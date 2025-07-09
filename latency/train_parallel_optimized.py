from DQNAgent_graph_parallel_shared import SharedParallelDQNAgent
from env_parallel_optimized import OptimizedGraphEnv, linear_alpha_schedule, exponential_alpha_schedule
from collections import deque
import random
import argparse
import os
import torch
import numpy as np
import wandb
import time
from datetime import datetime
import json

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, steps):
        self.buffer.append((state, action, reward, next_state, done, mask, steps))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class OptimizedStateManager:
    """Manages state arrays efficiently to avoid np.array() and np.append() bottlenecks"""
    
    def __init__(self, num_nodes, feature_dim):
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # Pre-allocate arrays
        self.state_buffer = np.zeros(2 * num_nodes * num_nodes + num_nodes, dtype=np.float32)
        self.mask_buffer = np.zeros(num_nodes, dtype=np.int32)
        self.degree_buffer = np.zeros(num_nodes, dtype=np.int32)
        
        # Calculate offsets for efficient array construction
        self.initial_graph_size = num_nodes * num_nodes
        self.current_graph_size = num_nodes * num_nodes
        self.degree_size = num_nodes
        
        self.initial_graph_offset = 0
        self.current_graph_offset = self.initial_graph_size
        self.degree_offset = self.initial_graph_size + self.current_graph_size
    
    def construct_state(self, state_dict):
        """Construct state array efficiently without np.append()"""
        # Get arrays from dict
        initial_graph = state_dict['initial_graph']
        current_graph = state_dict['graph']
        degree = state_dict['degree']
        
        # Fill pre-allocated buffer efficiently
        self.state_buffer[self.initial_graph_offset:self.current_graph_offset] = initial_graph.flatten()
        self.state_buffer[self.current_graph_offset:self.degree_offset] = current_graph.flatten()
        self.state_buffer[self.degree_offset:] = degree
        
        return self.state_buffer.copy()
    
    def update_mask(self, mask_data):
        """Update mask efficiently without np.array()"""
        if isinstance(mask_data, np.ndarray):
            return mask_data  # Already numpy array
        elif isinstance(mask_data, list):
            # Convert list to numpy array efficiently
            self.mask_buffer[:] = mask_data
            return self.mask_buffer
        else:
            return np.asarray(mask_data, dtype=np.int32)

def init_args():
    parser = argparse.ArgumentParser(description="Optimized Parallel DGRO Training")
    
    # Graph parameters
    parser.add_argument("--N", type=int, help="Number of nodes", default=400)
    parser.add_argument("--K", type=int, help="Degree constraint", default=3)
    parser.add_argument("--M", type=int, help="Number of partitions", default=4)
    
    # Training parameters
    parser.add_argument("--episodes", type=int, help="Number of episodes", default=50000)
    parser.add_argument("--bs", type=int, help="Batch size", default=32)
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=4)
    parser.add_argument("--decay_gamma", type=float, help="Q decay", default=0.9)
    parser.add_argument("--lr", type=float, help="Learning rate", default=5e-4)
    
    # Parallel DGRO specific parameters
    parser.add_argument("--alpha_start", type=float, help="Starting alpha for reward combination", default=0.5)
    parser.add_argument("--alpha_end", type=float, help="Ending alpha for reward combination", default=0.8)
    parser.add_argument("--alpha_schedule", type=str, help="Alpha schedule type", 
                        choices=['linear', 'exponential', 'constant'], default='linear')
    parser.add_argument("--sync_freq", type=int, help="Synchronization frequency", default=1)
    parser.add_argument("--target_update_freq", type=int, help="Target network update frequency", default=1000)
    
    # Other parameters
    parser.add_argument("--reward_mode", type=str, help="Reward mode", default='diameter')
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--num_sources", type=int, help="Number of diameter sources", default=1)
    parser.add_argument("--if_wandb", action='store_true', help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Create experiment name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_name = f'/pscratch/sd/s/swu264/SWARM/model/parallel_dgro_optimized_{current_time}'
    
    # Create experiment directory
    os.makedirs(args.experiment_name, exist_ok=True)
    
    # Save configuration
    config_file_path = os.path.join(args.experiment_name, 'config.json')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    return args

def create_alpha_schedule(args):
    """Create alpha scheduling function based on arguments"""
    if args.alpha_schedule == 'linear':
        return lambda episode: linear_alpha_schedule(episode, args.alpha_start, args.alpha_end, args.episodes)
    elif args.alpha_schedule == 'exponential':
        return lambda episode: exponential_alpha_schedule(episode, args.alpha_start, args.alpha_end)
    else:  # constant
        return lambda episode: args.alpha_start

def test_parallel_agent_optimized(args, agent, env, num_tests=1):
    """Optimized test function for parallel agent"""
    env.if_test = True
    total_diameter = 0
    
    # Pre-allocate state manager
    state_manager = OptimizedStateManager(args.N, args.feature_dim)
    
    for test_run in range(num_tests):
        # Generate partition masks and start IDs
        masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
        
        state_dict = env.reset(if_test=True, start_id=start_id, test_id=0)
        state = state_manager.construct_state(state_dict)
        
        mask = state_manager.update_mask(state_dict['mask'])
        done = False
        step_count = 0
        
        while not done and step_count < args.N * args.K // args.M:
            
            # Regenerate partition masks periodically
            if step_count % (args.N // args.M) == 0 and step_count != 0:
                masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
                mask = state_manager.update_mask([1] * args.N)
                for i in start_id:
                    mask[i] = 0
                env.start_id = start_id
            step_count += 1

            # Get actions from all partition agents
            actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                              K=args.K, epsilon=0.0)
            
            next_state_dict, rewards, done, info = env.step(actions)
            
            # Update state and masks efficiently
            mask = state_manager.update_mask(next_state_dict['mask'])
            state = state_manager.construct_state(next_state_dict)
            start_id = next_state_dict['start_id']
            
            if done:
                break
        
        # Compute final diameter
        try:
            import networkx as nx
            diameter = nx.diameter(env.graph, weight='weight')
            total_diameter += diameter
        except:
            total_diameter += float('inf')
    
    avg_diameter = total_diameter / num_tests
    env.if_test = False
    return avg_diameter

def train_parallel_dgro_optimized(args):
    """
    Optimized training loop with eliminated bottlenecks
    """
    print("Starting Optimized Parallel DGRO Training...")
    print(f"Configuration: N={args.N}, K={args.K}, M={args.M}, Episodes={args.episodes}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create alpha schedule
    alpha_schedule = create_alpha_schedule(args)
    
    # Initialize optimized environment
    env = OptimizedGraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                           alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    test_env = OptimizedGraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                                alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    
    # Initialize shared replay buffer
    replay_buffer = ReplayBuffer(1000000)
    
    # Initialize optimized state manager
    state_manager = OptimizedStateManager(args.N, args.feature_dim)
    
    # Initialize parallel DQN agent
    agent = SharedParallelDQNAgent(
        M=args.M,
        state_size=args.feature_dim,
        action_size=args.N,
        replay_buffer=replay_buffer,
        decay_gamma=args.decay_gamma,
        device=device,
        experiment_name=args.experiment_name,
        sync_freq=args.sync_freq,
        target_update_freq=args.target_update_freq
    )
    
    # Initialize Weights & Biases if requested
    if args.if_wandb:
        wandb.init(
            project='parallel-dgro-optimized',
            config=vars(args),
            name=f'parallel_dgro_optimized_N{args.N}_M{args.M}',
            save_code=True
        )
    
    # Training loop
    log_file_path = os.path.join(args.experiment_name, 'training.log')
    with open(log_file_path, 'w') as log_file:
        best_diameter = float('inf')
        update_count = 0
        
        for episode in range(args.episodes):
            # Update alpha schedule
            env.update_alpha(episode)
            
            # Generate partition masks and start IDs
            masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
            
            # Reset environment and construct state efficiently
            state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
            state = state_manager.construct_state(state_dict)
            
            mask = state_manager.update_mask(state_dict['mask'])
            
            # Episode variables
            episode_rewards = [0.0] * args.M
            done = False
            step_count = 0
            total_losses = []
            
            # Calculate epsilon for exploration
            epsilon = max((1 - update_count / 2000), 0.05)
            
            start_time = time.time()
            
            # Episode loop
            while not done and step_count < args.N * args.K // args.M:
                
                # Regenerate partition masks periodically
                if step_count % (args.N // args.M) == 0 and step_count != 0:
                    masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
                    # Use efficient mask update
                    mask = state_manager.update_mask([1] * args.N)
                    for i in start_id:
                        mask[i] = 0
                    env.start_id = start_id
                
                step_count += 1
                
                # All partition agents choose actions
                actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                                  K=args.K, epsilon=epsilon)
                
                # Environment step
                next_state_dict, individual_rewards, done, info = env.step(actions)

                # Update episode rewards
                for p in range(args.M):
                    episode_rewards[p] += individual_rewards[p]
                
                # Prepare next state efficiently
                next_mask = state_manager.update_mask(next_state_dict['mask'])
                next_state = state_manager.construct_state(next_state_dict)
                
                # Store experience for each partition in shared replay buffer
                agent.store_experience(
                    states=[state] * args.M,
                    actions=actions,
                    rewards=individual_rewards,
                    next_states=[next_state] * args.M,
                    dones=done,
                    masks=[next_mask] * args.M,
                    steps=1
                )
                
                # Update Q-networks if enough experiences are available
                if len(replay_buffer) > 4000:
                    loss = agent.learn(args.bs)
                    if loss > 0:
                        total_losses.append(loss)
                        update_count += 1
                
                # Update state efficiently
                state = next_state
                mask = next_mask
                start_id = next_state_dict['start_id']
            
            episode_time = time.time() - start_time
            
            # Logging and evaluation
            if episode < 1000:
                avg_loss = np.mean(total_losses) if total_losses else 0.0
                avg_episode_reward = np.mean(episode_rewards)
                
                # Test the agent
                test_diameter = test_parallel_agent_optimized(args, agent, test_env, num_tests=1)
                
                # Log results
                log_msg = (f"Episode {episode:6d}: Steps={step_count:3d}, "
                          f"Avg Reward={avg_episode_reward:8.2f}, Loss={avg_loss:8.4f}, "
                          f"Test Diameter={test_diameter:8.2f}, Alpha={env.alpha:.3f}, "
                          f"Time={episode_time:.2f}s")
                
                print(log_msg)
                log_file.write(log_msg + '\n')
                log_file.flush()
                
                # Save best model
                if test_diameter < best_diameter:
                    best_diameter = test_diameter
                    agent.save(episode)
                    print(f"New best diameter: {best_diameter:.2f}")
                
                # Weights & Biases logging
                if args.if_wandb:
                    wandb.log({
                        'episode': episode,
                        'avg_episode_reward': avg_episode_reward,
                        'test_diameter': test_diameter,
                        'avg_loss': avg_loss,
                        'alpha': env.alpha,
                        'epsilon': epsilon,
                        'global_reward': info.get('global_reward', 0),
                        'step_count': step_count,
                        'episode_time': episode_time
                    })
    
    print(f"Training completed. Best diameter: {best_diameter:.2f}")
    return agent, env

if __name__ == '__main__':
    args = init_args()
    
    print("Optimized Parallel DGRO - DQN Training")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("=" * 50)
    
    agent, env = train_parallel_dgro_optimized(args) 