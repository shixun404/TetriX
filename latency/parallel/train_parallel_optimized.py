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
        self.meaningful_buffer = deque(maxlen=capacity)  # Buffer for non-zero diameter changes
        self.zero_change_buffer = deque(maxlen=capacity//4)  # Smaller buffer for zero changes
        self.meaningful_ratio = 0.8  # Ratio of meaningful experiences in each batch
        self.diameter_threshold = 1e-6  # Minimum diameter change to consider meaningful
    
    def push(self, state, action, reward, next_state, done, mask, steps, diameter_change=None):
        experience = (state, action, reward, next_state, done, mask, steps)
        self.buffer.append(experience)
        
        # Separate experiences based on diameter change
        if diameter_change is not None:
            if abs(diameter_change) > getattr(self, 'diameter_threshold', 1e-6):  # Non-zero diameter change
                self.meaningful_buffer.append(experience)
            else:  # Zero diameter change
                self.zero_change_buffer.append(experience)
        else:
            # Fallback: use reward as proxy for meaningful change
            if abs(reward) > 1e-6:
                self.meaningful_buffer.append(experience)
            else:
                self.zero_change_buffer.append(experience)
    
    def sample(self, batch_size):
        # Calculate how many samples from each buffer
        meaningful_count = int(batch_size * self.meaningful_ratio)
        zero_change_count = batch_size - meaningful_count
        
        # Ensure we don't exceed buffer sizes
        meaningful_count = min(meaningful_count, len(self.meaningful_buffer))
        zero_change_count = min(zero_change_count, len(self.zero_change_buffer))
        
        # If one buffer is insufficient, compensate with the other
        if meaningful_count < int(batch_size * self.meaningful_ratio):
            zero_change_count = min(batch_size - meaningful_count, len(self.zero_change_buffer))
        if zero_change_count < batch_size - meaningful_count:
            meaningful_count = min(batch_size - zero_change_count, len(self.meaningful_buffer))
        
        samples = []
        
        # Sample from meaningful experiences
        if meaningful_count > 0 and len(self.meaningful_buffer) > 0:
            samples.extend(random.sample(self.meaningful_buffer, meaningful_count))
        
        # Sample from zero-change experiences
        if zero_change_count > 0 and len(self.zero_change_buffer) > 0:
            samples.extend(random.sample(self.zero_change_buffer, zero_change_count))
        
        # If we still don't have enough samples, fall back to regular buffer
        if len(samples) < batch_size:
            remaining = batch_size - len(samples)
            if len(self.buffer) >= remaining:
                additional_samples = random.sample(self.buffer, remaining)
                samples.extend(additional_samples)
        
        return samples[:batch_size]  # Ensure exact batch size
    
    def __len__(self):
        return len(self.buffer)
    
    def get_buffer_stats(self):
        """Return statistics about buffer composition"""
        return {
            'total': len(self.buffer),
            'meaningful': len(self.meaningful_buffer),
            'zero_change': len(self.zero_change_buffer),
            'meaningful_ratio': len(self.meaningful_buffer) / max(len(self.buffer), 1)
        }

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
    parser.add_argument("--N", type=int, help="Number of nodes", default=100)
    parser.add_argument("--K", type=int, help="Degree constraint", default=3)
    parser.add_argument("--M", type=int, help="Number of partitions", default=1)
    
    # Training parameters
    parser.add_argument("--episodes", type=int, help="Number of episodes", default=300)
    parser.add_argument("--bs", type=int, help="Batch size", default=32)
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=4)
    parser.add_argument("--decay_gamma", type=float, help="Q decay", default=0.9)
    parser.add_argument("--lr", type=float, help="Learning rate", default=5e-4)
    
    # Parallel DGRO specific parameters
    parser.add_argument("--alpha_start", type=float, help="Starting alpha for reward combination", default=1.0)
    parser.add_argument("--alpha_end", type=float, help="Ending alpha for reward combination", default=1.0)
    parser.add_argument("--alpha_schedule", type=str, help="Alpha schedule type", 
                        choices=['linear', 'exponential', 'constant'], default='linear')
    parser.add_argument("--sync_freq", type=int, help="Synchronization frequency", default=1)
    parser.add_argument("--target_update_freq", type=int, help="Target network update frequency", default=1000000)
    parser.add_argument("--weight_scale", type=float, help="Weight scale", default=1.0)
    parser.add_argument("--reward_normalization", type=str, help="Reward normalization", default='none')
    # Other parameters
    parser.add_argument("--reward_mode", type=str, help="Reward mode", default='diameter')
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--num_sources", type=int, help="Number of diameter sources", default=3)
    parser.add_argument("--if_wandb", action='store_true', help="Use Weights & Biases logging")
    
    # Meaningful sampling parameters
    parser.add_argument("--meaningful_ratio", type=float, help="Ratio of meaningful experiences in batch", default=0.8)
    parser.add_argument("--diameter_threshold", type=float, help="Minimum diameter change to consider meaningful", default=1)
    
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
        agent.generated_mask_count = 0
        masks, start_id, partition_start = agent.generate_masked_one_hot(args.N, args.M)
        
        state_dict = env.reset(if_test=True, start_id=start_id, test_id=0)
        state = state_manager.construct_state(state_dict)
        
        mask = state_manager.update_mask(state_dict['mask'])
        done = False
        step_count = 0
        
        while not done and step_count < args.N * args.K // args.M:
            
            # Regenerate partition masks periodically
            if step_count % (args.N // args.M) == 0 and step_count != 0:
                masks, start_id, partition_start = agent.generate_masked_one_hot(args.N, args.M)
                mask = state_manager.update_mask([1] * args.N)
                for i in start_id:
                    mask[i] = 0
                env.start_id = start_id
            

            if (step_count + 1) % (args.N // args.M) == 0:
                actions = [partition_start[(i + 1) % args.M] for i in range(args.M)]

            else:
                # All partition agents choose actions
                actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                                K=args.K, epsilon=0)
            step_count += 1
            
            next_state_dict, rewards, done, info = env.step(actions)
            # print('next_state_dict: ', next_state_dict)
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
    
    # Initialize shared replay buffer with meaningful sampling
    replay_buffer = ReplayBuffer(1000000)
    replay_buffer.meaningful_ratio = args.meaningful_ratio
    replay_buffer.diameter_threshold = args.diameter_threshold
    
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
        target_update_freq=args.target_update_freq,
        K=args.K
    )
    
    # Initialize Weights & Biases if requested
    if args.if_wandb:
        wandb.init(
            project=f'parallel-dgro-normalized-{args.reward_normalization}-feature{args.feature_dim}-bs{args.bs}',
            config=vars(args),
            name=f'N{args.N}_M{args.M}_K{args.K}_alpha{args.alpha_start}_norm{args.reward_normalization}_ws{args.weight_scale}',
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
            agent.generated_mask_count = 0
            masks, start_id, partition_start = agent.generate_masked_one_hot(args.N, args.M)
            
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
            epsilon = max((1 - episode / 150 ), 0.05)
            
            start_time = time.time()
            
            # Episode loop
            while not done and step_count < args.N * args.K // args.M:
                
                # Regenerate partition masks periodically
                if step_count % (args.N // args.M) == 0 and step_count != 0:
                    masks, start_id, partition_start = agent.generate_masked_one_hot(args.N, args.M)
                    # Use efficient mask update
                    # print('start_id: ', start_id, 'partition_start: ', partition_start)
                    mask = state_manager.update_mask([1] * args.N)
                    for i in start_id:
                        mask[i] = 0
                    env.start_id = start_id
                # print('step_count: ', step_count)
                
                
                if (step_count + 1) % (args.N // args.M) == 0:
                    # print('start_id: ', start_id, 'partition_start: ', partition_start)
                    actions = [partition_start[(i + 1) % args.M] for i in range(args.M)]

                else:
                    # All partition agents choose actions
                    actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                                    K=args.K, epsilon=epsilon)
                step_count += 1
                # Environment step
                next_state_dict, individual_rewards, done, info = env.step(actions)

                diameter_change = info.get('diameter_before', 0) - info.get('diameter_after', 0)
                # Update episode rewards
                for p in range(args.M):
                    episode_rewards[p] += individual_rewards[p]
                # print('next_state_dict: ', next_state_dict['mask'])
                # Prepare next state efficiently
                next_mask = state_manager.update_mask(next_state_dict['mask'])
                next_state = state_manager.construct_state(next_state_dict)
                
                # Store experience for each partition in shared replay buffer with diameter change
                for p in range(args.M):
                    replay_buffer.push(
                        state=state,
                        action=actions[p],
                        reward=individual_rewards[p],
                        next_state=next_state,
                        done=done,
                        mask=next_mask,
                        steps=1,
                        diameter_change=diameter_change
                    )
                
                # Update Q-networks if enough experiences are available
                if len(replay_buffer) > 2000:
                    loss = agent.learn(args.bs)
                    if loss > 0:
                        total_losses.append(loss)
                        update_count += 1
                
                # Update state efficiently
                state = next_state
                mask = next_mask
                start_id = next_state_dict['start_id']
                
                if update_count % 25 == 0 and update_count != 0:
                    avg_loss = np.mean(total_losses) if total_losses else 0.0
                    avg_episode_reward = np.mean(episode_rewards)
                    
                    # Test the agent
                    test_diameter = test_parallel_agent_optimized(args, agent, test_env, num_tests=1)
                    
                    # Get buffer statistics
                    buffer_stats = replay_buffer.get_buffer_stats()
                    
                    # Log results with buffer statistics
                    log_msg = (f"Episode {episode:6d}: Steps={step_count:3d}, "
                            f"Avg Reward={avg_episode_reward:8.2f}, Loss={avg_loss:8.4f}, "
                            f"Test Diameter={test_diameter:8.2f}, Alpha={env.alpha:.3f}, "
                            f"Meaningful Ratio={buffer_stats['meaningful_ratio']:.3f}, "
                            f"Buffer(Total:{buffer_stats['total']}, "
                            f"Meaningful:{buffer_stats['meaningful']}, "
                            f"Zero:{buffer_stats['zero_change']})"
                            )
                    
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
                            'meaningful_ratio': buffer_stats['meaningful_ratio'],
                            'buffer_total': buffer_stats['total'],
                            'buffer_meaningful': buffer_stats['meaningful'],
                            'buffer_zero_change': buffer_stats['zero_change'],
                            # 'episode_time': episode_time
                        })
            
            episode_time = time.time() - start_time
            # assert 0
            # Logging and evaluation
            # if episode < 1000:
            #     avg_loss = np.mean(total_losses) if total_losses else 0.0
            #     avg_episode_reward = np.mean(episode_rewards)
                
            #     # Test the agent
            #     test_diameter = test_parallel_agent_optimized(args, agent, test_env, num_tests=1)
                
            #     # Log results
            #     log_msg = (f"Episode {episode:6d}: Steps={step_count:3d}, "
            #               f"Avg Reward={avg_episode_reward:8.2f}, Loss={avg_loss:8.4f}, "
            #               f"Test Diameter={test_diameter:8.2f}, Alpha={env.alpha:.3f}, "
            #               f"Time={episode_time:.2f}s")
                
            #     print(log_msg)
            #     log_file.write(log_msg + '\n')
            #     log_file.flush()
                
            #     # Save best model
            #     if test_diameter < best_diameter:
            #         best_diameter = test_diameter
            #         agent.save(episode)
            #         print(f"New best diameter: {best_diameter:.2f}")
                
            #     # Weights & Biases logging
            #     if args.if_wandb:
            #         wandb.log({
            #             'episode': episode,
            #             'avg_episode_reward': avg_episode_reward,
            #             'test_diameter': test_diameter,
            #             'avg_loss': avg_loss,
            #             'alpha': env.alpha,
            #             'epsilon': epsilon,
            #             'global_reward': info.get('global_reward', 0),
            #             'step_count': step_count,
            #             'episode_time': episode_time
            #         })
    
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