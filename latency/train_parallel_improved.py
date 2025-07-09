from DQNAgent_graph_parallel_shared import SharedParallelDQNAgent
from env_parallel_improved import GraphEnv, linear_alpha_schedule, exponential_alpha_schedule
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

def init_args():
    parser = argparse.ArgumentParser(description="Parallel DGRO Training")
    
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
    args.experiment_name = f'/pscratch/sd/s/swu264/SWARM/model/parallel_dgro_{current_time}'
    
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

def test_parallel_agent(args, agent, env, num_tests=1):
    """Test the parallel agent"""
    env.if_test = True
    total_diameter = 0
    
    for test_run in range(num_tests):
        # Generate partition masks and start IDs
        masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
        
        state_dict = env.reset(if_test=True, start_id=start_id, test_id=0)
        state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
        state = np.append(state, state_dict['degree'])
        
        mask = np.array(state_dict['mask'])
        done = False
        step_count = 0
        
        while not done and step_count < args.N * args.K // args.M:
            
            # Key: Regenerate partition masks periodically for connectivity
            if step_count % (args.N // args.M) == 0 and step_count != 0:
                # Use ORIGINAL strategy for testing too (same as test_parallel.py)
                masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
                mask = [1 for i in range(args.N)]  # Reset to all available
                for i in start_id:
                    mask[i] = 0  # Only exclude start_ids
                env.start_id = start_id
            step_count += 1

            # Get actions from all partition agents
            actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                              K=args.K, epsilon=0.0)  # No exploration during testing
            
            next_state_dict, rewards, done, info = env.step(actions)
            
            # Update state and masks
            mask = np.array(next_state_dict['mask'])
            state = np.append(next_state_dict['initial_graph'].flatten(), next_state_dict['graph'].flatten())
            state = np.append(state, next_state_dict['degree'])
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

def train_parallel_dgro(args):
    """
    Main training loop implementing Algorithm 1: Parallel DGRO - DQN Training with Two-Level Rewards
    """
    print("Starting Parallel DGRO Training...")
    print(f"Configuration: N={args.N}, K={args.K}, M={args.M}, Episodes={args.episodes}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create alpha schedule
    alpha_schedule = create_alpha_schedule(args)
    
    # Initialize environment with alpha scheduling
    env = GraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                   alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    test_env = GraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                   alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    
    # Initialize shared replay buffer
    replay_buffer = ReplayBuffer(1000000)
    
    # Initialize parallel DQN agent with shared Q-network across partitions
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
            project='parallel-dgro',
            config=vars(args),
            name=f'parallel_dgro_N{args.N}_M{args.M}',
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
            
            # Reset environment with proper start_id
            state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
            state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
            state = np.append(state, state_dict['degree'])
            
            mask = np.array(state_dict['mask'])
            
            # Episode variables
            episode_rewards = [0.0] * args.M
            done = False
            step_count = 0
            total_losses = []
            
            # Calculate epsilon for exploration
            # epsilon = max(0.05, 1.0 - episode / (args.episodes * 0.3))
            epsilon = max((1 - update_count / 2000), 0.05)
            
            start_time = time.time()
            
            # Episode loop
            while not done and step_count < args.N * args.K // args.M:
                
                # CRITICAL: Regenerate partition masks periodically for connectivity
                # This ensures different partitions eventually connect to each other
                if step_count % (args.N // args.M) == 0 and step_count != 0:
                    # ORIGINAL STRATEGY: Reset mask completely to avoid "no valid actions"
                    # This temporarily relaxes degree constraints, which will be reapplied in env.step()
                    masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
                    mask = [1 for i in range(args.N)]  # Reset to all available
                    for i in start_id:
                        mask[i] = 0  # Only exclude start_ids
                    env.start_id = start_id
                    # if episode % 100 == 0:  # Only print occasionally to avoid spam
                    #     print(f"Episode {episode}, Step {step_count}: Regenerated partitions (original strategy)")
                    #     print(f"  New start_ids: {start_id}")
                    #     print(f"  Reset mask to: {sum(mask)} available actions")
                step_count += 1
                
                # All partition agents choose actions in parallel
                actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                                  K=args.K, epsilon=epsilon)
                
                # Environment step with two-level rewards
                next_state_dict, individual_rewards, done, info = env.step(actions)

                # Update episode rewards
                for p in range(args.M):
                    episode_rewards[p] += individual_rewards[p]
                
                # Prepare next state
                next_mask = np.array(next_state_dict['mask'])
                next_state = np.append(next_state_dict['initial_graph'].flatten(), 
                                     next_state_dict['graph'].flatten())
                next_state = np.append(next_state, next_state_dict['degree'])
                
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
                    # print(f"Step {step_count} loss: {loss:.4f}")

                    # # Logging and evaluation
                    # if update_count % 10 == 0:
                    #     avg_loss = np.mean(total_losses) if total_losses else 0.0
                    #     avg_episode_reward = np.mean(episode_rewards)
                        
                    #     # Test the agent
                    #     test_diameter = test_parallel_agent(args, agent, test_env, num_tests=1  )
                        
                    #     # Log results
                    #     log_msg = (f"Episode {episode:6d}: Steps={step_count:3d}, "
                    #             f"Avg Reward={avg_episode_reward:8.2f}, Loss={avg_loss:8.4f}, "
                    #             f"Test Diameter={test_diameter:8.2f}, Alpha={env.alpha:.3f}, "
                    #             f"Time={episode_time:.2f}s")
                        
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
                test_diameter = test_parallel_agent(args, agent, test_env, num_tests=1  )
                
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
                
                # Update state efficiently
                state = next_state
                mask = next_mask
                start_id = next_state_dict['start_id']
            
    
    print(f"Training completed. Best diameter: {best_diameter:.2f}")
    return agent, env

if __name__ == '__main__':
    args = init_args()
    
    print("Parallel DGRO - DQN Training with Shared Model")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("=" * 50)
    
    agent, env = train_parallel_dgro(args) 