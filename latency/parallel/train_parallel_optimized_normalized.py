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

class NormalizedReplayBuffer:
    """Replay buffer with reward normalization"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.reward_stats = {
            'mean': 0.0,
            'std': 1.0,
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0
        }
    
    def push(self, state, action, reward, next_state, done, mask, steps):
        # Update reward statistics
        self._update_reward_stats(reward)
        
        # Store normalized reward
        normalized_reward = self._normalize_reward(reward)
        self.buffer.append((state, action, normalized_reward, next_state, done, mask, steps))
    
    def _update_reward_stats(self, reward):
        """Update running statistics for reward normalization"""
        self.reward_stats['count'] += 1
        self.reward_stats['sum'] += reward
        self.reward_stats['sum_sq'] += reward * reward
        
        # Update mean and std
        if self.reward_stats['count'] > 1:
            self.reward_stats['mean'] = self.reward_stats['sum'] / self.reward_stats['count']
            variance = (self.reward_stats['sum_sq'] / self.reward_stats['count'] - 
                       self.reward_stats['mean'] ** 2)
            self.reward_stats['std'] = max(np.sqrt(variance), 1e-6)  # Avoid division by zero
    
    def _normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        if self.reward_stats['count'] < 10:  # Don't normalize until we have enough samples
            return reward
        
        # Check for NaN or infinite values
        if np.isnan(reward) or np.isinf(reward):
            print(f"Warning: Invalid reward value detected: {reward}")
            return 0.0
        
        # Avoid division by zero
        std = max(self.reward_stats['std'], 1e-6)
        normalized = (reward - self.reward_stats['mean']) / std
        
        # Clip to reasonable range
        return np.clip(normalized, -10, 10)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_reward_stats(self):
        return self.reward_stats.copy()

class AdaptiveRewardNormalizer:
    """Adaptive reward normalization based on alpha value"""
    def __init__(self, alpha_sensitivity=True):
        self.alpha_sensitivity = alpha_sensitivity
        self.diameter_scale = 1.0
        self.weight_scale = 1.0
        self.reward_history = deque(maxlen=1000)
        
    def normalize_reward_components(self, R_G, R_M_p, weight_sum, alpha, weight_scale=1.0):
        """Normalize individual reward components"""
        
        # Apply weight scaling
        scaled_weight = weight_sum * weight_scale
        
        # Estimate typical scales
        if len(self.reward_history) > 100:
            rewards = np.array(self.reward_history)
            reward_std = np.std(rewards)
            if reward_std > 1e-6:
                scale_factor = 1.0 / reward_std
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0
        
        # Scale components to similar magnitudes
        normalized_R_G = R_G * scale_factor
        normalized_R_M_p = R_M_p * scale_factor
        normalized_weight = scaled_weight * scale_factor
        
        # Combine with alpha weighting
        normalized_reward = (alpha * normalized_R_G + 
                            (1 - alpha) * normalized_R_M_p - 
                            normalized_weight)
        
        # Track reward history
        self.reward_history.append(normalized_reward)
        
        return normalized_reward

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
            return mask_data
        elif isinstance(mask_data, list):
            self.mask_buffer[:] = mask_data
            return self.mask_buffer
        else:
            return np.asarray(mask_data, dtype=np.int32)

def init_args():
    parser = argparse.ArgumentParser(description="Normalized Parallel DGRO Training")
    
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
    
    # Normalized reward parameters
    parser.add_argument("--reward_normalization", type=str, help="Reward normalization method", 
                        choices=['none', 'running', 'adaptive', 'clipping'], default='adaptive')
    parser.add_argument("--reward_clip", type=float, help="Reward clipping value", default=10.0)
    parser.add_argument("--weight_scale", type=float, help="Weight scaling factor for edge costs", default=1.0)
    
    # Parallel DGRO specific parameters
    parser.add_argument("--alpha_start", type=float, help="Starting alpha for reward combination", default=1.0)
    parser.add_argument("--alpha_end", type=float, help="Ending alpha for reward combination", default=1.0)
    parser.add_argument("--alpha_schedule", type=str, help="Alpha schedule type", 
                        choices=['linear', 'exponential', 'constant'], default='linear')
    parser.add_argument("--sync_freq", type=int, help="Synchronization frequency", default=1)
    parser.add_argument("--target_update_freq", type=int, help="Target network update frequency", default=1000000)
    
    # Other parameters
    parser.add_argument("--reward_mode", type=str, help="Reward mode", default='diameter')
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--num_sources", type=int, help="Number of diameter sources", default=3)
    parser.add_argument("--if_wandb", action='store_true', help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Create experiment name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_name = f'/pscratch/sd/s/swu264/SWARM/model/parallel_dgro_normalized_{current_time}'
    
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

class NormalizedGraphEnv(OptimizedGraphEnv):
    """Environment with normalized rewards"""
    
    def __init__(self, *args, normalization_method='adaptive', reward_clip=10.0, weight_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalization_method = normalization_method
        self.reward_clip = reward_clip
        self.weight_scale = weight_scale
        self.reward_normalizer = AdaptiveRewardNormalizer() if normalization_method == 'adaptive' else None
        
        # Running statistics for reward normalization
        self.reward_stats = {
            'global_mean': 0.0, 'global_std': 1.0, 'global_count': 0,
            'marginal_mean': 0.0, 'marginal_std': 1.0, 'marginal_count': 0,
            'weight_mean': 0.0, 'weight_std': 1.0, 'weight_count': 0
        }
    
    def _update_component_stats(self, R_G, R_M_p, weight_sum):
        """Update running statistics for reward components with numerical stability"""
        # Check for invalid values
        if np.isnan(R_G) or np.isinf(R_G):
            R_G = 0.0
        if np.isnan(R_M_p) or np.isinf(R_M_p):
            R_M_p = 0.0
        if np.isnan(weight_sum) or np.isinf(weight_sum):
            weight_sum = 100.0  # Default weight value
        
        # Update global reward stats
        self.reward_stats['global_count'] += 1
        n = self.reward_stats['global_count']
        delta = R_G - self.reward_stats['global_mean']
        self.reward_stats['global_mean'] += delta / n
        if n > 1:
            variance = ((n - 1) * self.reward_stats['global_std']**2 + delta * (R_G - self.reward_stats['global_mean'])) / n
            self.reward_stats['global_std'] = np.sqrt(max(variance, 1e-6))
        else:
            self.reward_stats['global_std'] = 1.0
        
        # Update marginal reward stats
        self.reward_stats['marginal_count'] += 1
        n = self.reward_stats['marginal_count']
        delta = R_M_p - self.reward_stats['marginal_mean']
        self.reward_stats['marginal_mean'] += delta / n
        if n > 1:
            variance = ((n - 1) * self.reward_stats['marginal_std']**2 + delta * (R_M_p - self.reward_stats['marginal_mean'])) / n
            self.reward_stats['marginal_std'] = np.sqrt(max(variance, 1e-6))
        else:
            self.reward_stats['marginal_std'] = 1.0
        
        # Update weight stats
        self.reward_stats['weight_count'] += 1
        n = self.reward_stats['weight_count']
        delta = weight_sum - self.reward_stats['weight_mean']
        self.reward_stats['weight_mean'] += delta / n
        if n > 1:
            variance = ((n - 1) * self.reward_stats['weight_std']**2 + delta * (weight_sum - self.reward_stats['weight_mean'])) / n
            self.reward_stats['weight_std'] = np.sqrt(max(variance, 1e-6))
        else:
            self.reward_stats['weight_std'] = 1.0
    
    def _normalize_reward_components(self, R_G, R_M_p, weight_sum):
        """Normalize reward components"""
        if self.normalization_method == 'none':
            return R_G, R_M_p, weight_sum
        
        # Avoid division by zero
        global_std = max(self.reward_stats['global_std'], 1e-6)
        marginal_std = max(self.reward_stats['marginal_std'], 1e-6)
        weight_std = max(self.reward_stats['weight_std'], 1e-6)
        
        # Normalize components
        normalized_R_G = (R_G - self.reward_stats['global_mean']) / global_std
        normalized_R_M_p = (R_M_p - self.reward_stats['marginal_mean']) / marginal_std
        normalized_weight = (weight_sum - self.reward_stats['weight_mean']) / weight_std
        
        return normalized_R_G, normalized_R_M_p, normalized_weight
    
    def step(self, action):
        """Step function with normalized rewards"""
        # Call parent step function
        state, original_rewards, done, info = super().step(action)
        
        if self.if_test or self.normalization_method == 'none':
            return state, original_rewards, done, info
        
        # Extract reward components
        R_G = info['global_reward']
        individual_rewards = info['individual_rewards']
        
        # Normalize rewards
        normalized_rewards = []
        for p in range(self.M):
            # Reconstruct marginal reward and weight
            start_node = self.start_id[p]
            end_node = action[p]
            
            # Get weight and apply scaling
            if self.initial_graph.has_edge(start_node, end_node):
                weight = self.initial_graph.edges[start_node, end_node]['weight'] * self.weight_scale
            else:
                weight = 1.0 * self.weight_scale
            
            # Estimate marginal reward (reverse calculation)
            original_reward = individual_rewards[p]
            R_M_p = (original_reward - self.alpha * R_G + weight) / (1 - self.alpha) if self.alpha != 1.0 else 0
            
            # Update statistics
            self._update_component_stats(R_G, R_M_p, weight)
            
            # Normalize components
            if self.normalization_method == 'adaptive':
                normalized_reward = self.reward_normalizer.normalize_reward_components(R_G, R_M_p, weight, self.alpha, self.weight_scale)
            else:  # running normalization
                norm_R_G, norm_R_M_p, norm_weight = self._normalize_reward_components(R_G, R_M_p, weight)
                normalized_reward = self.alpha * norm_R_G + (1 - self.alpha) * norm_R_M_p - norm_weight
            
            # Apply clipping
            if self.normalization_method == 'clipping':
                normalized_reward = np.clip(original_reward, -self.reward_clip, self.reward_clip)
            else:
                normalized_reward = np.clip(normalized_reward, -self.reward_clip, self.reward_clip)
            
            normalized_rewards.append(normalized_reward)
        
        # Update info with normalized rewards
        info['original_rewards'] = original_rewards
        info['normalized_rewards'] = normalized_rewards
        info['reward_stats'] = self.reward_stats.copy()
        
        return state, normalized_rewards, done, info

def test_parallel_agent_normalized(args, agent, env, num_tests=1):
    """Test function for normalized agent"""
    env.if_test = True
    total_diameter = 0
    
    state_manager = OptimizedStateManager(args.N, args.feature_dim)
    
    for test_run in range(num_tests):
        agent.generated_mask_count = 0
        masks, start_id, partition_start = agent.generate_masked_one_hot(args.N, args.M)
        
        state_dict = env.reset(if_test=True, start_id=start_id, test_id=0)
        state = state_manager.construct_state(state_dict)
        
        mask = state_manager.update_mask(state_dict['mask'])
        done = False
        step_count = 0
        
        while not done and step_count < args.N * args.K // args.M:
            if step_count % (args.N // args.M) == 0 and step_count != 0:
                masks, start_id, partition_start = agent.generate_masked_one_hot(args.N, args.M)
                mask = state_manager.update_mask([1] * args.N)
                for i in start_id:
                    mask[i] = 0
                env.start_id = start_id

            if (step_count + 1) % (args.N // args.M) == 0:
                actions = [partition_start[(i + 1) % args.M] for i in range(args.M)]
            else:
                actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                                K=args.K, epsilon=0)
            step_count += 1
            
            next_state_dict, rewards, done, info = env.step(actions)
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

def train_parallel_dgro_normalized(args):
    """Normalized training loop"""
    print("Starting Normalized Parallel DGRO Training...")
    print(f"Configuration: N={args.N}, K={args.K}, M={args.M}, Episodes={args.episodes}")
    print(f"Reward normalization: {args.reward_normalization}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create alpha schedule
    alpha_schedule = create_alpha_schedule(args)
    
    # Initialize normalized environment
    env = NormalizedGraphEnv(
        num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
        alpha=args.alpha_start, alpha_schedule=alpha_schedule,
        normalization_method=args.reward_normalization,
        reward_clip=args.reward_clip,
        weight_scale=args.weight_scale
    )
    test_env = NormalizedGraphEnv(
        num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
        alpha=args.alpha_start, alpha_schedule=alpha_schedule,
        normalization_method='none',  # No normalization for testing
        weight_scale=args.weight_scale
    )
    
    # Initialize replay buffer
    if args.reward_normalization == 'running':
        replay_buffer = NormalizedReplayBuffer(1000000)
    else:
        replay_buffer = NormalizedReplayBuffer(1000000)
    
    # Initialize state manager
    state_manager = OptimizedStateManager(args.N, args.feature_dim)
    
    # Initialize agent
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
    
    # Initialize Weights & Biases
    if args.if_wandb:
        wandb.init(
            project=f'parallel-dgro-normalized-{args.reward_normalization}',
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
            
            # Reset environment
            state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
            state = state_manager.construct_state(state_dict)
            mask = state_manager.update_mask(state_dict['mask'])
            
            # Episode variables
            episode_rewards = [0.0] * args.M
            done = False
            step_count = 0
            total_losses = []
            
            # Calculate epsilon
            epsilon = max((1 - episode / 150), 0.05)
            start_time = time.time()
            
            # Episode loop
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
                    actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                                    K=args.K, epsilon=epsilon)
                step_count += 1
                
                # Environment step
                next_state_dict, individual_rewards, done, info = env.step(actions)

                # Update episode rewards
                for p in range(args.M):
                    episode_rewards[p] += individual_rewards[p]
                
                # Prepare next state
                next_mask = state_manager.update_mask(next_state_dict['mask'])
                next_state = state_manager.construct_state(next_state_dict)
                
                # Store experience
                agent.store_experience(
                    states=[state] * args.M,
                    actions=actions,
                    rewards=individual_rewards,
                    next_states=[next_state] * args.M,
                    dones=done,
                    masks=[next_mask] * args.M,
                    steps=1
                )
                
                # Update Q-networks
                if len(replay_buffer) > 2000:
                    loss = agent.learn(args.bs)
                    if loss > 0:
                        total_losses.append(loss)
                        update_count += 1
                
                # Update state
                state = next_state
                mask = next_mask
                start_id = next_state_dict['start_id']
                
                # Logging
                if update_count % 25 == 0 and update_count != 0:
                    avg_loss = np.mean(total_losses) if total_losses else 0.0
                    avg_episode_reward = np.mean(episode_rewards)
                    
                    # Test the agent
                    test_diameter = test_parallel_agent_normalized(args, agent, test_env, num_tests=1)
                    
                    # Log results
                    log_msg = (f"Episode {episode:6d}: Steps={step_count:3d}, "
                            f"Avg Reward={avg_episode_reward:8.2f}, Loss={avg_loss:8.4f}, "
                            f"Test Diameter={test_diameter:8.2f}, Alpha={env.alpha:.3f}")
                    
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
                        log_data = {
                            'episode': episode,
                            'avg_episode_reward': avg_episode_reward,
                            'test_diameter': test_diameter,
                            'avg_loss': avg_loss,
                            'alpha': env.alpha,
                            'epsilon': epsilon,
                            'global_reward': info.get('global_reward', 0),
                            'step_count': step_count,
                        }
                        
                        # Add reward statistics if available
                        if 'reward_stats' in info:
                            stats = info['reward_stats']
                            log_data.update({
                                'reward_global_mean': stats.get('global_mean', 0),
                                'reward_global_std': stats.get('global_std', 1),
                                'reward_marginal_mean': stats.get('marginal_mean', 0),
                                'reward_marginal_std': stats.get('marginal_std', 1),
                                'reward_weight_mean': stats.get('weight_mean', 0),
                                'reward_weight_std': stats.get('weight_std', 1),
                            })
                        
                        wandb.log(log_data)
            
            episode_time = time.time() - start_time
    
    print(f"Training completed. Best diameter: {best_diameter:.2f}")
    return agent, env

if __name__ == '__main__':
    args = init_args()
    
    print("Normalized Parallel DGRO - DQN Training")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("=" * 50)
    
    agent, env = train_parallel_dgro_normalized(args) 