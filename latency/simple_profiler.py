#!/usr/bin/env python3
"""
Simple profiling script for train_parallel_improved.py
No argument conflicts - just profiles the training directly
"""

import cProfile
import pstats
import io
import time
import sys
import os
import torch
import numpy as np
import random

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_parallel_improved import (
    SharedParallelDQNAgent, GraphEnv, ReplayBuffer, 
    create_alpha_schedule, test_parallel_agent
)

def simple_profile_training():
    """Profile training with fixed parameters"""
    print("Simple Training Profiler")
    print("=" * 50)
    
    # Fixed parameters for profiling - no argparse conflicts
    class Args:
        def __init__(self):
            self.N = 100  # Smaller for profiling
            self.K = 3
            self.M = 2    # Fewer partitions
            self.episodes = 3  # Just 3 episodes
            self.bs = 32
            self.feature_dim = 4
            self.decay_gamma = 0.9
            self.lr = 5e-4
            self.alpha_start = 0.5
            self.alpha_end = 0.8
            self.alpha_schedule = 'linear'
            self.sync_freq = 1
            self.target_update_freq = 1000
            self.reward_mode = 'diameter'
            self.seed = 42
            self.num_sources = 1
            self.if_wandb = False
            self.experiment_name = '/tmp/profile_test'
    
    args = Args()
    
    # Create experiment directory
    os.makedirs(args.experiment_name, exist_ok=True)
    
    print(f"Profiling with: N={args.N}, M={args.M}, episodes={args.episodes}")
    
    # Profile the training function
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the training
    result = run_training_profile(args)
    
    pr.disable()
    
    # Save and display results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)  # Top 30 functions
    
    profile_output = s.getvalue()
    
    # Save to file
    with open('profile_results.txt', 'w') as f:
        f.write(profile_output)
    
    print("\ncProfile Results (Top 25):")
    print("=" * 80)
    lines = profile_output.split('\n')
    for line in lines[:30]:  # Print first 30 lines
        print(line)
    print("=" * 80)
    print("Full results saved to profile_results.txt")
    
    return result

def run_training_profile(args):
    """Run a simplified training loop for profiling"""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create alpha schedule
    alpha_schedule = create_alpha_schedule(args)
    
    # Initialize environment
    env = GraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                   alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    test_env = GraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                       alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    
    # Initialize shared replay buffer
    replay_buffer = ReplayBuffer(100000)  # Smaller buffer for profiling
    
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
    
    print("Starting training profile...")
    
    # Training loop
    update_count = 0
    
    for episode in range(args.episodes):
        print(f"Profiling episode {episode + 1}/{args.episodes}")
        
        # Update alpha schedule
        env.update_alpha(episode)
        
        # Generate partition masks and start IDs
        masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
        
        # Reset environment
        state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
        state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
        state = np.append(state, state_dict['degree'])
        
        mask = np.array(state_dict['mask'])
        
        # Episode variables
        episode_rewards = [0.0] * args.M
        done = False
        step_count = 0
        total_losses = []
        
        # Calculate epsilon
        epsilon = max((1 - update_count / 2000), 0.05)
        
        episode_start_time = time.time()
        
        # Episode loop
        while not done and step_count < args.N * args.K // args.M:
            
            # Regenerate partition masks periodically
            if step_count % (args.N // args.M) == 0 and step_count != 0:
                masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
                mask = [1 for i in range(args.N)]
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
            
            # Prepare next state
            next_mask = np.array(next_state_dict['mask'])
            next_state = np.append(next_state_dict['initial_graph'].flatten(), 
                                 next_state_dict['graph'].flatten())
            next_state = np.append(next_state, next_state_dict['degree'])
            
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
            if len(replay_buffer) > 100:  # Lower threshold for profiling
                loss = agent.learn(args.bs)
                if loss > 0:
                    total_losses.append(loss)
                    update_count += 1
            
            # Update state
            state = next_state
            mask = next_mask
            start_id = next_state_dict['start_id']
        
        episode_time = time.time() - episode_start_time
        
        # Quick evaluation
        avg_loss = np.mean(total_losses) if total_losses else 0.0
        avg_episode_reward = np.mean(episode_rewards)
        
        print(f"Episode {episode}: Steps={step_count}, "
              f"Avg Reward={avg_episode_reward:.2f}, Loss={avg_loss:.4f}, "
              f"Time={episode_time:.2f}s")
    
    print("Training profile complete!")
    return agent, env

def detailed_timing_profile():
    """Profile with detailed timing of each component"""
    print("\nDetailed Timing Profile")
    print("=" * 50)
    
    # Fixed parameters
    class Args:
        def __init__(self):
            self.N = 100
            self.K = 3
            self.M = 2
            self.episodes = 1
            self.bs = 32
            self.feature_dim = 4
            self.decay_gamma = 0.9
            self.lr = 5e-4
            self.alpha_start = 0.5
            self.alpha_end = 0.8
            self.alpha_schedule = 'linear'
            self.sync_freq = 1
            self.target_update_freq = 1000
            self.reward_mode = 'diameter'
            self.seed = 42
            self.num_sources = 1
            self.if_wandb = False
            self.experiment_name = '/tmp/timing_test'
    
    args = Args()
    os.makedirs(args.experiment_name, exist_ok=True)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_schedule = create_alpha_schedule(args)
    
    # Timing components
    timing_results = {}
    
    # Initialize environment
    start_time = time.time()
    env = GraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                   alpha=args.alpha_start, alpha_schedule=alpha_schedule)
    replay_buffer = ReplayBuffer(100000)
    timing_results['env_init'] = time.time() - start_time
    
    # Initialize agent
    start_time = time.time()
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
    timing_results['agent_init'] = time.time() - start_time
    
    # Time individual episode components
    masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
    
    # Environment reset
    start_time = time.time()
    state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
    state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
    state = np.append(state, state_dict['degree'])
    mask = np.array(state_dict['mask'])
    timing_results['env_reset'] = time.time() - start_time
    
    # Time 10 agent.act calls
    start_time = time.time()
    for i in range(10):
        actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                          K=args.K, epsilon=0.1)
    timing_results['agent_act_10x'] = time.time() - start_time
    
    # Time 10 environment steps
    start_time = time.time()
    for i in range(10):
        actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                          K=args.K, epsilon=0.1)
        next_state_dict, individual_rewards, done, info = env.step(actions)
        if done:
            state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
            state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
            state = np.append(state, state_dict['degree'])
            mask = np.array(state_dict['mask'])
        else:
            mask = np.array(next_state_dict['mask'])
            state = np.append(next_state_dict['initial_graph'].flatten(), 
                             next_state_dict['graph'].flatten())
            state = np.append(state, next_state_dict['degree'])
            start_id = next_state_dict['start_id']
    timing_results['env_step_10x'] = time.time() - start_time
    
    # Time learning updates
    if len(replay_buffer) > 100:
        start_time = time.time()
        for i in range(10):
            loss = agent.learn(args.bs)
        timing_results['agent_learn_10x'] = time.time() - start_time
    
    # Print results
    print("\nTiming Results:")
    print("=" * 50)
    for component, time_taken in timing_results.items():
        print(f"{component:<20}: {time_taken:.4f}s")
    
    return timing_results

if __name__ == '__main__':
    print("Choose profiling method:")
    print("1. cProfile (default)")
    print("2. Detailed timing")
    print("3. Both")
    
    choice = input("Enter choice (1-3, default=1): ").strip()
    
    if choice == '2':
        detailed_timing_profile()
    elif choice == '3':
        simple_profile_training()
        detailed_timing_profile()
    else:
        simple_profile_training() 