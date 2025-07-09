from DQNAgent_graph_parallel_shared import SharedParallelDQNAgent
from env_parallel_connectivity_fixed import GraphEnv, linear_alpha_schedule, exponential_alpha_schedule
from collections import deque
import random
import argparse
import os
import torch
import numpy as np
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

def test_connectivity_fixed(args):
    """Test the connectivity fix"""
    print("Testing Connectivity Fix...")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment WITHOUT ring initialization
    env = GraphEnv(num_nodes=args.N, K=args.K, M=args.M, num_sources=args.num_sources, 
                   alpha=args.alpha_start)
    
    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(1000000)
    agent = SharedParallelDQNAgent(
        M=args.M,
        state_size=args.feature_dim,
        action_size=args.N,
        replay_buffer=replay_buffer,
        decay_gamma=args.decay_gamma,
        device=device,
        experiment_name="connectivity_test"
    )
    
    # Generate partition masks and start IDs
    masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
    print(f"Initial start_ids: {start_id}")
    
    # Reset environment
    state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
    state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
    state = np.append(state, state_dict['degree'])
    
    mask = np.array(state_dict['mask'])
    
    # Episode variables
    done = False
    step_count = 0
    epsilon = 0.1
    
    print(f"Starting graph: {env.graph.number_of_edges()} edges")
    print(f"Testing with M={args.M} partitions")
    print()
    
    # Simulation loop with partition reallocation
    while not done and step_count < args.N * args.K // args.M:
        step_count += 1
        
        # CRITICAL: Regenerate partition masks periodically for connectivity
        # This ensures different partitions eventually connect to each other
        if step_count % (args.N // args.M) == 0 and step_count != 0:
            masks, start_id = agent.generate_masked_one_hot(args.N, args.M)
            mask = [1 for i in range(args.N)]
            for i in start_id:
                mask[i] = 0
            env.start_id = start_id
            print(f"Step {step_count}: Regenerated partitions")
            print(f"  New start_ids: {start_id}")
            print(f"  Current graph edges: {env.graph.number_of_edges()}")
            
            # Check current connectivity
            try:
                import networkx as nx
                diameter = nx.diameter(env.graph, weight='weight')
                print(f"  Current diameter: {diameter:.2f}")
            except:
                print(f"  Graph is disconnected (infinite diameter)")
            print()
        
        # All partition agents choose actions
        actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                          K=args.K, epsilon=epsilon)
        
        # Environment step
        next_state_dict, individual_rewards, done, info = env.step(actions)
        
        # Print step information
        if step_count % 10 == 0 or step_count <= 5:
            print(f"Step {step_count:3d}: Actions={actions}, Rewards={[f'{r:.3f}' for r in individual_rewards]}")
            print(f"         Edges={env.graph.number_of_edges():3d}, Connectivity={info['connectivity_check']}")
            if info['diameter_after'] != float('inf'):
                print(f"         Diameter={info['diameter_after']:.2f}")
            else:
                print(f"         Diameter=inf (disconnected)")
        
        # Update state
        mask = np.array(next_state_dict['mask'])
        state = np.append(next_state_dict['initial_graph'].flatten(), 
                         next_state_dict['graph'].flatten())
        state = np.append(state, next_state_dict['degree'])
        start_id = next_state_dict['start_id']
    
    # Final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Total steps: {step_count}")
    print(f"Final graph edges: {env.graph.number_of_edges()}")
    
    try:
        import networkx as nx
        final_diameter = nx.diameter(env.graph, weight='weight')
        print(f"Final diameter: {final_diameter:.2f}")
        print("✓ GRAPH IS CONNECTED!")
    except:
        print("✗ GRAPH IS DISCONNECTED (infinite diameter)")
    
    return env.graph

def init_args():
    parser = argparse.ArgumentParser(description="Connectivity Test")
    
    # Graph parameters
    parser.add_argument("--N", type=int, help="Number of nodes", default=400)
    parser.add_argument("--K", type=int, help="Degree constraint", default=3)
    parser.add_argument("--M", type=int, help="Number of partitions", default=4)
    
    # Training parameters
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=4)
    parser.add_argument("--decay_gamma", type=float, help="Q decay", default=0.9)
    
    # Parallel DGRO specific parameters
    parser.add_argument("--alpha_start", type=float, help="Starting alpha", default=0.5)
    
    # Other parameters
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--num_sources", type=int, help="Number of diameter sources", default=1)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init_args()
    
    print("Connectivity Fix Test")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("=" * 50)
    
    final_graph = test_connectivity_fixed(args)
    
    print("\nTest completed!") 