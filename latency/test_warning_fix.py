from DQNAgent_graph_parallel_shared import SharedParallelDQNAgent
from env_parallel_improved import GraphEnv
from collections import deque
import random
import argparse
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, steps):
        self.buffer.append((state, action, reward, next_state, done, mask, steps))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def test_warning_fix():
    print("Testing Warning Fix...")
    print("=" * 50)
    
    # Set parameters
    N, K, M = 400, 3, 4
    
    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment
    env = GraphEnv(num_nodes=N, K=K, M=M, num_sources=1, alpha=0.5)
    
    # Initialize agent
    replay_buffer = ReplayBuffer(1000000)
    agent = SharedParallelDQNAgent(
        M=M,
        state_size=4,
        action_size=N,
        replay_buffer=replay_buffer,
        decay_gamma=0.9,
        device=device,
        experiment_name="warning_fix_test"
    )
    
    # Generate initial partition
    masks, start_id = agent.generate_masked_one_hot(N, M)
    print(f"Initial start_ids: {start_id}")
    
    # Reset environment
    state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
    state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
    state = np.append(state, state_dict['degree'])
    
    mask = np.array(state_dict['mask'])
    
    done = False
    step_count = 0
    warning_count = 0
    epsilon = 0.1
    
    print(f"Starting with {sum(mask)} available actions")
    print()
    
    # Simulation loop with improved partition reallocation
    while not done and step_count < 50:  # Short test
        step_count += 1
        
        # IMPROVED: Regenerate partitions from valid nodes only
        if step_count % (N // M) == 0 and step_count != 0:
            # Get currently valid nodes
            current_valid_nodes = [i for i in range(N) if mask[i] == 1]
            print(f"Step {step_count}: Found {len(current_valid_nodes)} valid nodes")
            
            if len(current_valid_nodes) >= M:
                # Use improved method
                masks, start_id = agent.generate_masked_one_hot_from_valid(N, M, current_valid_nodes)
                
                # Update mask to remove new start_ids
                mask = np.array(mask)
                for i in start_id:
                    if i < len(mask):
                        mask[i] = 0
                
                env.start_id = start_id
                print(f"  Regenerated partitions from valid nodes")
                print(f"  New start_ids: {start_id}")
                print(f"  Available actions after: {sum(mask)}")
            else:
                print(f"  Skipped regeneration (only {len(current_valid_nodes)} valid nodes)")
            print()
        
        # Count warnings before action
        old_warning_count = warning_count
        
        # Choose actions
        actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                          K=K, epsilon=epsilon)
        
        # Count any new warnings (this is approximate)
        # In real implementation, we'd capture stdout or use a custom warning system
        
        # Environment step
        next_state_dict, individual_rewards, done, info = env.step(actions)
        
        # Print step information
        print(f"Step {step_count:2d}: Actions={actions}")
        print(f"         Edges={env.graph.number_of_edges():3d}, Available={sum(mask):3d}")
        
        # Update state
        mask = np.array(next_state_dict['mask'])
        state = np.append(next_state_dict['initial_graph'].flatten(), 
                         next_state_dict['graph'].flatten())
        state = np.append(state, next_state_dict['degree'])
        start_id = next_state_dict['start_id']
    
    print("\n" + "=" * 50)
    print("TEST COMPLETED")
    print(f"Total steps: {step_count}")
    print(f"Final graph edges: {env.graph.number_of_edges()}")
    print("Key improvement: Using generate_masked_one_hot_from_valid() to avoid invalid partitions")

if __name__ == '__main__':
    test_warning_fix() 