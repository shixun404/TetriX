#!/usr/bin/env python3
"""
Test script to verify the shared model parallel DGRO implementation
"""

import numpy as np
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_shared_model():
    """Test the shared model implementation"""
    print("Testing Shared Model Parallel DGRO...")
    print("=" * 50)
    
    try:
        # Import modules
        from env_parallel_improved import GraphEnv
        from DQNAgent_graph_parallel_shared import SharedParallelDQNAgent
        
        # Mock replay buffer
        class MockReplayBuffer:
            def __init__(self, capacity):
                self.capacity = capacity
                self.buffer = []
            def push(self, *args):
                self.buffer.append(args)
            def __len__(self):
                return len(self.buffer)
            def sample(self, batch_size):
                import random
                return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Test parameters
        N, K, M = 20, 2, 4
        feature_dim = 4
        
        # Create environment
        print(f"Creating environment: N={N}, K={K}, M={M}")
        env = GraphEnv(num_nodes=N, K=K, M=M, alpha=0.5)
        
        # Create shared agent
        print(f"Creating shared agent with feature_dim={feature_dim}")
        replay_buffer = MockReplayBuffer(1000)
        agent = SharedParallelDQNAgent(
            M=M,
            state_size=feature_dim,
            action_size=N,
            replay_buffer=replay_buffer,
            decay_gamma=0.9,
            device='cpu',  # Use CPU for testing
            experiment_name='test_shared'
        )
        
        print("✅ Agent and environment created successfully")
        print(f"   - Shared model device: {agent.model.device}")
        print(f"   - Number of partitions: {agent.M}")
        print(f"   - Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
        
        # Test episode simulation
        print("\nRunning test episode...")
        
        # Generate masks and start IDs
        masks, start_id = agent.generate_masked_one_hot(N, M)
        print(f"   - Start IDs: {start_id}")
        
        # Reset environment
        state_dict = env.reset(if_test=False, start_id=start_id, test_id=0)
        state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())
        state = np.append(state, state_dict['degree'])
        
        mask = np.array(state_dict['mask'])
        print(f"   - Initial state shape: {state.shape}")
        print(f"   - Initial diameter: {env.compute_diameter(env.graph)}")
        
        # Test action selection
        actions = agent.act(state, env.graph.degree, env.graph, mask, masks, start_id, 
                          K=K, epsilon=0.5)
        print(f"   - Actions selected: {actions}")
        
        # Test environment step
        next_state_dict, rewards, done, info = env.step(actions)
        print(f"   - Rewards: {rewards}")
        print(f"   - Global reward: {info['global_reward']}")
        print(f"   - Alpha: {info['alpha']}")
        
        # Test experience storage
        next_state = np.append(next_state_dict['initial_graph'].flatten(), 
                             next_state_dict['graph'].flatten())
        next_state = np.append(next_state, next_state_dict['degree'])
        next_mask = np.array(next_state_dict['mask'])
        
        agent.store_experience(
            states=[state] * M,
            actions=actions,
            rewards=rewards,
            next_states=[next_state] * M,
            dones=done,
            masks=[next_mask] * M,
            steps=1
        )
        
        print(f"   - Experience buffer size: {len(replay_buffer)}")
        
        # Test learning (if we have enough experiences)
        if len(replay_buffer) >= 8:  # Small batch for testing
            loss = agent.learn(batch_size=4)
            print(f"   - Learning loss: {loss:.4f}")
        else:
            print(f"   - Not enough experiences for learning ({len(replay_buffer)} < 8)")
        
        # Test save/load
        agent.save()
        print("   - Model saved successfully")
        
        agent.load()
        print("   - Model loaded successfully")
        
        print("\n✅ All tests passed!")
        print("🎉 Shared model implementation is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_shared_model()
    
    if success:
        print("\n" + "=" * 50)
        print("Ready to run training with shared model:")
        print("python train_parallel_improved.py --N 20 --K 2 --M 4 --episodes 1000")
        print("python train_parallel_improved.py --N 400 --K 3 --M 4 --episodes 50000")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 