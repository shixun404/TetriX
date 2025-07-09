#!/usr/bin/env python3
"""
Test script to verify the parallel DGRO fixes work correctly
"""

import numpy as np
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment_initialization():
    """Test that the environment initializes correctly with ring graph"""
    print("Testing environment initialization...")
    try:
        from env_parallel_improved import GraphEnv
        
        # Test environment creation
        env = GraphEnv(num_nodes=20, K=3, M=4, alpha=0.5)
        
        # Test reset with proper start_id handling
        state_dict = env.reset(if_test=False, start_id=None, test_id=0)
        
        print("✅ Environment initialization successful")
        print(f"   - Graph has {env.graph.number_of_nodes()} nodes")
        print(f"   - Graph has {env.graph.number_of_edges()} edges")
        print(f"   - Start IDs: {env.start_id}")
        
        # Test that graph is connected (non-infinite diameter)
        try:
            diameter = env.compute_diameter(env.graph)
            print(f"   - Initial diameter: {diameter}")
            if diameter == float('inf'):
                print("❌ Graph is still disconnected!")
                return False
        except Exception as e:
            print(f"❌ Diameter computation failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_agent_creation():
    """Test that the parallel agent creates correctly"""
    print("\nTesting agent creation...")
    try:
        # Mock replay buffer for testing
        class MockReplayBuffer:
            def __init__(self, capacity):
                self.capacity = capacity
                self.buffer = []
            def push(self, *args):
                pass
            def __len__(self):
                return 0
        
        from DQNAgent_graph_parallel_improved import ParallelDQNAgent
        
        # Test agent creation
        replay_buffer = MockReplayBuffer(1000)
        agent = ParallelDQNAgent(
            M=4,
            state_size=64,
            action_size=20,
            replay_buffer=replay_buffer,
            decay_gamma=0.9,
            device='cpu',  # Use CPU for testing
            experiment_name='test_experiment'
        )
        
        print("✅ Agent creation successful")
        print(f"   - Number of partitions: {agent.M}")
        print(f"   - Number of Q-networks: {len(agent.models)}")
        print(f"   - Number of target networks: {len(agent.target_models)}")
        
        # Test partition mask generation
        masks, start_ids = agent.generate_masked_one_hot(20, 4)
        print(f"   - Generated {len(masks)} partition masks")
        print(f"   - Start IDs: {start_ids}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_two_level_rewards():
    """Test that two-level rewards are computed correctly"""
    print("\nTesting two-level reward computation...")
    try:
        from env_parallel_improved import GraphEnv
        
        env = GraphEnv(num_nodes=12, K=2, M=3, alpha=0.5)
        state_dict = env.reset(if_test=False, start_id=[0, 4, 8], test_id=0)
        
        # Simulate parallel actions
        actions = [1, 5, 9]  # Each partition chooses an action
        
        # Test step function
        next_state_dict, rewards, done, info = env.step(actions)
        
        print("✅ Two-level reward computation successful")
        print(f"   - Global reward: {info['global_reward']}")
        print(f"   - Individual rewards: {rewards}")
        print(f"   - Alpha value: {info['alpha']}")
        
        # Check that we have one reward per partition
        if len(rewards) != env.M:
            print(f"❌ Expected {env.M} rewards, got {len(rewards)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Two-level reward test failed: {e}")
        return False

def main():
    print("Running Parallel DGRO Fix Tests")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        test_environment_initialization,
        test_agent_creation,
        test_two_level_rewards
    ]
    
    for test in tests:
        if not test():
            all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 All tests passed! The fixes should work correctly.")
        print("\nYou can now run:")
        print("python train_parallel_improved.py --N 20 --K 2 --M 4 --episodes 1000")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 