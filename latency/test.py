from DQNAgent_graph import DQNAgent
# from DQNAgent_greedy import DQNAgent
from collections import deque
from env import GraphEnv
import random
import argparse
import os
import torch
import numpy as np
from brute_force import *
from node2vec import Node2Vec

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, steps):
        self.buffer.append((state, action, reward, next_state, done, mask, steps))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def test(args, agent=None, env=None):

    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    if env is None:
        env = GraphEnv(num_nodes=args.N, K=args.K)
    if agent is None:
        agent = DQNAgent(state_size=args.feature_dim, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , device=device)

    test_reward = []
    test_diameter = []
    cnt_test = 0
    if_test = True
    for i in range(10):
        cnt_test += 1 if if_test else 0
        epsilon = 0 
        state_dict = env.reset(if_test=if_test)
        state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())  # Flatten the adjacency matrix to fit the network input
        state = np.append(state, state_dict['degree'])
        state = np.append(state, state_dict['start_id'])
        
        mask = state_dict['mask']
        total_reward = 0
        t = 0
        
        state_list_n = []
        reward_list_n = []
        action_list_n = []
        while True:
            t += 1
            
            action = agent.act(state, env.graph.degree, env.graph, mask, K=args.K, epsilon=epsilon)
            
            next_state_dict, reward, done, _ = env.step(action)
            state_list_n.append(state)
            reward_list_n.append(reward)
            action_list_n.append(action)
            mask = np.array(next_state_dict['mask'])
            next_state = np.append(next_state_dict['initial_graph'].flatten(), next_state_dict['graph'].flatten())
            next_state = np.append(next_state, next_state_dict['degree'])
            next_state = np.append(next_state, next_state_dict['start_id'])
            state = next_state
            total_reward += reward

            if done:
                break
        print(f"Test Graph {i:<4}: step = {t:<4} Total Reward = {total_reward:<8.2f} diameter = {env.cur_diameter:<4}", flush=True)
        if if_test:
            test_reward.append(total_reward)
            test_diameter.append(env.cur_diameter)
            if env.test_id == 9:
                print(sum(test_reward) / len(test_reward), sum(test_diameter) / len(test_reward))
                test_reward = []
                test_diameter = []

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument("--N", type=int, help="Number of nodes", default=100)
    parser.add_argument("--K", type=int, help="Degree", default=6)
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=64)
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    args = parser.parse_args()
    test(args)