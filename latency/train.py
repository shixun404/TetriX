from DQNAgent import DQNAgent
from collections import deque
from env import GraphEnv
import random
import argparse
import os
import torch
import numpy as np
from brute_force import *

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument("--N", type=int, help="Number of nodes", default=20)
    parser.add_argument("--K", type=int, help="Degree", default=4)
    parser.add_argument("--bs", type=int, help="Batch size", default=32)
    parser.add_argument("--lr", type=float, help="Learning rate", default=4e-5)
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    env = GraphEnv(num_nodes=args.N, K=args.K)
    agent = DQNAgent(state_size=2 * args.N * args.N + 1, action_size=args.N, replay_buffer=ReplayBuffer(10000)
                    , device=device)
    episodes = 5000
    batch_size = args.bs

    for episode in range(episodes):
        state_dict = env.reset()
        state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())  # Flatten the adjacency matrix to fit the network input
        state = np.append(state, state_dict['start_id'])
        total_reward = 0
        t = 0
        while True:
            t += 1
            # print(t, len(agent.replay_buffer))
            action = agent.act(state, env.graph.degree, args.K, epsilon=max((1 - episode/episodes), 0.05))
            next_state_dict, reward, done, _ = env.step(action)
            # print(next_state_dict['start_id'], reward)
            next_state = np.append(next_state_dict['initial_graph'].flatten(), next_state_dict['graph'].flatten())
            next_state = np.append(next_state, next_state_dict['start_id'])
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break
            
            if len(agent.replay_buffer) > batch_size:
                agent.learn(batch_size)
        _, best_diameter = find_regular_subgraph(env.initial_graph, args.K)
        print(f"Episode {episode+1:<4}: step = {t:<4} Total Reward = {total_reward:<6.2f} diameter = {env.cur_diameter:<4} brute_force = {best_diameter:<4} fully_diameter = {nx.diameter(env.initial_graph, weight='weight'):<4}", flush=True)
        # print(env.initial_adjacency_matrix)

if __name__ == '__main__':
    train()