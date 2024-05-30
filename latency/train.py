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
from test import test

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, steps):
        self.buffer.append((state, action, reward, next_state, done, mask, steps))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def get_graph_embeddings(graph, id, dim=64):
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=200, p=0.5, q=2)
    # Fit node2vec model (learn embeddings)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    vectors = model.wv.vectors
    vec_sum = np.sum(vectors, axis=0)

    return model.wv[f'{id}'], vec_sum, vectors

def train():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument("--N", type=int, help="Number of nodes", default=20)
    parser.add_argument("--K", type=int, help="Degree", default=4)
    parser.add_argument("--bs", type=int, help="Batch size", default=32)
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=64)
    parser.add_argument("--lr", type=float, help="Learning rate", default=5e-4)
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    env = GraphEnv(num_nodes=args.N, K=args.K)
    agent = DQNAgent(state_size=args.feature_dim, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , device=device)
    # agent = DQNAgent(state_size=args.N * args.N * 2 + 1, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
    #                 , device=device)
    episodes = 400000
    batch_size = args.bs
    epoch = 0
    
    for episode in range(episodes):
        epsilon = max((1 - episode * args.N * args.K / 10000), 0.05)
        state_dict = env.reset()
        
        state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())  # Flatten the adjacency matrix to fit the network input
        state = np.append(state, state_dict['degree'])
        state = np.append(state, state_dict['start_id'])
        
        
        mask = state_dict['mask']
        total_reward = 0
        t = 0
        total_loss = 0

        
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
            # if not if_test:
                # agent.replay_buffer.push(state_list_n[-args.N], action_list_n[-args.N],
                #                          sum(reward_list_n[-args.N:]), next_state, done, mask, args.N)
            agent.replay_buffer.push(state, action, reward, next_state, done, mask, 1)
            state = next_state
            total_reward += reward

            if done:
                break
            
            if len(agent.replay_buffer) > batch_size:
                cur_loss = agent.learn(batch_size)
                total_loss += cur_loss 
                epoch += 1
                print(f"Train Epoch {epoch:<4}: step = {t:<4} Cumulative Reward = {total_reward:<8.2f} loss = {cur_loss:<8.4f}")
                if epoch % 100 == 0:
                    test(args, agent)
                # _, best_diameter = find_regular_subgraph(env.initial_graph, args.K)
        total_loss /= t
        episode_name = 'Train'
        print(f"{episode_name} Episode {episode:<4}: step = {t:<4} Total Reward = {total_reward:<8.2f} diameter = {env.cur_diameter:<4} loss = {total_loss:<8.4f}")
        

if __name__ == '__main__':
    config = {
        'N': args.N,
        'fan_out': args.fan_out,
        'num_hop': args.num_hop,
        'lr': args.lr,
        'seed': args.seed,
        'bs': args.bs,
    }
    print(config)
    # assert 0
    wandb.init(
            project=f'gossip',
            sync_tensorboard=True,
            config=config,
            name=f"N={args.N}_fanout={args.fan_out}_numhop={args.num_hop}_seed={args.seed}.pth",
            # monitor_gym=True,
            save_code=True,
    )
    train()