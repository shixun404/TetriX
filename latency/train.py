from DQNAgent import DQNAgent
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
    
    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))
    
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
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    env = GraphEnv(num_nodes=args.N, K=args.K)
    agent = DQNAgent(state_size=args.feature_dim * 3, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , device=device)
    # agent = DQNAgent(state_size=args.N * args.N * 2 + 1, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
    #                 , device=device)
    episodes = 5000
    batch_size = args.bs

    test_reward = []
    test_diameter = []
    for episode in range(episodes):
        if_test = ((episode % 100 >= 90) and env.test_id < 10)
        
        epsilon = 0 if if_test else max((1 - episode/1000), 0.05)
        # epsilon = 0 if if_test else 0.01
        state_dict = env.reset(if_test=if_test)
        # pos_vec = np.zeros(args.N)
        # # pos_vec[state_dict['start_id']] = 1
        # state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())  # Flatten the adjacency matrix to fit the network input
        # state = np.append(state, state_dict['start_id'])
        # print(state.shape)
        
        mask = state_dict['mask']
        # state = np.append(state, pos_vec)
        total_reward = 0
        t = 0
        total_loss = 0

        
        vector_initial, vector_sum_initial, vectors = get_graph_embeddings(env.initial_graph, 0, args.feature_dim)
        # vector, vector_sum = get_graph_embeddings(env.graph, 0, args.feature_dim)
        selected_rows = vectors[mask]
        selected_sum = np.sum(selected_rows, axis=0)


        state = np.concatenate((vector_initial, vector_sum_initial, vector, vector_sum)).reshape(-1)
        state = np.concatenate((vector_initial, vector_sum_initial, selected_sum)).reshape(-1)
        while True:
            t += 1
            
            action = agent.act(state, env.graph.degree, env.graph, mask, K=args.K, epsilon=epsilon)
            
            next_state_dict, reward, done, _ = env.step(action)
            mask = np.array(next_state_dict['mask'])
            # next_state = np.append(next_state_dict['initial_graph'].flatten(), next_state_dict['graph'].flatten())
            # next_state = np.append(next_state, next_state_dict['start_id'])
            # vector, vector_sum, = get_graph_embeddings(env.graph, action, args.feature_dim)
            # next_state = np.concatenate((vector_initial, vector_sum_initial, vector, vector_sum)).reshape(-1)
            selected_rows = vectors[mask]
            selected_sum = np.sum(selected_rows, axis=0)
            next_state = np.concatenate((vector_initial, vector_sum_initial, selected_sum)).reshape(-1)
            if not if_test:
                agent.replay_buffer.push(state, action, reward, next_state, done, mask)
            state = next_state
            total_reward += reward

            if done:
                break
            
            if len(agent.replay_buffer) > batch_size and not if_test:
                total_loss += agent.learn(batch_size)
        _, best_diameter = find_regular_subgraph(env.initial_graph, args.K)
        episode_name = 'Test ' if if_test else 'Train'
        print(f"{episode_name} Episode {episode:<4}: step = {t:<4} Total Reward = {total_reward:<8.2f} diameter = {env.cur_diameter:<4} brute_force = {best_diameter:<4} fully_diameter = {nx.diameter(env.initial_graph, weight='weight'):<4} loss = {total_loss:<8.4f}", flush=True)
        # print(env.test_id)
        if if_test:
            test_reward.append(total_reward)
            test_diameter.append(env.cur_diameter)
            episode -= 1
            if env.test_id == 9:
                print(sum(test_reward) / len(test_reward), sum(test_diameter) / len(test_reward))
                test_reward = []
                test_diameter = []

        

        # print(env.initial_adjacency_matrix)

if __name__ == '__main__':
    train()