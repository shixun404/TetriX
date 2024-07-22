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
# from node2vec import Node2Vec
import time

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, steps):
        self.buffer.append((state, action, reward, next_state, done, mask, steps))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def test(args, num_tests=2, agent=None, env=None, log_file=None, if_plot=False):

    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    # device = torch.device("cpu")
    if env is None:
        env = GraphEnv(num_nodes=args.N, K=args.K)
    if agent is None:
        agent = DQNAgent(state_size=args.feature_dim, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , device=device)

    # test_reward = []
    max_test_diameter = []
    min_test_diameter = []
    # test_episode_lengths = []
    cnt_test = 0
    if_test = True
    time_list = []
    for i in range(num_tests):
        cnt_test += 1 if if_test else 0
        epsilon = 0
        diameter_list = []
        cumulative_time = 0    
        for start_id in range(1):
            # print("asdasdadasdsadadadadasdsad")
            state_dict = env.reset(if_test=if_test, start_id=start_id, test_id=i)
            state = np.append(state_dict['initial_graph'].flatten(), state_dict['graph'].flatten())  # Flatten the adjacency matrix to fit the network input
            state = np.append(state, state_dict['degree'])
            state = np.append(state, state_dict['start_id'])
            
            mask = state_dict['mask']
            total_reward = 0
            t = 0
            # state_list_n = []
            # reward_list_n = []
            # action_list_n = []
            if if_plot:
                positions = nx.circular_layout(env.initial_graph)
                fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(20, 35))  # Adjust figsize to fit your screen
                axes = axes.flatten()  # Flatten the array of axes
            last_time = time.time()
            while True:
                t += 1
                
                cur_time = time.time()
                
                action = agent.act(state, env.graph.degree, env.graph, mask, K=args.K, epsilon=epsilon)
                special_edge = (env.start_id, action)
                next_time = time.time()
                # print(t, next_time - cur_time)
                cumulative_time += next_time - cur_time
                next_state_dict, reward, done, _ = env.step(action)
                # state_list_n.append(state)
                # reward_list_n.append(reward)
                # action_list_n.append(action)
                mask = np.array(next_state_dict['mask'])
                next_state = np.append(next_state_dict['initial_graph'].flatten(), next_state_dict['graph'].flatten())
                next_state = np.append(next_state, next_state_dict['degree'])
                next_state = np.append(next_state, next_state_dict['start_id'])
                state = next_state
                total_reward += reward
                
                if if_plot:
                    ax = axes[t - 1]
                    special_edges = [edge for edge in env.graph.edges(data=True) if ((edge[0], edge[1]) == special_edge or (edge[1], edge[0]) == special_edge)]
                    nx.draw_networkx_nodes(env.graph, pos=positions, node_color='lightblue',  ax=ax)
                    edges = env.graph.edges(data=True)
                    nx.draw_networkx_edges(env.graph, pos=positions, edgelist=edges, width=[w['weight'] for (u, v, w) in edges],  ax=ax)
                    nx.draw_networkx_labels(env.graph, pos=positions, font_weight='bold', ax=ax)
                    edge_labels = {(u, v): w['weight'] for (u, v, w) in edges}
                    nx.draw_networkx_edge_labels(env.graph, pos=positions, edge_labels=edge_labels,  ax=ax)
                    nx.draw_networkx_edges(env.graph, pos=positions, edgelist=special_edges, 
                                arrows=True, width=[w['weight'] for (u, v, w) in special_edges], 
                                edge_color='red', ax=ax)
                    ax.set_title(f"Step {t} d={env.cur_diameter}")
                    ax.axis('off') 
                if done:
                    break
            cur_time = time.time()
            diameter_list.append(nx.diameter(env.graph, weight='weight'))
            # print('Test last step cacalculate diamater', time.time() - cur_time)
        if if_plot:
            plt.tight_layout()
            plt.savefig(f'figures/GNN_N=20_{i}.png')
        # test_reward.append(total_reward)
        max_test_diameter.append(max(diameter_list))
        min_test_diameter.append(min(diameter_list))
        # test_episode_lengths.append(t)
        time_list.append(cumulative_time / args.N)

    # print(f"Test Reward", test_reward, 'average=', sum(test_diameter) / len(test_diameter) )
    # log_file.write(f"Test Reward" + str(test_reward))
    print(f"Max Test Diameter", max_test_diameter)
    log_file.write(f"Max Test Diameter" + str(max_test_diameter))
    print(f"Min Test Diameter", min_test_diameter)
    log_file.write(f"Min Test Diameter" + str(min_test_diameter))
    # print(f"Test Episode Length", test_episode_lengths)
    # log_file.write(f"Test Episode Length" + str(test_episode_lengths))
    print(f"Test Time " + str(time_list), 'average_time = ', sum(time_list) / len(time_list) )
    log_file.write(f"Test Time" + str(time_list))

    return sum(min_test_diameter) / len(min_test_diameter)


def init():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # 添加参数
    parser.add_argument("--N", type=int, help="Number of nodes", default=20)
    parser.add_argument("--K", type=int, help="Degree", default=4)
    parser.add_argument("--bs", type=int, help="Batch size", default=32)
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=64)
    parser.add_argument("--decay_gamma", type=int, help="Q decay", default=0.9)
    parser.add_argument("--lr", type=float, help="Learning rate", default=5e-4)
    parser.add_argument("--reward_mode", type=str, help="Reward Mode", default='diameter')
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    args = parser.parse_args()
    config = {
        'num_nodes': args.N,
        'degree': args.K,
        'batch_size': args.bs,
        'gragh_embedding_dim': args.feature_dim,
        'seed': args.seed,
        'lr': args.lr,
        'decay_gamma':args.decay_gamma,
        'reward_mode':args.reward_mode
    }
    name = 'test'
    for k, v in config.items():
        name = name + f"{k}={v}_"
    name = name[:-1]
    args.experiment_name = name

    return args

if __name__ == '__main__':
    
    args = init()
    device = torch.device("cuda")
    # device = torch.device("cpu")
    env = GraphEnv(num_nodes=args.N, K=args.K)
    # assert 0
    agent = DQNAgent(state_size=args.feature_dim, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , decay_gamma=args.decay_gamma, device=device, experiment_name=args.experiment_name)
    # model_path = './num_nodes=20_degree=4_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9/num_nodes=20_degree=4_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_epoch=1200_testDiameter=7.32.pth'
    # model_path = './num_nodes=20_degree=4_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_reward_mode=diameter_1/num_nodes=20_degree=4_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_reward_mode=diameter_epoch=7900_testDiameter=6.8.pth'
    # model_path = './num_nodes=20_degree=4_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_reward_mode=diameter/num_nodes=20_degree=4_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_reward_mode=diameter_epoch=9700_testDiameter=6.5.pth'
    model_path = './num_nodes=100_degree=6_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9/num_nodes=100_degree=6_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_epoch=30900_testDiameter=23.4.pth'
    log_file_path = os.path.join(args.experiment_name, f'{args.experiment_name}.output')
    log_file = open(log_file_path, 'w')
    agent.load(model_path)
    test(args, env=env, agent=agent, log_file=log_file)