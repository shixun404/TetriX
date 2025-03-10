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
import json
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


def test(args, num_tests=1, agent=None, env=None, log_file=None, if_plot=False, seed=42, epsilon = 0):

    
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
        diameter_list = []
        cumulative_time = 0    
        for _ in range(3000):
            # for start_id in range(args.N):
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
                    next_state_dict, reward, done, _a = env.step(action)
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
            if _ % 100 == 0 or _ == 2999:
                plt.figure()
                plt.hist(diameter_list, bins=50)
                plt.title('Diameter Distribution RL N=100 K=3')
                plt.savefig(f'../sc_test/histo_seed={seed}/RL_1.png')
                with open(f'../sc_test/histo_seed={seed}/RL_1.txt', 'w') as f:
                    f.write(str(diameter_list))
                diameter_tensor = torch.tensor(diameter_list)
                print(f'diameter mean={diameter_tensor.mean()}, std={diameter_tensor.std()}, min={diameter_tensor.min()}, max={diameter_tensor.max()}')
            # print('Test last step cacalculate diamater', time.time() - cur_time)
        # print("graph in_degree = ", env.graph.in_degree(), "graph out_degree = ", env.graph.out_degree())
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

    return sum(min_test_diameter) / len(min_test_diameter), env.graph


def init(config_path="config.json"):
    parser = argparse.ArgumentParser(description="Process some integers.")
    # # 添加参数
    # parser.add_argument("--N", type=int, help="Number of nodes", default=20)
    # parser.add_argument("--K", type=int, help="Degree", default=4)
    # parser.add_argument("--bs", type=int, help="Batch size", default=32)
    # parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=64)
    # parser.add_argument("--decay_gamma", type=int, help="Q decay", default=0.9)
    # parser.add_argument("--lr", type=float, help="Learning rate", default=5e-4)
    # parser.add_argument("--reward_mode", type=str, help="Reward Mode", default='diameter')
    # parser.add_argument("--seed", type=int, help="Random seed", default=123)
    # parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    # args = parser.parse_args()

    # Default values
    default_config = {
        "N": 20,
        "K": 4,
        "bs": 32,
        "feature_dim": 64,
        "decay_gamma": 0.9,
        "lr": 5e-4,
        "reward_mode": "diameter",
        "seed": 123,
        "load_path": None,
        "if_wandb": False,
        "experiment_name": None  # Will be set dynamically
    }



    # Load JSON config if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            json_string = f.read()
            json_config = json.loads(json_string)
            json_config = json.loads(json_config)
            print(type(json_config), type({'a':1}))
            print(json_config)
            default_config.update(json_config)  # Override defaults with JSON values

    # Adding command-line arguments
    parser.add_argument("--N", type=int, help="Number of nodes", default=default_config["N"])
    parser.add_argument("--K", type=int, help="Degree", default=default_config["K"])
    parser.add_argument("--bs", type=int, help="Batch size", default=default_config["bs"])
    parser.add_argument("--feature_dim", type=int, help="Feature dimension", default=default_config["feature_dim"])
    parser.add_argument("--decay_gamma", type=float, help="Q decay", default=default_config["decay_gamma"])
    parser.add_argument("--lr", type=float, help="Learning rate", default=default_config["lr"])
    parser.add_argument("--reward_mode", type=str, help="Reward Mode", default=default_config["reward_mode"])
    parser.add_argument("--seed", type=int, help="Random seed", default=default_config["seed"])
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=default_config["load_path"])
    parser.add_argument("--if_wandb", type=bool, help="Whether to use Weights & Biases logging", default=default_config["if_wandb"])

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
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")


    return args

if __name__ == '__main__':
    
    args = init('/pscratch/sd/s/swu264/SWARM/model/20250309_030015/config.json')
    device = torch.device("cuda")
    # device = torch.device("cpu")
    env = GraphEnv(num_nodes=args.N, K=args.K)
    seed = 42
    # assert 0
    agent = DQNAgent(state_size=args.feature_dim, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , decay_gamma=args.decay_gamma, device=device, experiment_name=args.experiment_name)
    model_path = '/pscratch/sd/s/swu264/SWARM/model/20250309_030015/model.pth'
    log_file_path = os.path.join(args.experiment_name, f'{args.experiment_name}.output')
    log_file = open(log_file_path, 'w')
    agent.load(model_path)
    d, G = test(args, env=env, agent=agent, log_file=log_file, seed=seed, epsilon=0.01)
    with open(os.path.join('..', 'sc_test', f'best_test_graph_seed={seed}.pkl'), 'wb') as f:
        pkl.dump(G, f)