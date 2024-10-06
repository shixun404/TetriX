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
from test import test
import wandb
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

def init(path=None):
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
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=path)
    parser.add_argument("--if_wandb", type=bool, help="Wandb on or off", default=False)
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
    name = ''
    for k, v in config.items():
        name = name + f"{k}={v}_"
    name = name[:-1]
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_name = '/pscratch/sd/s/swu264/SWARM/model/' + str(current_time)

    # Create a directory for the experiment
    folder_path = args.experiment_name
    os.makedirs(folder_path, exist_ok=True)  # 'exist_ok=True' avoids an error if the folder already exists

    # Path for the configuration file
    config_file_path = os.path.join(folder_path, 'config.json')

    # Write configurations to a JSON file
    # Convert the Namespace to a dictionary
    args_dict = vars(args)

    # Convert the dictionary to a JSON string
    args_json = json.dumps(args_dict, indent=4)
    with open(config_file_path, 'w') as config_file:
        json.dump(args_json, config_file, indent=4)
    if args.if_wandb:
        wandb.init(
                project=f'gossip',
                sync_tensorboard=True,
                config=config,
                name=name,
                # monitor_gym=True,
                save_code=True,
        )
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    return args
def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    env = GraphEnv(num_nodes=args.N, K=args.K)
    # assert 0
    agent = DQNAgent(state_size=args.feature_dim, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
                    , decay_gamma=args.decay_gamma, device=device, experiment_name=args.experiment_name)
    # agent.load(args.load_path)
    # agent = DQNAgent(state_size=args.N * args.N * 2 + 1, action_size=args.N, replay_buffer=ReplayBuffer(1000000)
    #                 , device=device)
    episodes = 400000
    batch_size = args.bs
    epoch = 0
    log_file_path = os.path.join(args.experiment_name, 'output')
    log_file = open(log_file_path, 'w')
    best_test_diameter = 1e8
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
            # test_diameter = test(args, agent=agent, log_file=log_file)
            # assert 0
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
                if epoch % 100 == 0 or done:
                    print(f"Train Epoch {epoch:<4}: step = {t:<4} Cumulative Reward = {total_reward:<8.2f} loss = {cur_loss:<8.4f}")
                    log_file.write(f"Train Epoch {epoch:<4}: step = {t:<4} Cumulative Reward = {total_reward:<8.2f} loss = {cur_loss:<8.4f}")
                    test_diameter = test(args, agent=agent, log_file=log_file)
                    if test_diameter < best_test_diameter:
                        best_test_diameter = test_diameter
                        metadata = {}
                        metadata['epoch'] = epoch
                        metadata['testDiameter'] = best_test_diameter
                        agent.save()
                    print('Best Test Diameter', best_test_diameter)
                    log_file.write('Best Test Diameter'+ str(best_test_diameter))
                    log_file.flush()
                    if args.if_wandb:
                        wandb.log({ 
                            "loss": cur_loss, 
                            "test diameter": test_diameter})

        # total_loss /= t
        # episode_name = 'Train'
        # print(f"{episode_name} Episode {episode:<4}: step = {t:<4} Total Reward = {total_reward:<8.2f} diameter = {env.cur_diameter:<4} loss = {total_loss:<8.4f}")
        

if __name__ == '__main__':
    # load_path = './num_nodes=100_degree=6_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9/num_nodes=100_degree=6_batch_size=32_gragh_embedding_dim=64_seed=123_lr=0.0005_decay_gamma=0.9_epoch=30900_testDiameter=23.4.pth'
    load_path = None
    train(init(load_path))