import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
import copy

def graph_to_edge_index(G):
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list).t().contiguous()
    return edge_index

class DQNNetwork(nn.Module):
    def __init__(self, feature_dim, output_dim, T=2, N=20,
                    device=torch.device('cuda:0')):
        super(DQNNetwork, self).__init__()
        self.T = T
        self.N = N
        self.p = feature_dim
        self.device = device
        self.theta_1 = nn.Linear(1, feature_dim)
        self.theta_2 = nn.Linear(feature_dim, feature_dim)
        self.theta_3 = nn.Linear(feature_dim, feature_dim)
        self.theta_4 = nn.Linear(1, feature_dim)

        self.fc_theta_1 = nn.Linear(1, feature_dim)
        self.fc_theta_2 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_3 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_4 = nn.Linear(1, feature_dim)

        self.theta_6 = nn.Linear(feature_dim, feature_dim)
        self.theta_7 = nn.Linear(feature_dim, feature_dim)
        self.theta_8 = nn.Linear(feature_dim, feature_dim)
        self.theta_9 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_7 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_8 = nn.Linear(feature_dim, feature_dim)
        self.theta_5 = nn.Linear(1 + 4 * feature_dim, feature_dim)
        self.theta_10 = nn.Linear(feature_dim, feature_dim // 2)
        self.theta_11 = nn.Linear(feature_dim // 2, feature_dim // 2)
        self.output = nn.Linear(feature_dim // 2, 1)
        
    def forward(self, state, start_id):
        bs = state.shape[0]
        x_fc = torch.zeros(bs, self.N, self.p, device=self.device)
        m_fc = torch.ones(bs, self.N, 1, device=self.device)
        x = torch.zeros(bs, self.N, self.p, device=self.device)
        adj_fc = state[:, :self.N * self.N].reshape(-1, self.N, self.N)
        adj = state[:, self.N * self.N:self.N * self.N * 2].reshape(-1, self.N, self.N)
        adj_fc_ = adj_fc
        adj_fc = (adj_fc - 5) / 10
        
        adj = (adj - 5) / 10
        reshaped_adj = adj.reshape(-1, 1)
        theta_4_o = torch.relu(self.theta_4(reshaped_adj))
        theta_4_o = theta_4_o.reshape(bs, self.N, self.N, self.p)
        m = state[:, self.N * self.N * 2:].reshape(-1, self.N, 1)
        fc_reshaped_adj = adj_fc.reshape(-1, 1)
            
        fc_theta_4_o = torch.relu(self.fc_theta_4(fc_reshaped_adj))

        theta_1_o = self.theta_1(m)
        theta_3_i = torch.relu(theta_4_o).sum(dim=2)
        theta_3_o = self.theta_3(theta_3_i)
        for i in range(self.T):
            theta_2_o = self.theta_2(torch.bmm(adj, x))
            x = torch.relu(theta_1_o + theta_2_o + theta_3_o)

        summed_x = torch.sum(x, dim=1, keepdim=True)
        summed_x = summed_x.repeat(1, self.N, 1)
        theta_6_o = self.theta_6(summed_x)
        fc_selected_adj = torch.gather(adj_fc_, 1,
                                        start_id.view(-1, 1, 1).expand(
                                            bs, 1, self.N)).reshape(-1, self.N, 1)

        selected_elements = torch.gather(x, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.p))
        id_x = selected_elements.repeat(1, self.N, 1)
        fc_theta_8_o = self.fc_theta_8(x_fc)

        theta_7_o = self.theta_7(id_x)
        theta_8_o = self.theta_8(x)
        
        theta_5_in = torch.relu(torch.cat([fc_selected_adj,
                                            theta_6_o,
                                            fc_theta_8_o,
                                            theta_7_o,
                                            theta_8_o],
                                              dim=2))
        
        theta_10_in = torch.relu(self.theta_5(theta_5_in))
        theta_11_in = torch.relu(self.theta_10(theta_10_in))
        theta_11_out = torch.relu(self.theta_11(theta_11_in))
        output = self.output(theta_11_out).reshape(-1, self.N)
        
        return output

class SharedParallelDQNAgent:
    """
    Parallel DGRO with shared Q-network across all partitions
    所有partition共享同一个Q-network，减少内存使用并加快训练
    """
    def __init__(self, M, state_size, action_size, replay_buffer, decay_gamma, device, 
                 experiment_name, sync_freq=1, target_update_freq=1000, shared_buffer=True):
        
        self.state_size = state_size
        self.N = action_size
        self.M = M  # Number of partitions
        self.replay_buffer = replay_buffer
        self.device = device
        self.sync_freq = sync_freq
        self.target_update_freq = target_update_freq
        self.shared_buffer = shared_buffer
        self.decay_gamma = decay_gamma
        self.experiment_name = experiment_name
        self.update_counter = 0
        
        # 创建单个共享的Q-network和target network
        print(f"Creating shared Q-network for {M} partitions")
        
        # Main shared Q-network
        self.model = DQNNetwork(state_size, action_size, N=action_size, device=device).to(device)
        
        # Target Q-network
        self.target_model = DQNNetwork(state_size, action_size, N=action_size, device=device).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 单个optimizer和scheduler
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.95)
        
        self.criterion = nn.MSELoss()
        self.min_lr = 1e-5
        
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
    
    def generate_masked_one_hot(self, N, M, if_random=True):
        """Generate partition masks and starting nodes"""
        assert N % M == 0, "N must be divisible by M"
        partition_size = N // M

        indices = list(range(N))
        if if_random:
            random.shuffle(indices)

        masks = []
        start_id = []
        for i in range(M):
            part_indices = indices[i * partition_size: (i + 1) * partition_size]
            mask = torch.zeros(N)
            mask[part_indices] = 1
            masks.append(mask)
            start_id.append(indices[i * partition_size])
        
        return masks, start_id

    def act(self, state, degree, G, mask, masks, start_id, vector=None, K=4, epsilon=0.95):
        """
        Each partition agent chooses action using the shared Q-network
        完全按照原始DQNAgent_graph_parallel.py的逻辑实现
        """
        id = int(state[-1]) if len(state) > 0 else 0  # 使用state的最后一个元素（如果存在）
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        parallel_action = []
        mask_list = []
        
        for i in range(self.M):
            # 每次循环重新创建mask tensor（与原始版本一致）
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
            partition_mask = masks[i].to(dtype=torch.bool, device=self.device)
            final_mask = mask_tensor & partition_mask  # logical AND
            mask_list.append((torch.where(mask_tensor == 1)[0]).shape[0])  # 记录统计信息
            
            if random.random() > epsilon:
                # 使用共享的Q-network
                action_values = torch.where(final_mask, 
                                          self.model(state_tensor, torch.as_tensor([start_id[i]], device=self.device)), 
                                          torch.tensor(float('-inf')))
                action = torch.argmax(action_values).item()
            else:
                # 随机选择（原始逻辑）
                # print(f'partition {i}', mask_tensor, partition_mask, final_mask)
                valid_indices = torch.nonzero(final_mask, as_tuple=True)[0]
                action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
            unmasked_indices = torch.where(mask_tensor == 1)[0]
            masked_indices = torch.where(mask_tensor == 0)[0]
            # print(f'partition {i} start_id: {start_id[i]} action: {action} unmasked_indices: {masked_indices} len: {len(masked_indices)}')
            parallel_action.append(action)
        
        return parallel_action

    def store_experience(self, states, actions, rewards, next_states, dones, masks, steps):
        """
        Store experience for each partition in shared buffer
        """
        if self.shared_buffer:
            # Store all partition experiences in shared buffer
            for p in range(self.M):
                self.replay_buffer.push(
                    states[p] if isinstance(states, list) else states,
                    actions[p],
                    rewards[p],
                    next_states[p] if isinstance(next_states, list) else next_states,
                    dones,
                    masks[p] if isinstance(masks, list) else masks,
                    steps
                )

    def learn(self, batch_size):
        """
        Update the shared Q-network using experiences from all partitions
        更新共享的Q-network
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # 只更新一个共享网络
        loss = self._update_shared_network(batch_size)
        
        self.update_counter += 1
        
        # Periodic target network updates
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss

    def _update_shared_network(self, batch_size):
        """Update the shared Q-network"""
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, masks, steps = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        masks = torch.tensor(np.array(masks), dtype=torch.bool, device=self.device)
        steps = torch.tensor(steps, dtype=torch.float, device=self.device)
        
        # Create dummy start_ids for batch processing
        batch_size_actual = states.shape[0]
        dummy_start_ids = torch.zeros(batch_size_actual, dtype=torch.long, device=self.device)
        
        # Current Q-values using shared network
        current_q = self.model(states, dummy_start_ids).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values using target network
        with torch.no_grad():
            next_q_values = self.model(next_states, dummy_start_ids)
            next_q_values = torch.where(masks, next_q_values, torch.tensor(float('-inf')))
            next_q = next_q_values.max(1)[0]
        
        expected_q = rewards + (self.decay_gamma ** steps) * next_q * (1 - dones)
        
        loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr
                
        return loss.item()

    def _update_target_network(self):
        """Copy main network to target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, episode=None):
        """Save the shared Q-network"""
        if episode is not None:
            save_path = os.path.join(self.experiment_name, f'shared_model_episode_{episode}.pth')
        else:
            save_path = os.path.join(self.experiment_name, 'shared_model.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved shared model to {save_path}")

    def load(self, load_path=None, episode=None):
        """Load the shared Q-network"""
        if load_path is None:
            if episode is not None:
                filename = f'shared_model_episode_{episode}.pth'
            else:
                filename = 'shared_model.pth'
            load_path = os.path.join(self.experiment_name, filename)
        
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"Loaded shared model from {load_path}")
        else:
            print(f"Warning: Model file {load_path} not found")

# 为了向后兼容，创建别名
ParallelDQNAgent = SharedParallelDQNAgent
DQNAgent = SharedParallelDQNAgent 