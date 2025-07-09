import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR

from torch_geometric.nn import GCNConv
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
        reshaped_adj = adj.reshape(-1, 1) # [bs, N, N] --> [bs * N * N, 1]
        theta_4_o = torch.relu(self.theta_4(reshaped_adj)) # [bs * N * N, 1] --> [bs * N * N, p]
        theta_4_o = theta_4_o.reshape(bs, self.N, self.N, self.p) # [bs * N * N, p] --> [bs, N, N, p]
        # start_id = state[:, -1].to(torch.int64)
        m = state[:, self.N * self.N * 2:].reshape(-1, self.N, 1)
        fc_reshaped_adj = adj_fc.reshape(-1, 1)
            
        fc_theta_4_o = torch.relu(self.fc_theta_4(fc_reshaped_adj))

        theta_1_o = self.theta_1(m) # degree N, [bs, N, 1] --> [bs, N, p]
        theta_3_i = torch.relu(theta_4_o).sum(dim=2) # [bs, N, N, p] -- sum --> [bs, N, p]
        theta_3_o = self.theta_3(theta_3_i) # [bs, N, p] --> [bs, N, p]
        for i in range(self.T):
            theta_2_o = self.theta_2(torch.bmm(adj, x)) # (bs x N x p) <- (bs x N x N) @ (bs x N x p)
            x = torch.relu(theta_1_o + theta_2_o + theta_3_o) # [bs, N, p]

        summed_x = torch.sum(x, dim=1, keepdim=True) # [bs, N, p] --> [bs, 1, p]
        summed_x = summed_x.repeat(1, self.N, 1) # [bs, 1, p] --> [bs, N, p]
        theta_6_o = self.theta_6(summed_x) # [bs, N, p] --> [bs, N, p]
        fc_selected_adj = torch.gather(adj_fc_, 1,
                                        start_id.view(-1, 1, 1).expand(
                                            bs, 1, self.N)).reshape(-1, self.N, 1)


        selected_elements = torch.gather(x, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.p)) # x [bs, N, p], start_id [bs], selected_elements [bs, 1, p]
        
        id_x = selected_elements.repeat(1, self.N, 1) # selected_elements [bs, 1, p] --> id_x [bs, N, p]
        
        fc_theta_8_o = self.fc_theta_8(x_fc) # Not used, zero tensor [bs, N, p] --> [bs, N, p]

        theta_7_o = self.theta_7(id_x) # id_x [bs, N, p] --> theta_7_o [bs, N, p]
        theta_8_o = self.theta_8(x) # x [bs, N, p] --> theta_8_o [bs, N, p]
        
        # Concatenate along the last dimension
        theta_5_in = torch.relu(torch.cat([fc_selected_adj, # [bs, N, 1] | weight [start_id, i]
                                            theta_6_o,      # [bs, N, p] | global summary, repeat N times
                                            fc_theta_8_o,   # [bs, N, p] | not used
                                            theta_7_o,      # [bs, N, p] | start id feature vector
                                            theta_8_o],     # [bs, N, p] | all nodes feature vector 
                                              dim=2))       # [bs, N, 1 + 4p]
        
        theta_10_in = torch.relu(self.theta_5(theta_5_in)) # [bs, N, 1 + 4p] @ [1 + 4p, p] --> [bs, N, p]
        theta_11_in = torch.relu(self.theta_10(theta_10_in)) # [bs, N, p] @ [p, p/2] --> [bs, N, p/2]
        theta_11_out = torch.relu(self.theta_11(theta_11_in)) # [bs, N, p/2] @ [p/2, p/4] --> [bs, N, p/4]
        output = self.output(theta_11_out).reshape(-1, self.N) # [bs, N, p/4] --> [bs, N, 1]
        
        return output




class DQNAgent:
    def __init__(self, M, state_size, action_size, replay_buffer, decay_gamma, device, experiment_name):
        
        self.state_size = state_size
        self.N = action_size
        self.M = M
        self.replay_buffer = replay_buffer
        self.device = device
        self.model = DQNNetwork(state_size, action_size, N=action_size, device=device).to(self.device)
        # self.model = GCN(64, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.95)
        self.min_lr = 1e-5
        self.decay_gamma = decay_gamma
        self.experiment_name = experiment_name    
        if not os.path.exists(experiment_name):
            # Create the directory
            os.makedirs(experiment_name)
    
    def generate_masked_one_hot(self, N, M):
        assert N % M == 0, "N must be divisible by M"
        partition_size = N // M

        # Step 1: Random permutation
        indices = list(range(N))
        random.shuffle(indices)

        # Step 2 and 3: Partition and create masks
        masks = []
        start_id = []
        for i in range(M):
            part_indices = indices[i * partition_size: (i + 1) * partition_size]
            mask = torch.zeros(N)
            mask[part_indices] = 1
            masks.append(mask)
            start_id.append(indices[i * partition_size])  # Store the starting index of each partition
        
        return masks, start_id

    def act(self, state, degree, G, mask, masks, start_id, vector=None, K=4, epsilon=0.95):
        
        id = int(state[-1])
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        parallel_action = []
        mask_list = []
        for i in range(self.M):
            
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            partition_mask = masks[i].to(dtype=torch.bool, device=self.device)
            final_mask = mask & partition_mask  # logical AND
            mask_list.append((th.where(mask == 1)[0]).shape[0])
            if random.random() > epsilon:
                action_values = torch.where(final_mask, self.model(state, th.as_tensor([start_id[i]], device=self.device)), torch.tensor(float('-inf')))
                action = torch.argmax(action_values).item()
            else:
                valid_indices = torch.nonzero(final_mask, as_tuple=True)[0]
                action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
            parallel_action.append(action)
        
        return parallel_action

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, masks, steps = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        mask_ = masks
        masks = torch.tensor(masks, dtype=torch.bool, device=self.device)
        steps = torch.tensor(steps, dtype=torch.float, device=self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states)
        next_q_values = torch.where(masks, next_q_values, torch.tensor(float('-inf')))
        next_q = next_q_values.max(1)[0]
        
        expected_q = rewards + (self.decay_gamma ** steps) * next_q * (1 - dones)
        
        loss = self.criterion(current_q, expected_q)

        if loss == torch.tensor(float('-inf')):
            assert 0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr
        return loss.item()
    def save(self,):
        save_path = os.path.join(self.experiment_name, 'model.pth')
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path=None):
        if load_path is None:
            file_list = os.listdir(self.experiment_name)
            for file in file_list:
                if '.pth' in file:
                    load_path = file
                    break
            load_path = os.path.join(self.experiment_name, load_path)
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))