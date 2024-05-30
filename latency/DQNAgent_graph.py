import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.optim.lr_scheduler import StepLR

from torch_geometric.nn import GCNConv
def graph_to_edge_index(G):
    # Get the edges from the graph
    edge_list = list(G.edges())
    
    # Convert edge list to a tensor of shape [2, num_edges]
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

        # self.fc_theta_6 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_7 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_8 = nn.Linear(feature_dim, feature_dim)
        # self.fc_theta_9 = nn.Linear(feature_dim, feature_dim)

        # self.theta_5 = nn.Linear(1 + 6 * feature_dim, 1)
        self.theta_5 = nn.Linear(1 + 6 * feature_dim, feature_dim)
        self.theta_10 = nn.Linear(feature_dim, 32)
        self.theta_11 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        
        
    def forward(self, state):
        bs = state.shape[0]
        x_fc = torch.zeros(bs, self.N, self.p, device=self.device)
        m_fc = torch.ones(bs, self.N, 1, device=self.device)
        x = torch.zeros(bs, self.N, self.p, device=self.device)
        adj_fc = state[:, :self.N * self.N].reshape(-1, self.N, self.N)
        adj = state[:, self.N * self.N:self.N * self.N * 2].reshape(-1, self.N, self.N)
        adj_fc_ = adj_fc
        adj_fc = (adj_fc - 5) / 10
        
        adj = (adj - 5) / 10
        
        start_id = state[:, -1].to(torch.int64)
        m = state[:, self.N * self.N * 2:-1].reshape(-1, self.N, 1)

        theta_1_o = self.theta_1(m) #  (Nxp)
        for i in range(self.T):
            theta_2_o = self.theta_2(torch.bmm(adj, x)) # (NxNxp) + (Nxpxp)
            reshaped_adj = adj.reshape(-1, 1)
            
            theta_4_o = torch.relu(self.theta_4(reshaped_adj)) #  (NxNxp)
            theta_4_o = theta_4_o.reshape(bs, self.N, self.N, self.p)
            theta_3_i = torch.relu(theta_4_o).sum(dim=2) # (NxNxp)
            theta_3_o = self.theta_3(theta_3_i) #  (Nxpxp)
            # x = torch.relu(theta_1_o + theta_2_o + theta_3_o)
            x = torch.relu(theta_1_o + theta_2_o + theta_3_o)

        fc_theta_1_o = self.fc_theta_1(m_fc)
        for i in range(self.T):
            fc_theta_2_o = self.fc_theta_2(torch.bmm(adj_fc, x_fc))
            fc_reshaped_adj = adj_fc.reshape(-1, 1)
            
            fc_theta_4_o = torch.relu(self.fc_theta_4(fc_reshaped_adj))
            fc_theta_4_o = fc_theta_4_o.view(bs, self.N, self.N, self.p)
            fc_theta_3_i = torch.relu(fc_theta_4_o).sum(dim=2)
            fc_theta_3_o = self.fc_theta_3(fc_theta_3_i)
            x_fc = torch.relu(fc_theta_1_o + fc_theta_2_o + fc_theta_3_o)
        
        summed_x = torch.sum(x, dim=1, keepdim=True) # Nxp
        summed_x = summed_x.repeat(1, self.N, 1)
        fc_summed_x = torch.sum(x_fc, dim=1, keepdim=True) # Nxp
        fc_summed_x = fc_summed_x.repeat(1, self.N, 1)
        theta_9_o = self.theta_9(fc_summed_x) # Nxpxp
        theta_6_o = self.theta_6(summed_x) # Nxpxp

        # Gather selected elements based on indices
        fc_selected_elements = torch.gather(x_fc, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.p))
        fc_selected_adj = torch.gather(adj_fc_, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.N)).reshape(-1, self.N, 1)
        
        selected_elements = torch.gather(x, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.p))

        # Expand to match the original dimensions
        fc_id_x = fc_selected_elements.repeat(1, self.N, 1)
        
        id_x = selected_elements.repeat(1, self.N, 1)
        
        
        
        fc_theta_7_o = self.fc_theta_7(fc_id_x) # Nxpxp
        fc_theta_8_o = self.fc_theta_8(x_fc) # Nxpxp

        theta_7_o = self.theta_7(id_x) # Nxpxp
        theta_8_o = self.theta_8(x) # Nxpxp
        
        # Concatenate along the last dimension
        theta_5_in = torch.relu(torch.cat([fc_selected_adj, theta_6_o, fc_theta_7_o, fc_theta_8_o, theta_7_o, theta_8_o, theta_9_o], dim=2))
        
        theta_10_in = torch.relu(self.theta_5(theta_5_in)) # Nx(6p+1)x(6p+1)xp
        theta_11_in = torch.relu(self.theta_10(theta_10_in)) # Nxpxpx32
        theta_11_out = torch.relu(self.theta_11(theta_11_in)) # Nx32x32x32
        output = self.output(theta_11_out).reshape(-1, self.N) # Nx32x32
        
        # output = self.theta_5(theta_5_in).reshape(-1, self.N)
        # print(output[0])
        return output




class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer, device):
        
        self.state_size = state_size
        self.N = action_size
        self.replay_buffer = replay_buffer
        self.model = DQNNetwork(state_size, action_size, N=action_size).to(device)
        # self.model = GCN(64, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.device = device
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.95)
        self.min_lr = 1e-5

    def act(self, state, degree, G, mask, vector=None, K=4, epsilon=0.95):
        # epsilon = 0
        id = int(state[-1])
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            # action_values = self.model(state)
            # probabilities = torch.softmax(action_values, dim=0)
            # action_values = torch.where(mask, probabilities, torch.tensor(float(0)))
            # print(action_values)
            # assert 0
            
            action_values = torch.where(mask, self.model(state), torch.tensor(float('-inf')))
            action = torch.argmax(action_values).item()
            # action_values = torch.where(mask, state[-1, :self.N * self.N].reshape(self.N, self.N)[int(state[-1, -1])], torch.tensor(float('inf')))
            # action = torch.argmin(action_values).item()
            

            # Sample an action according to the probabilities
            # action = torch.multinomial(probabilities, 1).item()
            # print('argmax', action)
        else:
            non_zero_indices = [index for index, element in enumerate(mask) if element != 0]
            action = non_zero_indices[random.randrange(len(non_zero_indices))]
            # print('random', action, id, id_list)
        return action

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
        # next_q = next_q_values.min(1)[0]
        # print(masks, next_q)
        # print(len(self.replay_buffer))
        # for i in range(batch_size):
        #     if(next_q[i] == torch.tensor(float('-inf'))):
        #         print(i, masks[i], mask_[i])
        
        
        # expected_q = rewards + 0.99 * next_q * (1 - dones)
        expected_q = rewards + (0.1 ** steps) * next_q * (1 - dones)


        # print(current_q, expected_q)
        
        
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