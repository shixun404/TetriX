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
    def __init__(self, feature_dim, output_dim, T=5, N=20,
                    device=torch.device('cuda:0')):
        super(DQNNetwork, self).__init__()
        self.T = T
        self.N = 20
        self.p = feature_dim
        self.device = device
        # self.theta_5 = nn.Linear(1 + self.N * self.N, 1)
        self.theta_5 = nn.Linear(1 + self.N * self.N, 1)
        # self.theta_5 = nn.Linear(1 + self.N * self.N, feature_dim)
        # self.theta_10 = nn.Linear(feature_dim, 32)
        # self.theta_11 = nn.Linear(32, 32)
        # self.output = nn.Linear(32, 1)
        
        
    def forward(self, state):
        bs = state.shape[0]
        
        m_fc = torch.ones(bs, self.N, 1, device=self.device)
        
        adj_fc = state[:, :self.N * self.N].reshape(-1, self.N, self.N)
        adj = state[:, self.N * self.N:self.N * self.N * 2].reshape(-1, self.N, self.N)
        adj_fc_ = adj_fc
        # adj_fc = (adj_fc - 5) / 10
        # adj = (adj - 5) / 10
        
        start_id = state[:, -1].to(torch.int64)
        m = state[:, self.N * self.N * 2:-1].reshape(-1, self.N, 1)

        # Gather selected elements based on indices
        

        # Expand to match the original dimensions
        # adj_repeated = state[:, :self.N * self.N].reshape(-1,  1, self.N * self.N).repeat(1, self.N, 1)
        adj_repeated = adj_fc.reshape(-1,  1, self.N * self.N).repeat(1, self.N, 1)
        fc_selected_adj = torch.gather(adj_fc_, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.N)).reshape(-1, self.N, 1)

        
        # Concatenate along the last dimension
        
        theta_5_in = torch.relu(torch.cat([fc_selected_adj, adj_repeated], dim=2))
        # theta_5_in = torch.relu(torch.cat([theta_8_o, theta_9_o], dim=2))
        # theta_10_in = torch.relu(self.theta_5(theta_5_in))
        # theta_11_in = torch.relu(self.theta_10(theta_10_in))
        # theta_11_out = torch.relu(self.theta_11(theta_11_in))
        # output = self.output(theta_11_out).reshape(-1, self.N)
        output = self.theta_5(theta_5_in).reshape(-1, self.N)
        # print(output[0])
        return output




class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer, device):
        
        self.state_size = state_size
        self.N = action_size
        self.replay_buffer = replay_buffer
        self.model = DQNNetwork(state_size, action_size).to(device)
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
            # action_values = torch.where(mask, state[-1, :400].reshape(20, 20)[int(state[-1, -1])], torch.tensor(float('inf')))
            # action = torch.argmin(action_values).item()
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
        expected_q = rewards + (0.9 ** steps) * next_q * (1 - dones)


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