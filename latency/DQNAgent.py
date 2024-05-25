import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch_geometric.nn import GCNConv
def graph_to_edge_index(G):
    # Get the edges from the graph
    edge_list = list(G.edges())
    
    # Convert edge list to a tensor of shape [2, num_edges]
    edge_index = torch.tensor(edge_list).t().contiguous()
    
    return edge_index


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc1 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_classes)
    

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.out(x)
        return x

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

    def act(self, state, degree, G, mask, vector=None, K=4, epsilon=0.95):
        # epsilon = 0
        id = int(state[-1])
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            action_values = torch.where(mask, self.model(state), torch.tensor(float('-inf')))
            
            action = torch.argmax(action_values).item()
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
        states, actions, rewards, next_states, dones, masks = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        mask_ = masks
        masks = torch.tensor(masks, dtype=torch.bool, device=self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states)
        next_q_values = torch.where(masks, next_q_values, torch.tensor(float('-inf')))
        next_q = next_q_values.max(1)[0]
        # print(masks, next_q)
        # print(len(self.replay_buffer))
        # for i in range(batch_size):
        #     if(next_q[i] == torch.tensor(float('-inf'))):
        #         print(i, masks[i], mask_[i])
        
        
        expected_q = rewards + 0.99 * next_q * (1 - dones)


        # print(current_q, expected_q)
        
        
        loss = self.criterion(current_q, expected_q)

        if loss == torch.tensor(float('-inf')):
            assert 0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()