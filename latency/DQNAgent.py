import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer, device):
        
        self.state_size = state_size
        self.N = action_size
        self.replay_buffer = replay_buffer
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.device = device

    def act(self, state, degree, K=4, epsilon=0.95):
        id = int(state[-1])
        id_list = []
        # print(state)
        # assert 0
        for i in range(self.N):
            if degree[i] < K and i != id and state[self.N * self.N + id * self.N + i] == 0:
                id_list.append(i)
        
        if len(id_list) == 0:
            # print(degree)
            id_list.append(id)
        # if id in id_list:
        #     print(degree)
        #     assert 0
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_values = self.model(state)   
            # print(action_values) 
            # print(id_list, action_values[:, id_list])
            action = id_list[torch.argmax(action_values[:, id_list]).item()]
        else:
            action = id_list[random.randrange(len(id_list))]
        # print(id, action, degree[id], degree[action], degree)
        # print(degree)
        # if action == id:
        #     print('dqqn action', degree)
        #     assert 0
        return action

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_states).max(1)[0]
        expected_q = rewards + 0.99 * next_q * (1 - dones)

        loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()