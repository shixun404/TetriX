import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import networkx as nx
from collections import deque

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
batch_size = 64
memory_size = 10000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

class RingSelectionEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, N=500):
        super(RingSelectionEnv, self).__init__()
        self.N = N
        self.K = int(math.log2(N))
        self.action_space = spaces.Discrete(2)
        self.topology = nx.DiGraph()
        self.latency = torch.zeros(self.N, self.N)
        self.latency_mean = 0
        self.latency_std = 0
        self.topo_mean = 0
        self.topo_std = 0
        self.test_graphs = []
        self.test_id = -1
        self.graph.add_nodes_from(range(self.N))
        self.start_id = 0
        self.num_steps = 0
        self.if_test = False
        self.state = torch.zeros(6)
        self.num_random_ring = 0
        self.num_shortest_ring = 0
        self.ring_epsilon_greedy = 1

    def reset(self, latency):
        self.latency = latency
        self.latency_mean = latency.mean()
        self.latency_std = latency.std()

        self.topology = nx.DiGraph()
        self.topology.add_nodes_from(range(self.N))

        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0

    def step(self, action):
        nodes = list(self.latency.nodes())
        self.num_steps += 1
        if action:
            ring_nodes = nodes.copy()
            random.shuffle(ring_nodes)
            ring = [(ring_nodes[i], ring_nodes[(i+1) % len(ring_nodes)]) for i in range(len(ring_nodes))]
            self.num_random_ring += 1
            for u, v in ring:
                self.topology.add_edge(u, v, weight=self.latency[u][v])
        else:
            degree = [0 for i in range(self.N)]
            new_order = [i for i in range(self.N)]
            random.shuffle(new_order)
            current_node = new_order[0]
            self.num_shortest_ring += 1
            for _ in range(self.N):
                degree[current_node] += 1
                neighbors = list(range(self.N))
                neighbors.remove(current_node)
                neighbors.sort(key=lambda x: (degree[x], self.latency[current_node, x]))
                i = 0
                while (_ % self.N) != self.N - 1:
                    if self.topology.has_edge(current_node, neighbors[i % len(neighbors)]) or current_node == neighbors[i % len(neighbors)]:
                        i += 1
                        continue
                    else:
                        if random.random() <= self.ring_epsilon_greedy:
                            next_node = neighbors[i % len(neighbors)]
                            break
                    i += 1
                if _ % self.N == self.N - 1:
                    next_node = new_order[0]
                self.topology.add_edge(current_node, next_node)
                self.topology.edges[current_node, next_node]['weight'] = self.latency[current_node, next_node]
                current_node = next_node
            self.cur_diameter = nx.diameter(self.graph, weight='weight')
            reward = self.prev_diameter - self.cur_diameter

            edge_weights = [data['weight'] for _, _, data in self.topology.edges(data=True)]
            self.topo_mean = np.mean(edge_weights)
            self.topo_std = np.std(edge_weights)
            self.state = torch.as_tensor([self.latency_mean, self.latency_std,
                                          self.topo_mean, self.topo_std,
                                          self.num_random_ring, self.num_shortest_ring])
        done = self.num_steps >= self.K
        return self.state, reward, done

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon_start
        self.memory = deque(maxlen=memory_size)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * torch.max(self.target_model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach()
            target_f[0][action] = target
            self.model.zero_grad()
            loss = nn.functional.mse_loss(self.model(torch.FloatTensor(state).unsqueeze(0))[0][action], target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > epsilon_end:
            self.epsilon *= epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Example usage with log(N) episode length
N = 4  # Example matrix size, you can adjust this as needed
episode_length = int(math.log(N))  # Calculate episode length as log(N)
action_size = 2  # Actions: 0 or 1
state_size = N * N + 2  # N x N matrix plus two counts (0s and 1s)

agent = RLAgent(state_size, action_size)
num_episodes = 1000

for episode in range(num_episodes):
    matrix = np.random.randint(0, 2, (N, N))  # Initial random matrix
    count_0, count_1 = np.sum(matrix == 0), np.sum(matrix == 1)
    state = np.append(matrix.flatten(), [count_0, count_1])

    for t in range(episode_length):  # Log(N) steps per episode
        action = agent.act(state)
        # Define reward and next state based on your problem's logic
        reward = 0  # Placeholder reward, update as needed
        next_state = np.append(matrix.flatten(), [count_0, count_1])  # Update next state accordingly
        done = False  # Update this based on the end condition of your problem

        agent.memorize(state, action, reward, next_state, done)
        state = next_state

        agent.replay()  # Train the model in each step
        if done:
            agent.update_target_model()  # Update target model periodically
            break

    if episode % 10 == 0:
        print(f"Episode {episode} complete, epsilon: {agent.epsilon}")
