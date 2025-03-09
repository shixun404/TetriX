import gym
from gym import spaces
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque

class RingSelectionEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, N=500):
        super(RingSelectionEnv, self).__init__()
        self.N = N
        self.K = int(math.log2(N))
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.latency = None  # latency matrix
        self.latency_mean = 0
        self.latency_std = 0
        self.topo_mean = 0
        self.topo_std = 0

        self.topology = None  # current topology graph
        self.num_steps = 0

        self.state = np.zeros(6, dtype=np.float32)
        self.num_random_ring = 0
        self.num_shortest_ring = 0
        self.ring_epsilon_greedy = 1.0

        self.prev_diameter = 0
        self.cur_diameter = 0

    def reset(self, latency):
        self.latency = latency  # latency is a numpy array or torch tensor of size N x N
        self.latency_mean = self.latency.mean()
        self.latency_std = self.latency.std()

        self.topology = nx.DiGraph()
        self.topology.add_nodes_from(range(self.N))

        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0
        self.num_random_ring = 0
        self.num_shortest_ring = 0

        # Initialize state
        self.state = np.array([self.latency_mean, self.latency_std, 0, 0, 0, 0], dtype=np.float32)

        return self.state
 
    def step(self, action):
        nodes = list(range(self.N))
        self.num_steps += 1

        if action == 1:
            # Random ring
            ring_nodes = nodes.copy()
            random.shuffle(ring_nodes)
            ring = [(ring_nodes[i], ring_nodes[(i+1) % len(ring_nodes)]) for i in range(len(ring_nodes))]
            self.num_random_ring += 1
            for u, v in ring:
                self.topology.add_edge(u, v, weight=self.latency[u][v])
        else:
            # Shortest ring with some randomness
            degree = [0 for _ in range(self.N)]
            new_order = nodes.copy()
            random.shuffle(new_order)
            current_node = new_order[0]
            self.num_shortest_ring += 1
            ring = []
            for _ in range(self.N):
                if _ == self.N - 1:
                    self.topology.add_edge(current_node, new_order[0], weight=self.latency[current_node][new_order[0]])
                    break
                ring.append(current_node)
                degree[current_node] += 1
                neighbors = [i for i in range(self.N) if i != current_node and not self.topology.has_edge(current_node, i)]
                neighbors.sort(key=lambda x: (degree[x], self.latency[current_node][x]))
                next_node = None
                i = 0
                while True:
                    if i >= len(neighbors):
                        next_node = new_order[0]  # Close the ring
                        break
                    candidate = neighbors[i]
                    if random.random() <= self.ring_epsilon_greedy:
                        next_node = candidate
                        break
                    i += 1
                self.topology.add_edge(current_node, next_node, weight=self.latency[current_node][next_node])
                current_node = next_node
                

        # Compute reward
        try:
            self.cur_diameter = nx.diameter(self.topology, weight='weight')
        except nx.NetworkXError:
            # Graph not connected
            print("graph not connected!", self.num_steps, ring)
            self.cur_diameter = float('inf')
            assert 0
        reward = self.prev_diameter - self.cur_diameter
        self.prev_diameter = self.cur_diameter

        # Update state
        edge_weights = [data['weight'] for _, _, data in self.topology.edges(data=True)]
        if edge_weights:
            self.topo_mean = np.mean(edge_weights)
            self.topo_std = np.std(edge_weights)
        else:
            self.topo_mean = 0
            self.topo_std = 0
        self.state = np.array([self.latency_mean, self.latency_std,
                               self.topo_mean, self.topo_std,
                               self.num_random_ring, self.num_shortest_ring], dtype=np.float32)

        done = self.num_steps >= self.K

        return self.state, reward, done, {}

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def add_shortest_latency_edges(self, M):
        for u in range(self.N):
            # Get current outgoing neighbors and include self to avoid self-loops
            current_neighbors = list(self.topology.successors(u)) + [u]
            
            # Create a mask for candidate nodes (nodes not already connected from u)
            mask = torch.ones(self.N, dtype=torch.bool)
            mask[current_neighbors] = False  # Exclude current neighbors and self
            
            # Clone the latency tensor to avoid modifying the original
            candidate_latencies = torch.as_tensor(self.latency[u]).clone()
            
            # Set latencies of non-candidate nodes to infinity
            candidate_latencies[~mask] = 1e8
            
            # Determine the number of edges to add (cannot exceed available candidates)
            num_candidates = mask.sum().item()
            num_edges_to_add = min(M, num_candidates)
            if num_edges_to_add == 0:
                continue  # Skip if no candidates are available
            
            # Find the indices of the M smallest latencies
            _, indices = torch.topk(candidate_latencies, k=num_edges_to_add, largest=False)
            
            # Add edges to the topology
            for v in indices.tolist():
                self.topology.add_edge(u, v)

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0  # Initial epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 1e-3
        self.batch_size = 64
        self.memory_size = 10000
        self.target_update_freq = 10  # Update target network every N steps

        # Replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Q-Network
        self.policy_net = self.build_model().to(self.device)
        self.target_net = self.build_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.steps_done = 0

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        # Prepare tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def generate_latency_gaussian(N, mean=0.5, std=0.1, min_latency=0.1):
    # 使用 PyTorch 生成正态分布的随机数
    latency = torch.normal(mean, std, size=(N, N))
    # 截断最小值
    latency = torch.clamp(latency, min=min_latency)
    # 确保矩阵对称并对角线为零
    # latency = (latency + latency.T) / 2
    for i in range(N):
        for j in range(N):
            latency[i, j] = latency[j, i]
    latency.fill_diagonal_(0)
    # 转换为 NumPy 数组
    latency = latency.numpy()
    return latency

def generate_latency_uniform(N, min_latency=1, max_latency=11):
    # 使用 PyTorch 生成正态分布的随机数
    # latency = torch.normal(mean, std, size=(N, N))
    latency = torch.randint(min_latency, max_latency, (N, N))
    # 确保矩阵对称并对角线为零
    # latency = (latency + latency.T) / 2
    for i in range(N):
        for j in range(N):
            latency[i, j] = latency[j, i]
    latency.fill_diagonal_(0)
    # 转换为 NumPy 数组
    latency = latency.numpy()
    return latency

def train_agent(env, agent, num_episodes=1000):
    for e in range(num_episodes):
        # Generate a random latency matrix for each episode
        # latency = generate_latency_gaussian(env.N, mean=5, std=1, min_latency=1)
        latency = generate_latency_uniform(env.N, min_latency=1, max_latency=11)
        
        state = env.reset(latency)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            agent.steps_done += 1

        print(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained agent
    agent.save('dqn_agent.pth')

def test_agent(env, agent, num_episodes=10):
    agent.epsilon = 0.0  # 测试时不进行探索
    total_rewards_agent = []
    total_rewards_action_0 = []
    total_rewards_action_1 = []
    total_rewards_multi_ring_perigee = []
    diameter_agent = []
    diameter_action_0 = []
    diameter_action_1 = []
    diameter_multi_ring_perigee = []

    for e in range(num_episodes):
        # 为每个episode生成一个随机的延迟矩阵
        # latency = generate_latency_gaussian(env.N, mean=5, std=1, min_latency=1)
        latency = generate_latency_uniform(env.N, min_latency=1, max_latency=11)

        # 测试训练好的智能体
        state = env.reset(latency)
        done = False
        total_reward = 0
        action_list = []
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            action_list.append(action)

        total_rewards_agent.append(total_reward)
        diameter_agent.append(env.cur_diameter)

        # 测试始终选择动作0的策略
        state = env.reset(latency)
        done = False
        total_reward_action_0 = 0

        while not done:
            action = 0  # 始终选择动作0
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward_action_0 += reward

        total_rewards_action_0.append(total_reward_action_0)
        diameter_action_0.append(env.cur_diameter)

        # 测试始终选择动作1的策略
        state = env.reset(latency)
        done = False
        total_reward_action_1 = 0

        while not done:
            action = 1  # 始终选择动作1
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward_action_1 += reward

        total_rewards_action_1.append(total_reward_action_1)
        diameter_action_1.append(env.cur_diameter)

        state = env.reset(latency)
        done = False


        M = 2
        while env.num_steps <= env.K - M:
            action = 1  # 始终选择动作1
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward_action_1 += reward
        env.add_shortest_latency_edges(M)
        diameter_multi_ring_perigee.append(env.cur_diameter)

        print(f"Test Episode {e+1}/{num_episodes}")
        print(f"  Agent Cumulative Reward: {total_reward} Diameter: {diameter_agent[-1]} ", action_list)
        print(f"  Action 0 Cumulative Reward: {total_reward_action_0} Diameter: {diameter_action_0[-1]}")
        print(f"  Action 1 Cumulative Reward: {total_reward_action_1} Diameter: {diameter_action_1[-1]}")
        print(f"  Multi Ring Perigee Diameter: {diameter_multi_ring_perigee[-1]}")

    # 计算平均奖励
    avg_reward_agent = np.mean(total_rewards_agent)
    avg_reward_action_0 = np.mean(total_rewards_action_0)
    avg_reward_action_1 = np.mean(total_rewards_action_1)
    avg_reward_multi_ring_perigee = np.mean(diameter_multi_ring_perigee)

    print("\Test Result Summary")
    print(f"Agent Average Cumulative Reward: {avg_reward_agent} Diameter: {np.mean(diameter_agent)}")
    print(f"Action 0 Average Cumulative Reward: {avg_reward_action_0} Diameter: {np.mean(diameter_action_0)}")
    print(f"Action 1 Average Cumulative Reward: {avg_reward_action_1} Diameter: {np.mean(diameter_action_1)}")
    print(f"Multi Ring Perigee Diameter: {avg_reward_multi_ring_perigee}")

if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = RingSelectionEnv(N=100)
    agent = DQNAgent(state_size=6, action_size=2, device='cuda')

    # Load the trained agent
    agent.load('dqn_agent.pth')

    # Train the agent
    train_agent(env, agent, num_episodes=0)

    # Load the trained agent
    agent.load('dqn_agent.pth')

    # Test the agent
    test_agent(env, agent, num_episodes=10)
