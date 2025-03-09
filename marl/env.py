import numpy as np
from gymnasium import spaces
from gymnasium import Env

class P2PNetworkEnv(Env):
    def __init__(self, num_agents, max_neighbors):
        super(P2PNetworkEnv, self).__init__()
        self.num_agents = num_agents
        self.max_neighbors = max_neighbors
        
        # 定义状态和动作空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_agents, num_agents))
        self.action_space = spaces.MultiDiscrete([num_agents] * num_agents)

        # 初始化拓扑
        self.network = np.zeros((num_agents, num_agents))  # 邻接矩阵
        self.current_step = 0
        self.max_steps = 100

    def reset(self):
        self.network = np.zeros((self.num_agents, self.num_agents))
        self.current_step = 0
        return self.network

    def step(self, actions):
        rewards = []
        for agent_id, action in enumerate(actions):
            target_node = action  # 选择的邻居节点
            if target_node != agent_id:  # 避免自环
                self.network[agent_id, target_node] = 1  # 添加连接
                self.network[target_node, agent_id] = 1  # 双向连接
            
            # 奖励：例如基于全局直径优化
            reward = self.calculate_reward()
            rewards.append(reward)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.network, np.array(rewards), done, {}

    def calculate_reward(self):
        # 计算奖励：例如基于网络直径
        return -np.sum(self.network)  # 示例：惩罚过多连接
    
    def render(self, mode="human"):
        print("Network Topology:")
        print(self.network)


from pettingzoo.utils.env import ParallelEnv

class P2PParallelEnv(ParallelEnv):
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.env = P2PNetworkEnv(num_agents, max_neighbors=5)

    def reset(self):
        obs = self.env.reset()
        return {agent: obs for agent in self.agents}

    def step(self, actions):
        obs, rewards, done, _ = self.env.step(list(actions.values()))
        observations = {agent: obs for agent in self.agents}
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, dones, infos

    def render(self, mode="human"):
        self.env.render()


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 使用 PettingZoo 环境
env = P2PParallelEnv(num_agents=5)

# 定义 PPO 模型
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 测试模型
obs = env.reset()
for _ in range(10):
    actions = {agent: env.action_space.sample() for agent in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    env.render()
