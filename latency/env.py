import gym
from gym import spaces
import networkx as nx
import numpy as np
import random

class GraphEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, num_nodes=20, K = 4):
        super(GraphEnv, self).__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict({
            'graph': spaces.Box(low=0, high=1, shape=(num_nodes, num_nodes), dtype=np.uint8)
        })
        self.graph = nx.Graph()
        self.initial_graph = nx.complete_graph(self.num_nodes)
        self.graph.add_nodes_from(range(num_nodes))
        self.start_id = 0  # Starting node for edge connection
        self.num_steps = 0

    def step(self, action):
        # Ensure action is valid
        if action not in range(self.num_nodes + 1):
            raise ValueError("Invalid action: {}".format(action))

        # Add edge to the graph
        
        self.graph.add_edge(self.start_id, action)
        self.graph.edges[self.start_id, action]['weight'] = self.initial_graph.edges[self.start_id, action]['weight']
        self.graph.edges[action, self.start_id]['weight'] =  self.initial_graph.edges[action, self.start_id]['weight']
        
        # print(self.num_steps, self.start_id, action, self.graph.degree())
        
        # Calculate reward (diameter of the graph)
        try:
            self.cur_diameter = nx.diameter(self.graph, weight='weight')
        except:
            None
        if self.start_id == action:
            reward = 0
        elif nx.is_connected(self.graph):
            reward = (100 + self.prev_diameter - self.cur_diameter) / (max(self.prev_diameter, self.cur_diameter) - 10)
        else:
            reward = 0  # or some penalty if the graph isn't connected

        # Prepare state
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        self.prev_diameter = self.cur_diameter
        total_degree = 0
        self.num_steps += 1
        for i in range(self.num_nodes):
            total_degree += self.graph.degree[i]
        if self.start_id >= self.num_nodes or self.start_id == action or self.num_steps >= ((self.num_nodes - 2) * self.K / 2):
            # print("Finished", self.start_id, action, self.num_steps)
            done = True  # You can define your own condition
        else:
            done = False
        if num_steps % 2 == 0 and num_steps != 0:
            self.start_id = self.start_id + 1 # Move start to last connected node
        state = {'initial_graph': self.initial_adjacency_matrix, 'graph': adjacency_matrix, 'start_id': self.start_id}

        return state, reward, done, {}

    def reset(self):
        self.graph.clear()
        # self.initial_graph.clear()
        self.graph.add_nodes_from(range(self.num_nodes))
        self.start_id = 0
        # self.initial_graph = nx.complete_graph(self.num_nodes)
        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0
        for (u, v) in self.initial_graph.edges():
            self.initial_graph.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
        for (u, v) in self.initial_graph.edges():
            self.initial_graph.edges[v,u]['weight'] = self.initial_graph.edges[u,v]['weight']  # Assign random positive weights
        
        self.initial_adjacency_matrix = nx.to_numpy_array(self.initial_graph, nodelist=sorted(self.initial_graph.nodes()))
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.initial_graph.nodes()))
        return {'initial_graph': self.initial_adjacency_matrix, 
                'graph': adjacency_matrix, 'start_id': self.start_id}

    def render(self, mode='console'):
        if mode == 'console':
            print(nx.info(self.graph))