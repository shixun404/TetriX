import gym
from gym import spaces
import networkx as nx
import numpy as np
import random
import pickle as pkl
from math import exp
import os
from collections import Counter
# import torch
class GraphEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, num_nodes=500, K=8):
        super(GraphEnv, self).__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict({
            'graph': spaces.Box(low=0, high=1, shape=(num_nodes, num_nodes), dtype=np.uint8)
        })
        self.graph = nx.Graph()
        self.initial_graph = nx.complete_graph(self.num_nodes)
        self.test_graphs = []
        self.test_id = -1
        self.graph.add_nodes_from(range(num_nodes))
        self.start_id = 0  # Starting node for edge connection
        self.num_steps = 0
        self.mask = []
        self.if_test = False
        self.mean = 5
        self.std_dev = 1
        self.load_graph()

    def reset(self, if_test=False, start_id=0, test_id=0):
        self.if_test = if_test
        self.graph.clear()
        self.mask = [self.K for i in range(self.num_nodes)]
        mask = [1 for i in range(self.num_nodes)]
        mask[start_id] = 0
        self.initial_graph.clear() 
        self.graph.add_nodes_from(range(self.num_nodes))
        self.start_id = start_id
        self.test_id = test_id
        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0
        if if_test:        
            self.initial_graph = nx.Graph(self.test_graphs[self.test_id])  
        else:
            self.initial_graph = nx.complete_graph(self.num_nodes)
            for (u, v) in self.initial_graph.edges():
                # self.initial_graph.edges[u, v]['weight'] = random.randint(1, 10)  # Assign random positive weights
                self.initial_graph.edges[u, v]['weight'] = np.random.normal(self.mean,
                                                                             self.std_dev)
            for (u, v) in self.initial_graph.edges():
                self.initial_graph.edges[v,u]['weight'] = self.initial_graph.edges[u,v]['weight']  # Assign random positive weights
        self.initial_adjacency_matrix = nx.to_numpy_array(self.initial_graph, nodelist=sorted(self.initial_graph.nodes()))
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.initial_graph.nodes()))
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        return {'initial_graph': self.initial_adjacency_matrix, 
                'graph': adjacency_matrix, 'start_id': self.start_id, 'mask': mask, 'degree': degree}
    
    def load_graph(self, mode='gaussian'):
        # graph_name=f'G_N={self.num_nodes}_Gaussian.pkl'
        for i in range(1):
            graph_name = f'N={self.num_nodes}_{i}_{mode}.pkl'
            with open(os.path.join('.', 'test_graph', graph_name), 'rb') as f:
                self.test_graphs.append(pkl.load(f))
        # assert 0

    def step(self, action): 
        if action not in range(self.num_nodes + 1):
            raise ValueError("Invalid action: {}".format(action))
        self.graph.add_edge(self.start_id, action)
        self.graph.edges[self.start_id, action]['weight'] = self.initial_graph.edges[self.start_id, action]['weight']
        self.graph.edges[action, self.start_id]['weight'] =  self.initial_graph.edges[action, self.start_id]['weight']
        if self.if_test == False:
            try:
                # self.cur_diameter = nx.diameter(self.graph, weight='weight')
                shortest_length = nx.shortest_path_length(self.graph, source=self.start_id, weight='weight')
                self.cur_diameter = max(shortest_length.values())
            except:
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                # self.cur_diameter = nx.diameter(subgraph)
                shortest_length = nx.shortest_path_length(subgraph, source=0, weight='weight')
                self.cur_diameter = max(shortest_length.values())
                # self.cur_diameter = 0
            reward = self.prev_diameter - self.cur_diameter -   self.initial_graph.edges[self.start_id, action]['weight']
        else:
            reward = 0
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        self.prev_diameter = self.cur_diameter
        self.num_steps += 1
        
        self.mask[action] -= 1
        self.mask[self.start_id] -= 1
        
        
        max_value = np.max(self.mask)   
        min_value = np.min(self.mask)   
        mask = (self.mask == max_value).astype(int)

        i = 0
        for i in range(self.num_nodes):
            if self.graph.has_edge(action, i) or action == i:
                mask[i] = 0
        if min_value != 0 and sum(mask) == 0:
            for i in range(self.num_nodes):
                mask[i] = 1
            mask[action] = 0
        self.start_id = action
        
        if self.num_steps >= ((self.num_nodes) * self.K / 2) or sum(mask) == 0:
            mask = [1 for i in range(self.num_nodes)]
            done = True  # You can define your own condition
            
        else:
            done = False
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        state = {'initial_graph': self.initial_adjacency_matrix, 'graph': adjacency_matrix, 
                'start_id': self.start_id, 'mask': mask, 'degree': degree}

        return state, reward, done, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(nx.info(self.graph))