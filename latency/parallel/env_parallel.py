import gym
from gym import spaces
import networkx as nx
import numpy as np
import random
import pickle as pkl
from math import exp
import os
from collections import Counter
import torch as th
class GraphEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, num_nodes=500, K=8, num_sources=1, M=1):
        super(GraphEnv, self).__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.M = M
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict({
            'graph': spaces.Box(low=0, high=1, shape=(num_nodes, num_nodes), dtype=np.uint8)
        })
        self.graph = nx.DiGraph()
        self.initial_graph = nx.complete_graph(self.num_nodes)
        self.test_graphs = []
        self.test_id = -1
        self.graph.add_nodes_from(range(num_nodes))
        self.start_id = [i * (self.num_nodes // self.M) for i in range(self.M)]
        self.num_steps = 0
        self.mask = []
        self.if_test = False
        self.mean = 5
        self.std_dev = 1
        self.num_sources = num_sources
        self.load_graph()

    def reset(self, if_test=False, start_id=None, test_id=0):
        self.if_test = if_test
        self.graph.clear()
        self.mask = [self.K for i in range(self.num_nodes)]
        mask = [1 for i in range(self.num_nodes)]
        for i in range(self.M):
            mask[start_id[i]] = 0
        self.initial_graph.clear() 
        self.graph.add_nodes_from(range(self.num_nodes))
        self.start_id = start_id
        self.test_id = test_id
        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0
        with open('../sc_test/G_400_FABRIC.pkl', 'rb') as f:
            graph_background = pkl.load(f)
        if if_test:
            self.initial_graph = nx.Graph(self.test_graphs[self.test_id])  
        else:
            self.initial_graph = nx.complete_graph(self.num_nodes)
            self.initial_graph.add_nodes_from(range(self.num_nodes))
            for (u, v) in self.initial_graph.edges():
                self.initial_graph.edges[u, v]['weight'] = graph_background.edges[u, v]['weight']
            for (u, v) in self.initial_graph.edges():
                self.initial_graph.edges[v,u]['weight'] = self.initial_graph.edges[u,v]['weight']  # Assign random positive weights
        self.initial_adjacency_matrix = nx.to_numpy_array(self.initial_graph, nodelist=sorted(self.initial_graph.nodes()))
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.initial_graph.nodes()))
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        return {'initial_graph': self.initial_adjacency_matrix, 
                'graph': adjacency_matrix, 'start_id': self.start_id, 'mask': mask, 'degree': degree}
    
    def load_graph(self,):
        with open(f'../sc_test/G_400_FABRIC.pkl', 'rb') as f:
            graph_background = pkl.load(f)
            self.test_graphs.append(graph_background)

    def step(self, action): 
        weight_sum = 0
        for i in range(self.M):
            try:
                self.graph.add_edge(self.start_id[i], action[i])
            except:
                pass
            w = self.initial_graph.edges[self.start_id[i], action[i]]['weight']
            self.graph.edges[self.start_id[i], action[i]]['weight'] = w
            weight_sum += w
        if self.if_test == False:
            try:
                self.cur_diameter = 0
                for i in range(self.num_sources):
                    shortest_length = nx.shortest_path_length(self.graph, source=i, weight='weight')
                    self.cur_diameter = max(self.cur_diameter, max(shortest_length.values()))
            except:
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                self.cur_diameter = 0
                for i in range(self.num_sources):
                    shortest_length = nx.shortest_path_length(subgraph, source=i, weight='weight')
                    self.cur_diameter = max(self.cur_diameter, max(shortest_length.values()))
            reward = self.prev_diameter - self.cur_diameter - weight_sum
        else:
            reward = 0
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        self.prev_diameter = self.cur_diameter
        self.num_steps += 1
        for i in range(self.M):
            self.mask[action[i]] -= 1
            self.mask[self.start_id[i]] -= 1
        
        max_value = np.max(self.mask)   
        min_value = np.min(self.mask)   
        mask = (self.mask == max_value).astype(int)

        i = 0
        # for i in range(self.num_nodes):
        #     for j in range(self.M):
        #         if self.graph.has_edge(action[j], i) or action[j] == i:
        #             mask[i] = 0
        
        # if min_value != 0 and sum(self.mask) % (self.num_nodes * 2) == 0:
        #     print("reset", self.num_steps, sum(mask))
        #     for i in range(self.num_nodes):
        #         mask[i] = 1
        #     for i in range(self.M):
        #         mask[action[i]] = 0
        # print(self.num_steps, 'start_id', self.start_id, 'action', action, 'mask', sum(mask), 'self.mask', sum(self.mask),'min', min_value)
        for i in range(self.M):
            self.start_id[i] = action[i]
            mask[action[i]] = 0
        
        # print(self.num_steps, 'mask', sum(mask), th.where(th.as_tensor(mask) != 0), 'self.mask', sum(self.mask),  th.where(th.as_tensor(self.mask) != 0),'min', min_value)
        if self.num_steps >= ((self.num_nodes) * self.K // self.M):
            mask = [1 for i in range(self.num_nodes)]
            done = True
            
        else:
            done = False
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        state = {'initial_graph': self.initial_adjacency_matrix, 'graph': adjacency_matrix, 
                'start_id': self.start_id, 'mask': mask, 'degree': degree}

        return state, reward, done, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(nx.info(self.graph))