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
                self.initial_graph.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
            for (u, v) in self.initial_graph.edges():
                self.initial_graph.edges[v,u]['weight'] = self.initial_graph.edges[u,v]['weight']  # Assign random positive weights
        
        self.initial_adjacency_matrix = nx.to_numpy_array(self.initial_graph, nodelist=sorted(self.initial_graph.nodes()))
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.initial_graph.nodes()))
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        return {'initial_graph': self.initial_adjacency_matrix, 
                'graph': adjacency_matrix, 'start_id': self.start_id, 'mask': mask, 'degree': degree}
    
    def load_graph(self,):
        graph_name=f'G_N={self.num_nodes}.pkl'
        for i in range(50):
            graph_name = f'N={self.num_nodes}_{i}.pkl'
            if graph_name not in os.listdir(os.path.join('.', 'test_dataset')):
                test_graph = nx.complete_graph(self.num_nodes)
                for (u, v) in self.initial_graph.edges():
                    test_graph.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
                for (u, v) in self.initial_graph.edges():
                    test_graph.edges[v,u]['weight'] = test_graph.edges[u,v]['weight']  # Assign random positive weights
                with open(os.path.join('.', 'test_dataset', graph_name), 'wb') as f:
                    pkl.dump(test_graph, f)
                adj_matrix = nx.adjacency_matrix(test_graph, weight='weight')
                weights_list = adj_matrix.tocoo().data.tolist()
                weight_counts = Counter(weights_list)
                self.test_graphs.append(test_graph)
            else:
                with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
                    self.test_graphs.append(pkl.load(f))

    def step(self, action):
        # Ensure action is valid
        # print(self.num_steps, action)
        
        if action not in range(self.num_nodes + 1):
            raise ValueError("Invalid action: {}".format(action))

        # Add edge to the graph
        # print(self.start_id, action)
        # print(self.graph)
        self.graph.add_edge(self.start_id, action)
        # print(self.start_id, action, self.num_steps)
        self.graph.edges[self.start_id, action]['weight'] = self.initial_graph.edges[self.start_id, action]['weight']
        self.graph.edges[action, self.start_id]['weight'] =  self.initial_graph.edges[action, self.start_id]['weight']
        
        
        
        # Calculate reward (diameter of the graph)
        if self.if_test == False:
            try:
                self.cur_diameter = nx.diameter(self.graph, weight='weight')
            except:
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                self.cur_diameter = nx.diameter(subgraph)
                # self.cur_diameter = 0
            reward = self.prev_diameter - self.cur_diameter -  0.1 * self.initial_graph.edges[self.start_id, action]['weight']
        else:
            reward = 0

        # Prepare state
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        self.prev_diameter = self.cur_diameter
        self.num_steps += 1
       
        # if (self.num_steps % 2 == 0 and self.num_steps != 0) or self.graph.degree[self.start_id] > self.K:
        # self.start_id = (self.start_id + 1) % self.num_nodes # Move start to last connected node
        
        self.mask[action] -= 1
        self.mask[self.start_id] -= 1
        
        
        max_value = np.max(self.mask)   
        min_value = np.min(self.mask)   
        mask = (self.mask == max_value).astype(int)

        i = 0
        for i in range(self.num_nodes):
            # 检查是否存在从节点0到当前节点的边
            if self.graph.has_edge(action, i) or action == i:
                mask[i] = 0
        if min_value != 0 and sum(mask) == 0:
            for i in range(self.num_nodes):
                mask[i] = 1
            mask[action] = 0
        # print(self.num_steps, (self.num_nodes) * self.K / 2, 'mask_sum:',sum(self.mask), 'mask_max:', np.max(self.mask),
            #  f'edge {self.start_id} --> {action} ')
        # print(mask)
        self.start_id = action
        # print(mask)
        
        if self.num_steps >= ((self.num_nodes) * self.K / 2) or sum(mask) == 0:
            # print("Finished", self.start_id, action, self.num_steps, self.graph.degree())
            mask = [1 for i in range(self.num_nodes)]
            done = True  # You can define your own condition
            
            # assert 0
        else:
            done = False
        # print(self.num_steps, self.mask, mask)
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        state = {'initial_graph': self.initial_adjacency_matrix, 'graph': adjacency_matrix, 
                'start_id': self.start_id, 'mask': mask, 'degree': degree}

        return state, reward, done, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(nx.info(self.graph))