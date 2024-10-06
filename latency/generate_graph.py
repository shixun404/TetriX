import networkx as nx
import numpy as np
import os
import random
import pickle as pkl
from collections import Counter
N_list = [100]
seed = 1
random.seed(seed)
for i in range(500, 5001, 500):
    N_list.append(i)
mode = ['uniform', 'gaussian']
mode = mode[1]
for num_nodes in N_list:
    for i in range(1):
        graph_name = f'N={num_nodes}_{i}_{mode}.pkl'
        if graph_name not in os.listdir(os.path.join('.', 'ipdps_test')):
            test_graph = nx.complete_graph(num_nodes)
            for (u, v) in test_graph.edges():
                if mode == 'uniform':
                    test_graph.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
                else:
                    test_graph.edges[u,v]['weight'] = np.random.normal(self.mean,
                                                                             self.std_dev)  # Assign random positive weights
                #  test_graph.edges[u, v]['weight'] = np.random.normal(self.mean, self.std_dev)
            for (u, v) in test_graph.edges():
                test_graph.edges[v,u]['weight'] = test_graph.edges[u,v]['weight']  # Assign random positive weights
            with open(os.path.join('.', 'test_dataset', graph_name), 'wb') as f:
                pkl.dump(test_graph, f)
           