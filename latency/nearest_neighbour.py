import networkx as nx
import numpy as np
import random
import os
import pickle as pkl

def KNN(G, num_nodes, K):
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        current_node = i
        neighbors = list(G.neighbors(current_node))
        neighbors.sort(key=lambda x: (G.edges[i, x]['weight']))
        for k in range(K):
            subgraph.add_edge(i, neighbors[k])
            next_node = neighbors[k]
            subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
            subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
    # print(subgraph.degree())
    return nx.diameter(subgraph, weight='weight')


def KNN_with_degree_constraint(G, num_nodes, K, order):
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(num_nodes))
    for i in order:
        current_node = i
        neighbors = list(G.neighbors(current_node))
        neighbors.sort(key=lambda x: (G.edges[i, x]['weight']))
        degree = subgraph.degree()
        j = 0
        while degree[i] < K and j < len(neighbors):
            if subgraph.has_edge(current_node, neighbors[j]) or current_node == neighbors[j] or degree[neighbors[j]] >= K:
                a = 0
            else:
                subgraph.add_edge(i, neighbors[j])
                next_node = neighbors[j]
                subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
                subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
            j += 1
    # print(subgraph.degree())
    return nx.diameter(subgraph, weight='weight')

def perform_random_walk(G, num_nodes, start_node, num_steps):
    current_node = start_node
    visited = [current_node]
    degree = [0 for i in range(num_nodes)]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(num_nodes))
    for _ in range(num_steps - 1):
        degree[current_node] += 1
        neighbors = list(G.neighbors(current_node))
        # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
        neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
        
        # Move to the neighbor with the highest priority
        for i in range(len(neighbors)):
            if subgraph.has_edge(current_node, neighbors[i]) or current_node == neighbors[i]:
                continue
            else:
                next_node = neighbors[i]
                break
        subgraph.add_edge(current_node, next_node)
        subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
        subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
        current_node = next_node
        
    # print(subgraph.degree())
    return nx.diameter(subgraph, weight='weight')


# Main execution
if __name__ == '__main__':
    
    # plot_histo()
    
    N = 20
    k = 4
    num_steps = k * N // 2
    graph_name = f'G_N={N}.pkl'
    G = nx.Graph()
    
    diameter_list = []
    for i in range(10):
        graph_name = f'N={N}_{i}.pkl'
        with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
            G = pkl.load(f)
        diameter = perform_random_walk(G, N, 0, num_steps)
        diameter_list.append(diameter)
        print(f"Test G {i}: {diameter}")  
    print(sum(diameter_list) / 10)
    

    # diameter_list = []
    # for i in range(10):
    #     graph_name = f'N={N}_{i}.pkl'
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
    #     diameter = KNN(G, N, k)
    #     diameter_list.append(diameter)
    #     print(f"Test G {i}: {diameter}")  
    # print(sum(diameter_list) / 10)
    
    # diameter_list = []
    # cm_diameter_list = []
    # for i in range(10):
    #     graph_name = f'N={N}_{i}.pkl'
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
        
    #     best_diameter = 1e6
    #     cnt = 0
    #     cumulative_d = 0
        
    #     for id in range(N):
    #         order = [(j + id) % N for j in range(N)]
    #         try:
    #             diameter = KNN_with_degree_constraint(G, N, k, order)
    #             cnt += 1
    #             cumulative_d += diameter
    #         except:
    #             diameter = 1e6
    #         best_diameter = diameter if diameter < best_diameter else best_diameter
    #     diameter_list.append(best_diameter)
    #     cm_diameter_list.append(cumulative_d / cnt)
    #     print(f"Test G {i}: {best_diameter}, prob={cnt / 10}, avg_d={cumulative_d / cnt}")  
    # print(sum(diameter_list) / 10, sum(cm_diameter_list) / 10)
    