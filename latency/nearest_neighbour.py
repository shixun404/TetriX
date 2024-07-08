import networkx as nx
import numpy as np
import random
import os
import pickle as pkl
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time

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
                # print(G.edges[current_node, next_node]['weight'])
            j += 1
        # assert 0
            
    # print(subgraph.degree())
    return nx.diameter(subgraph, weight='weight')

def perform_random_walk(G, num_nodes, start_node, num_steps, gid, if_plot=False):
    
    current_node = start_node
    visited = [current_node]
    degree = [0 for i in range(num_nodes)]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(num_nodes))
    positions = nx.circular_layout(G)
    # nx.draw(subgraph, pos, ax=ax, node_color='lightblue', with_labels=True)
    # nx.draw(subgraph, pos, ax=ax, node_color='lightblue', with_labels=True)
    # Save the animation as a GIF file
    frames = []
    fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(20, 35))  # Adjust figsize to fit your screen
    axes = axes.flatten()  # Flatten the array of axes
    t = time.time()
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
        if if_plot:
            special_edge = (current_node, next_node)
            ax = axes[_]
            # nx.draw(subgraph, ax=ax)
            nx.draw_networkx_nodes(subgraph, pos=positions, node_color='lightblue',  ax=ax)
            edges = subgraph.edges(data=True)
            
            special_edges = [edge for edge in subgraph.edges(data=True) if ((edge[0], edge[1]) == special_edge or (edge[1], edge[0]) == special_edge)]
            nx.draw_networkx_edges(subgraph, pos=positions, edgelist=edges, width=[w['weight'] for (u, v, w) in edges],  ax=ax)
            nx.draw_networkx_edges(subgraph, pos=positions, edgelist=special_edges, 
                                width=[w['weight'] for (u, v, w) in special_edges], 
                                edge_color='red', ax=ax)
            nx.draw_networkx_labels(subgraph, pos=positions, font_weight='bold', ax=ax)
            edge_labels = {(u, v): w['weight'] for (u, v, w) in subgraph.edges(data=True)}
            nx.draw_networkx_edge_labels(subgraph, pos=positions, edge_labels=edge_labels,  ax=ax)

        if if_plot:
            try:
                d = nx.diameter(subgraph, weight='weight')
            except:
                # d = 'inf'
                largest_cc = max(nx.connected_components(subgraph), key=len)
                subgraph_largest_cc = subgraph.subgraph(largest_cc)
                d = nx.diameter(subgraph_largest_cc)
            ax.set_title(f"Step {_+1} d={d}")
            ax.axis('off') 
    if if_plot:
        plt.tight_layout()
        plt.savefig(f'nn_N=20_{gid}.png')
    print(time.time() - t) 
    d = nx.diameter(subgraph, weight='weight')
    return d


def chord(G, num_nodes, degree):
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        current_node = i
        j = 0
        
        while j < degree:
            next_node = (i + 2 ** j) % num_nodes
            subgraph.add_edge(current_node, next_node)
            subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
            subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
            j += 1

    print(subgraph.degree())
    return nx.diameter(subgraph, weight='weight')

def random_permutation(n):
    # Create a list of numbers from 0 to n-1
    numbers = list(range(n))
    # Shuffle the list in place to create a random permutation
    random.shuffle(numbers)
    return numbers

# # Example usage
# n = 10  # Generate a random permutation of numbers from 0 to 9
# perm = random_permutation(n)
# print(perm)

def K_ring(G, num_nodes, degree):
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(num_nodes))
    for i in range(degree // 2):
        numbers = list(range(num_nodes))
        random.shuffle(numbers)
        
        for j in range(num_nodes):
            current_node = numbers[j]
            next_node = numbers[(j + 1) % num_nodes]
            subgraph.add_edge(current_node, next_node)
            subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
            subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
            j += 1

    print(subgraph.degree())
    time = time.time()
    d = nx.diameter(subgraph, weight='weight')
    print(time.time() - time)
    return d

# Main execution
if __name__ == '__main__':
    
    # plot_histo()
    
    # N = 100
    # k = 6
    N = 500
    k = 8
    num_tests = 5
    num_steps = k * N // 2
    graph_name = f'G_N={N}.pkl'
    G = nx.Graph()
    diameter_list = []
    for i in range(num_tests):
        graph_name = f'N={N}_{i}.pkl'
        diameter = 1e9
        with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
            G = pkl.load(f)
        for start_id in range(4):
        # for start_id in range(1):
            diameter = min(diameter, perform_random_walk(G, N, start_id, num_steps, i))
        diameter_list.append(diameter)
        print(f"Test G {i}: {diameter}")  
    print(diameter_list)
    print(sum(diameter_list) / len(diameter_list))
    
    # diameter_list = []
    # for i in range(num_tests):
    #     graph_name = f'N={N}_{i}.pkl'
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
    #     diameter = chord(G, N, k / 2)
    #     diameter_list.append(diameter)
    #     print(f"Test G {i}: {diameter}")  
    # print(sum(diameter_list) / len(diameter_list))
    # assert 0

    # diameter_list = []
    # for i in range(num_tests):
    #     graph_name = f'N={N}_{i}.pkl'
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
    #     diameter = K_ring(G, N, k)
    #     diameter_list.append(diameter)
    #     print(f"Test G {i}: {diameter}")  
    # print(sum(diameter_list) / len(diameter_list))
    # assert 0

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
    # for i in range(num_tests):
    #     graph_name = f'N={N}_{i}.pkl'
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
    #     # print(nx.adjacency_matrix(G, weight='weight'))
    #     # assert 0
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
    #     print(f"Test G {i}: {best_diameter}, prob={cnt / num_tests}, avg_d={cumulative_d / cnt}")  
    # print(sum(diameter_list) / num_tests, sum(cm_diameter_list) / num_tests)
    