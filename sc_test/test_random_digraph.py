import numpy as np
import sys
import networkx as nx
import pickle as pkl
import os
import time
import random
import torch
import matplotlib.pyplot as plt
from test_nn import perform_random_walk_directed


def generate_fixed_degree_digraph(G, N, K, seed=42, perm=None):
    """
    Generate a directed graph where each node has exactly K in-degree and K out-degree.
    The edge weights are copied from the input graph G.

    Parameters:
    - G: Input fully connected networkx.DiGraph
    - K: Desired in-degree and out-degree per node
    - seed: Random seed for reproducibility (optional)

    Returns:
    - new_G: A new networkx.DiGraph with fixed in-degree and out-degree K
    """


    N = G.number_of_nodes()
    assert K < N, "K must be less than the number of nodes (N), otherwise a valid structure is impossible."

    # Initialize the new directed graph
    new_G = nx.DiGraph()
    new_G.add_nodes_from(G.nodes())

    if perm is not None:
        for i in range(N):
            # 下一个节点索引 (环状，最后连回第 0 个)
            next_i = (i + 1) % N
            new_G.add_edge(perm[i], perm[next_i], weight=G[perm[i]][perm[next_i]]['weight'])

    # Step 1: Assign exactly K outgoing and K incoming edges per node
    for node in G.nodes():
        # 1. Select K unique outgoing neighbors
        possible_out_neighbors = [n for n in G.nodes() if n != node]  # Avoid self-loops
        out_neighbors = random.sample(possible_out_neighbors, K)
        
        # 2. Select K unique incoming neighbors
        possible_in_neighbors = [n for n in G.nodes() if n != node]  # Avoid self-loops
        in_neighbors = random.sample(possible_in_neighbors, K)

        # Ensure the directed edges exist and assign the correct weight
        for target in out_neighbors:
            if G.has_edge(node, target):
                weight = G[node][target]['weight']
                new_G.add_edge(node, target, weight=weight)

        for source in in_neighbors:
            if G.has_edge(source, node):
                weight = G[source][node]['weight']
                new_G.add_edge(source, node, weight=weight)

    return new_G

def generate_weight_prioritized_digraph(G, N, K, seed=None, perm=None):
    """
    Generate a directed graph where each node has exactly K in-degree and K out-degree.
    The edge weights are preserved from the input graph G, and edges with smaller 
    weights are prioritized. If multiple edges have the same weight, selection is random.

    Parameters:
    - G: Input networkx.DiGraph (fully connected)
    - K: Desired in-degree and out-degree per node
    - seed: Random seed for reproducibility (optional)

    Returns:
    - new_G: A new networkx.DiGraph with fixed in-degree and out-degree K
    """


    N = G.number_of_nodes()
    assert K < N, "K must be less than N to ensure a valid degree-constrained graph."

    # Initialize the new directed graph
    new_G = nx.DiGraph()
    new_G.add_nodes_from(G.nodes())

        
    if perm is not None:
        for i in range(N):
            # 下一个节点索引 (环状，最后连回第 0 个)
            next_i = (i + 1) % N
            new_G.add_edge(perm[i], perm[next_i], weight=G[perm[i]][perm[next_i]]['weight'])

    # Step 1: Sort edges based on weight, breaking ties randomly
    all_edges = list(G.edges(data=True))
    random.shuffle(all_edges)  # Shuffle first to ensure tie-breaking is random
    all_edges.sort(key=lambda x: x[2]['weight'])  # Sort by weight (ascending)

    # Step 2: Initialize in-degree and out-degree trackers
    in_degree_count = {node: 0 for node in G.nodes()}
    out_degree_count = {node: 0 for node in G.nodes()}

    # Step 3: Iterate over sorted edges and select valid ones
    for u, v, data in all_edges:
        if in_degree_count[v] < K and out_degree_count[u] < K:
            new_G.add_edge(u, v, weight=data['weight'])
            in_degree_count[v] += 1
            out_degree_count[u] += 1

        # Stop if we have fulfilled the degree constraints for all nodes
        if all(in_degree_count[n] == K and out_degree_count[n] == K for n in G.nodes()):
            break

    return new_G

def generate_weight_prioritized_digraph(G, N, K, seed=None, perm_list=None, epsilon=0.95):
    """
    Generate a directed graph where:
    - Each node has exactly K outgoing edges (prioritizing lower weight edges)
    - Each node has exactly K incoming edges (enforced by constraints)
    - Edge weights are preserved from the input graph G
    - If multiple edges have the same weight, selection is random

    Parameters:
    - G: Input networkx.DiGraph (fully connected)
    - K: Desired in-degree and out-degree per node
    - seed: Random seed for reproducibility (optional)

    Returns:
    - new_G: A new networkx.DiGraph with fixed in-degree and out-degree K
    """
    N = G.number_of_nodes()
    assert K < N, "K must be less than N to ensure a valid degree-constrained graph."

    # Initialize the new directed graph
    new_G = nx.DiGraph()
    new_G.add_nodes_from(G.nodes())

    if perm_list is not None:
        for j in range(len(perm_list)):
            perm = perm_list[j]
            for i in range(N):
                # 下一个节点索引 (环状，最后连回第 0 个)
                next_i = (i + 1) % N
                new_G.add_edge(perm[i], perm[next_i], weight=G[perm[i]][perm[next_i]]['weight'])

    # Step 1: Initialize in-degree and out-degree counters
    in_degree_count = {node: 0 for node in G.nodes()}
    out_degree_count = {node: 0 for node in G.nodes()}

    # Step 2: Iterate over each node to assign outgoing edges first
    for node in G.nodes():
        # Get all outgoing edges from 'node' sorted by weight (ascending)
        outgoing_edges = [(v, G[node][v]['weight']) for v in G.nodes() if v != node]
        
        # Shuffle first to break ties randomly, then sort by weight
        random.shuffle(outgoing_edges)
        outgoing_edges.sort(key=lambda x: x[1])  # Sort by weight (low first)

        # Select up to K lowest-weight outgoing edges
        selected_outgoing = []
        while len(selected_outgoing) < K:
            for target, weight in outgoing_edges:
                if random.random() < epsilon:    
                    if len(selected_outgoing) < K and in_degree_count[target] < K:
                        selected_outgoing.append((node, target, weight))
                        in_degree_count[target] += 1  # Increase target node's in-degree

        # Add selected outgoing edges to new graph
        for u, v, weight in selected_outgoing:
            new_G.add_edge(u, v, weight=weight)
            out_degree_count[u] += 1  # Increase source node's out-degree

    return new_G


if __name__ == '__main__':
    N = 100  # Number of nodes
    K = 3   # Fixed in-degree and out-degree per node
    M = 1
    seed = 42
    epsilon = float(sys.argv[1])

    # Generate a fully connected directed graph
    with open(f'G_{N}.pkl', 'rb') as f:
        G = pkl.load(f)
    diameter_list = []
    random.seed(seed)
    perm_list = []
    for i in range(M):
        perm = random.sample(range(N), N)
        perm_list.append(perm)
    for i in range(3000):
        # new_G = generate_fixed_degree_digraph(G, N, K - 1, seed=seed, perm=perm)
        # new_G = generate_weight_prioritized_digraph(G, N, K - M, seed=seed, perm_list=perm_list)
        d, new_G = perform_random_walk_directed(G, N, 0, K * N, greedy=False, epsilon=epsilon)
        # new_G = generate_weight_prioritized_digraph(G, N, K - 1, seed=seed, perm=perm)
        diameter_list.append(d)
        if i % 500 == 0:
            diameter_tensor = torch.tensor(diameter_list)
            print(f"diameter mean={diameter_tensor.mean()}, std={diameter_tensor.std()}, max={diameter_tensor.max()}, min={diameter_tensor.min()}")
            plt.figure()
            plt.hist(diameter_list, bins=50)
            plt.xlabel('Diameter')
            plt.ylabel('Frequency')
            plt.title(f'Diameter Distribution for {N} nodes with K={K} DGRO epsilon={epsilon}')
            plt.savefig(f'diameter_distribution_{N}_{K}_DGRO_epsilon_greedy_eps={epsilon}.png')
            with open(f'diameter_list_{N}_{K}_DGRO_epsilon_greedy_eps={epsilon}.pkl', 'wb') as f:
                pkl.dump(diameter_list, f)
