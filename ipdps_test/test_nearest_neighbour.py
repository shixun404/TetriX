import networkx as nx
import numpy as np
import random
import os
import pickle as pkl
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time
import torch as th
import pickle as pkl

diameter_list = {
    'chord_shortest_ring': [],
    'chord_random_ring': [],
    'nearest_neighbour_shortest_ring': [],
    'nearest_neighbour_random_ring': [],
    'K_ring_shortest_ring': [],
    # 'K_ring_distributed': [],
    'K_ring_random_ring': [],
    'K_ring_greedy': [],
    'K_ring_epsilon_greedy': [],
}
def KNN(G, num_nodes, K, random_ring=True, chord=True, greedy=True):
    subgraph = nx.DiGraph()
    degree = [0 for i in range(num_nodes)]
    subgraph.add_nodes_from(range(num_nodes))
    current_node = 0
    new_order = [i for i in range(num_nodes)]
    random.shuffle(new_order)
    current_node = new_order[0]

    if random_ring is True:
        for i in range(num_nodes):
            # new_order.append(i)
            next_node = new_order[(i + 1) % num_nodes]
            subgraph.add_edge(current_node, next_node)
            subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
            current_node = next_node
    else:
        vis = [current_node]
        for _ in range(num_nodes):
            # print(current_node)
            # new_order.append(current_node)
            degree[current_node] += 1
            neighbors = list(G.neighbors(current_node))
            # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
            neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
            # print(neighbors)
            # assert 0
            # Move to the neighbor with the highest priority
            i = 0
            while _ != num_nodes - 1:
                if neighbors[i % len(neighbors)] in vis or current_node == neighbors[i % len(neighbors)]:    
                    i += 1
                    continue
                else:
                    # if random.random() < 0.99:
                    next_node = neighbors[i % len(neighbors)]
                    break
            if _ == num_nodes - 1:
                next_node = new_order[0]
            subgraph.add_edge(current_node, next_node)
            subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
            current_node = next_node
            vis.append(current_node)
        
    for i in range(num_nodes):
        current_node = new_order[i]
        neighbors = list(G.neighbors(current_node))
        neighbors.sort(key=lambda x: (G.edges[current_node, x]['weight']))
        # subgraph.add_edge(i, (i + 1) % num_nodes)
        # subgraph.edges[i, (i + 1) % num_nodes]['weight'] = G.edges[i, (i + 1) % num_nodes]['weight']
        # current_node = new_order[i]
        next_node = new_order[(i + 1) % num_nodes]
        # print(G.edges[current_node, next_node]['weight'])
        # subgraph.add_edge((i - 1) % num_nodes, i)
        # subgraph.edges[(i - 1) % num_nodes, i]['weight'] = G.edges[(i - 1) % num_nodes, i]['weight']
        k = 0
        id = 0
        while k < K - 1 and id < len(neighbors):
            if not chord:
                # print(id, len(neighbors))
                next_node = neighbors[id]
                id += 1
                if subgraph.in_degree()[next_node] >= K or next_node == current_node:
                    continue
            else:
                next_node = new_order[(i + 2 ** (k + 1)) % num_nodes]
            k += 1
            subgraph.add_edge(current_node, next_node)
            
            # print(G.edges[current_node, next_node]['weight'])
            subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
            # subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
        # assert 0
    # print(subgraph.degree())
    d = 0
    # print(subgraph.in_degree())
    # print(subgraph.out_degree())
    in_degrees = [d for n, d in subgraph.in_degree()]
    out_degrees = [d for n, d in subgraph.out_degree()]
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.hist(in_degrees, bins=range(min(in_degrees), max(in_degrees) + 2), edgecolor='black', alpha=0.7)
    # plt.title('In-Degree Distribution')
    # plt.xlabel('In-Degree')
    # plt.ylabel('Frequency')

    # # Plot histogram for out-degree
    # plt.subplot(1, 2, 2)
    # plt.hist(out_degrees, bins=range(min(out_degrees), max(out_degrees) + 2), edgecolor='black', alpha=0.7)
    # plt.title('Out-Degree Distribution')
    # plt.xlabel('Out-Degree')
    # plt.ylabel('Frequency')

    # plt.tight_layout()
    # plt.savefig("histo.pdf")
    try:
        d = nx.diameter(subgraph, weight='weight')
    except:
        scc = nx.strongly_connected_components(subgraph)
        cnt = 0
        for i in scc:
            cnt += len(i)
            # print(len(i), cnt)
        largest_cc = max(nx.strongly_connected_components(subgraph), key=len)
        subgraph = subgraph.subgraph(largest_cc)
        d = nx.diameter(subgraph)
    weight_sum = sum(data['weight'] for u, v, data in subgraph.edges(data=True))
    print("KNN sum of the graph:", weight_sum / K)
    return d


def generate_k_directed_rings(G, num_nodes, K, random_ring=True, greedy=True, num_random_ring=0):
    nodes = list(G.nodes())
    rings = []
    for _ in range(num_random_ring):
        ring_nodes = nodes.copy()
        random.shuffle(ring_nodes)
        ring = [(ring_nodes[i], ring_nodes[(i+1) % len(ring_nodes)]) for i in range(len(ring_nodes))]
        rings.append(ring)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    # if not random_ring:
    degree = [0 for i in range(num_nodes)]
    new_order = [i for i in range(num_nodes)]
    random.shuffle(new_order)
    current_node = new_order[0]
    # current_node = 0
    # for j in range(K - num_random_ring):
    #     new_order = []
    #     for _ in range(num_nodes):
    #         new_order.append(current_node)
    #         neighbors = list(G.neighbors(current_node))
    #         # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
    #         neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
            
    #         # Move to the neighbor with the highest priority
    #         i = 0
    #         while _ != num_nodes - 1:
    #             # if H.has_edge(current_node, neighbors[i]) or current_node == neighbors[i]:
    #             # print(i)
    #             if neighbors[i % len(neighbors)] in new_order or current_node == neighbors[i % len(neighbors)]:
    #                 i += 1
    #                 continue
    #             else:
    #                 if random.random() < 0.99:
    #                     next_node = neighbors[i % len(neighbors)]
    #                     break
    #             i += 1
    #         if _ == num_nodes - 1:
    #             # next_node = 0
    #             next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
    #             while next_node == current_node:
    #                 next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
    #         H.add_edge(current_node, next_node)
    #         H.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
    #         current_node = next_node
    #         degree[current_node] += 1
    for _ in range((K - num_random_ring) * num_nodes):
        # if _ % num_nodes == 0:
        #     new_order = []
        # new_order.append(current_node)
        degree[current_node] += 1
        neighbors = list(G.neighbors(current_node))
        # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
        neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
        
        # Move to the neighbor with the highest priority
        i = 0
        while (_ % num_nodes) != num_nodes - 1:
            if H.has_edge(current_node, neighbors[i % len(neighbors)]) or current_node == neighbors[i % len(neighbors)]:
            # if neighbors[i % len(neighbors)] in new_order or  neighbors[i % len(neighbors)] == current_node:
                i += 1
                continue
            else:
                if  greedy or random.random() < 0.5:
                    next_node = neighbors[i % len(neighbors)]
                    break
            i += 1
        
        if _ % num_nodes == num_nodes - 1:
            next_node = new_order[0]
            # next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
            # while next_node == current_node:
            #     next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
        # print(new_order, next_node, current_node)
        H.add_edge(current_node, next_node)
        H.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
        current_node = next_node
        
        
    for ring in rings:
        for u, v in ring:
            H.add_edge(u, v, weight=G[u][v]['weight'])
    # print(H.in_degree, H.out_degree)
    return H


def generate_k_directed_rings_distributed(G, num_nodes, K, stride=1):
    nodes = list(G.nodes())
    rings = []
    for _ in range(K):
        ring_nodes = nodes.copy()
        random.shuffle(ring_nodes)
        ring = [(ring_nodes[i], ring_nodes[(i+1) % len(ring_nodes)]) for i in range(len(ring_nodes))]
        rings.append(ring)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for ring in rings:
        for start_id in range(0, num_nodes, stride):
            end_id = min(start_id + stride, num_nodes)
            node_list = [v % num_nodes for u, v in ring[start_id:end_id - 1]]
            
            cur_node = ring[start_id][0]
            # print(start_id, end_id, cur_node, node_list)
            while node_list:
                # Find the node connected by the minimum weight edge
                min_weight = float('inf')
                next_node = None
                for node in node_list:
                    if G.has_edge(cur_node, node):
                        weight = G[cur_node][node]['weight']
                        if weight < min_weight:
                            min_weight = weight
                            next_node = node
                # Update current node and remove the selected node from the list
                H.add_edge(cur_node, next_node)
                H.edges[cur_node, next_node]['weight'] = G.edges[cur_node, next_node]['weight']
                cur_node = next_node
                node_list.remove(cur_node)
            next_node = ring[end_id - 1][1]
            H.add_edge(cur_node, next_node)
            H.edges[cur_node, next_node]['weight'] = G.edges[cur_node, next_node]['weight']
            # if stride > num_nodes:
            #     H.add_edge(next_node, ring[start_id][0])
            #     H.edges[next_node, ring[start_id][0]]['weight'] = G.edges[next_node, ring[start_id][0]]['weight']
    # for id, d in H.in_degree():
    #     if d > 4:
    #         print(id, d)
    return H


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

def perform_random_walk(G, num_nodes, start_node, num_steps, if_plot=False, greedy=True):
    
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
    # fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(20, 35))  # Adjust figsize to fit your screen
    # axes = axes.flatten()  # Flatten the array of axes
    t = time.time()
    for _ in range(num_steps - 1):
        if _ % num_nodes == 0:
            new_order = []
        new_order.append(current_node)
        degree[current_node] += 1
        neighbors = list(G.neighbors(current_node))
        # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
        neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
        
        # Move to the neighbor with the highest priority
        i = 0
        while (_ % num_nodes) != num_nodes - 1:
            if subgraph.has_edge(current_node, neighbors[i % len(neighbors)]) or current_node == neighbors[i % len(neighbors)]:
            # if neighbors[i % len(neighbors)] in new_order or  neighbors[i % len(neighbors)] == current_node:
                i += 1
                continue
            else:
                if  greedy or random.random() < 0.5:
                    next_node = neighbors[i % len(neighbors)]
                    break
            i += 1
        
        if _ % num_nodes == num_nodes - 1:
            next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
            while next_node == current_node:
                next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
        # print(new_order, next_node, current_node)
        subgraph.add_edge(current_node, next_node)
        subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
        subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
        current_node = next_node
        
        # if if_plot:
        #     special_edge = (current_node, next_node)
        #     ax = axes[_]
        #     # nx.draw(subgraph, ax=ax)
        #     nx.draw_networkx_nodes(subgraph, pos=positions, node_color='lightblue',  ax=ax)
        #     edges = subgraph.edges(data=True)
            
        #     special_edges = [edge for edge in subgraph.edges(data=True) if ((edge[0], edge[1]) == special_edge or (edge[1], edge[0]) == special_edge)]
        #     nx.draw_networkx_edges(subgraph, pos=positions, edgelist=edges, width=[w['weight'] for (u, v, w) in edges],  ax=ax)
        #     nx.draw_networkx_edges(subgraph, pos=positions, edgelist=special_edges, 
        #                         width=[w['weight'] for (u, v, w) in special_edges], 
        #                         edge_color='red', ax=ax)
        #     nx.draw_networkx_labels(subgraph, pos=positions, font_weight='bold', ax=ax)
        #     edge_labels = {(u, v): w['weight'] for (u, v, w) in subgraph.edges(data=True)}
        #     nx.draw_networkx_edge_labels(subgraph, pos=positions, edge_labels=edge_labels,  ax=ax)

        # if if_plot:
        #     try:
        #         d = nx.diameter(subgraph, weight='weight')
        #     except:
        #         # d = 'inf'
        #         largest_cc = max(nx.connected_components(subgraph), key=len)
        #         subgraph_largest_cc = subgraph.subgraph(largest_cc)
        #         d = nx.diameter(subgraph_largest_cc)
        #     ax.set_title(f"Step {_+1} d={d}")
        #     ax.axis('off') 
    # if if_plot:
    #     plt.tight_layout()
        # plt.savefig(f'nn_N=20_{gid}.png')
    # print(time.time() - t) 
    d = nx.diameter(subgraph, weight='weight')
    return d

def perform_random_walk_directed(G, num_nodes, start_node, num_steps, if_plot=False, greedy=True):
    
    current_node = start_node
    degree = [0 for i in range(num_nodes)]
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(range(num_nodes))
    for _ in range(num_steps - 1):
        if _ % num_nodes == 0:
            new_order = []
        new_order.append(current_node)
        degree[current_node] += 1
        neighbors = list(G.neighbors(current_node))
        # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
        neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
        
        # Move to the neighbor with the highest priority
        i = 0
        while (_ % num_nodes) != num_nodes - 1:
            if subgraph.has_edge(current_node, neighbors[i % len(neighbors)]) or current_node == neighbors[i % len(neighbors)]:
            # if neighbors[i % len(neighbors)] in new_order or  neighbors[i % len(neighbors)] == current_node:
                i += 1
                continue
            else:
                if  greedy or random.random() < 0.5:
                    next_node = neighbors[i % len(neighbors)]
                    break
            i += 1
        
        if _ % num_nodes == num_nodes - 1:
            # next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
            # while next_node == current_node:
            #     next_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
            next_node = start_node
        # print(new_order, next_node, current_node)
        subgraph.add_edge(current_node, next_node)
        subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
        # subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
        current_node = next_node
    # print(subgraph.in_degree, subgraph.out_degree)
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

    # print(subgraph.degree())
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
    # print(time.time() - time)
    return d

# Main execution

def test_synthetic_graph(num_tests, N, k, mode="FABRIC"):
    diameter_list = []
    N_ = N
    if mode == "FABRIC":
        N = ((N + 16) // 17) * 17
        k = int(np.log2(N))
    for i in range(num_tests):
        graph_name = f'N={N_}_{i}_{mode}.pkl'
        with open(os.path.join('.', 'test_graph', graph_name), 'rb') as f:
            G = pkl.load(f)
        test_methods(G, N, k)

def test_bitnode_graph(file_path, N, k):
    G = nx.complete_graph(N )
    with open(file_path, 'rb') as f:
        LinkDelay = np.load(f)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = int(LinkDelay[u, v])# Assign random positive weights
    for (u, v) in G.edges():
        G.edges[v,u]['weight'] = int(LinkDelay[v, u])  # Assign random positive weights
    test_methods(G, N, k)

def test_cluster(N, k, M):
    LinkDelay = th.zeros(N, N)
    n = N // M
    for i in range(M):
        for j in range(M):
            if i == j:
                LinkDelay[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.randint(low=10, high=20, size=(n, n))
                # LinkDelay[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.normal(mean=5, std=1, size=(n, n))
                # LinkDelay[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.ones(n, n)
            else:
                LinkDelay[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.ones(n, n) * 160
                # LinkDelay[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.normal(mean=160, std=10, size=(n, n))
    # LinkDelay = th.normal(mean=5, std=1, size=(N, N))
    # LinkDelay = th.randint(low=1, high=10, size=(N, N))
    G = nx.complete_graph(N )
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = int(LinkDelay[u, v]) # Assign random positive 

    for (u, v) in G.edges():
        G.edges[v,u]['weight'] = int(LinkDelay[v, u])  # Assign random positive weights
    test_methods(G, N, k)


def random_edges_weight_sum(G, k):
    total_weight_sum = 0
    min_weight_sum = 0

    for node in G.nodes():
        # Get the edges connected to the node
        edges = list(G.edges(node, data=True))
        
        if len(edges) >= k:
            # Randomly select 3 edges
            selected_edges = random.sample(edges, k)
        else:
            # If less than 3 edges, select all available edges
            selected_edges = edges
        
        selected_edges_sorted = sorted(selected_edges, key=lambda edge: edge[2]['weight'])

        # Calculate total weight sum for the selected edges
        total_weight_sum += sum(edge[2]['weight'] for edge in selected_edges)
        min_weight = 0
        i = 0
        # Calculate the minimum weight among the selected edges
        while i < (k // 2):
            min_weight += selected_edges_sorted[i][2]['weight']
            i += 1
        min_weight_sum += min_weight

    return total_weight_sum, min_weight_sum

def test_methods(G, N, k):
    num_steps = k * N // 2
    total_weight, min_weight = random_edges_weight_sum(G, k)
    print(total_weight / k, min_weight / (k // 2))
    print("Chord Random Ring")
    diameter_list['chord_random_ring'].append(KNN(G, N, k, random_ring=True, chord=True))
    print("Chord Shortest Ring")
    diameter_list['chord_shortest_ring'].append(KNN(G, N, k, random_ring=False, chord=True))
    # Assuming G is your graph
    print("NN Random Ring")
    diameter_list['nearest_neighbour_random_ring'].append(KNN(G, N, k, random_ring=True, chord=False))
    print("NN Shortest Ring")
    diameter_list['nearest_neighbour_shortest_ring'].append(KNN(G, N, k, random_ring=False, chord=False))
    assert 0
   
    for s in range(1, 2):
        random.seed(s)
        # diameter_list['K_ring_random_ring'].append(nx.diameter(generate_k_directed_rings(G, N, k, random_ring=True, num_random_ring=k), weight='weight'))
        # diameter_list['K_ring_shortest_ring'].append(nx.diameter(generate_k_directed_rings(G, N, k, random_ring=False, num_random_ring=k-1), weight='weight'))
        # # diameter_list['K_ring_distributed'].append(nx.diameter(generate_k_directed_rings_distributed(G, N, k, N_cluster=1), weight='weight'))
        # diameter_list['K_ring_greedy'].append(perform_random_walk(G, N, 0, num_steps))
        # diameter_list['K_ring_greedy'].append(perform_random_walk_directed(G, N, 0, num_steps * 2))
        # diameter_list['K_ring_epsilon_greedy'].append(perform_random_walk(G, N, 0, num_steps, greedy=False))
        
        # for i in range(k + 1):
        # # for i in range(1):
        #     if f'K_ring_{i}_random' not in diameter_list:
        #         diameter_list[f'K_ring_{i}_random'] = []
        #     diameter_list[f'K_ring_{i}_random'].append(nx.diameter(generate_k_directed_rings(G, N, k, 
        # random_ring=False, num_random_ring=i), weight='weight'))
        # for i in range(k + 1, 10):
        #     if f'K_ring_{i}_random' not in diameter_list:
        #         diameter_list[f'K_ring_{i}_random'] = []
        #     diameter_list[f'K_ring_{i}_random'].append(0)
        
        for stride in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            if f'K_ring_random_distributed_stride_{stride}' not in diameter_list:
                diameter_list[f'K_ring_random_distributed_stride_{stride}'] = []
            H = generate_k_directed_rings_distributed(G, N, k, stride=stride)
            d = nx.diameter(H, weight='weight')
            max_in_degree_node, max_in_degree = max(H.in_degree(), key=lambda x: x[1])
            max_out_degree_node, max_out_degree = max(H.out_degree(), key=lambda x: x[1])
            diameter_list[f'K_ring_random_distributed_stride_{stride}'].append(d)
            print(stride, d, f"max_in_degree={max_in_degree}, max_out_degree={max_out_degree}")
        print(diameter_list)

if __name__ == '__main__':
    
    N = 500
    k = 8
    M = 4
    file_path = '/global/homes/s/swu264/perigee/linkdelay.npy'
    N_list = [10]
    for i in range(50, 1001, 50):
        N_list.append(i)
    seed = 1
    # for N in range (1000, 5001, 1000):
    # for N in N_list:
    for N in [500]:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        num_tests = 1
        test_synthetic_graph(num_tests, N, int(np.log2(N)))
        # test_bitnode_graph(file_path, N, int(np.log2(N))) 
        # test_cluster(N, k, M)
        print(N, diameter_list)
        # assert 0
        # with open(f"N={N}_cluster_gaussian_exploration.pkl", 'wb') as f:
        #     pkl.dump(diameter_list, f)
        
    # for i in range(num_tests):
    #     graph_name = f'N={N}_{i}.pkl'
    #     diameter = 1e9
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
    #     for start_id in range(4):
    #     # for start_id in range(1):
    #         diameter = min(diameter, perform_random_walk(G, N, start_id, num_steps, i))
    #     diameter_list.append(diameter)
    #     print(f"Test G {i}: {diameter}")  
    # print(diameter_list)
    # print(sum(diameter_list) / len(diameter_list))
    
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
    # for i in range(num_tests):
    #     graph_name = f'N={N}_{i}.pkl'
    #     with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
    #         G = pkl.load(f)
    #     diameter = KNN(G, N, k)
    #     diameter_list.append(diameter)
    #     print(f"Test G {i}: {diameter}")  
    # print(diameter_list)
    # print(sum(diameter_list) / num_tests)

    
        # print(f"Test G {i}: {diameter}")  
    # print(diameter_list)
    # print(sum(diameter_list) / num_tests)
    
    
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
    