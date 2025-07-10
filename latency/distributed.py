import random
import networkx as nx
import torch as th
import pickle as pkl
import os
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import numpy as np
def directed_to_undirected(directed_graph, method='sum'):
    undirected_graph = nx.Graph()
    
    for u, v, data in directed_graph.edges(data=True):
        weight_uv = data['weight']
        
        if undirected_graph.has_edge(u, v):
            existing_weight = undirected_graph[u][v]['weight']
            
            if method == 'sum':
                new_weight = existing_weight + weight_uv
            elif method == 'average':
                new_weight = (existing_weight + weight_uv) / 2
            elif method == 'max':
                new_weight = max(existing_weight, weight_uv)
            elif method == 'min':
                new_weight = min(existing_weight, weight_uv)
            else:
                raise ValueError("Method must be one of 'sum', 'average', 'max', 'min'")
            
            undirected_graph[u][v]['weight'] = new_weight
        else:
            undirected_graph.add_edge(u, v, weight=weight_uv)
    
    return undirected_graph

def initialize_graph(N, K, if_random_ring=False, if_cluster=False, latency=None, num_random_rings=1):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    if latency is None:
        latency = nx.complete_graph(N)
        if not if_cluster:
            for (u, v) in latency.edges():
                latency.edges[u, v]['weight'] = np.random.normal(5, 1)
            for (u, v) in latency.edges():
                latency.edges[v,u]['weight'] = latency.edges[u,v]['weight']  # Assign random positive weights
        else:
            latency_mat = th.zeros(N, N)
            M = 4
            n = N // M
            for i in range(M):
                for j in range(M):
                    if i == j:
                        latency_mat[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.normal(mean=5, std=1, size=(n, n))
                    else:
                        latency_mat[i * n:(i + 1) * n, j * n:(j + 1) * n] = th.normal(mean=160, std=10, size=(n, n))
            for (u, v) in latency.edges():
                latency.edges[u,v]['weight'] = int(latency_mat[u, v])  # Assign random positive weights
                latency.edges[v,u]['weight'] = int(latency_mat[v, u])  # Assign random positive weights
                
    for i in range(N):
        connections = random.sample([j for j in range(N) if j != i], K - num_random_rings)
        for j in connections:
            graph.add_edge(i, j, weight=latency.edges[i, j]['weight'])
    random_rings = []
    if if_random_ring:
        for _ in range(num_random_rings):
            nodes = list(graph.nodes())
            random.shuffle(nodes)

            random_ring = {}
            # Connect each node to the next in the shuffled list, making a ring
            for i in range(len(nodes)):
                graph.add_edge(nodes[i], nodes[(i + 1) % len(nodes)], weight=latency.edges[nodes[i], nodes[(i + 1) % len(nodes)]]['weight'])
                random_ring[nodes[i]] = nodes[(i + 1) % len(nodes)]
                if len(list(graph.successors(nodes[i]))) > K:
                    print(f"Node {nodes[i]} has more than {K} successors: {len(graph.successors(nodes[i]))}")
                    assert 0
            random_rings.append(random_ring)
    return graph, latency, random_rings

def update_graph( graph, latency, random_rings, N, K, M, sample_sources=3,
                  if_diameter_directed=False, max_iterations=20, num_random_rings=1, num_drop=1):
    previous_diameter = 0
    try:
        previous_diameter = nx.diameter(graph, weight='weight')
    except:
        largest_cc = max(nx.strongly_connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        previous_diameter = nx.diameter(subgraph)

    a = th.zeros(max_iterations, N)
    
    x = [i for i in range(N)]
    sources = random.sample(graph.nodes(), sample_sources)
    diameter_list = []
    for itr in range(max_iterations):
        all_paths_lengths = []
        for source in sources:
            lengths = nx.single_source_dijkstra_path_length(graph, source)
            all_paths_lengths.append(lengths)
            a[itr][:len(list(lengths.values()))] += th.as_tensor(list(lengths.values())) / sample_sources
        successor = []
        for node in range(N):
            tmp = []
            if random_rings is not None:
                
                for i in graph.successors(node):
                    random_ring_flag = True
                    for ring_id in range(num_random_rings):
                        random_ring_flag = random_ring_flag and (i != random_rings[ring_id][node])
                    if random_ring_flag:
                        tmp.append(i)
                successor.append(tmp)

            else:
                successor.append( [ i for i in graph.successors(node)] )
            if len(successor[node]) > K:
                print(node, successor[node])
                assert 0
        scores = th.zeros(N, K)
        for lengths in all_paths_lengths:
            for node in range(N):
                neighbor_lengths = th.as_tensor([(latency.edges[neighbor, node]['weight'] + lengths.get(neighbor, float('inf'))) for neighbor in successor[node]])
                neighbor_lengths = neighbor_lengths - lengths.get(node, float('inf'))
                scores[node][:len(successor[node])] += th.as_tensor(neighbor_lengths)

        remaining_neighbors = []
        for i in range(N):
            try:
                top2_scores, top2_indices = th.topk(scores[i][:len(successor[i])], num_drop, dim=0)
                selected_neighbor = [successor[i][idx] for idx in top2_indices]
                for j in range(num_drop):
                    graph.remove_edge(i, selected_neighbor[j])
            except RuntimeError as e:
                pass
            
                
            remaining_neighbors.append(
                [n for j, n in enumerate(successor[i]) if j not in top2_indices]
            )
        for i in range(N):
            new_connections = []
            potential_new_connections = set(range(N)) - set(remaining_neighbors[i]) - set([i])
            new_connections = random.sample(potential_new_connections, K - len(remaining_neighbors[i]))
            for j in range(K - len(remaining_neighbors[i]) - num_random_rings):
                graph.add_edge(i, new_connections[j], weight=latency.edges[i, new_connections[j]]['weight'])
        for node in graph.nodes():
            if graph.in_degree(node) > M:
                excess_connections = graph.in_degree(node) - M
                if random_rings is not None:
                    incoming_edges = []
                    for (i, j) in graph.in_edges(node):
                        random_ring_flag = True
                        for ring_id in range(num_random_rings):
                            random_ring_flag = random_ring_flag and (node != random_rings[ring_id][i])
                        if random_ring_flag:
                            incoming_edges.append((i, node))
                else:
                    incoming_edges = list(graph.in_edges(node))
                edges_to_drop = random.sample(incoming_edges, excess_connections)
                graph.remove_edges_from(edges_to_drop)

        # Step 4: Check termination condition
        
        if if_diameter_directed:
            if_connected = nx.is_strongly_connected(graph)
            g = graph
        else:
            undirected_graph = directed_to_undirected(graph, method='average')
            if_connected = nx.is_strongly_connected(graph)
            g = undirected_graph
        try:
            current_diameter = nx.diameter(g, weight='weight')
        except:
            if if_diameter_directed:
                largest_cc = max(nx.strongly_connected_components(g), key=len)
            else:
                largest_cc = max(nx.connected_components(g), key=len)
            subgraph = g.subgraph(largest_cc)
            current_diameter = nx.diameter(subgraph, weight='weight')
        diameter_list.append(current_diameter)
        previous_diameter = current_diameter
        print(f'itr={itr}, current_diameter={current_diameter}')
    return a, diameter_list



if __name__ == '__main__':
    # Parameters and graph initialization
    N = 100  # Number of nodes
    K = 3   # Outgoing connections per node
    M = 4 # Maximum incoming connections per node
    num_drop = 1
    num_random_rings = 1
    num_tests = 1
    max_iterations = 65
    sample_sources = 8
    diameter_optimal = []
    for i in range(1):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        th.random.manual_seed(seed)

        # with open('../sc_test/G_400_FABRIC.pkl', 'rb') as f:
        with open(f'../sc_test/G_{N}_NORMAL.pkl', 'rb') as f:
            G = pkl.load(f)
        graph, latency, random_rings = initialize_graph(N, K, if_random_ring=True, if_cluster=False, latency=G, num_random_rings=num_random_rings)
        data,diameter_list = update_graph(graph, latency, random_rings, N, K, M, sample_sources=sample_sources, if_diameter_directed=True, max_iterations=max_iterations, num_drop=num_drop, num_random_rings=num_random_rings)
        diameter_optimal.append(min(diameter_list))
        plt.figure()
        fig, ax = plt.subplots()
        x = [j for j in range(N)]
        # for j in range(max_iterations):
        j = 1
        while j - 1 < max_iterations:
            y = data[j - 1].numpy()
            ax.plot(x, y, label=f'itr {j - 1}')
            j *= 4
        ax.legend()
        fig.savefig(f'N={N}_{seed}_bitnode_perigee.png')
        print(diameter_list, diameter_optimal)
        print(diameter_optimal)
        