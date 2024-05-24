import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def create_multi_ring_topology(num_nodes, num_rings):
    G = nx.Graph()
    nodes = list(range(num_nodes))
    for _ in range(num_rings):
        np.random.shuffle(nodes)
        # Ensure all nodes are connected in a ring
        ring_edges = [(nodes[i], nodes[(i + 1) % num_nodes]) for i in range(num_nodes)]
        G.add_edges_from(ring_edges)
    return G

def create_random_topology(num_nodes, degree):
    # Ensure the graph is connected
    while True:
        G = nx.random_regular_graph(degree, num_nodes)
        if nx.is_connected(G):
            break
    return G

def broadcast_simulation(G, start_node):
    visited = set()
    queue = deque([start_node])
    steps = 0

    while queue:
        current_size = len(queue)
        for _ in range(current_size):
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)
        steps += 1

    return steps

def simulate_all_nodes(G, num_nodes):
    total_steps = 0
    for start_node in range(num_nodes):
        total_steps += broadcast_simulation(G, start_node)
    return total_steps / num_nodes

def main():
    num_nodes = 1000
    num_rings = 10
    degree = 20

    # Create both topologies
    multi_ring_graph = create_multi_ring_topology(num_nodes, num_rings)
    random_graph = create_random_topology(num_nodes, degree)
    
    # Simulate broadcasts from all nodes
    multi_ring_avg_time = simulate_all_nodes(multi_ring_graph, num_nodes)
    random_avg_time = simulate_all_nodes(random_graph, num_nodes)
    
    print(f"Average multi-ring broadcast time: {multi_ring_avg_time}")
    print(f"Average random topology broadcast time: {random_avg_time}")

if __name__ == "__main__":
    main()
