import networkx as nx
import numpy as np
import random
random.seed(123)
# Create a fully connected graph with 20 nodes
num_nodes = 20
G = nx.complete_graph(num_nodes)

# Assign random weights to the edges between 1 and 10
for u, v in G.edges():
    G.edges[u, v]['weight'] = random.randint(1, 10)
    print(G.edge[u, v]['weight'], G.edge[v, u]['weight'])

# Function to perform a random walk with given rules


def perform_random_walk(G, start_node, num_steps):
    current_node = start_node
    visited = [current_node]
    degree = [0 for 0 in range(num_nodes)]
    subgraph.add_nodes_from(range(num_nodes))
    for _ in range(num_steps - 1):
        degree[current_node] += 1
        neighbors = list(G.neighbors(current_node))
        # Sort neighbors first by degree (lower degree first), then by latency (lower weight first)
        neighbors.sort(key=lambda x: (degree[x], G.edges[current_node, x]['weight']))
        
        # Move to the neighbor with the highest priority
        next_node = neighbors[0]
        subgraph.add_edge(self.start_id, action)
        subgraph.edges[current_node, next_node]['weight'] = G.edges[current_node, next_node]['weight']
        subgraph.edges[next_node, current_node]['weight'] = G.edges[next_node, current_node]['weight']
        current_node = next_node
        
    # Create the subgraph from the visited nodes
    # subgraph = G.subgraph(visited)
    return subgraph

# Perform the random walk starting from node 0 for 20 steps
subgraph = perform_random_walk(G, 0, 20)

# Output the edges and weights of the subgraph
# print(subgraph.edges(data=True))
print(nx.diameter(subgraph, weight='weight'))

