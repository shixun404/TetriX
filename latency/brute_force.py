import networkx as nx
import random
import itertools
import os
import pickle as pkl
from tqdm import tqdm
def create_fully_connected_graph(n):
    """ Create a fully connected weighted graph with n nodes """
    G = nx.complete_graph(n)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
    for (u, v) in G.edges():
        G.edges[v,u]['weight'] = G.edges[u,v]['weight']  # Assign random positive weights
    return G

def find_regular_subgraph(G, k,  graph_name=None, iterations=100):
    """ Attempt to find a k-regular subgraph with the minimum diameter from a fully connected graph """
    n = G.number_of_nodes()
    best_diameter = float('inf')
    diameter = float('inf')
    best_subgraph = None
    
    # pbar = tqdm(range(iterations),
    #         desc=f'Initial diameter: {diameter}',
    #         ncols=100,  # Width of the entire output
    #         colour='blue',  # Color of the progress bar
    #         miniters=1)  # Update the bar at least every iteration


    # for _ in pbar:
    for _ in range(iterations):
        # Randomly select edges to attempt to form a 4-regular graph
        possible_subgraph = nx.random_regular_graph(k, n)
        for u, v in possible_subgraph.edges():
            possible_subgraph.edges[u,v]['weight'] = G.edges[u,v]['weight']
            possible_subgraph.edges[v,u]['weight'] = G.edges[v,u]['weight']

        # pbar.set_description(f"id: {_:6d}, diameter: {diameter:4f}, best_diameter: {best_diameter:4f}")
        # Check if the graph is 4-regular and calculate diameter
        if all(d == k for n, d in possible_subgraph.degree()):
            try:
                diameter = nx.diameter(possible_subgraph, weight='weight')
                if diameter < best_diameter:
                    best_diameter = diameter
                    best_subgraph = possible_subgraph
                    # print(best_diameter)
                    # with open('best_'+graph_name, 'wb') as f:
                    #     pkl.dump(best_subgraph, f)
                    #     print('best')
            except nx.NetworkXError:
                # Graph is not connected, skip it
                continue
        
    return best_subgraph, best_diameter

# Main execution
N = 20
k = 4
graph_name = f'G_N={N}.pkl'
if graph_name not in os.listdir('.'):
    G = create_fully_connected_graph(20)
    with open(graph_name, 'wb') as f:
        pkl.dump(G, f)
else:
    with open(graph_name, 'rb') as f:
        G = pkl.load(f)
best_subgraph, best_diameter = find_regular_subgraph(G, k, graph_name)
print(f"Best diameter found: {best_diameter}")