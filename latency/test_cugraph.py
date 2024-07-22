import networkx as nx
# Import needed libraries
import os
import pickle as pkl
import matplotlib.pyplot as plt
# Set environment variable
# os.environ['NETWORKX_AUTOMATIC_BACKENDS'] = 'cugraph'

G = nx.Graph()
N = 500
i = 1
K = 8
sample = 3
nxcg_G = []
Gs = []
for i in range(sample):
    graph_name = f'N={N}_{i}.pkl'
    with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
        G = pkl.load(f)
    candidate_graph = nx.random_regular_graph(K, N)

    for u, v in candidate_graph.edges():
        candidate_graph.edges[u,v]['weight'] = G.edges[u,v]['weight']
        candidate_graph.edges[v,u]['weight'] = G.edges[v,u]['weight']
    # nxcg_G.append(nxcg.from_networkx(candidate_graph, preserve_all_attrs=True) )
    Gs.append(candidate_graph)
# print(nxcg_G.edges(data=True))
import time
t = time.time()
length = []

for i in range(sample):
    longest_length = []
    # length = dict(nx.all_pairs_bellman_ford_path_length(Gs[i], weight='weight'))
    plt.figure()
    for j in range(N):
        length = nx.shortest_path_length(Gs[i], source=j, weight='weight')
        longest_length.append(max(length.values()))
        # assert 0
    num_bins = len(set(longest_length))
    plt.hist(longest_length, bins=num_bins, edgecolor='black')
    # Add titles and labels
    plt.title('Histogram of Longest Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig(f'hist_longest_length_{i}.png')

execution_time = time.time() - t
print('CPU: ', execution_time / sample)