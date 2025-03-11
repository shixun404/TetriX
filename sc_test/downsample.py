import numpy as np
import networkx as nx
import pickle as pkl
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

# **Step 1: Load the original graph**
with open("G_400.pkl", "rb") as f:
    G_large = pkl.load(f)

# Convert to adjacency matrix (400x400)
nodes = list(G_large.nodes)
node_index = {node: i for i, node in enumerate(nodes)}  # Mapping nodes to indices

# Create adjacency matrix
A = np.zeros((len(nodes), len(nodes)))
for u, v, data in G_large.edges(data=True):
    A[node_index[u], node_index[v]] = data['weight']
    A[node_index[v], node_index[u]] = data['weight']  # Symmetric

# **Step 2: Perform SVD Decomposition**
k = 100  # Reduce to 100 dimensions
U, S, Vt = svds(A, k=k)  # SVD

# **Step 3: Construct the New 100x100 Graph**
# Use the first 100 rows to create the smaller matrix
A_small = U[:, -k:] @ np.diag(S) @ Vt[-k:, :]  # Low-rank approximation

# **Step 4: Construct the Graph**
G_small = nx.Graph()
new_nodes = nodes[:k]  # Select first 100 nodes
G_small.add_nodes_from(new_nodes)

# Add edges with weights from the new matrix
for i in range(k):
    for j in range(i + 1, k):
        weight = A_small[i, j]
        if weight > 0:  # Keep only positive weights
            G_small.add_edge(new_nodes[i], new_nodes[j], weight=weight)

# **Step 5: Save the Smaller Graph**
with open("G_100_downsample.pkl", "wb") as f:
    pkl.dump(G_small, f)

# # **Print Graph Info**
# print(nx.info(G_small))

# **Print First 10 Edges**
for edge in list(G_small.edges(data=True))[:10]:
    print(edge)

threshold = 0.01 * np.max(S)  # Singular values smaller than 1% of max are ignored
rank_approx = np.sum(S > threshold)

print(f"Estimated Rank: {rank_approx} / 100")

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(A_small, cmap="coolwarm", center=0)
plt.title("100x100 Matrix Heatmap")
plt.savefig('SVD_visualize')