import numpy as np
import heatmap
import torch
def construct_matrix_A(num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=int)

    # Connect each node to its 10 nearest neighbors in the ring
    for i in range(num_nodes):
        A[i, (i+1) % num_nodes] = 1
        A[i, (i-1) % num_nodes] = 1
        print(i + 1, i-1)
        for j in range(5):
            A[i, (i+2+j) % num_nodes] = 1
            A[i, (i-2-j) % num_nodes] = 1
            print( i + 2 + j, i -2 - j)
        # assert 0

    # Connect node 0 to all other nodes
    A[0, :] = 1

    return A

# Example usage
num_nodes = 1000
A = torch.as_tensor(construct_matrix_A(num_nodes)).T
A_10 = A
for i in range(9):
 A_10 = A_10 @ A
 A_10 = A_10.clamp(min=0, max=1)
print("After 10 hop", A_10.sum())
print(A[:10, :10])
print(A_10[:10, :10])
filename = 'N1000_topology.png'
heatmap.plot_heatmap(A, A_10, 1000, filename)