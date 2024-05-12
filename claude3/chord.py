import numpy as np
import heatmap
import torch
def construct_matrix_A(num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # Connect nodes within each group in a ring structure
    for i in range(1000):
        # A[i, i] = 1
        for j in range(10):
            A[i, (i + 2 ** j) % 1000] = 1
            # A[(i + 2 ** j) % 1000, i] = 1
    # print(A.sum())
    # print( torch.nonzero(torch.as_tensor(A[0])))
    # assert 0
    return A

# Example usage
num_nodes = 1000
# A = construct_matrix_A(num_nodes)
A = torch.as_tensor(construct_matrix_A(num_nodes), device=torch.device('cuda:0'), dtype=torch.float).T
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(eigenvalues[0:10])
print(eigenvectors[:, 0])
print((A @ eigenvectors[:,0].real - eigenvectors[:,0].real * eigenvalues[0].real).sum())
assert 0


print(A.sum(dim=0).float().mean())
# A = (A + A.T).clamp(min=0, max=1)
# print(A.sum(dim=0))
print(A.sum(dim=0).float().mean())
# assert 0
# A_10 = torch.matrix_power(A, 10).clamp(min=0, max=1)
# for i in range(9):
#  A_10 = A_10 @ A
#  A_10 = A_10.clamp(min=0, max=1)
# print("After 10 hop", A_10.sum())
# print(A[:10, :10])
# print(A_10[:10, :10])
A_i = A
for i in range(1, 21, 1):
    if i != 1:
        A_i = A_i @ A
    filename = f"chord/N1000_chord_A^{i}.png"
    heatmap.plot_heatmap(A_i, 1000, filename)