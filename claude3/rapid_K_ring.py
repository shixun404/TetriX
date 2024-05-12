import numpy as np
import heatmap
import torch
def construct_matrix_A(num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=int)

    # Connect nodes within each group in a ring structure
    K = 10
    seed = 123
    torch.manual_seed(seed)
    for i in range(K):
        order = torch.randperm(num_nodes)
        for j in range(num_nodes):
            A[order[j], order[(j + 1) % 1000]] = 1
    print(A.sum())
    return A

# Example usage
num_nodes = 1000
# A = construct_matrix_A(num_nodes)
A = torch.as_tensor(construct_matrix_A(num_nodes), device=torch.device('cuda:0'), dtype=torch.float)
print(A.sum(dim=0).float().mean())
# A = (A + A.T).clamp(min=0, max=1)
# print(A.sum(dim=0))
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(A.sum(dim=0).float().mean())
print(eigenvalues[0:10])
print(eigenvectors[:10, 0])
print((A @ eigenvectors[:,0].real - eigenvectors[:,0].real * eigenvalues[0].real).sum())
assert 0
A_i = A
for i in range(1, 21, 1):
    if i != 1:
        A_i = A_i @ A
    filename = f"rapid_k_ring/N1000_rapid_k_ring_A^{i}.png"
    heatmap.plot_heatmap(A_i, 1000, filename)