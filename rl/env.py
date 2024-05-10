import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl

def test_matrix_convergence(matrix, tol=1e-3, max_iters=1000):
    """
    Tests how many iterations it takes for a given matrix to converge.
    
    Args:
    - matrix (torch.Tensor): The matrix to test.
    - tol (float): Tolerance for convergence. Default is 1e-5.
    - max_iters (int): Maximum number of iterations to prevent infinite loops. Default is 1000.
    
    Returns:
    - int: Number of iterations it took for the matrix to converge.
    - bool: Whether the matrix converged within the maximum iterations.
    """
    current_matrix = matrix.clone()
    iteration = 0
    for iteration in range(max_iters):
        next_matrix = torch.matmul(current_matrix, matrix)
        # Calculate the Frobenius norm of the difference between current_matrix and next_matrix
        diff = torch.norm(next_matrix - current_matrix, p='fro')
        if diff <= tol:
            return iteration + 1, True
        current_matrix = next_matrix
    return iteration + 1, False

class MatrixNet(nn.Module):
    def __init__(self, n, input_shape=1):
        super(MatrixNet, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(input_shape, n * n)  # 第一个全连接层扩展维度
        self.relu1 = nn.ReLU()          # 第一个ReLU激活函数
        self.fc2 = nn.Linear(n * n, n * n)  # 第二个全连接层进一步扩展维度
        self.relu2 = nn.ReLU()          # 第二个ReLU激活函数
        self.fc3 = nn.Linear(n * n, n * n)  # 最后一个全连接层输出n*n维度
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = x.reshape(-1, self.n, self.n)
        x = torch.softmax(x, dim=2)  # 在每行应用softmax
        return x

def second_largest_eigenvalue(matrix):
    eigenvalues = torch.linalg.eigvals(matrix)
    top_two = torch.topk(eigenvalues.real, 2).values
    # return (1 - top_two[0]) ** 2  + top_two[1] ** 2  # Return the second largest eigenvalue
    return  top_two[1]** 2 / (top_two[0] ** 2)  # Return the second largest eigenvalue
    # return top_two[1] ** 2  # Return the second largest eigenvalue



# Parameters
n = 3  # Size of the matrix
batch_size = 128
input_shape = 2
torch.manual_seed(123)
model = MatrixNet(n, input_shape=input_shape)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
test_input = torch.randn(batch_size, input_shape)
# Training loop
loss_list = []
for _ in range(10000):
    optimizer.zero_grad()
    # train_data = torch.randn(batch_size, input_shape)
    # matrix = model(train_data)  # Random input vector
    matrix = model(test_input)  # Random input vector
    # print(matrix.shape)
    loss = 0
    for bid in range(batch_size):
        loss += second_largest_eigenvalue(matrix[bid])
    loss.backward()
    optimizer.step()

    loss_list.append(loss)
    if _ % 10 == 0:
        # Example usage with a matrix from the trained network
        test_matrix = model(test_input).detach()
        itr_sum = 0
        for bid in range(batch_size):
            iterations, converged = test_matrix_convergence(test_matrix[bid])
            loss += second_largest_eigenvalue(matrix[bid])
            itr_sum += iterations
        print(f'Iterations to converge: {itr_sum / batch_size}')
        print("Trained matrix:", matrix[0].T)
        print("Loss (second largest eigenvalue):", loss.item())
    
    with open('list.txt', 'wb') as f:
        pkl.dump(loss_list, f)

print(f"Converged: {converged} after {iterations} iterations.")


print("Trained matrix:", matrix.detach().numpy())
print("Loss (second largest eigenvalue):", loss.item())