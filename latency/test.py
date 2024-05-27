import torch

# Example tensor of shape [bs, n, feature_dim]
bs, n, feature_dim = 2, 5, 3
x = torch.randn((bs, n, feature_dim))

# Specific indices for each sample in the batch
indices = torch.tensor([1, 3])  # must be within range 0 to n-1

# Gather selected elements based on indices
selected_elements = torch.gather(x, 1, indices.view(-1, 1, 1).expand(bs, 1, feature_dim))

# Expand to match the original dimensions
expanded_elements = selected_elements.repeat(1, n, 1)

print(expanded_elements.shape)  # Expected shape: [bs, n, feature_dim]
print(expanded_elements)
print(x)
print(indices.view(-1, 1, 1).expand(bs, 1, feature_dim).shape)