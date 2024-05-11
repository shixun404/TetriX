import torch
import pickle as pkl
N = 150
test_inputs = torch.randn(128, N)
with open(f"test_{N}.pkl", 'wb') as f:
    pkl.dump(test_inputs, f)