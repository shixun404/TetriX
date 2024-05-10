import torch
import pickle as pkl
test_inputs = torch.randn(128, 16)
with open('test_16.pkl', 'wb') as f:
    pkl.dump(test_inputs, f)