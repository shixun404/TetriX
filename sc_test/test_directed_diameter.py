import pickle as pkl
import torch
import torch.nn.functional as F
import networkx as nx
import math
import random
import numpy as np
from directed_apprxo_diameter import nx_approx_diameter_directed
from test_nn import perform_random_walk
from utils import compare_graph_edges

if __name__ == "__main__":
    save_graph = False
    # 0) 选择设备: GPU 或 CPU
    seed = 42  # 你可以更改这个值
    random.seed(seed)  # 设置 Python random 库的种子
    np.random.seed(seed)  # 设置 NumPy 的种子
    torch.manual_seed(seed)  # 设置 PyTorch 的种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    N = 100
    # with open('G_100_seed=42.pkl', 'rb') as f:
    with open('G_400_FABRIC.pkl', 'rb') as f:
        graph = pkl.load(f)
    # with open('best_test_graph_seed=42.pkl', 'rb') as f:
    with open('FABRIC_400_best_test_graph.pkl', 'rb') as f:
        rl_graph = pkl.load(f)
    cnt = 0
    for i in range(N):
        for j in range(N):
            if rl_graph.has_edge(i, j):  # ✅ Correct way to check if the edge exists
                cnt += 1
                weight = rl_graph[i][j].get('weight', 0)  # ✅ Correct way to get the weight
                print(f"cnt={cnt}, i={i}, j={j}, rl_graph={weight}, G_100={graph[i][j].get('weight', 0)}")
                if weight != graph[i][j].get('weight', 0):  # ✅ Now using the correct weight retrieval
                    print(f"Mismatch at edge ({i}, {j}): rl_graph={weight}, G_100={graph[i][j].get('weight', 0)}")
                    assert 0  # Raise an assertion error to debug
    nodes = rl_graph.nodes()
    in_degrees = torch.tensor([rl_graph.in_degree(n) for n in nodes], dtype=torch.float)
    out_degrees = torch.tensor([rl_graph.out_degree(n) for n in nodes], dtype=torch.float)
    print(in_degrees.mean(), in_degrees.max(), in_degrees.min())
    print(out_degrees.mean(), out_degrees.max(), out_degrees.min())
    print(f"\n[NetworkX] Initial computed diameter = {nx.diameter(rl_graph, weight='weight')}")
