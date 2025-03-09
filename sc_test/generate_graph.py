import torch
import networkx as nx
import pickle

# 设定设备 (CUDA 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定节点数
N = 100  # 你可以更改 N 的值

# 生成边权矩阵 W_init
W_init = torch.randint(low=5, high=150, size=[N, N], device=device).float()

# 确保对角线 (self-loops) 设为无穷大 (表示无自环)
W_init.fill_diagonal_(float('inf'))

# 生成 networkx 图
G = nx.Graph()

# 添加节点
G.add_nodes_from(range(N))

# 添加边（忽略 inf，自环已排除）
for i in range(N):
    for j in range(i + 1, N):  # 避免重复无向边
        weight = W_init[i, j].item()  # 转换为 Python float
        if weight != float('inf'):  # 仅添加有限权重的边
            G.add_edge(i, j, weight=weight)

# 保存为 pkl 文件
with open("G_100.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph G_100.pkl saved successfully!")