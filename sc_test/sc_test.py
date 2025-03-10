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

############################################
# 一些辅助小函数: softmin, softmax
############################################
def softmin(x: torch.Tensor, beta: float = 10.0, dim: int = -1) -> torch.Tensor:
    """
    近似 min(x) 的可微函数:
      softmin_beta(x) = -1/beta * log( sum(exp(-beta*x)) ).
    当 beta 越大, 越逼近真正的 min.
    x  : 任意形状张量
    dim: 在哪个维度上做 softmin
    """
    return -1.0 / beta * torch.logsumexp(-beta * x, dim=dim)

def softmax(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """
    近似 max(x) 的可微函数:
      softmax_alpha(x) = 1/alpha * log( sum(exp(alpha*x)) ).
    当 alpha 越大, 越逼近真正的 max.
    x : 任意形状张量(将 flatten 后做全局 max).
    """
    return (1.0 / alpha) * torch.logsumexp(alpha * x.flatten(), dim=0)

############################################
# 用 networkx 计算“真实”直径 (带权)
############################################
def nx_approx_diameter(adj_01: torch.Tensor, W: torch.Tensor, N, degree=3) -> float:
    """
    用 networkx 计算图的(带权)直径:
      - adj_01: [N,N], 0~1 的邻接强度(阈值判断是否有边)
      - W:       [N,N], 边权, 若无边则应是 inf
    返回值:
      float型直径; 如果图不连通, 则可能为 inf.
    """
    # 先把数据转到 CPU + numpy
    adj_01_np = adj_01.cpu().numpy()
    W_np = W.cpu().numpy()
    
    N = adj_01_np.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    sG = nx.Graph()
    sG.add_nodes_from(range(N))
    # 给定最大度数限制
    max_degree = degree  # 这里的 degree 需要事先定义

    # 存储每个节点的度数
    node_degrees = {i: 0 for i in range(N)}

    # 给定一个小阈值, 超过它视为有边
    threshold = 1e-3
    # 存储所有可能的边
    edges = []
    # 存储每个节点的度数
    node_degrees = {i: 0 for i in range(N)}

    # 存储已经添加的边，避免重复
    added_edges = set()

    # 对每个节点单独排序，选择 `adj_01_np` 最高的边
    for i in range(N):
        # 获取所有可能的 (j, adj_01_np[i, j]) 对，并按 adj_01_np 降序排序
        neighbors = [(j, adj_01_np[i, j], W_np[i, j]) for j in range(N) if i != j and not math.isinf(W_np[i, j])]
        # 按 `adj_01_np` 取最大值
        neighbors.sort(key=lambda x: x[1], reverse=True)  # 按 adj_01_np[i,j] 递减排序
        # 尝试添加最多 `max_degree` 条边
        for j, adj_value, weight in neighbors:
            if adj_value > threshold and node_degrees[i] < max_degree and node_degrees[j] < max_degree and (i, j) not in added_edges and (j, i) not in added_edges:
                # 添加边
                
                G.add_edge(i, j, weight=weight)

                # 更新度数
                node_degrees[i] += 1
                node_degrees[j] += 1

                # 记录已添加的边，防止重复
                added_edges.add((i, j))
    for i in range(N):
        for j in range(N):
            if i != j and adj_01_np[i, j] > 0:
                sG.add_edge(i, j, weight=W_np[i, j])

    compare_graph_edges(sG, G, weight_key='weight')
    # 用 all_pairs_dijkstra_path_length 求所有点对距离, 再取最大
    # 若不连通, 则可能出现 inf
    # distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    # max_dist = 0.0
    # for i in distances:
    #     for j in distances[i]:
    #         if distances[i][j] > max_dist:
    #             max_dist = distances[i][j]
    
    # 如果发现有节点距离未覆盖, 说明不连通 => 直径=inf
    # if any(len(distances[i]) < N for i in distances):
    #     return float('inf')
    diameter = nx.diameter(G, weight='weight')
    return diameter

############################################
# 核心函数: 用K-hop + softmin/softmax估计直径, A∈[-1,1]
############################################
def soft_diameter(A: torch.Tensor,
                  W: torch.Tensor,
                  K: int = 3,
                  beta: float = 10.0,
                  alpha: float = 10.0,
                  eps: float = 1e-6, l=0.0) -> torch.Tensor:
    """
    基于 K-hop 近似 + softmin(min) + softmax(max) 的可微直径
    参数:
      A     : [N,N], 范围在 [-1,1], 需要 clamp 或其他方式保证范围
      W     : [N,N], 原始边权(若无边则 inf)
      K     : 最多 hop 数
      beta  : softmin 平滑参数 (越大越近似 min)
      alpha : softmax 平滑参数 (越大越近似 max)
      eps   : 防止除0
    返回:
      标量张量, 近似图的直径
    """
    # 1) 将 A clamp到 [-1,1], 然后映射到 [0,1]
    A_clamped = torch.clamp(A, -1.0, 1.0)
    A_soft = (A_clamped + 1.0) / 2.0  # [0,1]

    # 2) 构造 W_adj: 当 A_soft≈0 => 权重=inf
    W_adj = W / (A_soft + eps)
    # print("W:", W)
    # print("A_soft:", A_soft)
    # print("W_adj:", W_adj)

    # 3) 初始化 P => 1-hop 的最短距离近似
    P = W_adj.clone()

    # 4) 迭代 K-1 次做 softmin => 多步最短距离近似
    for _ in range(K - 1):
        # tmp[i,j,k] = P[i,k] + W_adj[k,j]
        tmp = P.unsqueeze(2) + W_adj.unsqueeze(0)
        # 在 k 维上做 softmin
        P = softmin(tmp, beta=beta, dim=2)
        # print(_, P)
    max_P = torch.max(P) + eps
    # 5) 用 softmax 近似 max => 近似直径
    D_approx = softmax(P, alpha=alpha) + l * P.mean() / max_P
    return D_approx


############################################
# Demo: 在 GPU 上训练, 用 networkx 对比结果
############################################
if __name__ == "__main__":
    save_graph = True
    # 0) 选择设备: GPU 或 CPU
    seed = 43  # 你可以更改这个值
    random.seed(seed)  # 设置 Python random 库的种子
    np.random.seed(seed)  # 设置 NumPy 的种子
    torch.manual_seed(seed)  # 设置 PyTorch 的种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    N = 100
    # 1) 初始 A, 大小4x4, 值落在 [-1,1], requires_grad=True
    
    # A_init = torch.rand(size=[N, N], device=device) * 2.0 - 1.0
    W_init = torch.randint(low=5, high=150, size=[N, N], device=device).float()
    W_init = (W_init + W_init.T) / 2.0  # 对称化
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(N))

    # 添加边（忽略 inf，自环已排除）
    for i in range(N):
        for j in range(N):  # 避免重复无向边
            weight = W_init[i, j].item()  # 转换为 Python float
            # if weight != float('inf'):  # 仅添加有限权重的边
            G.add_edge(i, j, weight=weight)
            G.add_edge(j, i, weight=weight)
    d_list = []
    for start_node in range(N):
        d, subgraph = perform_random_walk(G, N, start_node, (3) * N // 2)
        print("Perform random walk diameter:", d)
        d_list.append(d)
    print(torch.tensor(d_list).mean().item(), torch.tensor(d_list).std().item(), torch.tensor(d_list).max().item(), torch.tensor(d_list).min().item())
    assert 0
    A_init = torch.zeros((N, N), dtype=torch.float)

    # 建立 node -> index 的映射
    node_list = sorted(subgraph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # 遍历所有边，设置相应的矩阵元素为1
    for u, v in subgraph.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        A_init[i, j] = 1.0  # 有向边 (u->v)
        A_init[j, i] = 1.0  # 有向边 (u->v)
    A_torch = A_init * 2.0 - 1.0
    A_torch = torch.tensor(A_init, requires_grad=True, device=device)
    # 2) 原始权重矩阵 W, 若无边 => inf
    
    if save_graph:
        G = nx.Graph()

        # 添加节点
        G.add_nodes_from(range(N))

        # 添加边（忽略 inf，自环已排除）
        for i in range(N):
            for j in range(N):  # 避免重复无向边
                weight = W_init[i, j].item()  # 转换为 Python float
                # if weight != float('inf'):  # 仅添加有限权重的边
                G.add_edge(i, j, weight=weight)

        # 保存为 pkl 文件
        with open("G_100.pkl", "wb") as f:
            pkl.dump(G, f)

        print("Graph G_100.pkl saved successfully!")
        assert 0
    W_torch = torch.tensor(W_init, device=device)

    # 3) 训练超参
    K     = 5       # 最多 hop
    beta  = 10
    alpha = 10
    lr    = 0.01
    steps = 200

    optimizer = torch.optim.Adam([A_torch], lr=lr)

    # print("Initial A:", A_torch)
    A_optimized = A_torch.detach().cpu().clone()
    A_optimized_clamped = torch.clamp(A_optimized, -1, 1)
    A_optimized_01 = (A_optimized_clamped + 1.0) / 2.0  # [0,1]
    perm = random.sample(range(N), N)
    
    # 用 networkx 计算直径
    print(A_init.sum(dim=0))
    # assert 0
    initial_diameter_nx = nx_approx_diameter(A_init, W_init, N, degree = int(np.log2(N)) + 1)
    print(f"\n[NetworkX] Initial computed diameter = {initial_diameter_nx}")

    for step in range(steps+1):
        optimizer.zero_grad()
        D_k = soft_diameter(A_torch, W_torch, K=K, beta=beta, alpha=alpha, l=0.0)
        # 我们把近似直径当作loss
        loss = D_k
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f"Step={step}, ApproxDiameter={loss.item():.4f}")
        
        if step % 1 == 0:
            # print(A_torch.mean().item(), A_torch.std().item())
            A_optimized = A_torch.detach().cpu().clone()
            A_optimized_clamped = torch.clamp(A_optimized, -1, 1)
            A_optimized_01 = (A_optimized_clamped + 1.0) / 2.0  # [0,1]
            diameter_nx, test_G = nx_approx_diameter_directed(A_optimized_01, W_torch.cpu(), N, degree = int(np.log2(N)) - 1, perm=perm)
            print(f"\n[NetworkX] Computed diameter = {diameter_nx}")

    # ============ 用 networkx 验证优化后邻接矩阵对应的“真实”直径 =============
    # 把 A_torch 移回 CPU, clamp 到 [-1,1], 然后映射到 [0,1]
    A_optimized = A_torch.detach().cpu().clone()
    A_optimized_clamped = torch.clamp(A_optimized, -1, 1)
    A_optimized_01 = (A_optimized_clamped + 1.0) / 2.0  # [0,1]

    # 用 networkx 计算直径
    # print((A_init - A_torch).mean().item(), (A_init - A_torch).std().item())
    # diameter_nx = nx_approx_diameter(A_optimized_01, W_torch.cpu(), N, degree = 7)
    diameter_nx, test_G = nx_approx_diameter_directed(A_optimized_01, W_torch.cpu(), N, degree = int(np.log2(N)) - 1, perm=perm)
    print(f"\n[NetworkX] Initial computed diameter = {initial_diameter_nx}")
    print(f"\n[NetworkX] Computed diameter = {diameter_nx}")
    
    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if test_G.has_edge(i, j):  # ✅ Correct way to check if the edge exists
                weight = test_G[i][j].get('weight', 0)  # ✅ Correct way to get the weight
                subgraph.add_edge(i, j, weight=weight)

                if weight != W_init[i, j]:  # ✅ Now using the correct weight retrieval
                    print(f"Mismatch at edge ({i}, {j}): test_G={weight}, W_init={W_init[i, j]}")
                    assert 0  # Raise an assertion error to debug

    d = nx.diameter(subgraph, weight='weight')
    # print("Optimized A in [-1,1]:")
    # print(A_optimized)
