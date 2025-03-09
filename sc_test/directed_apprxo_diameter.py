import torch
import networkx as nx
import math
import random

def nx_approx_diameter_directed(adj_01: torch.Tensor,
                                      W: torch.Tensor,
                                      N: int,
                                      degree: int = 3, perm=None) -> float:
    """
    用 networkx 计算图的(带权)直径 (有向图):
    每个节点的入度和出度均不能超过 degree。

    参数:
      adj_01: [N,N], 0~1 的邻接强度(阈值判断是否有边)
      W     : [N,N], 边权, 若无边则应是 inf
      N     : 节点数量
      degree: 每个节点的(入度, 出度)最大限制

    返回值:
      float 型直径; 如果图不连通, 则可能为 inf.
    """

    # 转到 CPU + numpy
    adj_01_np = adj_01.cpu().numpy()
    W_np = W.cpu().numpy()

    # 创建有向图 DiGraph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    # random.seed(seed)
    # perm = random.sample(range(N), N)
    # print("Random permutation:", perm)

    # 按照该排列依次添加有向边
    # 第 i 个节点指向第 i+1 个节点
    if perm is not None:
        for i in range(N):
            # 下一个节点索引 (环状，最后连回第 0 个)
            next_i = (i + 1) % N
            G.add_edge(perm[i], perm[next_i], weight=W_np[perm[i], perm[next_i]])

    # 记录入度和出度
    out_degrees = {i: 0 for i in range(N)}
    in_degrees  = {i: 0 for i in range(N)}

    # 小阈值, 超过它才视为可能有边
    threshold = 1e-3

    # 若想去重，可以用集合记录已加的边
    added_edges = set()

    # 对每个节点 i，根据 adj_01_np[i,j] 大小排序，优先选强度大的
    for i in range(N):
        neighbors = []
        for j in range(N):
            if i != j and not math.isinf(W_np[i, j]):
                # 邻接强度
                adj_value = adj_01_np[i, j]
                neighbors.append((j, adj_value, W_np[i, j]))
        
        # 按邻接强度从大到小排序 (如果想从小到大, 去掉 reverse=True)
        neighbors.sort(key=lambda x: x[1], reverse=True)

        # 逐个尝试添加边 i->j
        for j, adj_value, weight in neighbors:
            # 入度和出度都不能超限
            if adj_value > threshold:
                if out_degrees[i] < degree and in_degrees[j] < degree:
                    if (i, j) not in added_edges:
                        if weight == 0:
                            print(f"Warning: weight=0 for edge ({i}, {j})")
                            assert 0
                        G.add_edge(i, j, weight=weight)
                        added_edges.add((i, j))
                        out_degrees[i] += 1
                        in_degrees[j] += 1
                else:
                    # 如果任意一方超限，就跳过
                    pass

    # 用 all_pairs_dijkstra_path_length 求所有点对距离, 再取最大 => 直径
    distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    max_dist = 0.0
    for src_node, dist_dict in distances.items():
        for tgt_node, dist_val in dist_dict.items():
            if dist_val > max_dist:
                max_dist = dist_val

    # 若有节点无法到达 => 不连通 => 直径=inf
    for src_node in range(N):
        # 如果从 src_node 出发能到达的节点数 < N, 则说明不连通
        if len(distances[src_node]) < N:
            return float('inf')
    # print(G.in_degree(), G.out_degree())
    return max_dist, G
