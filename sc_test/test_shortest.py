import random
import pickle as pkl
import numpy as np
import networkx as nx
def generate_subgraph_with_hamiltonian_cycles(G, M, K):
    """
    在 fully connected 图 G (N个节点) 中:
    - 添加 M 条随机哈密顿环 (Hamiltonian Cycles)
    - 每个节点额外添加 (K - M) 条基于最近邻的边，确保连通性
    
    参数:
    - G: 完全图 (networkx Graph), N个节点
    - M: 生成 M 条哈密顿环
    - K: 每个节点的目标度数 (必须满足 K >= M)

    返回:
    - subG: 生成的子图，包含 M 条哈密顿环和 K-M 条最近邻连接
    """
    N = G.number_of_nodes()
    assert K >= M, "K 必须至少等于 M，否则无法满足度数要求"

    # 初始化子图
    subG = nx.Graph()
    subG.add_nodes_from(G.nodes())

    # Step 1: 生成 M 条哈密顿环
    for _ in range(M):
        # 随机生成一个哈密顿环的节点顺序
        hamiltonian_cycle = random.sample(G.nodes(), N)
        # 形成环结构
        for i in range(N):
            u, v = hamiltonian_cycle[i], hamiltonian_cycle[(i + 1) % N]
            if not subG.has_edge(u, v):
                subG.add_edge(u, v, weight=G[u][v]['weight'])

    # Step 2: 添加 (K-M) 条基于最近邻的连接
    for node in G.nodes():
        # 找到当前节点已有的邻居
        existing_neighbors = set(subG.neighbors(node))
        num_missing_edges = K - len(existing_neighbors)  # 还需要添加的边数

        if num_missing_edges > 0:
            # 计算所有可能的候选邻居，并按欧几里得距离排序（假设 G 有位置坐标）
            all_neighbors = sorted(
                G.nodes(), key=lambda x: G[node][x]['weight']
            )

            # 选择最近邻的 (K-M) 个未连接的节点
            new_neighbors = [n for n in all_neighbors if n != node and n not in existing_neighbors][:num_missing_edges]
            
            # 添加最近邻边
            for neighbor in new_neighbors:
                subG.add_edge(node, neighbor, weight=G[node][neighbor]['weight'])

    return subG

# 示例用法
if __name__ == "__main__":
    N = 100  # 总节点数
    M = 2   # 生成 2 条哈密顿环
    K = 6   # 每个节点至少 4 条边
    # 生成完全连通图
    random.seed(42)
    with open('G_100.pkl', 'rb') as f:
        G = pkl.load(f)
    # G = pkl.load("G_100.pkl", 'rb')
    
    # 生成子图
    for M in range(1, 2):
        for _ in range(20):
            subG = generate_subgraph_with_hamiltonian_cycles(G, M, K)

            # 输出结果
            print("Edges in the generated subgraph:")
            print(nx.diameter(subG, weight='weight'))