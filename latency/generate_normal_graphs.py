import networkx as nx
import numpy as np
import pickle as pkl
import os
import random

def generate_normal_weighted_graph(num_nodes, mu=100, sigma=15, seed=None):
    """
    生成权重服从正态分布的完全图
    
    Args:
        num_nodes: 节点数量
        mu: 正态分布均值
        sigma: 正态分布标准差
        seed: 随机种子
    
    Returns:
        networkx.Graph: 带权重的完全图
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # 创建完全图
    graph = nx.complete_graph(num_nodes)
    
    # 为每条边分配正态分布权重
    for (u, v) in graph.edges():
        # 生成正态分布权重，确保权重为正数
        weight = max(25, np.random.normal(mu, sigma))
        graph.edges[u, v]['weight'] = weight
        # 无向图，设置对称权重
        graph.edges[v, u]['weight'] = weight
    
    return graph

def generate_test_graphs(num_graphs=5, num_nodes=100, mu=100, sigma=15, base_seed=42):
    """
    生成多个测试图并保存
    
    Args:
        num_graphs: 要生成的图数量
        num_nodes: 每个图的节点数
        mu: 正态分布均值  
        sigma: 正态分布标准差
        base_seed: 基础随机种子
    
    Returns:
        list: 生成的图列表
    """
    test_graphs = []
    
    # 确保目录存在
    os.makedirs('../sc_test', exist_ok=True)
    
    for i in range(num_graphs):
        seed = base_seed + i
        graph = generate_normal_weighted_graph(num_nodes, mu, sigma, seed)
        test_graphs.append(graph)
        
        # 保存每个测试图
        filename = f'../sc_test/G_{num_nodes}_normal_test_{i}.pkl'
        with open(filename, 'wb') as f:
            pkl.dump(graph, f)
        
        print(f"Generated test graph {i}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # 显示一些统计信息
        weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        print(f"  Weight stats: mean={np.mean(weights):.2f}, std={np.std(weights):.2f}, min={np.min(weights):.2f}, max={np.max(weights):.2f}")
    
    return test_graphs

def save_background_graph(num_nodes=100, mu=100, sigma=15, seed=12345):
    """
    生成并保存训练用的背景图（训练时每次随机生成时的参考模板）
    """
    os.makedirs('../sc_test', exist_ok=True)
    
    graph = generate_normal_weighted_graph(num_nodes, mu, sigma, seed)
    filename = f'../sc_test/G_{num_nodes}_NORMAL.pkl'
    
    with open(filename, 'wb') as f:
        pkl.dump(graph, f)
    
    print(f"Saved background graph to {filename}")
    weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    print(f"Background graph weight stats: mean={np.mean(weights):.2f}, std={np.std(weights):.2f}")
    
    return graph

if __name__ == "__main__":
    print("Generating N=100 graphs with normal distribution weights (μ=100, σ=50)...")
    
    # 生成5个测试图
    test_graphs = generate_test_graphs(
        num_graphs=5, 
        num_nodes=100, 
        mu=100, 
        sigma=15, 
        base_seed=42
    )
    
    # 生成训练用背景图
    bg_graph = save_background_graph(
        num_nodes=100, 
        mu=100, 
        sigma=15, 
        seed=12345
    )
    
    print(f"\nGeneration complete!")
    print(f"Test graphs: 5 files saved as G_100_normal_test_*.pkl")
    print(f"Background graph: G_100_NORMAL.pkl") 