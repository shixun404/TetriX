#!/usr/bin/env python3
"""
示例：使用N=100节点和正态分布权重(μ=100, σ=50)的图环境

此脚本展示了：
1. 如何初始化环境
2. 如何生成测试图（如果不存在）
3. 如何运行简单的训练和测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_parallel_optimized import OptimizedGraphEnv
from generate_normal_graphs import generate_test_graphs, save_background_graph
import numpy as np
import networkx as nx

def setup_graphs():
    """设置图文件（如果不存在则生成）"""
    print("Setting up N=100 graphs with normal distribution weights (μ=100, σ=50)...")
    
    # 检查是否需要生成图
    test_graphs_exist = True
    for i in range(5):
        filename = f'../sc_test/G_100_normal_test_{i}.pkl'
        if not os.path.exists(filename):
            test_graphs_exist = False
            break
    
    bg_graph_exists = os.path.exists('../sc_test/G_100_NORMAL.pkl')
    
    if not test_graphs_exist:
        print("Generating test graphs...")
        generate_test_graphs(num_graphs=5, num_nodes=100, mu=100, sigma=50, base_seed=42)
    else:
        print("Test graphs already exist.")
    
    if not bg_graph_exists:
        print("Generating background graph...")
        save_background_graph(num_nodes=100, mu=100, sigma=50, seed=12345)
    else:
        print("Background graph already exists.")
    
    print("Graph setup complete!\n")

def test_environment():
    """测试环境基本功能"""
    print("Testing environment functionality...")
    
    # 创建环境实例
    env = OptimizedGraphEnv(
        num_nodes=100,
        K=4,          # 每个节点最多4条边
        M=4,          # 4个并行agent
        num_sources=1,
        alpha=0.5,
        weight_mu=100,
        weight_sigma=50
    )
    
    print(f"Environment created with {env.num_nodes} nodes")
    
    # 测试训练模式
    print("\n=== Training Mode Test ===")
    state_dict = env.reset(if_test=False, start_id=[0, 25, 50, 75])
    print(f"Training reset complete. Graph shape: {state_dict['graph'].shape}")
    
    # 显示初始图的权重统计
    if env.initial_graph is not None:
        weights = [data['weight'] for _, _, data in env.initial_graph.edges(data=True)]
        print(f"Initial graph weights - Mean: {np.mean(weights):.2f}, Std: {np.std(weights):.2f}")
        print(f"Weight range: [{np.min(weights):.2f}, {np.max(weights):.2f}]")
    else:
        print("Initial graph is None")
    
    # 测试几步动作
    print("\nTesting actions...")
    for step in range(3):
        # 随机选择动作（避免选择起始节点）
        actions = []
        for i in range(env.M):
            available_nodes = [j for j in range(env.num_nodes) if j != env.start_id[i]]
            action = np.random.choice(available_nodes)
            actions.append(action)
        
        next_state, rewards, done, info = env.step(actions)
        print(f"Step {step+1}: Actions={actions}, Rewards={[f'{r:.2f}' for r in rewards]}, Done={done}")
        
        if done:
            break
    
    # 测试评估模式  
    print("\n=== Test Mode ===")
    for test_id in range(min(3, len(env.test_graphs))):
        state_dict = env.reset(if_test=True, test_id=test_id, start_id=[0, 25, 50, 75])
        if env.initial_graph is not None:
            weights = [data['weight'] for _, _, data in env.initial_graph.edges(data=True)]
            print(f"Test graph {test_id} - Mean weight: {np.mean(weights):.2f}, Std: {np.std(weights):.2f}")
        else:
            print(f"Test graph {test_id} - Initial graph is None")

def compare_graph_properties():
    """比较不同图的性质"""
    print("\n=== Graph Properties Comparison ===")
    
    env = OptimizedGraphEnv(num_nodes=100, weight_mu=100, weight_sigma=50)
    
    # 生成几个不同的训练图
    print("Comparing different training graphs:")
    for i in range(3):
        graph = env._generate_normal_weighted_graph(seed=1000+i)
        weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        
        print(f"Graph {i+1}:")
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print(f"  Weight stats: μ={np.mean(weights):.2f}, σ={np.std(weights):.2f}")
        print(f"  Weight range: [{np.min(weights):.2f}, {np.max(weights):.2f}]")
        
        # 计算图的直径（用于验证）
        try:
            diameter = nx.diameter(graph, weight='weight')
            print(f"  Graph diameter: {diameter:.2f}")
        except:
            print(f"  Graph diameter: disconnected")
        print()

if __name__ == "__main__":
    print("N=100 Environment Example with Normal Distribution Weights")
    print("=" * 60)
    
    # 设置图文件
    setup_graphs()
    
    # 测试环境
    test_environment()
    
    # 比较图性质
    compare_graph_properties()
    
    print("Example complete! You can now use the environment for training.")
    print("\nTo use in your training scripts:")
    print("from env_parallel_optimized import OptimizedGraphEnv")
    print("env = OptimizedGraphEnv(num_nodes=100, weight_mu=100, weight_sigma=50)") 