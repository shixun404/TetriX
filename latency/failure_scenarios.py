#!/usr/bin/env python3
"""
实现三种网络故障场景的模拟函数：
1. 延迟噪声 (latency fluctuation)
2. 链路失效 (link failure) 
3. 节点 churn
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from scipy import stats
from typing import Dict, List, Tuple
import os

class FailureScenarios:
    def __init__(self, graph: nx.Graph, seed: int = 42):
        """
        初始化故障场景模拟器
        
        Args:
            graph: 原始图
            seed: 随机种子
        """
        self.original_graph = graph.copy()
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def latency_fluctuation(self, scenario: str = "mild") -> nx.Graph:
        """
        1) 延迟噪声场景：对边权重添加乘性对数正态噪声
        
        w'_e = w_e * exp(N(0, σ²))
        
        Args:
            scenario: "mild" (σ=0.10), "moderate" (σ=0.30), "severe" (σ=0.60)
        
        Returns:
            带噪声的图副本
        """
        G = self.original_graph.copy()
        
        # 设置噪声参数
        sigma_map = {
            "mild": 0.10,      # ≈±10-20% 抖动
            "moderate": 0.30,
            "severe": 0.60
        }
        sigma = sigma_map.get(scenario, 0.10)
        
        # 对每条边添加乘性对数正态噪声
        for u, v in G.edges():
            original_weight = G.edges[u, v]['weight']
            # 乘性对数正态噪声: w' = w * exp(N(0, σ²))
            noise_factor = np.exp(np.random.normal(0, sigma))
            G.edges[u, v]['weight'] = original_weight * noise_factor
        
        return G
    
    def latency_fluctuation_burst(self, scenario: str = "mild") -> nx.Graph:
        """
        延迟噪声的突发/拥塞脉冲变种 (ON-OFF burst)
        
        以概率 p_burst 进入"拥塞态"，期间把该边延迟乘以系数 B
        
        Args:
            scenario: "mild", "moderate", "severe"
        
        Returns:
            带突发噪声的图副本
        """
        G = self.original_graph.copy()
        
        # 设置突发参数
        burst_params = {
            "mild": {"p_burst": 0.02, "T_burst": 10, "B": 2},
            "moderate": {"p_burst": 0.05, "T_burst": 30, "B": 3},
            "severe": {"p_burst": 0.10, "T_burst": 60, "B": 5}
        }
        params = burst_params.get(scenario, burst_params["mild"])
        
        # 对每条边检查是否进入拥塞态
        for u, v in G.edges():
            if np.random.random() < params["p_burst"]:
                # 进入拥塞态，延迟乘以系数B
                original_weight = G.edges[u, v]['weight']
                G.edges[u, v]['weight'] = original_weight * params["B"]
        
        return G
    
    def link_failure(self, scenario: str = "mild", simulation_time: float = 1.0) -> nx.Graph:
        """
        2) 链路失效场景：基于泊松故障 + 指数修复的Markov模型
        
        每条边独立：故障到达率 λ_e，修复率 μ_e
        失效时将该边权置为∞
        
        Args:
            scenario: "mild", "moderate", "severe"
            simulation_time: 模拟时间长度（小时）
        
        Returns:
            有链路失效的图副本
        """
        G = self.original_graph.copy()
        
        # 设置故障参数 (per hour)
        failure_params = {
            "mild": {"lambda_e": 1/24, "mttr_min": 5},      # 日均一次故障，5分钟修复
            "moderate": {"lambda_e": 1/6, "mttr_min": 15},   # 6小时一次，15分钟修复  
            "severe": {"lambda_e": 1/1, "mttr_min": 30}      # 每小时一次，30分钟修复
        }
        params = failure_params.get(scenario, failure_params["mild"])
        
        # 计算修复率 μ_e = 1/MTTR (per hour)
        mu_e = 1.0 / (params["mttr_min"] / 60.0)  # 转换为小时
        
        # 对每条边模拟故障
        failed_edges = set()
        for u, v in G.edges():
            # 检查在simulation_time内是否发生故障
            # 泊松过程：P(故障) = 1 - exp(-λt)
            failure_prob = 1 - np.exp(-params["lambda_e"] * simulation_time)
            
            if np.random.random() < failure_prob:
                # 发生故障，检查是否已修复
                # 指数修复：P(修复) = 1 - exp(-μt)
                repair_prob = 1 - np.exp(-mu_e * simulation_time)
                
                if np.random.random() > repair_prob:
                    # 未修复，标记为失效
                    failed_edges.add((u, v))
        
        # 将失效的边权重设为无穷大
        for u, v in failed_edges:
            G.edges[u, v]['weight'] = float('inf')
        
        return G
    
    def gilbert_elliott_failure(self, scenario: str = "mild") -> nx.Graph:
        """
        链路失效的Gilbert-Elliott两态模型变种
        
        Good↔Bad 转移，Bad状态下延迟增加
        
        Args:
            scenario: "mild", "moderate", "severe"
        
        Returns:
            应用Gilbert-Elliott模型的图副本
        """
        G = self.original_graph.copy()
        
        # 设置转移概率
        ge_params = {
            "mild": {"p_gb": 0.01, "p_bg": 0.2, "bad_factor": 3},
            "moderate": {"p_gb": 0.03, "p_bg": 0.1, "bad_factor": 5},
            "severe": {"p_gb": 0.05, "p_bg": 0.05, "bad_factor": 10}
        }
        params = ge_params.get(scenario, ge_params["mild"])
        
        # 对每条边应用两态模型
        for u, v in G.edges():
            # 简化：直接采样当前状态（Good=0, Bad=1）
            # 平稳分布：P(Bad) = p_gb / (p_gb + p_bg)
            p_bad = params["p_gb"] / (params["p_gb"] + params["p_bg"])
            
            if np.random.random() < p_bad:
                # Bad状态：延迟增加
                original_weight = G.edges[u, v]['weight']
                G.edges[u, v]['weight'] = original_weight * params["bad_factor"]
        
        return G
    
    def node_churn(self, scenario: str = "mild", simulation_time: float = 1.0) -> nx.Graph:
        """
        3) 节点churn场景：节点加入/离开过程
        
        泊松到达率 λ_n，在线时长服从Weibull或Pareto分布
        离开时移除节点及其相邻边
        
        Args:
            scenario: "mild", "moderate", "severe"  
            simulation_time: 模拟时间长度（小时）
        
        Returns:
            经过节点churn的图副本
        """
        G = self.original_graph.copy()
        
        # 设置churn参数
        churn_params = {
            "mild": {"lambda_n": 0.01, "distribution": "weibull", "k": 1.5, "theta": 3},
            "moderate": {"lambda_n": 0.05, "distribution": "weibull", "k": 1.2, "theta": 2},
            "severe": {"lambda_n": 0.10, "distribution": "pareto", "alpha": 2, "xmin": 1}
        }
        params = churn_params.get(scenario, churn_params["mild"])
        
        nodes_to_remove = set()
        
        # 对每个节点模拟churn过程
        for node in list(G.nodes()):
            # 检查是否发生离开事件
            departure_prob = 1 - np.exp(-params["lambda_n"] * simulation_time)
            
            if np.random.random() < departure_prob:
                # 发生离开，采样在线时长
                if params["distribution"] == "weibull":
                    # Weibull分布：online_time ~ Weibull(k, θ)
                    online_time = np.random.weibull(params["k"]) * params["theta"]
                else:  # pareto
                    # Pareto分布：online_time ~ Pareto(α, x_min)
                    online_time = (np.random.pareto(params["alpha"]) + 1) * params["xmin"]
                
                # 检查是否在simulation_time内回来
                if online_time < simulation_time:
                    # 节点在模拟期间离线，但会回来，暂时不移除
                    pass
                else:
                    # 节点在模拟期间持续离线
                    nodes_to_remove.add(node)
        
        # 移除离线的节点
        G.remove_nodes_from(nodes_to_remove)
        
        return G

def test_failure_scenarios(G: nx.Graph, num_samples: int = 100, save_dir: str = "failure_analysis"):
    """
    测试所有故障场景并生成histogram
    
    Args:
        G: 原始图
        num_samples: 每种场景的采样次数
        save_dir: 保存结果的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    scenarios = FailureScenarios(G)
    original_diameter = nx.diameter(G, weight='weight')
    
    print(f"Original graph diameter: {original_diameter:.2f}")
    print(f"Graph info: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 定义所有测试场景
    test_scenarios = [
        # 延迟噪声
        ("latency_fluctuation", "mild", "Latency Fluctuation (Mild σ=0.10)"),
        ("latency_fluctuation", "moderate", "Latency Fluctuation (Moderate σ=0.30)"),
        ("latency_fluctuation", "severe", "Latency Fluctuation (Severe σ=0.60)"),
        ("latency_fluctuation_burst", "mild", "Latency Burst (Mild p=0.02, B=2)"),
        ("latency_fluctuation_burst", "moderate", "Latency Burst (Moderate p=0.05, B=3)"),
        ("latency_fluctuation_burst", "severe", "Latency Burst (Severe p=0.10, B=5)"),
        
        # 链路失效
        ("link_failure", "mild", "Link Failure (Mild λ=1/24h, MTTR=5min)"),
        ("link_failure", "moderate", "Link Failure (Moderate λ=1/6h, MTTR=15min)"),
        ("link_failure", "severe", "Link Failure (Severe λ=1/1h, MTTR=30min)"),
        ("gilbert_elliott_failure", "mild", "Gilbert-Elliott (Mild p_gb=0.01, p_bg=0.2)"),
        ("gilbert_elliott_failure", "moderate", "Gilbert-Elliott (Moderate p_gb=0.03, p_bg=0.1)"),
        ("gilbert_elliott_failure", "severe", "Gilbert-Elliott (Severe p_gb=0.05, p_bg=0.05)"),
        
        # 节点churn
        ("node_churn", "mild", "Node Churn (Mild λ=0.01, Weibull k=1.5, θ=3h)"),
        ("node_churn", "moderate", "Node Churn (Moderate λ=0.05, Weibull k=1.2, θ=2h)"),
        ("node_churn", "severe", "Node Churn (Severe λ=0.10, Pareto α=2, x_min=1h)")
    ]
    
    # 创建总览图
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    axes = axes.flatten()
    
    results = {}
    
    for idx, (method, severity, title) in enumerate(test_scenarios):
        print(f"\nTesting {title}...")
        diameter_list = []
        
        for i in range(num_samples):
            try:
                # 获取对应的方法
                method_func = getattr(scenarios, method)
                G_modified = method_func(severity)
                
                # 计算直径
                if G_modified.number_of_nodes() > 1 and nx.is_connected(G_modified):
                    diameter = nx.diameter(G_modified, weight='weight')
                    # 过滤掉无穷大的直径（表示图不连通）
                    if np.isfinite(diameter):
                        diameter_list.append(diameter)
                
                if i % 20 == 0:
                    print(f"  Sample {i}: {len(diameter_list)} valid diameters so far")
                    
            except Exception as e:
                if i % 20 == 0:
                    print(f"  Sample {i}: Failed - {str(e)}")
                continue
        
        # 绘制histogram
        if diameter_list:
            ax = axes[idx]
            ax.hist(diameter_list, bins=min(20, len(set(diameter_list))), 
                   edgecolor='black', alpha=0.7, density=True)
            ax.axvline(original_diameter, color='blue', linestyle='--', 
                      label=f'Original: {original_diameter:.2f}')
            
            if len(diameter_list) > 0:
                mean_diameter = np.mean(diameter_list)
                ax.axvline(mean_diameter, color='red', linestyle='--', 
                          label=f'Mean: {mean_diameter:.2f}')
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Diameter')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 保存统计信息
            results[f"{method}_{severity}"] = {
                'title': title,
                'diameter_list': diameter_list,
                'original_diameter': original_diameter,
                'mean_diameter': np.mean(diameter_list) if diameter_list else None,
                'std_diameter': np.std(diameter_list) if len(diameter_list) > 1 else None,
                'success_rate': len(diameter_list) / num_samples
            }
            
            print(f"  Results: {len(diameter_list)}/{num_samples} successful samples")
            print(f"  Mean diameter: {np.mean(diameter_list):.2f}")
            
        else:
            ax = axes[idx]
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{title} (Failed)", fontsize=10)
            
            results[f"{method}_{severity}"] = {
                'title': title,
                'diameter_list': [],
                'success_rate': 0
            }
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_failure_scenarios_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存结果
    import pickle
    with open(f"{save_dir}/failure_scenarios_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # 打印总结
    print("\n" + "="*80)
    print("FAILURE SCENARIOS ANALYSIS SUMMARY")
    print("="*80)
    
    for key, result in results.items():
        if result['success_rate'] > 0:
            print(f"\n{result['title']}:")
            print(f"  Success rate: {result['success_rate']*100:.1f}%")
            if result.get('mean_diameter'):
                print(f"  Mean diameter: {result['mean_diameter']:.2f}")
                print(f"  Std diameter: {result.get('std_diameter', 0):.2f}")
        else:
            print(f"\n{result['title']}: FAILED (no valid samples)")
    
    return results

if __name__ == "__main__":
    # 测试用例
    print("Creating test graph...")
    G = nx.complete_graph(20)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(1, 10)
    
    print("Running failure scenarios test...")
    results = test_failure_scenarios(G, num_samples=100)
    print("Test completed!")
