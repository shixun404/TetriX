import gym
from gym import spaces
import networkx as nx
import numpy as np
import random
import pickle as pkl
from math import exp
import os
from collections import Counter
import torch as th
import torch
import copy

class GraphEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    
    def __init__(self, num_nodes=500, K=8, num_sources=1, M=1, alpha=0.5, alpha_schedule=None):
        super(GraphEnv, self).__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.M = M  # Number of partitions
        self.alpha = alpha  # Balance between global and local rewards
        self.alpha_schedule = alpha_schedule  # Optional schedule for alpha
        self.current_episode = 0
        
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict({
            'graph': spaces.Box(low=0, high=1, shape=(num_nodes, num_nodes), dtype=np.uint8)
        })
        
        self.graph = nx.DiGraph()
        self.initial_graph = nx.complete_graph(self.num_nodes)
        self.test_graphs = []
        self.test_id = -1
        self.graph.add_nodes_from(range(num_nodes))
        self.start_id = [i * (self.num_nodes // self.M) for i in range(self.M)]
        self.num_steps = 0
        self.mask = []
        self.if_test = False
        self.mean = 5
        self.std_dev = 1
        self.num_sources = num_sources
        self.load_graph()

    def update_alpha(self, episode):
        """Update alpha according to schedule if provided"""
        self.current_episode = episode
        if self.alpha_schedule is not None:
            self.alpha = self.alpha_schedule(episode)

    def reset(self, if_test=False, start_id=None, test_id=0):
        self.if_test = if_test
        self.graph.clear()
        self.mask = [self.K for i in range(self.num_nodes)]
        mask = [1 for i in range(self.num_nodes)]
        
        # Generate default start_ids if None is provided
        if start_id is None:
            start_id = [i * (self.num_nodes // self.M) for i in range(self.M)]
        
        for i in range(self.M):
            mask[start_id[i]] = 0
        self.initial_graph.clear() 
        self.graph.add_nodes_from(range(self.num_nodes))
        self.start_id = start_id
        self.test_id = test_id
        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0
        
        # Load background graph
        with open('../sc_test/G_400_FABRIC.pkl', 'rb') as f:
            graph_background = pkl.load(f)
            
        if if_test:
            self.initial_graph = nx.Graph(self.test_graphs[self.test_id])  
        else:
            self.initial_graph = nx.complete_graph(self.num_nodes)
            self.initial_graph.add_nodes_from(range(self.num_nodes))
            for (u, v) in self.initial_graph.edges():
                self.initial_graph.edges[u, v]['weight'] = graph_background.edges[u, v]['weight']
            for (u, v) in self.initial_graph.edges():
                self.initial_graph.edges[v,u]['weight'] = self.initial_graph.edges[u,v]['weight']
                
        self.initial_adjacency_matrix = nx.to_numpy_array(self.initial_graph, nodelist=sorted(self.initial_graph.nodes()))
        
        # Initialize graph with a random ring to ensure connectivity
        # self._initialize_ring_graph()
        
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.initial_graph.nodes()))
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        
        return {'initial_graph': self.initial_adjacency_matrix, 
                'graph': adjacency_matrix, 'start_id': self.start_id, 'mask': mask, 'degree': degree}
    
    def load_graph(self):
        with open(f'../sc_test/G_400_FABRIC.pkl', 'rb') as f:
            graph_background = pkl.load(f)
            self.test_graphs.append(graph_background)

    def _initialize_ring_graph(self):
        """Initialize the graph with a random ring to ensure connectivity"""
        # Create a random permutation of nodes
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
        
        # Create ring edges
        for i in range(self.num_nodes):
            next_node = nodes[(i + 1) % self.num_nodes]
            self.graph.add_edge(nodes[i], next_node)
            
            # Copy weight from initial graph
            if self.initial_graph.has_edge(nodes[i], next_node):
                weight = self.initial_graph.edges[nodes[i], next_node]['weight']
            else:
                weight = self.initial_graph.edges[next_node, nodes[i]]['weight']  # undirected
            
            self.graph.edges[nodes[i], next_node]['weight'] = weight
            
            # Update mask (each node uses one edge)
            self.mask[nodes[i]] -= 1
            self.mask[next_node] -= 1

    def compute_diameter(self, graph):
        """Compute diameter of the graph with weighted edges"""
        try:
            diameter = 0
            for i in range(self.num_sources):
                shortest_length = nx.shortest_path_length(graph, source=i, weight='weight')
                diameter = max(diameter, max(shortest_length.values()))
            return diameter
        except:
            # Handle disconnected graphs
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            diameter = 0
            for i in range(self.num_sources):
                if i in subgraph:
                    shortest_length = nx.shortest_path_length(subgraph, source=i, weight='weight')
                    diameter = max(diameter, max(shortest_length.values()))
            return diameter

    def step(self, action): 
        """
        Implements Algorithm 1: Parallel DGRO with two-level rewards
        
        Args:
            action: List of actions, one per partition
            
        Returns:
            state: Next state
            rewards: List of rewards, one per partition (implementing two-level reward)
            done: Whether episode is finished
            info: Additional information
        """
        # Store graph state before actions
        graph_before = copy.deepcopy(self.graph)
        diameter_before = self.compute_diameter(graph_before) if not self.if_test else 0
        
        # Apply all actions to get final graph G_{t+1}
        weight_sum = []
        for i in range(self.M):
            try:
                self.graph.add_edge(self.start_id[i], action[i])
            except:
                pass
            w = self.initial_graph.edges[self.start_id[i], action[i]]['weight']
            self.graph.edges[self.start_id[i], action[i]]['weight'] = w
            weight_sum.append(w)
            
        # Compute diameter after all actions
        diameter_after = self.compute_diameter(self.graph) if not self.if_test else 0
        
        # Compute global reward R_G
        R_G = diameter_before - diameter_after
        
        # Compute individual rewards for each partition
        individual_rewards = []
        
        if not self.if_test:
            for p in range(self.M):
                # Create graph without partition p's action (G_{t+1} \ a_p^t)
                graph_without_p = copy.deepcopy(graph_before)
                
                # Add all other partitions' actions except partition p
                for i in range(self.M):
                    if i != p:
                        try:
                            graph_without_p.add_edge(self.start_id[i], action[i])
                            w = self.initial_graph.edges[self.start_id[i], action[i]]['weight']
                            graph_without_p.edges[self.start_id[i], action[i]]['weight'] = w
                        except:
                            pass
                
                # Compute marginal reward R_M^(p)
                diameter_without_p = self.compute_diameter(graph_without_p)
                R_M_p = diameter_without_p - diameter_after
                
                # Combine global and marginal rewards
                individual_reward = self.alpha * R_G + (1 - self.alpha) * R_M_p - weight_sum[p]
                individual_rewards.append(individual_reward)
        else:
            # For testing, just return zero rewards
            individual_rewards = [0.0] * self.M
        
        # Update environment state
        adjacency_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        self.prev_diameter = diameter_after
        self.num_steps += 1
        
        # Update masks and constraints
        for i in range(self.M):
            self.mask[action[i]] -= 1
            self.mask[self.start_id[i]] -= 1
        
        max_value = np.max(self.mask)   
        min_value = np.min(self.mask)   
        mask = (self.mask == max_value).astype(int)
        
        # Update start positions
        for i in range(self.M):
            self.start_id[i] = action[i]
            mask[action[i]] = 0
        # masked_indices = torch.where(torch.as_tensor(mask) == 0)[0]
        # print(f'masked_indices: {masked_indices}')
        
        # Check termination condition
        if self.num_steps >= ((self.num_nodes) * self.K // self.M):
            mask = [1 for i in range(self.num_nodes)]
            done = True
        else:
            done = False
            
        degree = [self.K - self.mask[i] for i in range(self.num_nodes)]
        state = {
            'initial_graph': self.initial_adjacency_matrix, 
            'graph': adjacency_matrix, 
            'start_id': self.start_id, 
            'mask': mask, 
            'degree': degree
        }

        info = {
            'global_reward': R_G,
            'individual_rewards': individual_rewards,
            'diameter_before': diameter_before,
            'diameter_after': diameter_after,
            'alpha': self.alpha
        }

        return state, individual_rewards, done, info

    def render(self, mode='console'):
        if mode == 'console':
            print(nx.info(self.graph))

# Example alpha scheduling functions
def linear_alpha_schedule(episode, start_alpha=0.5, end_alpha=0.8, max_episodes=10000):
    """Linear interpolation from start_alpha to end_alpha"""
    progress = min(episode / max_episodes, 1.0)
    return start_alpha + (end_alpha - start_alpha) * progress

def exponential_alpha_schedule(episode, start_alpha=0.5, end_alpha=0.8, decay_rate=0.001):
    """Exponential approach to end_alpha"""
    return end_alpha - (end_alpha - start_alpha) * np.exp(-decay_rate * episode) 