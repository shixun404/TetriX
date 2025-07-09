import gym
import numpy as np
import networkx as nx
import random
import pickle as pkl
import copy

class OptimizedGraphEnv(gym.Env):
    """Optimized version of GraphEnv that eliminates deepcopy operations"""
    
    metadata = {'render.modes': ['console']}
    
    def __init__(self, num_nodes=500, K=8, num_sources=1, M=1, alpha=0.5, alpha_schedule=None):
        super(OptimizedGraphEnv, self).__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.num_sources = num_sources
        self.M = M
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        
        # Pre-allocate arrays for better performance
        self.mask = np.zeros(self.num_nodes, dtype=int)
        self.start_id = [0] * self.M
        self.degree_cache = np.zeros(self.num_nodes, dtype=int)
        
        # Cache for graph operations
        self.adjacency_cache = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        self.edge_weights_cache = {}
        
        # Initialize test graphs
        self.test_graphs = []
        
        # Pre-compute diameter cache for efficiency
        self.diameter_cache = {}
        
        # Initialize graphs
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        self.initial_graph = None
        self.initial_adjacency_matrix = None
        
        # Track edges for efficient removal
        self.current_edges = set()
        
        # Load background graph once
        self._load_background_graph()
        
    def _load_background_graph(self):
        """Load background graph once to avoid repeated I/O"""
        try:
            with open('../sc_test/G_400_FABRIC.pkl', 'rb') as f:
                self.graph_background = pkl.load(f)
        except:
            # Fallback if file doesn't exist
            self.graph_background = nx.complete_graph(self.num_nodes)
            for (u, v) in self.graph_background.edges():
                self.graph_background.edges[u, v]['weight'] = 1.0
        
    def update_alpha(self, episode):
        """Update alpha parameter based on schedule"""
        if self.alpha_schedule:
            self.alpha = self.alpha_schedule(episode)
    
    def reset(self, if_test=False, start_id=None, test_id=0):
        """Reset environment with optimized operations"""
        # Clear current graph efficiently
        self.graph.clear_edges()
        self.current_edges.clear()
        
        # Reset masks and positions
        self.mask.fill(self.K)  # Use fill instead of list comprehension
        
        if start_id is None:
            self.start_id = random.sample(range(self.num_nodes), self.M)
        else:
            self.start_id = start_id.copy()
        
        # Set environment state
        self.if_test = if_test
        self.test_id = test_id
        self.prev_diameter = 0
        self.cur_diameter = 0
        self.num_steps = 0
        
        # Initialize graph efficiently
        if if_test and self.test_graphs:
            self.initial_graph = self.test_graphs[self.test_id]
        else:
            self.initial_graph = nx.complete_graph(self.num_nodes)
            # Add weights efficiently
            for (u, v) in self.initial_graph.edges():
                if self.graph_background.has_edge(u, v):
                    weight = self.graph_background.edges[u, v]['weight']
                else:
                    weight = 1.0
                self.initial_graph.edges[u, v]['weight'] = weight
                self.initial_graph.edges[v, u]['weight'] = weight
        
        # Cache adjacency matrix
        self.initial_adjacency_matrix = nx.to_numpy_array(
            self.initial_graph, 
            nodelist=sorted(self.initial_graph.nodes())
        )
        
        # Cache current adjacency matrix (empty initially)
        self.adjacency_cache.fill(0)
        
        # Prepare mask efficiently
        mask = np.ones(self.num_nodes, dtype=int)
        for start in self.start_id:
            mask[start] = 0
        
        # Pre-compute degree array
        self.degree_cache = self.K - self.mask
        
        return {
            'initial_graph': self.initial_adjacency_matrix,
            'graph': self.adjacency_cache.copy(),
            'start_id': self.start_id,
            'mask': mask,
            'degree': self.degree_cache.copy()
        }
    
    def _compute_diameter_optimized(self, graph, use_cache=True):
        """Optimized diameter computation with caching"""
        if self.if_test:
            return 0
        
        # Use edge set as cache key for small graphs
        if use_cache and len(graph.edges()) < 1000:
            edge_key = tuple(sorted(graph.edges()))
            if edge_key in self.diameter_cache:
                return self.diameter_cache[edge_key]
        
        try:
            diameter = 0
            for i in range(self.num_sources):
                if i in graph.nodes():
                    shortest_length = nx.shortest_path_length(graph, source=i, weight='weight')
                    diameter = max(diameter, max(shortest_length.values()))
            
            # Cache result
            if use_cache and len(graph.edges()) < 1000:
                self.diameter_cache[edge_key] = diameter
            
            return diameter
        except:
            # Handle disconnected graphs
            if len(graph.nodes()) == 0:
                return float('inf')
            
            try:
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                diameter = 0
                for i in range(self.num_sources):
                    if i in subgraph:
                        shortest_length = nx.shortest_path_length(subgraph, source=i, weight='weight')
                        diameter = max(diameter, max(shortest_length.values()))
                return diameter
            except:
                return float('inf')
    
    def step(self, action):
        """Optimized step function eliminating deepcopy operations"""
        # Pre-compute diameter before actions (only if needed)
        if not self.if_test:
            diameter_before = self._compute_diameter_optimized(self.graph)
        else:
            diameter_before = 0
        
        # Apply actions efficiently - add edges directly
        weight_sum = []
        new_edges = []
        
        for i in range(self.M):
            start_node = self.start_id[i]
            end_node = action[i]
            
            # Add edge if it doesn't exist
            if not self.graph.has_edge(start_node, end_node):
                # Get weight from initial graph
                if self.initial_graph.has_edge(start_node, end_node):
                    weight = self.initial_graph.edges[start_node, end_node]['weight']
                else:
                    weight = 1.0
                
                self.graph.add_edge(start_node, end_node, weight=weight)
                self.current_edges.add((start_node, end_node))
                new_edges.append((start_node, end_node))
                weight_sum.append(weight)
            else:
                weight_sum.append(0)  # Edge already exists
        
        # Compute diameter after all actions
        if not self.if_test:
            diameter_after = self._compute_diameter_optimized(self.graph)
        else:
            diameter_after = 0
        
        # Compute global reward
        R_G = diameter_before - diameter_after
        
        # Compute individual rewards efficiently (avoid deepcopy)
        individual_rewards = []
        
        if not self.if_test:
            for p in range(self.M):
                # Instead of deepcopy, temporarily remove edge and compute diameter
                start_node = self.start_id[p]
                end_node = action[p]
                
                # Remove this partition's edge temporarily
                if self.graph.has_edge(start_node, end_node):
                    edge_data = self.graph.edges[start_node, end_node]
                    self.graph.remove_edge(start_node, end_node)
                    
                    # Compute diameter without this edge
                    diameter_without_p = self._compute_diameter_optimized(self.graph, use_cache=False)
                    
                    # Restore edge
                    self.graph.add_edge(start_node, end_node, **edge_data)
                    
                    # Compute marginal reward
                    R_M_p = diameter_without_p - diameter_after
                else:
                    R_M_p = 0
                
                # Combine rewards
                individual_reward = self.alpha * R_G + (1 - self.alpha) * R_M_p - weight_sum[p]
                individual_rewards.append(individual_reward)
        else:
            individual_rewards = [0.0] * self.M
        
        # Update environment state efficiently
        self.prev_diameter = diameter_after
        self.num_steps += 1
        
        # Update masks efficiently
        for i in range(self.M):
            self.mask[action[i]] -= 1
            self.mask[self.start_id[i]] -= 1
        
        # Compute new mask efficiently
        max_value = np.max(self.mask)
        mask = (self.mask == max_value).astype(int)
        
        # Update start positions
        for i in range(self.M):
            self.start_id[i] = action[i]
            mask[action[i]] = 0
        
        # Check termination
        done = self.num_steps >= (self.num_nodes * self.K // self.M)
        if done:
            mask = np.ones(self.num_nodes, dtype=int)
        
        # Update degree cache
        self.degree_cache = self.K - self.mask
        
        # Update adjacency cache
        self.adjacency_cache = nx.to_numpy_array(
            self.graph, 
            nodelist=sorted(self.graph.nodes())
        )
        
        # Return optimized state
        state = {
            'initial_graph': self.initial_adjacency_matrix,
            'graph': self.adjacency_cache,
            'start_id': self.start_id,
            'mask': mask,
            'degree': self.degree_cache.copy()
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

# Keep the original classes for backward compatibility
class GraphEnv(OptimizedGraphEnv):
    """Backward compatibility alias"""
    pass

def linear_alpha_schedule(episode, start_alpha=0.5, end_alpha=0.8, max_episodes=10000):
    """Linear interpolation from start_alpha to end_alpha"""
    progress = min(episode / max_episodes, 1.0)
    return start_alpha + (end_alpha - start_alpha) * progress

def exponential_alpha_schedule(episode, start_alpha=0.5, end_alpha=0.8, decay_rate=0.001):
    """Exponential approach to end_alpha"""
    return end_alpha - (end_alpha - start_alpha) * np.exp(-decay_rate * episode) 