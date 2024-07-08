import numpy as np
import networkx as nx
import pickle as pkl
import random
import os
import math
class Simulated_Annealing():
    def __init__(self, num_vertices=20, K=4, initial_temp=100, 
                        final_temp=1, cooling_rate=0.95, iterations_per_temp=1000):
        self.num_vertices = num_vertices  # Number of vertices in the graph
        self.K = K              # Each vertex should have exactly K neighbors
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.G = nx.complete_graph(self.num_vertices)
        self.generate_weights()
        # Create a fully connected weighted graph
    
    def generate_weights(self,):
        for (u, v) in self.G.edges():
            self.G[u][v]['weight'] = np.random.randint(1, 10)
    
    def is_k_regular(self, graph):
        return all(dict(graph.degree()).values()) == [self.K] * len(graph.nodes())

    def calculate_diameter(self, graph):
        # Using all-pairs shortest path to calculate the diameter
        try:
            return nx.diameter(graph, weight='weight')
        except nx.NetworkXError:
            # If the graph is not connected
            return float('inf')

    def create_k_regular_graph(self,):
        candidate_graph = nx.random_regular_graph(self.K, self.num_vertices)
        for u, v in candidate_graph.edges():
            candidate_graph.edges[u,v]['weight'] = self.G.edges[u,v]['weight']
            candidate_graph.edges[v,u]['weight'] = self.G.edges[v,u]['weight']
        return candidate_graph
    
    def mutate(self, graph):
        # Randomly add or remove an edge and repair to maintain K-regularity
        u, v = np.random.choice(self.num_vertices, 2, replace=False)
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
        else:
            graph.add_edge(u, v)
            graph.edges[u, v]['weight'] = self.G.edges[u, v]['weight']
        # Repair logic to maintain K-regularity needs to be added here

        self.repair_k_regular(graph)


    def run(self,):
        current_solution = self.create_k_regular_graph()
        current_diameter = self.calculate_diameter(current_solution)
        best_solution = None
        best_diameter = float('inf')
        temperature = self.initial_temp
    
        while temperature > self.final_temp:
            for _ in range(self.iterations_per_temp):
                new_solution = current_solution.copy()
                self.mutate(new_solution)
                new_diameter = self.calculate_diameter(new_solution)
                delta_diameter = new_diameter - current_diameter
                
                if delta_diameter < 0 or random.random() < math.exp(-delta_diameter / temperature):
                    current_solution = new_solution
                    current_diameter = new_diameter

                if new_diameter < best_diameter:
                    best_diameter = new_diameter
                    best_solution = new_solution.copy()
                    
                    # assert 0
            # adj_matrix = nx.to_numpy_array(best_solution, weight='weight')
            # print([new_solution.degree(i) for i in range(self.num_vertices)])
            # print(adj_matrix)
            temperature *= self.cooling_rate
            
            print(f'Temperature {temperature}, Best Diameter: {best_diameter}')
            
        return best_solution, best_diameter

    def repair_k_regular(self, graph):
        # Step 1: Identify nodes with too few or too many connections
        surplus_nodes = [node for node in graph.nodes() if graph.degree(node) > self.K]

        # Step 2: Remove excess edges from surplus nodes
        while surplus_nodes:
            # print(len(surplus_nodes))
            node = surplus_nodes.pop()
            while graph.degree(node) > self.K:
                neighbors = list(graph.neighbors(node))
                # Remove edge to a neighbor that won't go below K connections
                prev_degree = graph.degree(node)
                for neighbor in neighbors:
                    graph.remove_edge(node, neighbor)
                    break
                if prev_degree == graph.degree(node):
                    break
        # print("surplus finished")
        # Step 3: Add missing edges to deficit nodes
        deficit_nodes = [node for node in graph.nodes() if graph.degree(node) < self.K]
        while deficit_nodes:
            node = deficit_nodes.pop()
            while graph.degree(node) < self.K:
                prev_degree = graph.degree(node)
                potential_partners = [n for n in deficit_nodes if not graph.has_edge(node, n) and n != node]
                for partner in potential_partners:
                    graph.add_edge(node, partner)
                    graph.edges[node, partner]['weight'] = self.G.edges[node, partner]['weight']
                    # Update the deficit list
                    if graph.degree(partner) == self.K:
                        deficit_nodes.remove(partner)
                    break
                if prev_degree == graph.degree(node):
                    break
        return graph

if __name__ == '__main__':
    seed = 123
    N = 20
    K = 4
    num_tests = 10
    np.random.seed(seed)
    random.seed(seed)
    diameter_list = []
    for i in range(num_tests):
        ga = Simulated_Annealing()
        graph_name = f'N={N}_{i}.pkl'
        with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
            G = pkl.load(f)
        ga.G = G
        best_solution, best_diameter = ga.run()
        
        diameter_list.append(best_diameter)
        print(f"Test G {i} - GA best diameter: {best_diameter}") 
        with open(f"solution/SA/Simulated_Annealing_N={N}_K={K}_id={i}.pkl", "wb") as f:
            pkl.dump(best_solution, f)
    print(sum(diameter_list) / len(diameter_list))
    print(diameter_list)