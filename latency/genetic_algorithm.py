import numpy as np
import networkx as nx
import pickle as pkl
import os
import time
import random
class Genetic_Algorithm():
    def __init__(self, num_vertices=20, K=4, population_size=200, 
                        num_generations=1, mutation_rate=0.2):
        self.num_vertices = num_vertices  # Number of vertices in the graph
        self.K = K              # Each vertex should have exactly K neighbors
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
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
        t = time.time()
        candidate_graph = nx.random_regular_graph(self.K, self.num_vertices)
        print(time.time() - t)
        
        for u, v in candidate_graph.edges():
            candidate_graph.edges[u,v]['weight'] = self.G.edges[u,v]['weight']
            candidate_graph.edges[v,u]['weight'] = self.G.edges[v,u]['weight']
        p = []
        t = time.time()
        for i in range(10):
            p.append(nx.diameter(candidate_graph, weight='weight'))
        print((time.time() - t ) / 10)
        assert 0
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

    def fitness(self, graph):
        # Lower diameter means better fitness
        return 1 / self.calculate_diameter(graph)

    def run(self,):
        population = [self.create_k_regular_graph() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = float('-inf')
        fitness_scores = [self.fitness(graph) for graph in population]
        best_fitness = np.mean(fitness_scores)
        best_solution = population[0]
        print(f'Generation 0, Best Diameter: {1 / best_fitness}')
        execution_time = 0
        last_update_generation = 0
        start_time = time.time()
        for generation in range(self.num_generations):
            if generation - last_update_generation > 300:
                break

            # Selection
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]
            selected_population = sorted_population[:self.population_size // 2]  # keep the best half
            
            # Crossover (can be added here)

            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(self.population_size // 2, 2, replace=False)
                parent1 = selected_population[parent1]
                parent2 = selected_population[parent2]
                # Generate two offspring from each pair of parents
                offspring = self.crossover(parent1, parent2)
                new_population.extend([offspring])

            # Ensure the population does not exceed the desired size

            population = new_population[:self.population_size]
            
            # Mutation
            for individual in population:
                if np.random.rand() < self.mutation_rate:
                    self.mutate(individual)
            
            # Evaluate fitness
            fitness_scores = [self.fitness(graph) for graph in population]

            # Update best solution
            current_best = population[np.argmax(fitness_scores)]
            current_best_fitness = max(fitness_scores)
            if current_best_fitness > best_fitness:
                best_solution = current_best
                best_fitness = current_best_fitness
                last_update_generation = generation
                execution_time = time.time() - start_time
            print(f'Generation {generation}, Best Diameter: {1 / best_fitness}', 'time to find best:', execution_time)
            
        return best_solution, 1 / best_fitness, execution_time

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


    def crossover(self, parent1, parent2):
        # Initialize offspring as an empty graph with the same nodes
        offspring = nx.Graph()
        offspring.add_nodes_from(parent1.nodes())
        
        # List edges from both parents
        edges1 = list(parent1.edges(data='weight'))
        edges2 = list(parent2.edges(data='weight'))
        
        # Randomly select edges from each parent
        chosen_edges = []
        while offspring.number_of_edges() < parent1.number_of_edges():
            if np.random.rand() > 0.5 and edges1:
                edge = edges1.pop(np.random.randint(len(edges1)))
            elif edges2:
                edge = edges2.pop(np.random.randint(len(edges2)))
            else:
                break
            if not offspring.has_edge(edge[0], edge[1]) and (offspring.degree[edge[0]] < self.K \
                                                   or offspring.degree[edge[1]] < self.K):
                offspring.add_edge(edge[0], edge[1])
                offspring.edges[edge[0], edge[1]]['weight'] = edge[2]
            # print(len(offspring.edges), parent1.number_of_edges())
        # print('asdasdas')
        
        # It's possible the offspring is not fully K-regular at this point
        # Apply repair mechanism to adjust the graph
        self.repair_k_regular(offspring)
        
        return offspring

if __name__ == '__main__':
    seed = 123
    N = 500
    K = 8
    num_tests = 5
    random.seed(seed)
    np.random.seed(seed)
    diameter_list = []
    execution_time = []
    running_time = []
    for i in range(num_tests):

        ga = Genetic_Algorithm(num_vertices=N, K=K)
        graph_name = f'N={N}_{i}.pkl'
        with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
            G = pkl.load(f)
        ga.G = G
        start_time = time.time()
        best_solution, best_diameter, t = ga.run()
        execution_time.append(t)
        running_time.append(time.time() - start_time)
        diameter_list.append(best_diameter)
        print(running_time)
        print(f"Test G {i} - GA best diameter: {best_diameter}") 
        # assert 0
        with open(f"solution/GA/GA_N={N}_K={K}_id={i}.pkl", "wb") as f:
            pkl.dump(best_solution, f)
        
    
    print(diameter_list, sum(diameter_list) / len(diameter_list))
    print(execution_time, sum(execution_time) / len(execution_time))
    print(running_time, sum(running_time) / len(running_time))