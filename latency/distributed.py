import random
import networkx as nx
import torch as th

def initialize_graph(N, K):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    latency = nx.complete_graph(N)
    for (u, v) in latency.edges():
        latency.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
        # test_graph.edges[u, v]['weight'] = np.random.normal(self.mean, self.std_dev)
    for (u, v) in latency.edges():
        latency.edges[v,u]['weight'] = latency.edges[u,v]['weight']  # Assign random positive weights
    for i in range(N):
        connections = random.sample([j for j in range(N) if j != i], K)
        for j in connections:
            graph.add_edge(i, j, weight=latency.edges[i, j]['weight'])
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    random_ring = {}
    # Connect each node to the next in the shuffled list, making a ring
    for i in range(len(nodes)):
        graph.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
        random_ring[nodes[i]] = nodes[(i + 1) % len(nodes)]
    return graph, latency, random_ring

def update_graph( graph, latency, random_ring, N, K, M, sample_sources=10):
    previous_diameter = 0
    try:
        previous_diameter = nx.diameter(graph, weight='weight')
    except:
        largest_cc = max(nx.strongly_connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        previous_diameter = nx.diameter(subgraph)
    # main loop to update the graph
    
    for itr in range(20000):
        # Step 1: Randomly select 3 source nodes
        sources = random.sample(graph.nodes(), sample_sources)
        
        # Step 2: Compute the shortest paths from these sources
        all_paths_lengths = []
        for source in sources:
            lengths = nx.single_source_dijkstra_path_length(graph, source)
            all_paths_lengths.append(lengths)
        successor = []
        for node in range(N):
            successor.append( [ i for i in graph.successors(node) if i != random_ring[node]] )
            # # successors = list(graph.successors(node))
            # print(list(graph.successors(node)))
            # assert 0
        # Step 3: Update scores (example update mechanism)
       
        scores = th.zeros(N, K)
        for lengths in all_paths_lengths:
            # if len(successor[node]) >= K - 2:
            for node in range(N):
                neighbor_lengths = th.as_tensor([lengths.get(neighbor, float('inf')) for neighbor in successor[node]])
                neighbor_lengths = neighbor_lengths - neighbor_lengths.min()
                scores[node][:len(successor[node])] += th.as_tensor(neighbor_lengths)
            

        top2_scores, top2_indices = th.topk(scores, 2, dim=1)

        # Step 2: Find the corresponding outgoing connected nodes id
        selected_neighbors = [[successor[i][idx] for idx in top2_indices[i]] for i in range(N)]
        remaining_neighbors = []
        for i in range(N):
            if len(successor[i]) == K:
                for j in range(2):
                    # graph.edges[i, selected_neighbors[i][j]]['weights'] = 0
                    graph.remove_edge(i, selected_neighbors[i][j])
                    
                remaining_neighbors.append(
                    [n for j, n in enumerate(successor[i]) if j not in top2_indices[i]]
                )
            else:
                remaining_neighbors.append(
                    [n for n in successor[i]]
                )

        # Step 3: Drop the selected two outgoing connections
        # Assume 'dropping' means we don't consider these for the next step of reconnection

        # Step 4: Randomly select two new nodes as new outgoing connections
        # Ensuring that the newly selected nodes are not the same as any current ones
        for i in range(N):
            new_connections = []
            potential_new_connections = set(range(K)) - set(remaining_neighbors[i]) - set([i])
            new_connections = random.sample(potential_new_connections, 2)
            # print(new_connections)
            # assert 0
            if len(successor[i]) == K:
                for j in range(2):
                    # print(i, new_connections[j])
                    # print(graph.edges[i, new_connections[j]]['weights'])
                    # print(latency.edges[i, new_connections[j]]['weights'])
                    # graph.edges[i, new_connections[j]]['weights'] = latency.edges[i, new_connections[j]]['weights']
                    graph.add_edge(i, new_connections[j], weight=latency.edges[i, new_connections[j]]['weight'])
        # Step 5: Check and drop incoming connections if necessary
        for node in graph.nodes():
            if graph.in_degree(node) > M:
                excess_connections = graph.in_degree(node) - M
                # print( list(graph.in_edges(node)))
                # assert 0
                incoming_edges = [(i, node) for (i, j) in graph.in_edges(node) if node != random_ring[i]]
                edges_to_drop = random.sample(incoming_edges, excess_connections)
                graph.remove_edges_from(edges_to_drop)

        # Step 4: Check termination condition
        if_connected = nx.is_strongly_connected(graph)
        try:
            current_diameter = nx.diameter(graph, weight='weight')
        except:
            largest_cc = max(nx.strongly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            current_diameter = nx.diameter(subgraph, weight='weight')
        # if current_diameter == previous_diameter:
        #     print("Termination condition met: Diameter unchanged.")
        #     break
        print(f"itr {itr}: if_connected={if_connected}, D={current_diameter}")
        previous_diameter = current_diameter


if __name__ == '__main__':
    # Parameters and graph initialization
    N = 500  # Number of nodes
    K = 8   # Outgoing connections per node
    M = 20 # Maximum incoming connections per node
    graph, latency, random_ring = initialize_graph(N, K)

    # Compute scores
    update_graph(graph, latency, random_ring, N, K, M)