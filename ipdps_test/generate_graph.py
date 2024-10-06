import networkx as nx
import numpy as np
import os
import random
import pickle as pkl
from collections import Counter
def FABRIC(node_per_cluster=4):
    location_name = [
    "RUTG", "EDUKY", "MASS", "SRI", "PRIN", "TOKY", "CERN", 
    "DALL", "STAR", "UTAH", "MICH", "MAX", "AMST", "PSC", 
    "GATECH", "HAWI", "NCSA"
    ]
    locations = {}
    locations["RUTG"] = {"RUTG":0, "EDUKY":11.5, "MASS":9.3, "SRI":37, "PRIN":1.2, "TOKY":84.6, "CERN":46, 
        "DALL":18, "STAR":12, "UTAH":27.5, "MICH":14.4, "MAX":4.25, "AMST":45.09, "PSC":6.06, 
        "GATECH":10.08, "HAWI":57.9, "NCSA":22.3}
    locations["EDUKY"] = {"RUTG":0, "EDUKY":0, "MASS":15, "SRI":30.2, "PRIN":16.85, "TOKY":80, "CERN":51, "DALL":23.1, "STAR":4.75, "UTAH":20.1, "MICH":7.3, "MAX":9.38, "AMST":70.85, "PSC":11.3, 
        "GATECH":15.25, "HAWI":50.72, "NCSA":15.1}

    locations["MASS"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":42.5, "PRIN":9.5, "TOKY":88.6, "CERN":60.8, "DALL":21.3, "STAR":15.2, "UTAH":28.8, "MICH":15.93, "MAX":9.38, "AMST":60.7, "PSC":11.2, 
        "GATECH":15.3, "HAWI":61.2, "NCSA":23.8}
    locations["SRI"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":37.5, "TOKY":53.5, "CERN":82.4, "DALL":22.3, "STAR":25.5, "UTAH":10.2, "MICH":28, "MAX":37.3, "AMST":81.5, "PSC":40, 
        "GATECH":15.3, "HAWI":43, "NCSA":35.8}
    locations["PRIN"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":84.8, "CERN":46.1, "DALL":18.25, "STAR":12.1, "UTAH":27.5, "MICH":14.6, "MAX":4.3, "AMST":45.5, "PSC":6.26, 
        "GATECH":10.3, "HAWI":71.7, "NCSA":22.46}
    locations["TOKY"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":130.1, "DALL":67, "STAR":75, "UTAH":55.65, "MICH":77.8, "MAX":81, "AMST":128.7, "PSC":80.9, 
        "GATECH":85, "HAWI":74, "NCSA":83}
    locations["CERN"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":69.5, "STAR":57, "UTAH":72.3, "MICH":59.45, "MAX":44, "AMST":8.3, "PSC":57.5, 
        "GATECH":52.3, "HAWI":116.6, "NCSA":67}
    locations["DALL"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":17.5, "UTAH":24, "MICH":12.2, "MAX":15.5, "AMST":61.4, "PSC":17.4, 
        "GATECH":21.4, "HAWI":41.7, "NCSA":12.4}
    locations["STAR"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":15.45, "MICH":2.6, "MAX":8.3, "AMST":56, "PSC":10.1, 
        "GATECH":14.4, "HAWI":46, "NCSA":10.4}
    locations["UTAH"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":18, "MAX":24.46, "AMST":71.4, "PSC":25.6, 
        "GATECH":29.5, "HAWI":44.4, "NCSA":25.77}
    locations["MICH"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":10.8, "AMST":58.6, "PSC":12.7, 
        "GATECH":16.7, "HAWI":62.26, "NCSA":12.92}
    locations["MAX"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":0, "AMST":63.92, "PSC":4.26, 
        "GATECH":8.28, "HAWI":57.8, "NCSA":18.6}
    locations["AMST"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":0, "AMST":0, "PSC":50.3, 
        "GATECH":54.2, "HAWI":115.7, "NCSA":66.45}
    locations["PSC"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":0, "AMST":0, "PSC":0, 
        "GATECH":10.13, "HAWI":59.72, "NCSA":20.46}
    locations["GATECH"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":0, "AMST":0, "PSC":0, 
        "GATECH":0, "HAWI":80.14, "NCSA":24.5}
    locations["HAWI"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":0, "AMST":0, "PSC":0, 
        "GATECH":0, "HAWI":0, "NCSA":74}
    locations["NCSA"] = {"RUTG":0, "EDUKY":0, "MASS":0, "SRI":0, "PRIN":0, "TOKY":0, "CERN":0, "DALL":0, "STAR":0, "UTAH":0, "MICH":0, "MAX":0, "AMST":0, "PSC":0, 
        "GATECH":0, "HAWI":0, "NCSA":0}

    for i in range(len(location_name)):
        for j in range(i):
            locations[location_name[i]][location_name[j]] = locations[location_name[j]][location_name[i]]
    num_nodes = node_per_cluster * 17
    test_graph = nx.complete_graph(num_nodes)
    for i in range(num_nodes):
        src = location_name[i // node_per_cluster]
        for j in range(num_nodes):
            if i != j:
                dst = location_name[j // node_per_cluster]
                latency = np.random.normal(5, 1) + locations[src][dst]
                test_graph.edges[i,j]['weight'] = latency 

    return test_graph


if __name__ == "__main__":
    N_list = [10]
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    for i in range(50, 1001, 50):
        N_list.append(i)
    mode = ['uniform', 'gaussian', 'uniform_3', 'FABRIC']
    mode = mode[-1]

    for num_nodes in N_list:
        for i in range(1):
            graph_name = f'N={num_nodes}_{i}_{mode}.pkl'
            if mode == 'FABRIC':
                test_graph = FABRIC((num_nodes + 16) // 17)
                import networkx as nx
                import matplotlib.pyplot as plt

                # Example: Create a random graph with weighted edges
                # Extract the weights
                weights = [data['weight'] for u, v, data in test_graph.edges(data=True)]

                # Plot the histogram
                plt.figure(figsize=(8, 6))
                plt.hist(weights, bins=10, edgecolor='black')
                plt.title('Histogram of Edge Weight Distribution')
                plt.xlabel('Weight')
                plt.ylabel('Frequency')
                plt.savefig(f"./FABRIC/FABRIC_{num_nodes}.png")

            else:
                test_graph = nx.complete_graph(num_nodes)
                for (u, v) in test_graph.edges():
                    if mode == 'uniform':
                        test_graph.edges[u,v]['weight'] = random.randint(1, 10)
                    elif mode == 'uniform_3':
                        test_graph.edges[u,v]['weight'] = random.randint(4, 7)
                    else:
                        test_graph.edges[u,v]['weight'] = max(1, np.random.normal(5, 1))
                    # if u == v:
                    #     test_graph.edges[u,v]['weight'] = 1e6
            for (u, v) in test_graph.edges():
                test_graph.edges[v,u]['weight'] = test_graph.edges[u,v]['weight']  # Assign random positive weights
            with open(os.path.join('.', 'test_graph', graph_name), 'wb') as f:
                pkl.dump(test_graph, f)
            