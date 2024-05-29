import networkx as nx
import random
import itertools
import os
import pickle as pkl
import numpy as np
import torch as th
from tqdm import tqdm
from scipy import stats
import statistics
import matplotlib.pyplot as plt
def create_fully_connected_graph(n):
    """ Create a fully connected weighted graph with n nodes """
    G = nx.complete_graph(n)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(1, 10)  # Assign random positive weights
    for (u, v) in G.edges():
        G.edges[v,u]['weight'] = G.edges[u,v]['weight']  # Assign random positive weights
    return G

def find_regular_subgraph(G, k,  graph_name=None, iterations=10000):
    """ Attempt to find a k-regular subgraph with the minimum diameter from a fully connected graph """
    n = G.number_of_nodes()
    best_diameter = float('inf')
    diameter = float('inf')
    best_subgraph = None
    
    # pbar = tqdm(range(iterations),
    #         desc=f'Initial diameter: {diameter}',
    #         ncols=100,  # Width of the entire output
    #         colour='blue',  # Color of the progress bar
    #         miniters=1)  # Update the bar at least every iteration


    # for _ in pbar:
    diameter_list = []
    for _ in range(iterations):
        # Randomly select edges to attempt to form a 4-regular graph
        possible_subgraph = nx.random_regular_graph(k, n)
        for u, v in possible_subgraph.edges():
            possible_subgraph.edges[u,v]['weight'] = G.edges[u,v]['weight']
            possible_subgraph.edges[v,u]['weight'] = G.edges[v,u]['weight']
        # pbar.set_description(f"id: {_:6d}, diameter: {diameter:4f}, best_diameter: {best_diameter:4f}")
        # Check if the graph is 4-regular and calculate diameter
        try:
            diameter = nx.diameter(possible_subgraph, weight='weight')
            diameter_list.append(diameter)
            if diameter < best_diameter:
                best_diameter = diameter
                best_subgraph = possible_subgraph
        except nx.NetworkXError:
            # Graph is not connected, skip it
            continue
    histogram = [0 for i in range(0, 100)]
    for i in diameter_list:
        histogram[i] += 1
    # mean_value = np.mean(diameter_list)

    # # Calculating the median
    # median_value = np.median(diameter_list)

    # # Calculating the mode using scipy.stats (returns the smallest mode in case of multi-modal data)
    # mode_value = stats.mode(diameter_list)[0][0]  # stats.mode returns a ModeResult object

    # # Alternatively, calculating the mode using the statistics module (also handles multi-modal lists well)
    # mode_value_statistics = statistics.mode(diameter_list)  # This throws an error if there's no single mode

    # # Output results
    # print("Mean:", mean_value)
    # print("Median:", median_value)
    # print("Mode (scipy.stats):", mode_value)
    # print("Mode (statistics):", mode_value_statistics)
    return best_subgraph, best_diameter, histogram

def plot_histo(means=None, std_devs=None):
    # Indices for the bars
    indices = np.arange(len(means))

    # Creating the bar graph
    plt.figure(figsize=(10, 5))
    plt.bar(indices, means, color='skyblue', label='Mean Values')

    # Adding error bars for the standard deviation
    plt.errorbar(indices, means, yerr=std_devs, fmt='+', color='red', label='Standard Deviation')

    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Bar Graph with Standard Deviation')
    plt.legend()

    plt.savefig('histogram.png')

def plot_pdf(sample_counts, N, k):
    # Calculate total number of samples
    total_samples = np.sum(sample_counts)

    # Normalize the counts to get probabilities
    probabilities = sample_counts / total_samples

    # Create an array of indices (i.e., the values)
    values = np.arange(len(sample_counts))

    # Plotting the PDF
    plt.figure(figsize=(10, 5))
    plt.bar(values, probabilities, color='blue')  # Using a bar chart
    plt.xlabel('Diameter')
    plt.ylabel('Probability')
    plt.title('Probability Density Function (PDF)')
    plt.grid(True)
    plt.savefig(f'pdf_{N}_{k}.pdf', bbox_inches='tight')

def plot_cdf(sample_counts, N, k):
    total_samples = np.sum(sample_counts)
    probabilities = sample_counts / total_samples

    # Calculate the CDF using cumulative sum
    cdf = np.cumsum(probabilities)

    # Values for plotting (same as for the PDF)
    values = np.arange(len(probabilities))

    # Plotting the CDF
    plt.figure(figsize=(10, 5))
    plt.plot(values, cdf, drawstyle='steps-post', marker='o', color='red', label='Random 4-Regular Graph')
    plt.xlabel('Diameter')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'cdf_{N}_{k}.pdf', bbox_inches='tight')

# Main execution
if __name__ == '__main__':
    
    # plot_histo()
    
    N = 20
    k = 4
    histo_tensor = th.zeros(10, 100)
    # histo_tensor = np.loadtxt('tensor_data.txt')
    # histo_tensor = th.from_numpy(histo_tensor)
    for i in range(10):
        graph_name = f'N={N}_{i}.pkl'
        with open(os.path.join('.', 'test_dataset', graph_name), 'rb') as f:
            G = pkl.load(f)
        best_subgraph, best_diameter, histo = find_regular_subgraph(G, k, graph_name)
        histo_tensor[i] = th.as_tensor(histo)
        print(f"Best diameter found: {best_diameter}")  
    histo_avg = th.mean(histo_tensor, dim=0).reshape(-1)[0:30]
    histo_std = th.std(histo_tensor, dim=0).reshape(-1)[0:30]
    np.savetxt('tensor_data_{N}_{k}.txt', histo_tensor.numpy(), fmt='%.6f')
    plot_histo(histo_avg, histo_std)
    plot_pdf(histo_avg.numpy(), N, k)
    plot_cdf(histo_avg.numpy(), N, k)
        