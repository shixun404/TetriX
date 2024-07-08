import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sns
color = sns.color_palette(n_colors=5)
# Data for the two methods
# q_learning_data = [6, 7, 7, 5, 8, 9, 9, 8, 7, 8]
q_learning_data_without_exploration = [7, 7, 7, 7, 9, 8, 9, 8, 7, 6]
# q_learning_data = [6, 6, 6, 6, 8, 8, 8, 8, 7, 7]
q_learning_data = [6, 6, 6, 5, 8, 7, 7, 7, 7, 6]
# q_learning_data = [6, 6, 6, 5, 8, 7, 7, 7, 7, 6,
#                     6, 7, 6, 6, 6, 6, 7, 5, 7, 7,
#                       8, 6, 6, 7, 6, 8, 6, 7, 6, 6,
#                         7, 8, 7, 6, 6, 6, 7, 6, 7, 7,
#                           7, 8, 6, 5, 6, 6, 6, 8, 7, 5]


nearest_neighbor_data_without_exploration = [9, 6, 6, 6, 9, 8, 8, 9, 8, 8]
nearest_neighbor_data = [6, 6, 6, 5, 9, 7, 8, 7, 7, 6]

# ga = [7.0, 7.0, 6.0, 6.0, 9.0, 9.0, 8.0, 8.0, 7.0, 8.0]
ga = [6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 7.0, 8.0, 7.0, 7.0]
sa = [12, 11, 11, 12, 13, 13, 13, 13, 11, 12]
values = [min(q_learning_data),max(q_learning_data),
             min(nearest_neighbor_data), max(nearest_neighbor_data),
             min(ga), max(ga),
             min(sa), max(sa)]
# Calculating the mean for each method
q_learning_mean = np.mean(q_learning_data)
nearest_neighbor_mean = np.mean(nearest_neighbor_data)

q_learning_data.append(q_learning_mean)
nearest_neighbor_data.append(nearest_neighbor_mean)
q_learning_data_without_exploration.append(np.mean(q_learning_data_without_exploration))
nearest_neighbor_data_without_exploration.append(np.mean(nearest_neighbor_data_without_exploration))
ga.append(np.mean(ga))
sa.append(np.mean(sa))
print(q_learning_data)
print(nearest_neighbor_data)
print(ga)
print(sa)

# Data points for each method
# x = np.arange(len(q_learning_data))  # the label locations
x = np.array([i * 2 for i in range(len(q_learning_data))])
print(x.shape, len(q_learning_data))

# Creating the bar plot for each data point
plt.figure(figsize=(12, 6))
plt.bar(x - 0.6, sa, width=0.4, label='Simulated Annealing',edgecolor='black',  color=color[3])
plt.bar(x - 0.2, ga, width=0.4, label='Genetic Algorithm', edgecolor='black', color=color[2])


plt.bar(x + 0.2, nearest_neighbor_data_without_exploration, width=0.4, hatch='//', edgecolor=color[1], label='Nearest Neighbor with exploration', color="none")
plt.bar(x + 0.6, q_learning_data_without_exploration, width=0.4, hatch='//', edgecolor=color[0], label='Q-Learning with exploration', color="none")

plt.bar(x + 0.2, nearest_neighbor_data, width=0.4, label='Nearest Neighbor',edgecolor='black',  color=color[1])
plt.bar(x + 0.6, q_learning_data, width=0.4, label='Q-Learning',edgecolor='black',  color=color[0])


# Adding labels and title
plt.xlabel('Graph')
plt.ylabel('Diameter')
plt.title('Comparison of Graph Diameter Optimization Methods Across 10 Test Graphs')
plt.xticks(x, [f'Graph {i+1}' for i in range(len(q_learning_data) - 1)] + ['Avg'])
plt.ylim(min(values) - 1, max(values) + 1) 
plt.legend()


# Show the plot
plt.savefig('NN_vs_Qlearning.pdf', bbox_inches='tight')