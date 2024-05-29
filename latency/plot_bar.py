import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sns
color = sns.color_palette(n_colors=5)
# Data for the two methods
q_learning_data = [6, 7, 7, 5, 8, 9, 9, 8, 7, 8]
nearest_neighbor_data = [9, 6, 6, 6, 9, 8, 8, 9, 8, 8]
values = [min(q_learning_data),max(q_learning_data),
             min(nearest_neighbor_data), max(nearest_neighbor_data)]
# Calculating the mean for each method
q_learning_mean = np.mean(q_learning_data)
nearest_neighbor_mean = np.mean(nearest_neighbor_data)

q_learning_data.append(q_learning_mean)
nearest_neighbor_data.append(nearest_neighbor_mean)
# Data points for each method
x = np.arange(len(q_learning_data))  # the label locations

# Creating the bar plot for each data point
plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, q_learning_data, width=0.4, label='Q-Learning', color=color[0])
plt.bar(x + 0.2, nearest_neighbor_data, width=0.4, label='Nearest Neighbor', color=color[1])

# Adding labels and title
plt.xlabel('Graph')
plt.ylabel('Diameter')
plt.title('Comparison of Q-Learning and Nearest Neighbor Methods')
plt.xticks(x, [f'Graph {i+1}' for i in range(len(q_learning_data) - 1)] + ['Avg'])
plt.ylim(min(values) - 1, max(values) + 1) 
plt.legend()


# Show the plot
plt.savefig('NN_vs_Qlearning.pdf', bbox_inches='tight')