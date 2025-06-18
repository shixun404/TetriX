import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sns
color = sns.color_palette(n_colors=5)
q_learning_data_without_exploration = [7, 7, 7, 7, 9, 8, 9, 8, 7, 6]
q_learning_data = [6, 6, 6, 5, 8, 7, 7, 7, 7, 6]

nearest_neighbor_data_without_exploration = [9, 6, 6, 6, 9, 8, 8, 9, 8, 8]
nearest_neighbor_data = [6, 6, 6, 5, 9, 7, 8, 7, 7, 6]
ga = [6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 7.0, 8.0, 7.0, 7.0]
sa = [12, 11, 11, 12, 13, 13, 13, 13, 11, 12]

t_q_without_exploration = [0.003, 0.005, 1000]
t_q = [0.08, 1.5, 4000]
t_ga = [113, 2160, 2e5]
t_nn = [0.07, 1.3, 250]
t_nn_without_exploration = [0.008, 0.013, 0.05]
# Calculating the mean for each method
q_learning_data = [np.mean(q_learning_data)]
nearest_neighbor_data = [np.mean(nearest_neighbor_data)]

# q_learning_data = q_learning_mean
# nearest_neighbor_data = nearest_neighbor_mean
q_learning_data_without_exploration = [np.mean(q_learning_data_without_exploration)]
nearest_neighbor_data_without_exploration = [np.mean(nearest_neighbor_data_without_exploration)]
ga = [np.mean(ga)]
sa = [np.mean(sa)]

q_learning_data_without_exploration = np.append(q_learning_data_without_exploration, [np.mean([22, 24, 24, 24, 23])])

q_learning_data = np.append(q_learning_data, np.mean([20, 21, 21, 20, 20]))

nearest_neighbor_data_without_exploration = np.append(nearest_neighbor_data_without_exploration, [np.mean([26, 24, 24, 25, 25])])
nearest_neighbor_data_without_exploration = np.append(nearest_neighbor_data_without_exploration, [8])
nearest_neighbor_data = np.append(nearest_neighbor_data, [np.mean([24, 24, 24, 24, 24])])
nearest_neighbor_data = np.append(nearest_neighbor_data, [7])

ga = np.append(ga, [np.mean([21, 21, 21, 21, 21])])
ga = np.append(ga, [20])
ga_without = [15, 24, 21]
print(q_learning_data)

# assert 0
print(nearest_neighbor_data)

x = np.array([i * 2 for i in range(ga.size)])
print(x.shape, q_learning_data.size)

# Creating the bar plot for each data point
plt.figure(figsize=(12, 8))
fig, ax = plt.subplots()
# plt.bar(x - 0.6, sa, width=0.4, label='Simulated Annealing',edgecolor='black',  color=color[3])

values = [min(q_learning_data),max(q_learning_data),
             min(nearest_neighbor_data), max(nearest_neighbor_data),
             min(ga), max(ga),
             min(sa), max(sa)]
bars = []

b = plt.bar(x - 0.6, ga_without, width=0.4, label='Random', hatch='//', edgecolor=color[2], color="none")
bars.append(b)
b = plt.bar(x - 0.6, ga, width=0.4, label='Genetic Algorithm w/ exploration over 1e5 graphs', edgecolor='black', color=color[2])
bars.append(b)
b = plt.bar(x - 0.2, nearest_neighbor_data_without_exploration, width=0.4, hatch='//', edgecolor=color[1], label='d-Hamilton + greedy w/o exploration', color="none")
bars.append(b)
b = plt.bar(x + 0.2, q_learning_data_without_exploration, width=0.4, hatch='//', edgecolor=color[0], label='d-Hamilton + Q-Learning w/o exploration', color="none")
bars.append(b)
b = plt.bar(x - 0.2, nearest_neighbor_data, width=0.4, label='d-Hamilton + greedy w/ exploration',edgecolor='black',  color=color[1])
bars.append(b)
b = plt.bar(x + 0.2, q_learning_data, width=0.4, label='d-Hamilton + Q-Learning w/ exploration',edgecolor='black',  color=color[0])
bars.append(b)

for bar in bars:
    for b in bar:
        yval = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

# Adding labels and title
plt.xlabel('Graph')
plt.ylabel('Diameter')
plt.title('Comparison of Graph Diameter Optimization Methods')

plt.ylim(min(values) - 1, max(values) + 4) 
plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 0.5))


x = np.array([i * 2 for i in range(q_learning_data.size + 1)])
ax2 = ax.twinx()
# ax2.plot(x, t_q_without_exploration, label='Q-Learning w/o exploration', color='k', linestyle='--', marker='*')
ax2.plot(x, t_ga, color='k', label='Genetic Algorithm',  linestyle='-.', marker='o')
ax2.plot(x, t_q, color='k', label='Q-Learning', marker='+')
ax2.plot(x, t_nn, color='k', label='d-Hamilton Cycle w/ greedy',linestyle='--', marker='*')


ax2.set_yscale('log')

# Labels for the second y-axis
ax2.set_ylabel('Execution Time (seconds)')
ax2.set_xticks(x, ['N=20\n K=4\n w={1,2, ..., 10}', 'N=100\n K=6\n  w={4, 5, 6}', 
                #    'N=500\n K=8\n  w={1,2, ..., 10}'
                ])
ax2.legend(loc='upper right', bbox_to_anchor=(-0.2, 0))
# Show the plot
plt.savefig('DifferentN&time.pdf', bbox_inches='tight')