import torch as th
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data_perigee = []

with open(f"N=5000_cluster_guassian.pkl", "rb") as f:
    data = pkl.load(f)

with open(f"N=1000_bitnode.pkl", "rb") as f:
    data_bitnode = pkl.load(f)
    for key in data_bitnode.keys():
        data[key].append(data_bitnode[key][0])

for i in range(1, 6, 1):
    with open(f"N={i}000_0_Gaussian_perigee.pkl", "rb") as f:
        data_perigee.append(th.as_tensor(pkl.load(f), dtype=th.float32).mean().item())
with open(f"N=1000_0_bitnode_perigee.pkl", "rb") as f:
    data_bitnode_perigee = pkl.load(f)
    data_perigee.append(th.as_tensor(data_bitnode_perigee, dtype=th.float32).mean().item())
print(data)
print(data_perigee)

data['Chord'] = data['chord_random_ring']
data['Nearest Neihbour'] = data['nearest_neighbour_random_ring']
data['RAPID'] = data['K_ring_random_ring']
data['Perigee'] = data_perigee
data['Ours'] = data['K_shortest_ring']

labels = ['Chord', 'Nearest Neihbour', 'RAPID', 'Perigee', 'Ours']
hatches = ["x", "/", ".", "\\",'']
tab10 = sns.color_palette("tab10", n_colors=5)
face_colors = tab10
x = np.arange(6)
x_tickslabel = ['1000', '2000', '3000', '4000', '5000', 'Bitnode\n1000']
width = 0.15

fig, ax = plt.subplots(figsize=(20, 10))
for i, label in enumerate(labels):
    ax.bar(x + i*width, data[label], width, label=label, color=face_colors[i])
ax.set_ylabel('Diameter (ms)')
ax.set_title('Diameter of different algorithms')
ax.set_xticks(x + 2.5*width)
ax.set_xticklabels(x_tickslabel)
ax.legend()
plt.tight_layout()
plt.savefig("diameter.png", bbox_inches='tight')
