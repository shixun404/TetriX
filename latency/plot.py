import matplotlib.pyplot as plt
import numpy as np

# Parameters
n = 7
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
radius = 1.0

# Centers for the four diagrams (unicast + 3 gossip rounds)
centers = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])

# Pre-compute node positions for each sub-plot
positions = {c: np.column_stack((np.cos(angles), np.sin(angles))) * radius + np.array([c, 0.0]) for c in centers}
all_y = np.concatenate([pos[:, 1] for pos in positions.values()])
y_pad = 0.4  # Extra space above/below to avoid marker clipping
y_min, y_max = all_y.min() - y_pad, all_y.max() + y_pad

# Edge definitions
unicast_edges = [(0, j) for j in range(1, n)]

round1_edges = [(0, 1), (0, 3)]
round2_edges = round1_edges + [(1, 5), (3, 2)]
round3_edges = round2_edges + [(5, 6), (2, 4)]
round4_edges = round3_edges + [(4, 6)]

edges_per_round = {
    centers[1]: round1_edges,
    centers[2]: round2_edges,
    centers[3]: round3_edges,
    centers[4]: round4_edges
}

# Node labels showing the round they first received the message
labels_per_round = {
    centers[1]: {0: 'S', 1: '1', 3: '3'},
    centers[2]: {0: 'S', 1: '1', 3: '3', 5: '5', 2: '2'},
    centers[3]: {0: 'S', 1: '1', 3: '3', 5: '5', 2: '2', 6: '6', 4: '4'},
    centers[4]: {0: 'S', 1: '1', 3: '3', 5: '5', 2: '2', 6: '6', 4: '4'}
}

# Matplotlib global font adjustments
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold'
})

fig, ax = plt.subplots(figsize=(16, 5), facecolor='white')

# ---- Unicast diagram ----
c = centers[0]
pos = positions[c]
ax.scatter(pos[:, 0], pos[:, 1], s=350, zorder=3)
ax.scatter(pos[0, 0], pos[0, 1], s=450, facecolors='none', linewidths=3.0, zorder=4)

for u, v in unicast_edges:
    ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
            linewidth=3.5, solid_capstyle='round')

ax.text(pos[0, 0], pos[0, 1], 'S',
                va='center', ha='center', fontsize=14, fontweight='bold', color='white')

for i in range(1, n):
    ax.text(pos[i, 0], pos[i, 1], str(i), va='center', ha='center', fontsize=14, fontweight='bold', color='white')

ax.text((pos[0, 0] + pos[4, 0]) / 2, (pos[0, 1] + pos[4, 1]) / 2, 'Time Out',
                va='center', ha='center', fontsize=14, rotation=15, fontweight='bold', color='black')


ax.text(c, 2, 'Unicast\n(Degree N-1)', fontsize=16, fontweight='bold', ha='center')

# ---- Gossip diagrams ----
round_titles = {centers[1]: 'Gossip Round 1\nDegree 2',
                centers[2]: 'Gossip Round 2\nDegree 2',
                centers[3]: 'Gossip Round 3\nDegree 2',
                centers[4]: 'Gossip Round 4\nDegree 2',
                }

for c in centers[1:]:
    pos = positions[c]

    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], s=350, zorder=3)
    ax.scatter(pos[0, 0], pos[0, 1], s=450, facecolors='none', linewidths=3.0, zorder=4)

    # Draw edges for this round
    for u, v in edges_per_round[c]:
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                linewidth=2.2, linestyle='dashed', solid_capstyle='round')

    # Annotate nodes
    for idx, lab in labels_per_round[c].items():
        ax.text(pos[idx, 0], pos[idx, 1], lab,
                va='center', ha='center', fontsize=14, fontweight='bold', color='white')

    # Title
    ax.text(c, 2, round_titles[c], fontsize=16, fontweight='bold', ha='center')

# Add dashed vertical separator
separator_x = (centers[0] + centers[1]) / 2  # mid-way between Unicast and Gossip
ax.axvline(x=separator_x, linestyle='--', linewidth=2.0, color='grey')

# General styling
ax.set_aspect('equal')
ax.set_ylim(y_min, y_max)

ax.axis('off')





plt.tight_layout()
plt.savefig('figures/latency_diagram.png', dpi=300, bbox_inches='tight')
