import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
import seaborn as sns
sys.path.append(os.path.abspath('../scripts'))
# import utils
import torch as th
plt.rc('font', size=20, weight='bold')
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams["font.family"] = "Times New Roman"
plt.tight_layout()

def plot_heatmap(data, data_power, N=16, num_hop=4, filename=None):
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    colors = ["gainsboro", "red"] 
    cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

    fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    sns_plot = sns.heatmap(data.detach().cpu(), cmap=cmap, vmax=1, vmin=0,
        square=True, linewidths=0, cbar_kws={"shrink": .5}, ax=ax[0], xticklabels=N//4, yticklabels=N//4)
    sns_plot = sns.heatmap(data_power.detach().cpu(), cmap=cmap, vmax=1, vmin=0,
    	square=True, linewidths=0, cbar_kws={"shrink": .5}, ax=ax[1], xticklabels=N//4, yticklabels=N//4)
    for i in range(2):
        colorbar = ax[i].collections[0].colorbar
        colorbar.set_ticks([0.25,0.75])
        colorbar.set_ticklabels(['0', '1'])
    ax[0].set_title("A")
    ax[1].set_title(f"A^{num_hop}")
    plt.suptitle(filename.split('/')[-1].split('.')[0])
    fig.subplots_adjust(hspace=0.1, wspace = 0)
    if filename is not None:
        fig.savefig(filename,  bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    filename = 'figs/test.png'
    # print(filename.split('/')[-1].split('.')[0])
    # assert 0
    a = th.randn(100, 100)
    plot_heatmap(a, a, 100, 7, filename)