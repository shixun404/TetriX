import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

def plot_heatmap(data, data_power, N=16, filename=None):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(ncols=2, figsize=(20, 20))
    sns_plot = sns.heatmap(data.detach().cpu(), cmap=cmap, vmax=1, vmin=0,
            square=True, linewidths=.001, cbar_kws={"shrink": .5}, ax=ax[0])
    sns_plot = sns.heatmap(data_power.detach().cpu(), cmap=cmap, vmax=1, vmin=0,
            square=True, linewidths=.001, cbar_kws={"shrink": .5}, ax=ax[1])
    ax[0].set_title("A")
    ax[1].set_title("A^10")
    if filename is not None:
        fig.savefig(filename)
