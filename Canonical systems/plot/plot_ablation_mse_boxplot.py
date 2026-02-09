import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import random
import math
from sklearn.preprocessing import MinMaxScaler
import warnings
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.stats import mannwhitneyu
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.transforms import blended_transform_factory

# data=pd.read_csv('ablation_study/mse_results/fixed/fixed_ablation_mse_all_dt0.5_xyrange1_grid10.csv')
# data=pd.read_csv('ablation_study/mse_results/cyc/cyc_ablation_mse_all_dt0.5_xyrange1.5_grid10.csv')
# data=pd.read_csv('ablation_study/mse_results/vdp/vdp_ablation_mse_all_dt0.4_xyrange3_grid10.csv')
data=pd.read_csv('ablation_study/mse_results/duffing/duffing_ablation_mse_all_dt0.3_xyrange4_grid10.csv')

full = data['HANDI'].values  # ★
A = data['ablationA'].values            # ①
O = data['ortho'].values            # ②
size = 15

arial_font = FontProperties(fname="ablation_study/fonts/arial.ttf", size=size)
symbol_font = FontProperties(family='DejaVu Sans', size=size) 
rcParams['font.family'] = arial_font.get_name()

data_dict = {
    '★': full,
    '①':A,   
    '②': O
}
df = pd.DataFrame(data_dict)
df_long = df.melt(var_name='Method', value_name='Error')

fill_colors = ['#F1C77E', '#A4D2F0', '#89D3B2']
edge_colors = ['#B07C00', '#2C6DA1', '#00694C']

fig, ax = plt.subplots(figsize=(6, 4))

positions = [0.1, 0.6, 1.1]
# positions = [0.1, 0.4, 0.7]

bp = ax.boxplot(
    list(data_dict.values()),
    positions=positions,
    widths=0.2,
    patch_artist=True,
    showfliers=False
)

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(fill_colors[i])
    patch.set_edgecolor(edge_colors[i])
    patch.set_linewidth(1.5)

for median in bp['medians']:
    i = bp['medians'].index(median)
    median.set_color(edge_colors[i])
    median.set_linewidth(1.5)

for whisker in bp['whiskers']:
    i = bp['whiskers'].index(whisker) // 2
    whisker.set_color(edge_colors[i])
    whisker.set_linewidth(1.5)

for cap in bp['caps']:
    i = bp['caps'].index(cap) // 2
    cap.set_color(edge_colors[i])
    cap.set_linewidth(1.5)


trans = blended_transform_factory(ax.transData, ax.transAxes)
xtick_labels = list(data_dict.keys())
for i, label in enumerate(xtick_labels):
    ax.text(
        positions[i], -0.09,
        label,
        transform=trans,
        ha='center', va='center',
        fontsize=30,
        fontproperties=symbol_font,
        color='black' if i == 0 else 'black',
        clip_on=False
    )

# ax.set_ylim(1e-8, 0.025) #fixed
# ax.set_ylim(3*1e-5, 0.03)  # cyc
# ax.set_ylim(5*1e-3, 15)  #vdp
ax.set_ylim(5*1e-7, 0.5)  #duffing

ax.set_yscale('log')
ax.set_xticks([])

trans_y = ax.get_yaxis_transform()
# yticks = [1e-7,1e-5,1e-3]  #fixed
# yticks = [1e-4,1e-3,1e-2]  #cyc
# yticks = [1e-2,1e-1,1e-0]  #vdp
yticks = [1e-6,1e-4,1e-2]  #duffing
ax.set_yticks(yticks)

ax.tick_params(axis='y', which='minor', length=0)
ax.tick_params(axis='y', which='major', labelsize=22, length=4, width=1.5)
# ax.tick_params(axis='x', which='major', labelsize=14, length=8, width=1.5)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
description = [
    ("★", "Full\nframework"),
    ("①", "Stage A\nablated"),
    ("②", "orthogonal\nablated")
]

# plt.subplots_adjust(left=0.3, bottom=0.2, right=0.8, top=0.95)
sns.despine(top=True, right=True)
plt.savefig("ablation_study/mse_results/duffing_boxplot_0118.svg", format='svg', bbox_inches='tight')
# plt.show()
