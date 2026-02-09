import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

system = 'fixpoint'
file_path = f'plot/nrmse_files/nrmse_sindy_5_{system}.xlsx'
df = pd.read_excel(file_path)
dt = df['dt']
methods = ["PSE","SINDy-SR3","SINDy", "HANDI"]
data = [df[method] for method in methods]
colors = ["#D8A0A7", "#C9A1CB", "#d06569", "#4FA8D5"]
fig, ax = plt.subplots(figsize=(6, 4))

for i, method in enumerate(methods):
    name = methods[i]
    color = colors[i]
    ax.plot(dt, data[i], label=name, linewidth=5, color=color, marker='*', markersize=35,
            markerfacecolor=color, markeredgecolor="#FFFAFA", linestyle='-')


y_max = max(np.max(data[0]), np.max(data[1]))
y_max_rounded = np.ceil(y_max / 0.04) * 0.04
ax.set_yticks(np.linspace(0, y_max_rounded, 3))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))  # fix cyc0.2、duffing vdp 0.1
ax.tick_params(axis='x', which='major', labelsize=20, length=8, width=1.5)
ax.tick_params(axis='y', which='major', labelsize=20, length=8, width=1.5)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)


for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(False)
    ax.spines[side].set_linewidth(1.5)


xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_min=0.05
y_min=-0.25   # fixed -0.25、 duffing -0.015 、vdp -0.03 、cyc -0.15
ax.set_xlim(left=x_min)  
ax.set_xlim(right=xlim[1]+0.02)
ax.set_ylim(bottom=y_min)
ax.set_ylim(top=ylim[1]+0.20)  # fixed 0.2、duffing 0.015 、vdp 0.02 、cyc 0.10

ax.plot([x_min, xlim[1]], [y_min, y_min], 'k-', linewidth=1.5) 
ax.plot([x_min, x_min], [y_min, ylim[1]], 'k-', linewidth=1.5)   
ax.annotate('', xy=(xlim[1]+0.02, y_min), xytext=(x_min, y_min),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=20,shrinkA=0,
                shrinkB=0),
            clip_on=False)
ax.annotate('', xy=(x_min, ylim[1]+0.20), xytext=(x_min, y_min),  # fixed 0.1、duffing 0.015 、vdp 0.01 、cyc 0.05
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=20,shrinkA=0,
                shrinkB=0),
            clip_on=False)



ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')

# ax.legend(fontsize=28, loc='upper left', frameon=False)
plt.tight_layout()

plt.savefig(f'plot/nrmse_{system}_new_0106.svg', format='svg', bbox_inches='tight', dpi=300)
plt.show()