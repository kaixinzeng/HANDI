import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from matplotlib import cm

def set_half_tick(ax, min_val, max_val,max_ticks=5):
    start = np.ceil(min_val / 0.5) * 0.5
    end = np.floor(max_val / 0.5) * 0.5
    
    if end > -start:
        end=-start
    else:
        start=-end

    if start > end:
        center = round((min_val + max_val) / 2 / 0.5) * 0.5
        return [center]

    full_ticks = np.arange(start, end + 0.01, 0.5)

    n = len(full_ticks)
    if n <= max_ticks:
        return full_ticks
    indices = np.linspace(0, n - 1, max_ticks).round().astype(int)
    ticks = full_ticks[indices]

    return ticks



dt=0.4
n=5
system ='cyc'
# eig_values = np.load(f'observation_duffing_{dt}_{n}/real/{dt}_eigvals.npy')
# data_points = np.load(f'observation_duffing_{dt}_{n}/real/{dt}.npy')
# X = np.load(f'observation_duffing_{dt}_{n}/{dt}_X.npy')
# Y = np.load(f'observation_duffing_{dt}_{n}/{dt}_Y.npy')

# eig_values = np.load(f'observation_vdp_{dt}_5/mode/{dt}_eigvals.npy')
# data_points = np.load(f'observation_vdp_{dt}_5/mode/{dt}.npy')
# X = np.load(f'observation_vdp_{dt}_5/{dt}_X.npy')
# Y = np.load(f'observation_vdp_{dt}_5/{dt}_Y.npy')

eig_values = np.load(f'observation_cyc_{dt}_5/mode/{dt}_eigvals.npy')
data_points = np.load(f'observation_cyc_{dt}_5/mode/{dt}.npy')
Y = np.load(f'observation_cyc_{dt}_5/{dt}_Y.npy')
X = np.load(f'observation_cyc_{dt}_5/{dt}_X.npy')

# eig_values = np.load(f'observation_fix_{dt}_5/mode/{dt}_eigvals.npy')
# data_points = np.load(f'observation_fix_{dt}_5/mode/{dt}.npy')
# X = np.load(f'observation_fix_{dt}_5/{dt}_X.npy')
# Y = np.load(f'observation_fix_{dt}_5/{dt}_Y.npy')

X = X.reshape(-1)  # shape: (num_points * T,)
Y = Y.reshape(-1)  # shape: (num_points * T,)
# N = 300
# Z1 = data_points[:, 0].reshape(N, N)
# Z2 = data_points[:, 1].reshape(N, N)
# Z3 = data_points[:, 2].reshape(N, N)
# Z4 = data_points[:, 3].reshape(N, N)
# Z5 = data_points[:, 4].reshape(N, N)
# Z6 = data_points[:, 5].reshape(N, N)
# Z7 = data_points[:, 6].reshape(N, N)
# Z8 = data_points[:, 7].reshape(N, N)
# Z9 = data_points[:, 8].reshape(N, N)
# Z10 = data_points[:,9].reshape(N, N)
Z1 = data_points[:, 0]
Z2 = data_points[:, 1]
Z3 = data_points[:, 2]
Z4 = data_points[:, 3]
Z5 = data_points[:, 4]
Z6 = data_points[:, 5]
Z7 = data_points[:, 6]
Z8 = data_points[:, 7]
Z9 = data_points[:, 8]
Z10 = data_points[:,9]
z = [Z1, Z2, Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10]

x_min, x_max = X.min(), X.max()
y_min, y_max = Y.min(), Y.max()

for i in range(11,12): # vdp 13, cyc 11, duffing 0, fix 1

    c = data_points[:, i]
    epsilon = 1e-3

    mask_near_zero = np.abs(c) < epsilon
    fig, ax = plt.subplots(figsize=(9,6))

    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16)/255 for i in range(0, lv, lv // 3))

    color_hex_blue = "#03699C"
    color_hex_red = "#cf171d"

    color_rgb_blue = hex_to_rgb(color_hex_blue)
    color_rgb_red = hex_to_rgb(color_hex_red)

    color_list = [color_rgb_blue,
                (1, 1, 1)]

    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=256)
    sc1 =  plt.scatter(X, Y, c=data_points[:,i], cmap=cmap, s=15)

    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sc1, cax=cax)
    
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.locator = LinearLocator(numticks=3)
    cbar.update_ticks()

    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=30)

    val = eig_values[i]
    title = f"Eigenvalue{i}_{val.real:.2f}{'+'if val.imag >= 0 else'-'}{abs(val.imag):.2f}j"

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)


    plt.tight_layout()
    plt.savefig(f'{system}_mode_{dt}_{i}_{title}.svg',format='svg', bbox_inches='tight', dpi=300)
    plt.show()
