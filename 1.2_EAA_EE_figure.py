import numpy as np
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
mpl_config = {
    'font.family': 'serif',
    'font.size': 18,
    'mathtext.fontset': 'cm',
}
plt.rcParams.update(mpl_config)
del mpl_config

from module_entropy import *

model_name = 'EAA'


if __name__ == '__main__':
    L = fibonacci(14)
    beta = fibonacci(13) / fibonacci(14)
    phi = 0
    params = [] # (V, mu)
    for i in range(15):
        params.append((1, 0.5 + 0.1 * i))
    for i in range(15):
        params.append((1 + 0.2 * i, 2 - 0.1 * i))
    for i in range(15):
        params.append((4 - 0.2 * i, 0.5))

    # for preview
    text = 'preview'
    steps = 1000
    dt = 200

    with np.load(f"data/1.2_EAA_EE_{text}_L_{L}.npz") as data:
        S_array = data["S_array"]
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    plt.plot(np.arange(steps) * dt, S_array[0, :], 'tab:red', linewidth=2, label=r"$P_1$") # A extended
    plt.plot(np.arange(steps) * dt, S_array[15, :], 'tab:green', linewidth=2, label=r"$P_2$") # B critical
    plt.plot(np.arange(steps) * dt, S_array[30, :], 'tab:blue', linewidth=2, label=r"$P_3$") # C localized
    plt.legend(fontsize=16, frameon=False)
    from matplotlib.ticker import ScalarFormatter
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(x_formatter)
    plt.xlabel(r"$t$", fontsize=19)
    plt.xlim([0, 2e5])

    # plt.ylabel(r"$S$")
    fig.text(0.085, 0.57, r'$S$', 
             fontsize=19, rotation='vertical', ha='center', va='center')
    
    plt.ylim(ymin=0)
    plt.yticks([0, 60, 120])
    plt.tight_layout()
    plt.savefig(f"fig/1.2_EAA_EE_preview_L_{L}.pdf")
    plt.show()

    

