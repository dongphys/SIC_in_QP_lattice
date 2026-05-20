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

from module_SIC import *

model_name = 'Raman'


if __name__ == "__main__":
    np.random.seed(123)
    L = fibonacci(14)
    beta = fibonacci(13) / fibonacci(14)
    t0 = 1
    tso = 0.3
    lbd_array = np.array([0.7, 0.9, 1.1, 1.8, 2.0, 2.2, 2.9, 3.1, 3.3])
    phi = 0

    pre = 4e5
    steps = 10
    dt = 5000

    data = np.load(f"data/A2.1_Raman_SIC_L_{L}.npz")
    SIC_array=data['SIC_array']

    plt.figure(figsize=(6.6, 4.2))
    lineE = [0, 0, 0]
    lineC = [0, 0, 0]
    lineL = [0, 0, 0]
    for i in range(3):
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, 3))
        lineE[i], = plt.plot(np.arange(2 * L // 2 - 1), np.mean(SIC_array[i, :, :] / np.log(2), 0), label=rf'$V = {0.7 + 0.2 * i:.1f}$', linewidth=2.5, color=colors[i])
    for i in range(3):
        colors = plt.cm.Greens(np.linspace(0.25, 0.75, 3))
        lineC[i], = plt.plot(np.arange(2 * L // 2 - 1), np.mean(SIC_array[3 + i, :, :] / np.log(2), 0), label=rf'$V = {1.8 + 0.2 * i:.1f}$', linestyle='--', linewidth=2.5, color=colors[i])
    for i in range(3):
        colors = plt.cm.Blues(np.linspace(0.25, 0.75, 3))
        lineL[i], = plt.plot(np.arange(2 * L // 2 - 1), np.mean(SIC_array[6 + i, :, :] / np.log(2), 0), label=rf'$V = {2.9 + 0.2 * i:.1f}$', linestyle=':', linewidth=2.5, color=colors[i])

    # plt.legend([lineE[2], lineC[2], lineL[2]], ['Extended', 'Critical', 'Localized'], fontsize=14, frameon=False)
    plt.legend(fontsize=12, frameon=False)
    plt.xlabel(r'$x$')
    plt.xlim([0, L])
    plt.xticks([0, 300, 600])
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$')
    plt.ylim([0, 2.05])
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig(f"fig/A2.1_Raman_SIC_all_L_{L}.pdf")
    plt.show()