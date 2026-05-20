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

model_name = 'EAA'


if __name__ == "__main__":
    np.random.seed(500)
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

    pre = 2e5
    steps = 10
    dt = 5000
    with np.load(f"data/2.1_EAA_SIC_L_{L}.npz") as data:
        SIC_array = data['SIC_array']

    plt.figure(figsize=(6.6, 4.2))
    lineE = [0, 0, 0]
    lineC = [0, 0, 0]
    lineL = [0, 0, 0]
    for i in range(3):
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, 3))
        lineE[i], = plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[i, :, :] / np.log(2), 0), label=rf'$(V, \mu) = (1.0, {0.5 + 0.1 * i})$', linewidth=2.5, color=colors[i])
    for i in range(3):
        colors = plt.cm.Greens(np.linspace(0.25, 0.75, 3))
        lineC[i], = plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[15 + i, :, :] / np.log(2), 0), label=rf'$(V, \mu) = ({1.0 + 0.2 * i}, {2.0 - 0.1 * i})$', linestyle='--', linewidth=2.5, color=colors[i])
    for i in range(3):
        colors = plt.cm.Blues(np.linspace(0.25, 0.75, 3))
        lineL[i], = plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[30 + i, :, :] / np.log(2), 0), label=rf'$(V, \mu) = ({4.0 - 0.2 * i}, 0.5)$', linestyle=':', linewidth=2.5, color=colors[i])

    plt.legend([lineE[2], lineC[2], lineL[2]], ['Extended', 'Critical', 'Localized'], fontsize=14, frameon=False)
    plt.xlabel(r'$x$')
    plt.xlim([0, 610])
    plt.xticks([0, 300, 600])
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$')
    plt.ylim([0, 2.05])
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig("fig/2.1_EAA_SIC_all.pdf")
    plt.show()