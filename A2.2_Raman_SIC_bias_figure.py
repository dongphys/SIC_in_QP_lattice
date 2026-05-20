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


def vis_SIC_of_E(L, lbd, bias_array):
    file_name = f'A2.2_Raman_SIC_L_{L}_lbd_{lbd}_bias'
    data = np.load("data/" + file_name + ".npz")
    SIC_array=data['SIC_array']
    fig = plt.figure(figsize=(2.8, 2))
    for idx, bias in enumerate(bias_array):
        colors = plt.cm.Reds(np.linspace(0.25, 1, len(bias_array)))
        plt.plot(np.arange(2 * L // 2 - 1), np.mean(SIC_array[idx, :, :] / np.log(2), 0), linewidth=2, label=rf"$\Delta={bias:d}$", color=colors[idx])
    plt.xlabel(r'', fontsize=16)
    fig.text(0.56, 0.18, r'$x$', 
             fontsize=15, ha='center', va='center')
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$', fontsize=16)
    plt.xlim([0, L])
    plt.xticks([0, 600], fontsize=15)
    plt.ylim([0, 2.05])
    plt.yticks([0, 2], fontsize=15)
    plt.tight_layout()
    plt.savefig("fig/A2.2_Raman_SIC_bias_E.pdf")
    plt.show()


def vis_SIC_of_C(L, lbd, bias_array):
    file_name = f'A2.2_Raman_SIC_L_{L}_lbd_{lbd}_bias'
    data = np.load("data/" + file_name + ".npz")
    SIC_array=data['SIC_array']
    plt.figure(figsize=(6.6, 4.2))
    for idx, bias in enumerate(bias_array):
        colors = plt.cm.Greens(np.linspace(0.25, 1, len(bias_array)))
        plt.plot(np.arange(2 * L // 2 - 1), np.mean(SIC_array[idx, :, :] / np.log(2), 0), linewidth=2, label=rf"$\Delta={bias // 2:d}$", color=colors[idx])
    plt.legend(loc='lower center', bbox_to_anchor=(0.45, 0), fontsize=12, frameon=False)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$')
    plt.xlim([0, L])
    plt.xticks([0, 300, 600])
    plt.ylim([0, 2.05])
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig("fig/A2.2_Raman_SIC_bias_C.pdf")
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    L = fibonacci(14)
    beta = fibonacci(13) / fibonacci(14)
    t0 = 1
    tso = 0.3
    phi = 0

    pre = 4e5
    steps = 10
    dt = 5000

    bias_array = np.arange(-20, 21, 10) * 2

    lbd =  0.3 # E
    vis_SIC_of_E(L, lbd, bias_array)

    lbd = 2.0 # C
    vis_SIC_of_C(L, lbd, bias_array)