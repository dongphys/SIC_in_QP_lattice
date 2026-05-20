import numpy as np
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
mpl_config = {
    'font.family': 'serif',
    'font.size': 20,
    'mathtext.fontset': 'cm',
}
plt.rcParams.update(mpl_config)
del mpl_config

from module_entropy import *

model_name = 'EAA'


def cal_Ssat(S_array):
    S_sat_array = np.zeros(np.size(S_array, 0), dtype=np.complex128)
    for i in range(np.size(S_array, 0)):
        S_sat_array[i] = np.mean(S_array[i, 1000:])
    return S_sat_array

def cal_ve(S_array, dt):
    ve_array = np.zeros(np.size(S_array, 0), dtype=np.complex128)
    for i in range(np.size(S_array, 0)):
        x = np.arange(np.size(S_array, 1)) * dt
        coefficients = np.polyfit(x, S_array[i, :], deg=1)
        ve_array[i] = coefficients[0]
    return ve_array

if __name__ == "__main__":
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

    text = 'Ssat'
    steps = 2000
    dt = 2000
    with np.load(f'data/1.2_EAA_EE_{text}_L_{L}.npz') as data:
        S_array = data['S_array']
    Ssat_array = cal_Ssat(S_array)

    text = 've'
    steps = 600
    dt = 20
    with np.load(f'data/1.2_EAA_EE_{text}_L_{L}.npz') as data:
        S_array = data['S_array']
    ve_array = cal_ve(S_array, dt)

    fig, ax1 = plt.subplots(figsize=(8.2, 3.8))
    ax1.plot(np.arange(len(params)), Ssat_array, marker='o', markersize=6, linewidth=2.7, color='tab:blue')
    ax1.set_xticks([0, 15, 30, 45], [r'$P_1$', r'$P_2$', r'$P_3$', r'$P_1$'])
    ax1.axvline(x=5, color='k', linewidth=3, linestyle=':')
    ax1.axvline(x=22.5, color='k', linewidth=3, linestyle=':')
    ax1.axvline(x=40, color='k', linewidth=3, linestyle=':')
    ax1.set_xlim([0, len(params)])
    ax1.set_ylabel(r'$S_\mathrm{sat}$', color='tab:blue')
    ax1.set_ylim([0, 125])
    ax1.set_yticks([0, 60, 120])
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(params)), ve_array, marker='^', markersize=7, linewidth=2.7, color='tab:green')
    ax2.set_ylabel(r'$v_e$', color='tab:green')
    ax2.set_ylim(ymin=0)
    from matplotlib.ticker import ScalarFormatter
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2, 2))
    ax2.yaxis.set_major_formatter(y_formatter)
    fig.text(0.94, 0.83, r'$\times 10^{-3}$', 
            color='tab:green', fontsize=22, ha='center', va='center')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    plt.tight_layout()
    plt.savefig(f'fig/1.3_EAA_ssat_and_ve_L_{L}.pdf')
    plt.show()