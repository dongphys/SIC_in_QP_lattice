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

model_name = 'EAA'


def fibonacci(n):
    if not(type(n) == int) or n < 0:
        return
    if n == 0 or n == 1:
        return 1
    else:
        f = fibonacci(n - 1) + fibonacci(n - 2)
        return f
    

def gen_H_Harper_real(L, mu, V, beta, phi=0):
    Ham = np.zeros((L, L), dtype=np.complex128)
    for i in range(L - 1):
        Ham[i, i + 1] = 1 + mu * np.cos(2 * np.pi * (i + 1 + 0.5) * beta + phi)
    Ham[L - 1, 0] = 1 + mu * np.cos(2 * np.pi * (L + 0.5) * beta + phi)
    Ham += Ham.conj().T
    for i in range(L):
        Ham[i, i] = V * np.cos(2 * np.pi * (i + 1) * beta + phi)
    return Ham


if __name__ == '__main__':
    L = fibonacci(14)
    mu = 2
    V = 1  # C
    beta = fibonacci(13) / fibonacci(14)
    phi = 0

    Ham = gen_H_Harper_real(L, mu, V, beta, phi)
    IDZs = np.zeros(L - 1)
    for i in range(L - 1):
        if np.abs(Ham[i, i + 1]) > 0.03:
            IDZs[i] = np.nan
    e, psi = np.linalg.eigh(Ham)
    psi = np.abs(psi) ** 2

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(range(L), psi[:, L // 2])
    ax1.set_xlim([0, 610])
    ax1.set_xlabel(r'site index $j$')
    ax1.set_ylim(ymin=0)
    ax1.set_yticks([0, 0.06])
    # ax1.set_ylabel(r'$|u_{m,j}|^2$')
    fig.text(0.07, 0.58, r'$|u_{m,j}|^2$', 
                fontsize=19, rotation='vertical', ha='center', va='center')
    # ax1.set_yticks([])
    ax2 = ax1.twinx()
    ax2.scatter(range(L - 1), IDZs, s=15, c='r')
    ax2.set_ylim([-0.001, 0.1])
    ax2.set_yticks([])
    fig.text(0.75, 0.8, r'$m=305$', 
                fontsize=20, ha='center', va='center')
    plt.tight_layout()
    plt.savefig('fig/3.1_phi_IDZs.pdf')
    plt.show()

    for i in range(L - 1):
        if IDZs[i] == 0:
            print(i)