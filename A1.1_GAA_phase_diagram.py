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

model_name = 'GAA'


def cal_FD_of_all_state(h):
    L = len(h)
    result = np.zeros((L, 2))
    E, VR = np.linalg.eig(h)
    result[:, 0] = E
    IPR = np.sum(np.abs(VR) ** 4, axis=0)
    eta = - np.log(IPR) / np.log(L)
    result[:, 1] = eta
    return result


def vis_energy_spectrum(L, x_array, FD):
    data = np.load(f'data/A1.1_GAA_spectrum_L_{L}.npz')
    FD = data['FD']
    N = np.size(FD, 0)
    L = np.size(FD, 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i in range(L):
        p = plt.scatter(x_array, FD[:, i, 0], c=FD[:, i, 1], cmap=plt.cm.jet, vmin=0, vmax=1, s=0.5)
    cbar = plt.colorbar()
    cbar.set_ticks([0.0, 0.5, 1])
    fig.text(0.85, 0.96, r'$\eta$', 
             fontsize=20, ha='center', va='center')
    ax.axline((0, 4), (1, 0), color='k', linewidth=2)
    plt.xlim([0, 3])
    plt.xticks([0, 1, 2, 3])
    plt.xlabel(r"$\lambda / |t|$")
    plt.ylim([-12, 12])
    plt.yticks([-12, 0, 12])
    fig.text(0.065, 0.57, r'$E$', 
                fontsize=19, rotation='vertical', ha='center', va='center')
    plt.tight_layout()
    plt.savefig("fig/" + f"A1.1_GAA_spectrum_L_{L}.pdf")
    plt.savefig("fig/" + f"A1.1_GAA_spectrum_L_{L}.png", dpi=500)
    plt.show()


def fibonacci(n):
    if not(type(n) == int) or n < 0:
        return
    if n == 0 or n == 1:
        return 1
    else:
        f = fibonacci(n - 1) + fibonacci(n - 2)
        return f


def gen_H_GAA_real(L, t0, lbd, a, beta, phi=0):
    Ham = np.zeros((L, L), dtype=np.complex128)
    for i in range(L - 1):
        Ham[i + 1, i] = - t0
        Ham[i, i + 1] = - t0
    Ham[L - 1, 0] = - t0
    Ham[0, L - 1] = - t0
    for i in range(L):
        Ham[i, i] = 2 * lbd * np.cos(2 * np.pi * beta * (i + 1) + phi) / (1 - a * np.cos(2 * np.pi * beta * (i + 1) + phi))
    return Ham


if __name__ == '__main__':
    L = fibonacci(14)
    beta = fibonacci(13) / fibonacci(14)
    t0 = 1
    lbd_array = np.arange(0, 3 + 0.001, 0.05)
    a = 0.5
    phi = 0

    FD = np.zeros((len(lbd_array), L, 2))
    for i, lbd in enumerate(lbd_array):
        Ham = gen_H_GAA_real(L, t0, lbd, a, beta, phi)
        FD[i, :, :] = cal_FD_of_all_state(Ham)
        print(f'{i + 1} / {len(lbd_array)}')
    np.savez(f'data/A1.1_GAA_spectrum_L_{L}.npz', FD=FD)
    with np.load(f'data/A1.1_GAA_spectrum_L_{L}.npz') as data:
        FD = data['FD']
    vis_energy_spectrum(L, lbd_array, FD)