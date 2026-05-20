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


def cal_SIC_of_x(L, mu, V, bias_array, beta, phi, pre, steps, dt):
    H_evo = gen_H_Harper_real(L, mu, V, beta, phi)
    SIC_array = np.zeros((len(bias_array), steps, L // 2 - 1))
    for i, bias in enumerate(bias_array):
        print(f"bias = {bias:d} ({i + 1} / {len(bias_array)})")
        H_ent = gen_H_entangle(L, bias)
        filled_indices = np.arange(0 , L // 2)
        if np.isin(L // 2 + bias, filled_indices) == False:
            filled_indices = np.append(filled_indices, [L])
        system = FreeFermionSystem(L + 1, filled=filled_indices)
        system.evol_sys(H_ent, np.pi/4)
        system.evol_sys(H_evo, pre)
        random_array = np.random.rand(steps)
        for j in range(steps):
            for x in range(L // 2 - 1):
                E_list = np.arange(L // 2 - x + bias, L // 2 + x + 1 + bias, 1) #
                for k in range(len(E_list)):
                    if E_list[k] >= L:
                        E_list[k] -= L
                    elif E_list[k] < 0:
                        E_list[k] += L
                S_E = system.entropy(E_list)
                S_R = system.entropy([L])
                S_ER = system.entropy(np.append(E_list, L))
                SIC_array[i, j, x] = S_E + S_R - S_ER
            system.evol_sys(H_evo, dt * random_array[j])

    file_name = f"2.2_EAA_SIC_L_{L}_V_{V}_mu_{mu}_bias"
    np.savez("data//" + file_name + ".npz", SIC_array=SIC_array)


def vis_SIC_of_E(L, mu, V, bias_array):
    file_name = f"2.2_EAA_SIC_L_{L}_V_{V}_mu_{mu}_bias"
    data = np.load("data//" + file_name + ".npz")
    SIC_array=data['SIC_array']
    fig = plt.figure(figsize=(2.8, 2))
    for idx, bias in enumerate(bias_array):
        if idx % 2 == 0:
            colors = plt.cm.Reds(np.linspace(0.25, 1, len(bias_array)))
            plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[idx, :, :] / np.log(2), 0), linewidth=2, label=rf"$\Delta={bias:d}$", color=colors[idx])
    plt.xlabel(r'', fontsize=16)
    fig.text(0.56, 0.18, r'$x$', 
             fontsize=15, ha='center', va='center')
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$', fontsize=16)
    plt.xlim([0, 610])
    plt.xticks([0, 600], fontsize=15)
    plt.ylim([0, 2.05])
    plt.yticks([0, 2], fontsize=15)
    plt.tight_layout()
    plt.savefig("fig/2.2_EAA_SIC_bias_E.pdf")
    plt.show()


def vis_SIC_of_C(L, mu, V, bias_array):
    file_name = f"2.2_EAA_SIC_L_{L}_V_{V}_mu_{mu}_bias"
    data = np.load("data//" + file_name + ".npz")
    SIC_array=data['SIC_array']
    plt.figure(figsize=(6.6, 4.2))
    for idx, bias in enumerate(bias_array):
        if idx % 2 == 0:
            colors = plt.cm.Greens(np.linspace(0.25, 1, len(bias_array)))
            plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[idx, :, :] / np.log(2), 0), linewidth=2, label=rf"$\Delta={bias:d}$", color=colors[idx])
    plt.legend(loc='lower center', bbox_to_anchor=(0.45, 0), fontsize=12, frameon=False)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$')
    plt.xlim([0, 610])
    plt.xticks([0, 300, 600])
    plt.ylim([0, 2.05])
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig("fig/2.2_EAA_SIC_bias_C.pdf")
    plt.show()


def vis_SIC_of_L(L, mu, V, bias_array):
    file_name = f"2.2_EAA_SIC_L_{L}_V_{V}_mu_{mu}_bias"
    data = np.load("data//" + file_name + ".npz")
    SIC_array=data['SIC_array']
    plt.figure(figsize=(3.5, 3.5))
    for idx, bias in enumerate(bias_array):
        if idx % 2 == 0:
            colors = plt.cm.Blues(np.linspace(0.25, 1, len(bias_array)))
            plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[idx, :, :] / np.log(2), 0), linewidth=2, label=rf"$\Delta={bias:d}$", color=colors[idx])
    plt.xlabel(r'$x=|A|$')
    plt.ylabel(r'$MI$')
    plt.xlim([0, 610])
    plt.xticks([0, 300, 600])
    plt.ylim([0, 2.05])
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    # plt.savefig("fig/2.2_EAA_SIC_bias_L.pdf")
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    state_name = "bipartite_state"
    L = fibonacci(14)
    beta = fibonacci(13) / fibonacci(14)
    phi = 0
    pre = 2e5
    steps = 10
    dt = 5000

    bias_array = np.arange(-20, 21, 5)

    start_time = time.time()

    mu = 0.5
    V = 1.0 # E
    cal_SIC_of_x(L, mu, V, bias_array, beta, phi, pre, steps, dt)
    vis_SIC_of_E(L, mu, V, bias_array)

    mu = 2.0
    V = 1.0 # C
    cal_SIC_of_x(L, mu, V, bias_array, beta, phi, pre, steps, dt)
    vis_SIC_of_C(L, mu, V, bias_array)

    # mu = 0.5
    # V = 4.0 # L
    # cal_SIC_of_x(L, mu, V, bias_array, beta, phi, pre, steps, dt)
    # vis_SIC_of_L(L, mu, V, bias_array)

    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")
