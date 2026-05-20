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

model_name = 'GAA'


def cal_SIC_of_x(L, t0, lbd, a, bias_array, beta, phi, pre, steps, dt):
    H_evo = gen_H_GAA_real(L, t0, lbd, a, beta, phi)
    SIC_array = np.zeros((len(bias_array), steps, L // 2 - 1))
    for i, bias in enumerate(bias_array):
        print(f"bias = {bias:d} ({i + 1} / {len(bias_array)})")
        H_ent = gen_H_entangle(L, bias)
        filled_indices = np.arange(0 , L // 2)
        if np.isin(L // 2 + bias, filled_indices) == False:
            filled_indices = np.append(filled_indices, [L])
        system = FreeFermionSystem(L + 1, filled=filled_indices)
        system.evol_sys(H_ent, np.pi / 4)
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

    file_name = f"A1.2_GAA_SIC_L_{L}_lbd_{lbd}_a_{a}_bias"
    np.savez("data/" + file_name + ".npz", SIC_array=SIC_array)

def vis_SIC(L, t0, lbd, a, bias_array):
    file_name = f"A1.2_GAA_SIC_L_{L}_lbd_{lbd}_a_{a}_bias"
    data = np.load("data/" + file_name + ".npz")
    SIC_array=data['SIC_array']
    plt.figure(figsize=(6.6, 4))
    for idx, bias in enumerate(bias_array):
        colors = plt.cm.Oranges(np.linspace(0.25, 1, len(bias_array)))
        plt.plot(np.arange(L // 2 - 1) * 2, np.mean(SIC_array[idx, :, :] / np.log(2), 0), linewidth=2, label=rf"$\Delta={bias:d}$", color=colors[idx])
    plt.legend(loc='lower right', fontsize=12, frameon=False)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mathrm{MI} (x, t \rightarrow \infty)$')
    plt.xlim([0, L])
    plt.xticks([0, 300, 600])
    plt.ylim([0, 2.05])
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig("fig/A1.2_GAA_SIC_ME.pdf")
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    L = fibonacci(14)
    beta = fibonacci(13) / fibonacci(14)
    t0 = 1
    lbd = 1
    a = 0.5
    phi = 0

    pre = 2e5
    steps = 10
    dt = 5000

    bias_array = np.arange(-20, 21, 10)

    start_time = time.time()
    cal_SIC_of_x(L, t0, lbd, a, bias_array, beta, phi, pre, steps, dt)
    vis_SIC(L, t0, lbd, a, bias_array)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")
