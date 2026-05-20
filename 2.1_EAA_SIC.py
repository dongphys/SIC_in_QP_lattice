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


def cal_SIC_of_x(L, params, beta, phi, pre, steps, dt):
    H_ent = gen_H_entangle(L)
    filled_indices = np.arange(0 , L // 2)
    if np.isin(L // 2, filled_indices) == False:
        filled_indices = np.append(filled_indices, [L])

    SIC_array = np.zeros((len(params), steps, L // 2 - 1))
    for i, p in enumerate(params):
        V = p[0]
        mu = p[1]
        print(f"(V, \mu) = {p} ({i + 1} / {len(params)})")
        H_evo = gen_H_Harper_real(L, mu, V, beta, phi)
        system = FreeFermionSystem(L + 1, filled=filled_indices)
        system.evol_sys(H_ent, np.pi/4)
        system.evol_sys(H_evo, pre)
        random_array = np.random.rand(steps)
        for j in range(steps):
            for x in range(L // 2 - 1):
                E_list = np.arange(L // 2 - x, L // 2 + x + 1, 1)
                S_E = system.entropy(E_list)
                S_R = system.entropy([L])
                S_ER = system.entropy(np.append(E_list, L))
                SIC_array[i, j, x] = S_E + S_R - S_ER
            system.evol_sys(H_evo, dt * random_array[j])

    np.savez(f"data/2.1_EAA_SIC_L_{L}.npz", SIC_array=SIC_array)


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

    cal_SIC_of_x(L, params, beta, phi, pre, steps, dt)



