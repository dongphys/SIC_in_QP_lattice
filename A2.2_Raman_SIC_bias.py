import numpy as np
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
mpl_config = {
    'font.family': 'sans-serif',
    'font.size': 18,
    'mathtext.fontset': 'cm',
}
plt.rcParams.update(mpl_config)
del mpl_config

from module_SIC import *

model_name = "Raman"


def cal_SIC_of_x(L, t0, tso, lbd, bias_array, beta, phi, pre, steps, dt):
    H_evo = gen_H_Raman_real(L, tso, lbd, beta, t0, phi)
    SIC_array = np.zeros((len(bias_array), steps, 2 * L // 2 - 1))

    for i, bias in enumerate(bias_array):
        print(f"bias = {bias:d} ({i + 1} / {len(bias_array)})")
        H_ent = gen_H_entangle(2 * L, bias)
        filled_indices = np.arange(1, 2 * L // 2, 2)
        if np.isin(2 * L // 2 + bias, filled_indices) == False:
            filled_indices = np.append(filled_indices, [2 * L])
        system = FreeFermionSystem(2 * L + 1, filled=filled_indices)
        system.evol_sys(H_ent, np.pi / 4)
        system.evol_sys(H_evo, pre)
        random_array = np.random.rand(steps)
        for j in range(steps):
            for x in range(2 * L // 2 - 1):
                E_list = np.arange(2 * L // 2 - x + bias, 2 * L // 2 + x + 1 + bias, 1) #
                for k in range(len(E_list)):
                    if E_list[k] >= 2 * L:
                        E_list[k] -= 2 * L
                    elif E_list[k] < 0:
                        E_list[k] += 2 * L
                S_E = system.entropy(E_list)
                S_R = system.entropy([2 * L])
                S_ER = system.entropy(np.append(E_list, 2 * L))
                SIC_array[i, j, x] = S_E + S_R - S_ER
            system.evol_sys(H_evo, dt * random_array[j])

    file_name = f'A2.2_Raman_SIC_L_{L}_lbd_{lbd}_bias'
    np.savez("data//" + file_name + ".npz", SIC_array=SIC_array)

    
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
    start_time = time.time()
    cal_SIC_of_x(L, t0, tso, lbd, bias_array, beta, phi, pre, steps, dt)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")
    
    lbd = 2.0 # C
    start_time = time.time()
    cal_SIC_of_x(L, t0, tso, lbd, bias_array, beta, phi, pre, steps, dt)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")
