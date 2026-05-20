import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time

from module_SIC import *

model_name = "Raman"

    
def cal_SIC_of_x(L, t0, tso, lbd_array, beta, phi, pre, steps, dt):
    H_ent = gen_H_entangle(2 * L)
    filled_indices = np.arange(1 , 2 * L // 2, 2)
    if np.isin(2 * L // 2, filled_indices) == False:
        filled_indices = np.append(filled_indices, [2 * L])

    SIC_array = np.zeros((len(lbd_array), steps, 2 * L // 2 - 1))
    for i, lbd in enumerate(lbd_array):
        print(f"lbd = {lbd:.2f} ({i + 1} / {len(lbd_array)})")
        H_evo = gen_H_Raman_real(L, tso, lbd, beta, t0, phi)
        system = FreeFermionSystem(2 * L + 1, filled=filled_indices)
        system.evol_sys(H_ent, np.pi / 4)
        system.evol_sys(H_evo, pre)
        random_array = np.random.rand(steps)
        for j in range(steps):
            for x in range(2 * L // 2 - 1):
                E_list = np.arange(2 * L // 2 - x, 2 * L // 2 + x + 1, 1)
                S_E = system.entropy(E_list)
                S_R = system.entropy([2 * L])
                S_ER = system.entropy(np.append(E_list, 2 * L))
                SIC_array[i, j, x] = S_E + S_R - S_ER
            system.evol_sys(H_evo, dt * random_array[j])

    np.savez(f"data/A2.1_Raman_SIC_L_{L}.npz", SIC_array=SIC_array)


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

    start_time = time.time()
    cal_SIC_of_x(L, t0, tso, lbd_array, beta, phi, pre, steps, dt)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")
