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

model_name = 'EAA'


def gen_H_entangle(L, bias):
    H = np.zeros((L + 1, L + 1), dtype=np.complex128)

    H[bias, L] = 1
    H[L, bias] = 1
    return H


def cal_SIC_of_t(L, mu, V, beta, phi, x0, bias, steps, dt):
    H_ent = gen_H_entangle(L, bias)
    filled_indices = np.arange(L // 2)
    if np.isin(bias, filled_indices) == False:
        filled_indices = np.append(filled_indices, [L])
        
    SIC_array = np.zeros(steps)
    H_evo = gen_H_Harper_real(L, mu, V, beta, phi)
    system = FreeFermionSystem(L + 1, filled=filled_indices)
    system.evol_sys(H_ent, np.pi / 4)
    E_list = np.arange(bias - x0, bias + x0 + 1, 1)  #
    for j in range(steps):
        S_E = system.entropy(E_list)
        S_R = system.entropy([L])
        S_ER = system.entropy(np.append(E_list, L))
        SIC_array[j] = S_E + S_R - S_ER
        system.evol_sys(H_evo, dt)
    SIC_array /= np.log(2)

    np.savez(f'data/3.2_EAA_SIC_L_{L}_bias_{bias}.npz', SIC_array=SIC_array)


if __name__ == '__main__':
    L = fibonacci(14)
    mu = 2.0
    V = 1.0  # C
    beta = fibonacci(13) / fibonacci(14)
    phi = 0

    steps = 4500
    dt = 20

    x0 = 12
    bias = 187
    cal_SIC_of_t(L, mu, V, beta, phi, x0, bias, steps, dt)
    print('Finished #1')

    x0 = 46
    bias = 303
    cal_SIC_of_t(L, mu, V, beta, phi, x0, bias, steps, dt)
    print('Finished #2')



