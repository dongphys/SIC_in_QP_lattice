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

from module_entropy import *

model_name = 'EAA'


def cal_S_of_t(L, params, beta, steps, dt, text, phi=0):
    S_array = np.zeros((len(params), steps), dtype=np.complex128)
    sub_system = np.arange(L // 2)
    filled_indices = np.arange(L // 2)

    for i, p in enumerate(params):
        V = p[0]
        mu = p[1]
        print(f"(V, mu) = ({V:.2f}, {mu:.2f}) ({i + 1} / {len(params)})")
        H_evo = gen_H_Harper_real(L, mu, V, beta, phi)
        system = FreeFermionSystem(L, filled=filled_indices)
        for j in range(steps):
            S_array[i, j] = system.entropy(sub_system)
            system.evol_sys(H_evo, dt)

    np.savez(f"data/1.2_EAA_EE_{text}_L_{L}.npz", S_array=S_array)


if __name__ == '__main__':
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

    # # for preview
    # text = 'preview'
    # steps = 1000
    # dt = 200

    # for calculating Ssat
    text = 'Ssat'
    steps = 2000
    dt = 2000

    # # for calculating ve
    # text = 've'
    # steps = 600
    # dt = 20

    cal_S_of_t(L, params, beta, steps, dt, text, phi=0)