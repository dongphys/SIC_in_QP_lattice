import numpy as np
from scipy.linalg import expm


class FreeFermionSystem:
    def __init__(self, L, filled):
        init_state = np.zeros((L, len(filled)), dtype=np.complex128)
        for i, idx in enumerate(filled):
            init_state[idx, i] = 1
        self.psi = init_state
        return

    def evol_sys(self, H_evo, t):
        self.psi = expm( - 1j * H_evo * t) @ self.psi
        
    def entropy(self, subsys):
        Cor = self.psi @ self.psi.conj().T
        Cor_subsys = Cor[subsys, :][:, subsys]
        xi, _ = np.linalg.eig(Cor_subsys)
        ee = np.nansum( - xi * np.log(xi) - (1 - xi) * np.log(1 - xi))
        return ee
    
    def entropy_density(self, subsys):
        Cor = self.psi @ self.psi.conj().T
        Cor_subsys = Cor[subsys, :][:, subsys]
        xi, _ = np.linalg.eig(Cor_subsys)
        ee = np.nansum( - xi * np.log(xi) - (1 - xi) * np.log(1 - xi)) / len(subsys)
        return ee


def fibonacci(n):
    if not(type(n) == int) or n < 0:
        return
    if n == 0 or n == 1:
        return 1
    else:
        f = fibonacci(n - 1) + fibonacci(n - 2)
        return f
# F14 = 610


def gen_H_Raman_real(L, tso, Mz, beta, t0=1, phi=0):
    Ham = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    
    for i in range(L - 1):
        Ham[2 * i + 2, 2 * i] = t0  # 0, 2, 4, ... down
        Ham[2 * i + 3, 2 * i + 1] = - t0  # 1, 3, 5, ... up
        Ham[2 * i + 1, 2 * i + 2] = tso
        Ham[2 * i + 3, 2 * i] = - tso

    Ham[0, 2 * (L - 1)] = t0  # PBC
    Ham[1, 2 * (L - 1) + 1] = - t0
    Ham[2 * (L - 1) + 1, 0] = tso
    Ham[1, 2 * (L - 1)] = - tso

    Ham += Ham.conj().T
    for i in range(L):
        Ham[2 * i, 2 * i] = - Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)
        Ham[2 * i + 1, 2 * i + 1] = Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)

    return Ham


def gen_H_Harper_real(L, mu, V, beta, phi=0):
    Ham = np.zeros((L, L), dtype=np.complex128)
    for i in range(L - 1):
        Ham[i, i + 1] = 1 + mu * np.cos(2 * np.pi * (i + 1 + 0.5) * beta + phi)
    Ham[L - 1, 0] = 1 + mu * np.cos(2 * np.pi * (L + 0.5) * beta + phi)
    Ham += Ham.conj().T
    for i in range(L):
        Ham[i, i] = V * np.cos(2 * np.pi * (i + 1) * beta + phi)
    return Ham