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

from module_SIC import *

model_name = 'EAA'


def gen_H_entangle(L, bias):
    H = np.zeros((L + 1, L + 1), dtype=np.complex128)

    H[bias, L] = 1
    H[L, bias] = 1
    return H


if __name__ == '__main__':
    L = fibonacci(14)
    mu = 2.0
    V = 1.0  # C
    beta = fibonacci(13) / fibonacci(14)
    phi = 0

    # # for fft 4096 1/20
    # steps = 4500
    # dt = 20
    
    # for fitting
    steps = 3000
    dt = 20

    bias = 187
    from scipy.optimize import curve_fit
    omega = 2.0 * np.pi * 0.00034
    with np.load(f'data/3.2_EAA_SIC_L_{L}_bias_{bias}.npz') as data:
        SIC_array = data['SIC_array']
    x_data = np.arange(0, 60000, 20)
    y_data = SIC_array[:3000]
    B0 = (np.max(y_data[250:3000]) - np.min(y_data[250:3000])) / 2
    C0 = np.mean(SIC_array[250:3000])
    t0 = 1
    def constrained_model(x, B, C, t1, t2):
        A = 2.0 - B - C
        return A * np.exp(-x / t1) + B * np.cos(omega * x) + C * np.exp(-x / t2)
    p0 = [B0, C0, t0, t0]
    bounds = ([-np.inf, -np.inf, 1e-12, 1e-12], [np.inf, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(constrained_model, x_data, y_data, p0=p0, bounds=bounds)
    B_fit, C_fit, t1_fit, t2_fit = popt
    A_fit = 2.0 - B_fit - C_fit
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = constrained_model(x_fit, B_fit, C_fit, t1_fit, t2_fit)

    with np.load(f'data/3.2_EAA_SIC_L_{L}_bias_{bias}.npz') as data:
        SIC_array = data['SIC_array']
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.plot(np.arange(steps//1) * dt, SIC_array[:steps//1], linewidth=2)
    plt.plot(x_fit, y_fit, 'r--', linewidth=3, alpha = 0.6)
    plt.xlim([0, steps * dt])
    plt.xlabel(r'$t$')
    plt.ylim([0, 2.1])
    plt.yticks([0, 1, 2])
    plt.ylabel(r'$\mathrm{MI} (L_B/2, t)$')
    from matplotlib.ticker import ScalarFormatter
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(x_formatter)
    plt.tight_layout()
    plt.savefig('fig/3.2.1_EAA_SIC_of_t.pdf')
    plt.show()

    from scipy.fft import fft, fftfreq
    fs = 1 / 20 # dt=20
    N = 4096
    avg = np.mean(SIC_array[250:(250 + N)])
    SIC_array -= avg
    yf = fft(SIC_array[250:(250 + N)])
    xf = fftfreq(N, 1 / fs)[:N // 2 // 20]

    fig, ax = plt.subplots(figsize=(5, 4))
    plt.plot(xf, 2.0/N * np.abs(yf[:N // 2 // 20]), linewidth=2)
    plt.xlim([0, xf[-1]])
    plt.xticks([0, 0.4e-3, 0.8e-3, 1.2e-3])
    plt.xlabel('Freq.')
    plt.ylim([0, 0.16])
    plt.yticks([0, 0.16])
    fig.text(0.15, 0.58, 'Amplitude', 
            fontsize=20, rotation='vertical', ha='center', va='center')
    from matplotlib.ticker import ScalarFormatter
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(x_formatter)
    plt.tight_layout()
    plt.savefig('fig/3.2.1_EAA_SIC_fft.pdf')
    plt.show()


    bias = 303
    omega = 2.0 * np.pi * 0.000076
    with np.load(f'data/3.2_EAA_SIC_L_{L}_bias_{bias}.npz') as data:
        SIC_array = data['SIC_array']
    x_data = np.arange(0, 60000, 20)
    y_data = SIC_array[:3000]
    B0 = (np.max(y_data[250:3000]) - np.min(y_data[250:3000])) / 2
    C0 = np.mean(SIC_array[250:3000])
    t0 = 1
    def constrained_model(x, B, C, t1, t2):
        A = 2.0 - B - C
        return A * np.exp(-x / t1) + B * np.cos(omega * x) + C * np.exp(-x / t2)
    p0 = [B0, C0, t0, t0]
    bounds = ([-np.inf, -np.inf, 1e-12, 1e-12], [np.inf, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(constrained_model, x_data, y_data, p0=p0, bounds=bounds)
    B_fit, C_fit, t1_fit, t2_fit = popt
    A_fit = 2.0 - B_fit - C_fit
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = constrained_model(x_fit, B_fit, C_fit, t1_fit, t2_fit)

    with np.load(f'data/3.2_EAA_SIC_L_{L}_bias_{bias}.npz') as data:
        SIC_array = data['SIC_array']
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.plot(np.arange(steps//1) * dt, SIC_array[:steps//1], color='tab:green', linewidth=2)
    plt.plot(x_fit, y_fit, 'r--', linewidth=3, alpha = 0.6)
    plt.xlim([0, steps * dt])
    plt.xlabel(r'$t$')
    plt.ylim([0, 2.1])
    plt.yticks([0, 1, 2])
    plt.ylabel(r'$\mathrm{MI} (L_C/2, t)$')
    from matplotlib.ticker import ScalarFormatter
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(x_formatter)
    plt.tight_layout()
    plt.savefig('fig/3.2.2_EAA_SIC_of_t.pdf')
    plt.show()

    from scipy.fft import fft, fftfreq
    fs = 1 / 20 # dt=20
    N = 4096
    avg = np.mean(SIC_array[250:(250 + N)])
    SIC_array -= avg
    yf = fft(SIC_array[250:(250 + N)])
    xf = fftfreq(N, 1 / fs)[:N // 2 // 20]

    fig, ax = plt.subplots(figsize=(5, 4))
    plt.plot(xf, 2.0/N * np.abs(yf[:N // 2 // 20]), color='tab:green', linewidth=2)
    plt.xlim([0, xf[-1]])
    plt.xticks([0, 0.4e-3, 0.8e-3, 1.2e-3])
    plt.xlabel(r'$f$')
    plt.ylim([0, 0.16])
    plt.yticks([0, 0.16])
    fig.text(0.15, 0.58, 'Amplitude', 
            fontsize=20, rotation='vertical', ha='center', va='center')
    from matplotlib.ticker import ScalarFormatter
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(x_formatter)
    plt.tight_layout()
    plt.savefig('fig/3.2.2_EAA_SIC_fft.pdf')
    plt.show()