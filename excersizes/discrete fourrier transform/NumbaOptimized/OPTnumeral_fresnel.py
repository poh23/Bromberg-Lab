import numpy as np
import matplotlib.pyplot as plt
import OPTfft_funcs
import numba

import matplotlib
matplotlib.use('TkAgg')

@numba.njit
def closest(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx

@numba.njit
def find_sigma(x, y):
    # Find index of y closest to 1/e^2 of max value
    max_val = np.max(y)
    y_closest_idx = closest(y, max_val / np.e**2)
    sigma = x[y_closest_idx]
    return sigma

def fresnel_approximation_1d(func, x, d, lamda=1, H0=1):
    energy_before_propagation = np.sum(np.abs(func) ** 2)
    nu, F = OPTfft_funcs.fourier_transform_1d(func, x)
    H0 *= np.exp(-1j * (2 * np.pi / lamda) * d)
    H = H0 * np.exp(1j * np.pi * lamda * d * nu ** 2)
    X, G = OPTfft_funcs.inverse_fourier_transform_1d(H * F, nu)
    energy_after_propagation = np.sum(np.abs(G) ** 2)
    return X, G

def fresnel_approximation_2d(func, x, y, d, lamda=1, H0=1):
    X1, Y1 = np.meshgrid(x, y)
    init_energy = np.sum(np.abs(func(X1, Y1)) ** 2)
    print(f'Initial energy - {init_energy}')

    VX, VY, F = OPTfft_funcs.fourier_transform_2d_continuous(func, x, y)
    dx = x[1] - x[0]
    numerical_intensity_normalization = dx ** 4 * len(F) ** 2
    print(f'Energy after FFT - {np.sum(np.abs(F) ** 2) / numerical_intensity_normalization}')

    H0 *= np.exp(-1j * (2 * np.pi / lamda) * d)
    H = H0 * np.exp(1j * np.pi * lamda * d * (VX ** 2 + VY ** 2))
    vx = VX[0, :]
    vy = VY[:, 0]
    X, Y, G = OPTfft_funcs.inverse_fourier_transform_2d(H * F, vx, vy)

    intensity_after_propagation = np.sum(np.abs(G) ** 2)
    print(f'Energy after propagation - {intensity_after_propagation}')
    return X, Y, G

@numba.njit
def analytic_fresnel_approx_of_1d_gaussian(x, d, sigma, lamda):
    w = 1 + (d**2 * lamda**2 / (4 * np.pi**2 * sigma**4))
    I = (1 / np.sqrt(w)) * np.exp(-x**2 / (sigma**2 * w))
    return I

@numba.njit
def analytic_fresnel_approx_of_2d_gaussian(X, Y, d, sigma, lamda):
    theta_0 = lamda / (np.pi * sigma)
    new_sigma_squared = sigma**2 + (theta_0 * d)**2
    I = ((sigma**2) / new_sigma_squared) * np.exp(-2 * (X**2 + Y**2) / new_sigma_squared)
    return I

def propagation_of_rect(square_width=15, num_samples=1000, rect_width=1):
    d = 10
    lamda = 0.5
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.where((np.abs(X) <= rect_width / 2) & (np.abs(Y) <= rect_width / 2), 1, 0)
    X, Y, G = fresnel_approximation_2d(func, x, x, d, lamda=lamda)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1))
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plane Wave Intensity')

    numerical_intensity = np.abs(G) ** 2
    plt.subplot(1, 2, 2)
    plt.imshow(numerical_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Numerical Fresnel Approximation at d = {d}')
    plt.tight_layout()
    plt.show()
