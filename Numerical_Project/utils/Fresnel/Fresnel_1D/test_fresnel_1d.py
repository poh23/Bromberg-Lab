import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from fresnel_approximation_1d import fresnel_approximation_1d

# Add the parent directory of Numerical_Project to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from Numerical_Project.utils.closest import closest
from Numerical_Project.utils.find_sigma import find_sigma


def analytic_fresnel_approx_of_1d_gaussian(x, d, sigma, lamda):
    w = 1 + (d**2 * lamda**2 / (4 * np.pi**2 * sigma**4))
    I = (1 / np.sqrt(w)) * np.exp(-x**2 / (sigma**2 * w))
    return I

def fresnel_approx_of_gaussian_1d(square_width=3e-3, num_samples=2 ** 12, sigma=50e-6):
    lamda = 532e-9
    rayleigh_length = np.pi * sigma ** 2 / lamda
    d = 1 * rayleigh_length

    print(f'Rayleigh length sigma- {np.sqrt(2) * sigma}')

    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = np.exp(-x ** 2 /(2 * sigma ** 2))

    X, G = fresnel_approximation_1d(func, x, d, lamda=lamda)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Gaussian Aperture
    plt.subplot(1, 2, 1)
    plt.plot(X, np.abs(func) ** 2, label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Intensity')
    plt.title('Gaussian Aperture')

    # Subplot 2: Numerical Fresnel Approximation
    numerical_intensity = np.abs(G) ** 2
    analytic_intensity = analytic_fresnel_approx_of_1d_gaussian(X, d, sigma, lamda)
    plt.subplot(1, 2, 2)
    plt.plot(X, numerical_intensity, label='Numeric Intensity')
    plt.plot(X, analytic_intensity, linestyle='--', label='Analytic Intensity')
    plt.xlabel('X')
    plt.ylabel('Intensity')
    plt.title(f'Numerical and Analytic Fresnel Approximation at d = {d}')
    print(f'energy of analytic propagation - {np.sum(analytic_intensity)}')

    x0 = closest(X, 0)
    sigma_of_numeric = find_sigma(X[x0:], numerical_intensity[x0:])
    print(f'sigma of numeric - {sigma_of_numeric}')
    sigma_of_analytic = find_sigma(X[x0:], analytic_intensity[x0:])
    print(f'sigma of analytic - {sigma_of_analytic}')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

fresnel_approx_of_gaussian_1d()