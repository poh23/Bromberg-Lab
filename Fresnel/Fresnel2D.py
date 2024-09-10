import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, fftfreq, ifft2
from Fourier2D import numerical_ft_2d as numfft2d
from Fourier2D import numerical_ifft_2d as inumfft2d
from Fourier2D import normalize_ft as normalizefft
from Fourier2D import gaussian_2d as gaussian_2d


def Fresnel(f, x, y, wavelength_tot, d):
    F, k_x, k_y = numfft2d(f, x, y)
    H = np.exp(1j * 4 * (1 / np.pi) * wavelength_tot * d * (np.power(k_x, 2) + np.power(k_y, 2)))
    FH = F * H
    g = inumfft2d(FH)
    return g


# Plot the results
def plot_expressions_2d(x, y, f, g, title):
    int_g = np.power(np.abs(g), 2)
    int_f = np.power(np.abs(f), 2)
    int_diff = int_g - int_f

    plt.figure(figsize=(16, 8))

    # Plot original function
    plt.subplot(2, 2, 1)
    original_contour = plt.contourf(x, y, int_f, cmap='viridis')
    plt.colorbar(original_contour)
    plt.title('OG Plane Wave at Z=0')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot propagated function
    plt.subplot(2, 2, 2)
    numerical_contour = plt.contourf(x, y, int_g, cmap='plasma')
    plt.colorbar(numerical_contour)
    plt.title(f'Numerically Propagated Plane Wave at Z={d}')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot difference between OG and Propagated
    plt.subplot(2, 2, 3)
    difference_contour = plt.contourf(x, y, int_diff, cmap='inferno')
    plt.colorbar(difference_contour)
    plt.title('Difference (Analytical - Numerical)')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Generate grid
x = np.linspace(-50, 50, 2000)
y = np.linspace(-50, 50, 2000)
X, Y = np.meshgrid(x, y)

# Specify Gaussian
gaussian_amplitude = 1
gaussian_mean = 0
gaussian_std = 1

f = gaussian_2d(X, Y, gaussian_amplitude, gaussian_mean * (2 * np.pi), gaussian_std * (2 * np.pi))

# Wave Characteristics
wavelength_tot = 1

# propagation distance
d = 0.1

g = Fresnel(f, x, y, wavelength_tot, d)
g_analytic = 1



plot_expressions_2d(x, y, f, g, "Fresnel Propagation")
