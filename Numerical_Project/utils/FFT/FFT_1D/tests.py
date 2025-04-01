import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from fourier_transform_1d import fourier_transform_1d
from inverse_fourier_transform_1d import inverse_fourier_transform_1d

matplotlib.use('TkAgg')


# Test the 1D Fourier transform of a rect function
def ft_and_ift_of_rect():
    N = 2**10

    # Define the function f(x)
    x = np.linspace(-1, 1, N)
    f = np.where((x >= -0.5) & (x <= 0.5), 1, 0)
    plt.figure(1)
    plt.plot(x, f)

    k, g = fourier_transform_1d(f, x)
    plt.figure(2)
    plt.plot(k, np.real(g), label='Numerical')
    plt.plot(k, np.sinc(k), linestyle='--', label='Analytic')
    plt.xlim(-30, 30)
    plt.legend()

    y, h = inverse_fourier_transform_1d(g, k)

    plt.figure(3)
    plt.plot(y, np.real(h), label='Numerical transform')
    plt.plot(x, f, linestyle='--', label='Analytical')
    plt.legend()

    plt.show()

def ft_and_ift_of_gaussian():
    N = 2**11
    mean = 0
    sigma = 50e-6

    # Define the function f(x)
    square_width = 3e-3
    x = np.linspace(-50, 50, N, endpoint=False)
    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    f = np.exp(-(x - mean) ** 2 / (sigma ** 2))
    dx = x[1] - x[0]
    plt.subplot(2, 3, 1)
    plt.plot(x, f)

    nu, g = fourier_transform_1d(f, x)
    plt.subplot(2, 2, 2)
    plt.plot(nu, np.real(g), label='Numerical Real Part')
    analytic_g = sigma * np.sqrt(np.pi) * np.exp(-(nu ** 2) * (sigma ** 2) * np.pi ** 2) * np.exp(-1j * 2 * np.pi * nu * mean)
    plt.plot(nu, np.real(analytic_g), linestyle='--', label='Analytic')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(nu, np.imag(g), label='Numerical Imaginary Part')
    plt.plot(nu, np.imag(analytic_g), linestyle='--', label='Analytic Imaginary Part')
    plt.legend()

    y, h = inverse_fourier_transform_1d(g, nu)
    plt.subplot(2, 2, 4)
    plt.plot(y, np.real(h), label='Numerical Inverse real part')
    plt.plot(x, f, linestyle='--', label='Original Function')
    plt.legend()

    print(f'energy before 1d FFT_1D - {np.sum(np.abs(f)**2)}')
    print(f'energy after 1d FFT_1D - {1/(N*(dx**2)) * np.sum(np.abs(g) ** 2)}')
    print(f'energy after 1d IFFT - {np.sum(np.abs(h) ** 2)}')

    plt.show()