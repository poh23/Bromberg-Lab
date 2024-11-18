import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from Numerical_Project.utils.heatmap_generator import FFT_2d_heatmap, XY_2d_heatmap

# setting path
sys.path.append('../../../')
from Numerical_Project.utils.FFT.FFT_2D.fourier_transform_2d import fourier_transform_2d
from Numerical_Project.utils.FFT.FFT_2D.inverse_fourier_transform_2d import inverse_fourier_transform_2d
from Numerical_Project.utils.closest import closest

matplotlib.use('TkAgg')

def ft_of_2d_gaussian(square_width=10, num_samples=500, sigma=0.6):
    mean_x = 1
    # The more sigma is smaller than the square width, the difference between the intensities is smaller
    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    X,Y = np.meshgrid(x, x)
    f = np.exp(-((X-mean_x) ** 2 + Y ** 2) / sigma ** 2) * np.cos(20 * np.pi * X) * np.cos(4 * np.pi * Y)

    # Compute the numerical Fourier transform
    VX, VY, FFT = fourier_transform_2d(f, x, x)

    # Create a single figure for all subplots
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # Subplot 1: Gaussian Aperture
    plt.subplot(2, 3, 1)
    F = np.abs(f) ** 2
    ax1 = XY_2d_heatmap(ax1, fig, X, Y, F, 'Gaussian Aperture')

    # Subplot 2: Numerical Fourier Transform
    numerical_intensity = np.abs(FFT) ** 2
    ax2 = FFT_2d_heatmap(ax2, fig, VX, VY, np.abs(FFT), 'Numerical FFT_2D Fourier Transform Intensity')

    # Subplot 3: Analytical Fourier Transform
    G = 0.25 * (sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY+2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY+2)**2)))
    ax3 = FFT_2d_heatmap(ax3, fig, VX, VY, np.abs(G)**2, 'Analytical FFT_2D Fourier Transform Intensity')

    # Subplot 4: Difference between Numerical and Analytical
    difference = np.abs(numerical_intensity - np.abs(G) ** 2)
    ax4 = FFT_2d_heatmap(ax4, fig, VX, VY, difference, 'Difference: Numerical vs. Analytical', cmap='hot')

    plt.subplot(2, 3, 5)
    ax5 = FFT_2d_heatmap(ax5, fig, VX, VY, np.imag(FFT), 'Imaginary part of Numerical FFT_2D', cmap='viridis')

    vx = VX[0, :]
    vy = closest(VY[:, 0], 10)
    ax6.plot(vx, np.imag(FFT[:, vy]), label='f_hat Imaginary part at vy=10')
    ax6.plot(vx, np.imag(G[:, vy]), linestyle='--', label='Analytic f_hat Imag part at vy=10')
    ax6.legend()

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()


# Call the function
ft_of_2d_gaussian()

def ift_of_2d_gaussian(square_width=20, num_samples=2**9, sigma=1):
    # The more sigma is smaller than the square width, the difference between the intensities is smaller
    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    X, Y = np.meshgrid(x, x)
    f = np.exp(-(X ** 2 + Y ** 2) / sigma ** 2) * np.cos(4 * np.pi * X) * np.cos(2 * np.pi * Y)

    # Compute the numerical Fourier transform
    VX, VY, FFT = fourier_transform_2d(f, x, x)

    vx = VX[0, :]
    vy = VY[:, 0]
    # Compute the numerical inverse Fourier transform
    IX, IY, IFFT = inverse_fourier_transform_2d(FFT, vx, vy)

    # Create a single figure for all subplots
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Subplot 1: Gaussian Aperture
    plt.subplot(1, 3, 1)
    F = np.abs(f) ** 2
    ax1 = XY_2d_heatmap(ax1, fig, X, Y, F, 'Gaussian Aperture')

    dx = x[1] - x[0]
    numerical_intensity_normalization = dx ** 4 * num_samples ** 2
    print(f'energy after FFT_1D - {(np.sum(np.abs(FFT)**2)/numerical_intensity_normalization)}')

    # Subplot 2: Numerical Inverse Fourier Transform on Fourier transform
    numerical_intensity = np.abs(IFFT) ** 2
    print(f'energy after inverse FFT_1D - {np.sum(numerical_intensity)}')
    print(f'energy after divided by energy before = {np.sum(np.abs(FFT)**2)/np.sum(F)}')
    ax2 = XY_2d_heatmap(ax2, fig, IX, IY, numerical_intensity, 'Numerical Inverse Fourier Transform in FFT_2D')

    # Subplot 4: Difference between Numerical and Analytical
    difference = np.abs(numerical_intensity - F)
    ax3 = XY_2d_heatmap(ax3, fig, IX, IY, difference, 'Difference: Original vs. Numerical IFT of FT', cmap='hot')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# Call the function
# ift_of_2d_gaussian()
