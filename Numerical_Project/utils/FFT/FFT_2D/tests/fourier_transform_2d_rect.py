import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from Numerical_Project.utils.heatmap_generator import FFT_2d_heatmap, XY_2d_heatmap

# setting path
sys.path.append('../../../')
from Numerical_Project.utils.FFT.FFT_2D.fourier_transform_2d import fourier_transform_2d
from Numerical_Project.utils.FFT.FFT_2D.inverse_fourier_transform_2d import inverse_fourier_transform_2d


matplotlib.use('TkAgg')


def analytical_fourier_transform_rect(VX, VY, width, height):
    # Compute the analytical solution of the FFT_2D Fourier Transform of a rectangle
    sinc_x = 0.5 * np.sinc(width * (VX-1)) + 0.5 * np.sinc(width * (VX+1))
    sinc_y = np.sinc(height * VY)
    F = width * height * sinc_x * sinc_y  # Adjusted based on the new definition

    return F


def plot_comparison(func, x, y, width=1, height=1):
    # Perform the FFT_2D Fourier transform
    X, Y = np.meshgrid(x, y)
    f = func(X, Y)
    VX, VY, G = fourier_transform_2d(f, x, y)

    # Compute the analytical Fourier transform
    analytical_intensity = np.abs(analytical_fourier_transform_rect(VX, VY, width, height))

    # Perform the inverse FFT_2D Fourier transform
    X, Y, reconstructed = inverse_fourier_transform_2d(G, VX[0, :], VY[:, 0])

    # Plotting the numerical Fourier transform intensity
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Numerical Solution Plot
    ax1 = FFT_2d_heatmap(ax1, fig, VX, VY, np.abs(G), 'Numerical FFT_2D Fourier Transform Intensity')

    # Analytical Solution Plot
    ax2 = FFT_2d_heatmap(ax2, fig, VX, VY, analytical_intensity, 'Analytical FFT_2D Fourier Transform Intensity')

    # Starting Function Plot
    ax3 = XY_2d_heatmap(ax3, fig, X, Y, func(X, Y), 'Original Function')

    # Reconstructed Function Plot
    ax4 = XY_2d_heatmap(ax4, fig, X, Y, np.real(reconstructed), 'Reconstructed Function from Inverse FFT_2D FFT_1D')

    plt.tight_layout()
    plt.show()

def fft2_of_rect_continuous():
    width = 1
    height = 1
    # Define the x and y ranges
    x = np.linspace(-5, 5, 2**8)
    y = np.linspace(-5, 5, 2**8)

    # Define the rectangular function correctly using X and Y grids
    func = lambda X, Y: np.where((np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2), 1, 0) * np.cos(2*np.pi*X)

    # Plot the Fourier transform, analytical solution, and reconstruction
    plot_comparison(func, x, y, width, height)

# Call the function to execute
fft2_of_rect_continuous()
