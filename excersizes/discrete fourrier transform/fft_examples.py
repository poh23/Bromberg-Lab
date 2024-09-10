import fft_funcs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def ft_and_ift_of_rect():
    N = 2048

    # Define the function f(x)
    f = lambda x: np.where((x >= -0.5) & (x <= 0.5), 1, 0)
    x = np.linspace(-1, 1, N)
    plt.figure(1)
    plt.plot(x, f(x))

    k, g = fft_funcs.fourier_transform_1d(f, x)
    plt.figure(2)
    plt.plot(k, np.real(g), label='Numerical')
    plt.plot(k, np.sinc(k), linestyle='--', label='Analytic')
    plt.xlim(-30, 30)
    plt.legend()

    y, h = fft_funcs.inverse_fourier_transform_1d(g, k)

    plt.figure(3)
    plt.plot(y, np.real(h), label='Numerical transform')
    plt.plot(x, f(x), linestyle='--', label='Analytical')
    plt.legend()

    plt.show()


def ft_of_gaussian():
    N = 20000
    mean = 0
    sigma = 2

    # Define the function f(x)
    f = lambda x: np.exp(-(x-mean)**2/(sigma**2))*np.cos(2*np.pi*5*x)
    x = np.linspace(-100, 100, N, endpoint=False)
    plt.subplot(1, 2, 1)
    plt.plot(x, f(x))

    k, g = fft_funcs.fourier_transform_1d(f, x)
    plt.subplot(1, 2, 2)
    plt.plot(k, np.real(g), label='Numerical')
    num_g = 0.5 * sigma*np.sqrt(np.pi) * np.exp(-((k-5)**2)*(sigma**2)*np.pi**2)*np.exp(-1j*k*mean) + 0.5 * sigma*np.sqrt(np.pi) * np.exp(-((k+5)**2)*(sigma**2)*np.pi**2)*np.exp(-1j*k*mean)
    plt.plot(k, np.real(num_g), linestyle='--', label='Analytic')
    # plt.xlim(-10, 10)
    plt.legend()
    plt.show()

def ft_of_delta():
    N = 5000

    # Define the function f(x)
    f = lambda x: np.where((x >= -0.001) & (x <= 0.001), 1000, 0)
    x = np.linspace(-1, 1, N)
    plt.figure(1)
    plt.plot(x, f(x))

    k, g = fft_funcs.fourier_transform_1d(f, x)
    plt.figure(2)
    plt.plot(k, np.imag(g), label='Numerical')
    num_g = np.exp(-1j*k)
    plt.plot(k, np.imag(num_g), linestyle='--', label='Analytic')
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()


## 2D

## 2D rect
def analytical_fourier_transform_rect(VX, VY, width, height):
    """
    Calculate the analytical 2D Fourier transform of a 2D rectangular function.

    Parameters:
    - VX: 2D array of frequency components in the x direction.
    - VY: 2D array of frequency components in the y direction.
    - width: Width of the rectangular function.
    - height: Height of the rectangular function.

    Returns:
    - F: 2D array of the analytical Fourier transform values.
    """
    # Compute the analytical solution of the 2D Fourier Transform of a rectangle
    sinc_x = 0.5 * np.sinc(width * (VX-1)) + 0.5 * np.sinc(width * (VX+1))
    sinc_y = np.sinc(height * VY)
    F = width * height * sinc_x * sinc_y  # Adjusted based on the new definition

    return F


def plot_comparison(func, x, y, width=1, height=1):
    """
    Plots the intensity of the 2D Fourier transform of a given function,
    its analytical solution, and the reconstructed function using the inverse transform.

    Parameters:
    - func: A callable function representing the 2D function to be transformed.
    - x: 1D array of x-coordinates.
    - y: 1D array of y-coordinates.
    - width: Width of the rectangle (for analytical solution).
    - height: Height of the rectangle (for analytical solution).
    """
    # Perform the 2D Fourier transform
    VX, VY, G = fft_funcs.fourier_transform_2d_continuous(func, x, y)

    # Compute the analytical Fourier transform
    analytical_intensity = np.abs(analytical_fourier_transform_rect(VX, VY, width, height))

    # Perform the inverse 2D Fourier transform
    X, Y, reconstructed = fft_funcs.inverse_fourier_transform_2d(G, VX[:, 0], VY[0, :])

    # Plotting the numerical Fourier transform intensity
    plt.figure(figsize=(18, 6))

    # Numerical Solution Plot
    plt.subplot(1, 4, 1)
    plt.imshow(np.abs(G), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('Frequency $\\nu_x$')
    plt.ylabel('Frequency $\\nu_y$')
    plt.title('Numerical 2D Fourier Transform Intensity')

    # Analytical Solution Plot
    plt.subplot(1, 4, 2)
    plt.imshow(analytical_intensity, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('Frequency $\\nu_x$')
    plt.ylabel('Frequency $\\nu_y$')
    plt.title('Analytical 2D Fourier Transform Intensity')

    # Starting Function Plot
    plt.subplot(1, 4, 3)
    plt.imshow(func(X,Y), extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Amplitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Function')

    # Reconstructed Function Plot
    plt.subplot(1, 4, 4)
    plt.imshow(np.real(reconstructed), extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Amplitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Reconstructed Function from Inverse 2D FFT')

    plt.tight_layout()
    plt.show()

def fft2_of_rect_continuous():
    width = 1
    height = 1
    # Define the x and y ranges
    x = np.linspace(-5, 5, 256)
    y = np.linspace(-5, 5, 256)

    # Define the rectangular function correctly using X and Y grids
    func = lambda X, Y: np.where((np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2), 1, 0) * np.cos(2*np.pi*X)

    # Plot the Fourier transform, analytical solution, and reconstruction
    plot_comparison(func, x, y, width, height)

# Call the function to execute
#fft2_of_rect_continuous()

def ft_of_2d_gaussian(square_width=10, num_samples=500, sigma=0.6):
    # The more sigma is smaller than the square width, the difference between the intensities is smaller
    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    func = lambda X, Y: np.exp(-(X ** 2 + Y ** 2) / sigma ** 2) * np.cos(20 * np.pi * X) * np.cos(4 * np.pi * Y)

    # Compute the numerical Fourier transform
    VX, VY, FFT = fft_funcs.fourier_transform_2d_continuous(func, x, x)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 12))

    # Subplot 1: Gaussian Aperture
    plt.subplot(2, 2, 1)
    X1, Y1 = np.meshgrid(x, x, indexing='ij')
    F = func(X1, Y1) ** 2
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Aperture')

    # Subplot 2: Numerical Fourier Transform
    numerical_intensity = np.real(FFT) ** 2
    plt.subplot(2, 2, 2)
    plt.imshow(numerical_intensity, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Numerical Fourier Transform in 2D')

    # Subplot 3: Analytical Fourier Transform
    G = 0.25 * (sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY+2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY+2)**2)))
    plt.subplot(2, 2, 3)
    plt.imshow(np.real(G) ** 2, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Analytic Fourier Transform in 2D')

    # Subplot 4: Difference between Numerical and Analytical
    difference = np.abs(numerical_intensity - np.real(G) ** 2)
    plt.subplot(2, 2, 4)
    plt.imshow(difference, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='hot')
    plt.colorbar(label='Difference Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Difference: Numerical vs. Analytical')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# Call the function
# ft_of_2d_gaussian()

def ft_and_ift_2d_gaussian(square_width=10, num_samples=1000, sigma=1):
    # The more sigma is smaller than the square width, the difference between the intensities is smaller
    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    func = lambda X, Y: np.exp(-(X ** 2 + Y ** 2) / sigma ** 2) * np.cos(4 * np.pi * X) * np.cos(2 * np.pi * Y)

    # Compute the numerical Fourier transform
    VX, VY, FFT = fft_funcs.fourier_transform_2d_continuous(func, x, x)

    # Compute the numerical inverse Fourier transform
    IX, IY, IFFT = fft_funcs.inverse_fourier_transform_2d(FFT, VX[:, 0], VY[0, :])

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 12))

    # Subplot 1: Gaussian Aperture
    plt.subplot(1, 3, 1)
    X1, Y1 = np.meshgrid(x, x, indexing='ij')
    F = func(X1, Y1) ** 2
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Aperture')

    # Subplot 2: Numerical Inverse Fourier Transform on Fourier transform
    numerical_intensity = np.real(IFFT) ** 2
    plt.subplot(1, 3, 2)
    plt.imshow(numerical_intensity, extent=[IX.min(), IX.max(), IY.min(), IY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Numerical Inverse Fourier Transform in 2D')

    # Subplot 4: Difference between Numerical and Analytical
    difference = np.abs(numerical_intensity - F)
    plt.subplot(1, 3, 3)
    plt.imshow(difference, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='hot')
    plt.colorbar(label='Difference Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Difference: Original vs. Numerical IFT of FT')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# Call the function
ft_and_ift_2d_gaussian()