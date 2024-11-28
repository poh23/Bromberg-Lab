from scipy.fft import fftfreq, fftshift
from pyfftw.interfaces.scipy_fft import fft2
import numpy as np

def fourier_transform_2d(f, x, y):
    """
    Calculate the FFT_2D Fourier transform of a given function over a grid defined by x and y,
    approximating the continuous Fourier transform defined with integrals over infinite bounds,
    and applying a Hanning window to reduce edge effects.

    Parameters:
    - func: A callable function that takes two arrays (x, y) and returns a FFT_2D array.
    - x: FFT_1D array of x-coordinates.
    - y: FFT_1D array of y-coordinates.

    Returns:
    - vx, vy: FFT_2D arrays of shifted frequency components in the x and y directions.
    - g: FFT_2D array of Fourier coefficients after applying normalization and phase correction.
    """

    # Discretize space (x, y)
    x0, dx = x[0], x[1] - x[0]
    y0, dy = y[0], y[1] - y[0]

    # Generate a FFT_2D grid of x and y coordinates
    X, Y = np.meshgrid(x, y)

    # Calculate the FFT_2D FFT_1D of the windowed function
    g = fft2(fftshift(f))

    # Continuous frequency normalization factors (in cycles per unit distance)
    vx = fftfreq(f.shape[0], dx)  # Frequencies in cycles per unit distance
    vy = fftfreq(f.shape[1], dy)

    # Scale frequencies to match the continuous Fourier transform definition
    vx = fftshift(vx)
    vy = fftshift(vy)
    VX, VY = np.meshgrid(vx, vy)

    g = fftshift(g) * dx * dy

    return VX, VY, g