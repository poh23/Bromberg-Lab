from scipy.fft import fft, fftfreq, ifft, fftshift, fft2, ifftshift, ifft2
import numpy as np
import matplotlib.pyplot as plt


def fourier_transform_1d(func, x):
    """
    Perform a 1D Fourier Transform matching the continuous definition:
    F(ν) = ∫ f(x) exp(i 2π ν x) dx

    Parameters:
    - func: A callable function of x.
    - x: 1D array of spatial coordinates.

    Returns:
    - nu: 1D array of frequency components in cycles per unit length.
    - G: 1D array of Fourier coefficients.
    """
    # Discretize space x
    x0, dx = x[0], x[1] - x[0]
    f = func(x)

    # Perform FFT
    G = fftshift(fft(fftshift(f)))

    # Frequency components as per the continuous definition
    nu = fftshift(fftfreq(f.size, dx))  # frequencies in cycles per unit length

    # Multiply by external factor for normalization and phase correction
    G *= dx

    return nu, G
def inverse_fourier_transform_1d(func, v):
    """
    Performs the inverse 1D Fourier transform to match the continuous definition:
    f(x) = ∫ F(v) * exp(-i * 2π * v * x) dv

    Parameters:
    - func: 1D array of Fourier coefficients (complex).
    - v: 1D array of frequency components (in cycles per unit).

    Returns:
    - x: 1D array of spatial coordinates.
    - f_reconstructed: 1D array of the reconstructed function in the spatial domain.
    """
    # Step size in frequency domain
    dv = v[1] - v[0]

    # Inverse FFT with the correct scaling
    func_unshifted = ifftshift(func)  # Unshift to perform ifft correctly
    f_reconstructed = ifft(func_unshifted) * len(v) * dv

    # Generate spatial coordinates x corresponding to frequencies
    x = np.linspace(-0.5 / dv, 0.5 / dv, len(v))

    # Center the zero frequency component
    f_reconstructed = fftshift(f_reconstructed)

    return x, np.real(f_reconstructed)


## 2D fourrier transforms

def fourier_transform_2d_continuous(func, x, y, window=False):
    """
    Calculate the 2D Fourier transform of a given function over a grid defined by x and y,
    approximating the continuous Fourier transform defined with integrals over infinite bounds,
    and applying a Hanning window to reduce edge effects.

    Parameters:
    - func: A callable function that takes two arrays (x, y) and returns a 2D array.
    - x: 1D array of x-coordinates.
    - y: 1D array of y-coordinates.

    Returns:
    - vx, vy: 2D arrays of shifted frequency components in the x and y directions.
    - g: 2D array of Fourier coefficients after applying normalization and phase correction.
    """

    # Discretize space (x, y)
    x0, dx = x[0], x[1] - x[0]
    y0, dy = y[0], y[1] - y[0]

    # Generate a 2D grid of x and y coordinates
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the 2D grid
    f = func(X, Y)

    if window:
        # Create Hanning windows for both dimensions
        hanning_x = np.hanning(len(x))
        hanning_y = np.hanning(len(y))
        hanning_2d = np.outer(hanning_x, hanning_y)  # Create the 2D Hanning window

        # Apply the 2D Hanning window to the function
        f = f * hanning_2d

    # Calculate the 2D FFT of the windowed function
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


def inverse_fourier_transform_2d(func, vx, vy):
    """
    Performs the inverse 2D Fourier transform to match the continuous definition:
    f(x, y) = ∫ ∫ F(vx, vy) * exp(-i * 2π * (vx * x + vy * y)) dvx * dvy

    Parameters:
    - func: 2D array of Fourier coefficients (complex).
    - vx: 1D array of frequency components in x direction (in cycles per unit).
    - vy: 1D array of frequency components in y direction (in cycles per unit).

    Returns:
    - x: 2D array of x spatial coordinates.
    - y: 2D array of y spatial coordinates.
    - f_reconstructed: 2D array of the reconstructed function in the spatial domain.
    """
    # Step sizes in frequency domain
    dvx = vx[1] - vx[0]
    dvy = vy[1] - vy[0]

    # Unshift and perform ifft2
    func_unshifted = ifftshift(ifftshift(func, axes=0), axes=1)
    f_reconstructed = ifft2(func_unshifted) * len(vx) * len(vy) * dvx * dvy

    # Generate spatial coordinates x and y
    x = np.linspace(-0.5 / dvx, 0.5 / dvx, len(vx))
    y = np.linspace(-0.5 / dvy, 0.5 / dvy, len(vy))
    x, y = np.meshgrid(x, y)

    # Center the zero frequency component
    f_reconstructed = fftshift(fftshift(f_reconstructed, axes=0), axes=1)

    return x, y, f_reconstructed

def closest(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx