from scipy.fft import fftshift, ifftshift, ifft2
import numpy as np

def inverse_fourier_transform_2d(func, vx, vy):
    """
    Performs the inverse FFT_2D Fourier transform to match the continuous definition:
    f(x, y) = ∫ ∫ F(vx, vy) * exp(-i * 2π * (vx * x + vy * y)) dvx * dvy

    Parameters:
    - func: FFT_2D array of Fourier coefficients (complex).
    - vx: FFT_1D array of frequency components in x direction (in cycles per unit).
    - vy: FFT_1D array of frequency components in y direction (in cycles per unit).

    Returns:
    - x: FFT_2D array of x spatial coordinates.
    - y: FFT_2D array of y spatial coordinates.
    - f_reconstructed: FFT_2D array of the reconstructed function in the spatial domain.
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
