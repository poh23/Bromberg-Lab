from scipy.fft import ifft, fftshift, ifftshift
import numpy as np

def inverse_fourier_transform_1d(func, v):
    """
    Performs the inverse FFT_1D Fourier transform to match the continuous definition:
    f(x) = ∫ F(v) * exp(-i * 2π * v * x) dv

    Parameters:
    - func: FFT_1D array of Fourier coefficients (complex).
    - v: FFT_1D array of frequency components (in cycles per unit).

    Returns:
    - x: FFT_1D array of spatial coordinates.
    - f_reconstructed: FFT_1D array of the reconstructed function in the spatial domain.
    """
    # Step size in frequency domain
    dv = v[1] - v[0]

    # Inverse FFT_1D with the correct scaling
    func_unshifted = ifftshift(func)  # Unshift to perform ifft correctly
    f_reconstructed = ifft(func_unshifted) * len(v) * dv

    # Generate spatial coordinates x corresponding to frequencies
    x = np.linspace(-0.5 / dv, 0.5 / dv, len(v))

    # Center the zero frequency component
    f_reconstructed = fftshift(f_reconstructed)

    return x, f_reconstructed
