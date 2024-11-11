from scipy.fft import fft, fftfreq, fftshift
def fourier_transform_1d(f, x):
    """
    Perform a FFT_1D Fourier Transform matching the continuous definition:
    F(ν) = ∫ f(x) exp(i 2π ν x) dx

    Parameters:
    - func: A callable function of x.
    - x: FFT_1D array of spatial coordinates.

    Returns:
    - nu: FFT_1D array of frequency components in cycles per unit length.
    - G: FFT_1D array of Fourier coefficients.
    """
    # Discretize space x
    x0, dx = x[0], x[1] - x[0]

    # Perform FFT_1D
    G = fftshift(fft(fftshift(f)))

    # Frequency components as per the continuous definition
    nu = fftshift(fftfreq(f.size, dx))  # frequencies in cycles per unit length

    # Multiply by external factor for normalization and phase correction
    G *= dx

    return nu, G