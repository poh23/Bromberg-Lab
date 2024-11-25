import numpy as np
import sys

# setting path
sys.path.append('../../../')
from Numerical_Project.utils.FFT.FFT_1D.fourier_transform_1d import fourier_transform_1d
from Numerical_Project.utils.FFT.FFT_1D.inverse_fourier_transform_1d import inverse_fourier_transform_1d


def fresnel_approximation_1d(func, x, d, refractive_index=1, lamda=1, H0=1):
    """
    Perform a Fresnel approximation for FFT_1D propagation of a wavefront.

    Parameters:
    - func: A callable function of x.
    - x: FFT_1D array of spatial coordinates.
    - d: Propagation distance.
    - lamda: Wavelength of the light.
    - H0: Initial amplitude factor.

    Returns:
    - X: FFT_1D array of spatial coordinates.
    - G: FFT_1D array of the propagated wavefront.
    """
    energy_before_propagation = np.sum(np.abs(func) ** 2)
    # Find F(v) weight function to put in fresnel by transforming input func
    nu, F = fourier_transform_1d(func, x)

    # Calculate the transfer function approximation
    H0 *= np.exp(-1j * (2 * np.pi * refractive_index / lamda) * d)
    H = H0 * np.exp(1j * np.pi * lamda / refractive_index * d * nu ** 2)

    # Calculate output by inverse transform of the product H(v) * F(v)
    X, G = inverse_fourier_transform_1d(H * F, nu)

    energy_after_propagation = np.sum(np.abs(G) ** 2)

    return X, G
