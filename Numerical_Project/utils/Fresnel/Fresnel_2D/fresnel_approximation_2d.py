import numpy as np
import sys

# setting path
sys.path.append('../../../')
from Numerical_Project.utils.FFT.FFT_2D.fourier_transform_2d import fourier_transform_2d
from Numerical_Project.utils.FFT.FFT_2D.inverse_fourier_transform_2d import inverse_fourier_transform_2d


def fresnel_approximation_2d(f, x, y, d, lamda=1, H0=1):

    init_energy = np.sum(np.abs(f) ** 2)
    print(f'initial energy - {init_energy}')

    # Find F(vx, vy) weight function to put in fresnel by transforming input func
    VX, VY, F = fourier_transform_2d(f, x, y)

    dx = x[1] - x[0]
    numerical_intensity_normalization = dx ** 4 * len(F) ** 2
    print(f'energy after FFT - {(np.sum(np.abs(F) ** 2) / numerical_intensity_normalization)}')

    # calculate the transfer function approximation
    H0 *= np.exp(-1j * (2 * np.pi / lamda) * d)
    H = H0 * np.exp(1j * np.pi * lamda * d * (VX ** 2 + VY ** 2))

    # calculate output by inverse transform of the product H(vx, vy)*F(vx,vy)
    vx = VX[0, :]
    vy = VY[:, 0]
    X, Y, G = inverse_fourier_transform_2d(H * F, vx, vy)

    intensity_after_propagation = np.sum(np.abs(G) ** 2)
    print(f'energy after propagation - {intensity_after_propagation}')

    return X, Y, G