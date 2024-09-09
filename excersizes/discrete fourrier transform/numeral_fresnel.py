import numpy as np
import matplotlib.pyplot as plt
import fft_funcs
import warnings

def fresnel_approximation(func, x, y, d, lamda=1, H0=1):
    # Find F(vx, vy) weight function to put in fresnel by transforming input func
    VX, VY, F = fft_funcs.fourier_transform_2d_continuous(func, x, y)

    # Find indices where F is not zero
    non_zero_indices = np.abs(F) > 1e-3  # Use a threshold to consider non-zero values, adjust as necessary

    # Calculate vx^2 + vy^2 only where F is non-zero
    vx_vy_squared = (VX ** 2 + VY ** 2)[non_zero_indices]

    # Find the maximum value of vx^2 + vy^2 where F is not zero
    max_v_squared = np.max(vx_vy_squared) if vx_vy_squared.size > 0 else 0
    threshold = 1 / lamda ** 2

    # Issue a warning if the approximation does not hold
    if max_v_squared >= 0.1 * threshold:  # Adjust the threshold factor (0.1) as needed
        warnings.warn(
            f"The Fresnel approximation may not hold: max(vx^2 + vy^2) = {max_v_squared:.2e} "
            f"is not much smaller than 1/λ^2 = {threshold:.2e}.",
            UserWarning
        )

    # calculate the transfer function approximation
    H0 *= np.exp(-1j * d / lamda)
    H = H0 * np.exp(1j * np.pi * lamda * d * (VX ** 2 + VY ** 2))

    # calculate output by inverse transform of the product H(vx, vy)*F(vx,vy)
    X, Y, G = fft_funcs.inverse_fourier_transform_2d(H * F, VX[:, 0], VY[0, :])

    return X, Y, G


## fresnel diffraction from a gaussian aperture


def fresnel_approx_of_gaussian(square_width=15, num_samples=1000, sigma=5):
    d = 200

    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.exp(-(X**2 + Y**2) / sigma**2)

    X, Y, G = fresnel_approximation(func, x, x, d)

    # Create a single figure for all subplots
    plt.figure(figsize=(18, 6))

    # Subplot 1: Gaussian Aperture
    plt.subplot(1, 3, 1)
    X1, Y1 = np.meshgrid(x, x, indexing='ij')
    F = func(X1, Y1)
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Aperture')

    # Subplot 2: Numerical Fresnel Approximation
    numerical_intensity = np.real(G)**2
    plt.subplot(1, 3, 2)
    plt.imshow(numerical_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Numerical Fresnel Approximation at d = {d}')

    # Subplot 3: Analytic Fresnel Approximation
    R, Nf = analytic_fresnel_approx_of_gaussian(X, Y, d, sigma, 1)
    plt.subplot(1, 3, 3)
    plt.imshow(np.real(R), extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Analytic at d = {d}, N_F = {round(Nf,2)}')

    # Show all subplots in one figure
    plt.show()


def analytic_fresnel_approx_of_gaussian(X, Y, d, sigma, lamda):
    theta_0 = lamda/(np.pi*sigma)
    new_sigma_squared = sigma**2 + (theta_0+d)**2
    Nf = np.pi*(sigma**2)/(2*lamda*d)
    I = (sigma**2)/new_sigma_squared * np.exp(-2*(X**2+Y**2)/new_sigma_squared**2)
    return I, Nf

#fresnel_approx_of_gaussian()


