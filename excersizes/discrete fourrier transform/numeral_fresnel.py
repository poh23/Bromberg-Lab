import numpy as np
import matplotlib.pyplot as plt
import fft_funcs

import matplotlib

matplotlib.use('TkAgg')

# todo: check perseval on various edge cases
# todo: compare normalized slices of numerical output, analytical output and input` qa

def fresnel_approximation(func, x, y, d, lamda=1, H0=1):

    X1, Y1 = np.meshgrid(x, y)
    init_energy = np.sum(np.abs(func(X1, Y1)) ** 2)
    print(f'initial energy - {init_energy}')

    # Find F(vx, vy) weight function to put in fresnel by transforming input func
    VX, VY, F = fft_funcs.fourier_transform_2d_continuous(func, x, y)

    dx = x[1] - x[0]
    numerical_intensity_normalization = dx ** 4 * len(F) ** 2
    print(f'energy after FFT - {(np.sum(np.abs(F) ** 2) / numerical_intensity_normalization)}')

    # calculate the transfer function approximation
    H0 *= np.exp(-1j * (2 * np.pi / lamda) * d)
    H = H0 * np.exp(1j * np.pi * lamda * d * (VX ** 2 + VY ** 2))

    # calculate output by inverse transform of the product H(vx, vy)*F(vx,vy)
    vx = VX[0, :]
    vy = VY[:, 0]
    X, Y, G = fft_funcs.inverse_fourier_transform_2d(H * F, vx, vy)

    intensity_after_propagation = np.sum(np.abs(G) ** 2)
    print(f'energy after propagation - {intensity_after_propagation}')

    return X, Y, G

## propagtaion without any approximations
def real_propagation(func, x, y, L, wl=1):
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]


    # X, Y range centered around 0
    X = (np.arange(1, Nx + 1) - (Nx / 2 + 0.5)) * dx
    Y = (np.arange(1, Ny + 1) - (Ny / 2 + 0.5)) * dy
    XX, YY = np.meshgrid(X, Y)

    # k-space grid
    fs = 1 / (XX.max() - XX.min())
    freq_x = fs * np.arange(-Nx // 2, Nx // 2)
    fs = 1 / (YY.max() - YY.min())
    freq_y = fs * np.arange(-Ny // 2, Ny // 2)
    freq_XXs, freq_YYs = np.meshgrid(freq_x, freq_y)
    light_k = 2 * np.pi / wl
    k_xx = freq_XXs * 2 * np.pi
    k_yy = freq_YYs * 2 * np.pi

    k_z_sqr = light_k ** 2 - (k_xx ** 2 + k_yy ** 2)
    # Remove all the negative component, as they represent evanescent waves, see Fourier Optics page 58
    np.maximum(k_z_sqr, 0, out=k_z_sqr)
    k_z = np.sqrt(k_z_sqr)

    # Initial
    E_gaus_init = func(XX, YY)

    E_K = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_gaus_init)))
    # Apply the transfer function of free-space, see Fourier Optics page 74
    # normal forward motion is with + in the exponent
    prop_mat = np.exp(+1j * k_z * L)
    E_K *= prop_mat
    E_out = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_K)))

    return XX,YY,E_out

## fresnel diffraction from a gaussian aperture


def fresnel_approx_of_gaussian(square_width=3e-3, num_samples=1024, sigma=50e-6):

    lamda = 532e-9
    rayleigh_length = np.pi * sigma ** 2 / lamda
    d = 5 * rayleigh_length

    print(f'Rayleigh length sigma- {np.sqrt(2) * sigma}')

    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.exp(-(X**2 + Y**2) / sigma**2)

    X, Y, G = fresnel_approximation(func, x, x, d, lamda)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 12))

    # Subplot 1: Gaussian Aperture
    plt.subplot(2, 3, 1)
    F = func(X, Y)
    plt.imshow(np.abs(F)**2, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Aperture')

    # Subplot 2: Numerical Fresnel Approximation
    numerical_intensity = np.abs(G)**2
    plt.subplot(2, 3, 2)
    plt.imshow(numerical_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Numerical Fresnel Approximation at d = {d}')

    # Subplot 3: Analytic Fresnel Approximation
    analytic_intensity, Nf = analytic_fresnel_approx_of_gaussian(X, Y, d, sigma, lamda)
    plt.subplot(2, 3, 3)
    plt.imshow(analytic_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Analytic at d = {d}, N_F = {round(Nf,2)}')

    print(f'energy after propagation of analytic solution - {np.sum(analytic_intensity)}')

    # Subplot 4: difference between analytic propagation and numeric
    analytic_numeric_Diff = np.abs(analytic_intensity - numerical_intensity)
    plt.subplot(2, 3, 4)
    plt.imshow(analytic_numeric_Diff, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Analytic vs Numeric difference')

    # Subplot 5: x slice
    x_slice = 0
    plt.subplot(2, 3, 5)
    y = Y[:, 0]
    y0 = fft_funcs.closest(y, 0)
    x = fft_funcs.closest(X[0, :], x_slice)
    plt.plot(y[y0:], analytic_intensity[x, y0:], linestyle='--',
             label=f'Analytic Fresnel Intensity at x={x_slice} only positive y')
    plt.plot(y[y0:], numerical_intensity[x, y0:], label=f'Numeric Fresnel Intensity at x={x_slice} only positive y')
    print(f'area under numerical x slice - {np.sum(numerical_intensity[x, y0:])}')
    print(f'area under analytic x slice - {np.sum(analytic_intensity[x, y0:])}')
    plt.legend()

    # Subplot 6: y slice
    y_slice = 0
    plt.subplot(2, 3, 6)
    x = X[0, :]
    x0 = fft_funcs.closest(x, 0)
    y = fft_funcs.closest(Y[:, 0], y_slice)
    plt.plot(x[x0:], analytic_intensity[x0:, y], linestyle='--', label=f'Analytic Fresnel Intensity at y={y_slice} only positive x')
    plt.plot(x[x0:], numerical_intensity[x0:, y], label=f'Numeric Fresnel Intensity at y={y_slice} only positive x')
    print(f'area under numerical y slice - {np.sum(numerical_intensity[x0:, y])}')
    print(f'area under analytic y slice - {np.sum(analytic_intensity[x0:, y])}')

    sigma_of_analytic = find_sigma(x[x0:], analytic_intensity[x0:, y])
    print(f'sigma of analytic - {sigma_of_analytic}')

    sigma_of_numeric = find_sigma(x[x0:], numerical_intensity[x0:, y])
    print(f'sigma of numeric - {sigma_of_numeric}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Numeric vs Analytic sliced difference')

    plt.show()


def analytic_fresnel_approx_of_gaussian(X, Y, d, sigma, lamda):
    theta_0 = lamda/(np.pi*sigma)
    new_sigma_squared = sigma**2 + (theta_0*d)**2
    # Nf = np.pi*(sigma**2)/(2*lamda*d)
    Nf = 0
    I = ((sigma**2)/new_sigma_squared) * np.exp(-2*(X**2+Y**2)/new_sigma_squared)
    return I, Nf

def find_sigma(x, y):
    # find index of y closest to 1/e^2 of max value
    max_val = np.max(y)
    y_closest_idx = fft_funcs.closest(y, max_val/np.e**2)
    sigma = x[y_closest_idx]
    return sigma

fresnel_approx_of_gaussian()


def propagation_of_rect(square_width=15, num_samples=1000, rect_width = 1):
    d = 10
    lamda = 0.5

    # Generate spatial grid
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.where((np.abs(X) <= rect_width / 2) & (np.abs(Y) <= rect_width / 2), 1, 0)

    # Compute Fresnel approximation
    X, Y, G = fresnel_approximation(func, x, x, d, lamda=lamda)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Plane Wave
    plt.subplot(1, 2, 1)
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1))  # Use absolute value for visualization
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plane Wave Intensity')

    # Subplot 2: Numerical Fresnel Approximation
    numerical_intensity = np.abs(G) ** 2  # Corrected intensity calculation
    plt.subplot(1, 2, 2)
    plt.imshow(numerical_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Numerical Fresnel Approximation at d = {d}')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# Call the function
# propagation_of_rect()

def propagate_circ(square_width=3e-3, num_samples=1024, radius = 50e-6):
    lamda = 532e-9
    d = 5e-2

    # Generate spatial grid
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.where((X**2 + Y**2 <= radius**2), 1, 0)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Plane Wave
    plt.subplot(2, 2, 1)
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1))**2
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Initial Plane Wave Intensity')

    # Subplot 2: Numerical propagation computed exactly
    X, Y, G = real_propagation(func, x, x, d, wl=lamda)
    numerical_exact_intensity = np.abs(G) ** 2
    plt.subplot(2, 2, 2)
    plt.imshow(numerical_exact_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Numerical Exact Propagation at d = {d}')

    # Subplot 3: Numerical propagation computed using fresnel approximation
    X1, Y1, H = fresnel_approximation(func, x, x, d, lamda=lamda)
    numerical_intensity = np.abs(H) ** 2
    plt.subplot(2, 2, 3)
    plt.imshow(numerical_intensity, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Numerical Fresnel Approximation at d = {d}')

    # Subplot 4: difference between exact and numeric
    analytic_numeric_Diff = np.abs(numerical_exact_intensity - numerical_intensity)
    plt.subplot(2, 2, 4)
    plt.imshow(analytic_numeric_Diff, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Exact vs Numeric difference')


    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# propagate_circ()