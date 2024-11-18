import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from Numerical_Project.utils.Fresnel.Fresnel_2D.fresnel_approximation_2d import fresnel_approximation_2d
from Numerical_Project.utils.find_sigma import find_sigma
from Numerical_Project.utils.closest import closest
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap

# setting path
sys.path.append('../../../')

matplotlib.use('TkAgg')

def analytic_fresnel_approx_of_2d_gaussian(X, Y, d, sigma, lamda):
    theta_0 = lamda/(np.pi*sigma)
    new_sigma_squared = sigma**2 + (theta_0*d)**2
    I = ((sigma**2)/new_sigma_squared) * np.exp(-2*(X**2+Y**2)/new_sigma_squared)
    return I

def fresnel_approx_of_gaussian_2d(square_width=3e-3, num_samples=1024, sigma=50e-6):

    lamda = 532e-9
    rayleigh_length = np.pi * sigma ** 2 / lamda
    d = rayleigh_length

    print(f'Rayleigh length sigma- {np.sqrt(2) * sigma}')

    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    X, Y = np.meshgrid(x, x)
    f = np.exp(-(X**2 + Y**2) / sigma**2)

    X, Y, G = fresnel_approximation_2d(f, x, x, d, lamda)

    # Create a single figure for all subplots
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # Subplot 1: Gaussian Aperture
    plt.subplot(2, 3, 1)
    ax1 = XY_2d_heatmap(ax1, fig, X, Y, np.abs(f)**2, 'Gaussian Aperture')

    # Subplot 2: Numerical Fresnel Approximation
    numerical_intensity = np.abs(G)**2
    ax2 = XY_2d_heatmap(ax2, fig, X, Y, numerical_intensity, f'Numerical Fresnel Approximation at d = {d}')

    # Subplot 3: Analytic Fresnel Approximation
    analytic_intensity = analytic_fresnel_approx_of_2d_gaussian(X, Y, d, sigma, lamda)
    ax3 = XY_2d_heatmap(ax3, fig, X, Y, analytic_intensity, f'Analytic Propagation at d = {d}')

    print(f'energy after propagation of analytic solution - {np.sum(analytic_intensity)}')

    # Subplot 4: difference between analytic propagation and numeric
    analytic_numeric_Diff = np.abs(analytic_intensity - numerical_intensity)
    ax4 = XY_2d_heatmap(ax4, fig, X, Y, analytic_numeric_Diff, 'Analytic vs Numeric difference', cmap='hot')

    # Subplot 5: x slice
    x_slice = 0
    y = Y[:, 0]
    y0 = closest(y, 0)
    x = closest(X[0, :], x_slice)
    ax5.plot(y[y0:], analytic_intensity[x, y0:], linestyle='--',
             label=f'Analytic Fresnel Intensity at x={x_slice} only positive y')
    ax5.plot(y[y0:], numerical_intensity[x, y0:], label=f'Numeric Fresnel Intensity at x={x_slice} only positive y')
    print(f'area under numerical x slice - {np.sum(numerical_intensity[x, y0:])}')
    print(f'area under analytic x slice - {np.sum(analytic_intensity[x, y0:])}')
    ax5.legend()

    # Subplot 6: y slice
    y_slice = 0
    x = X[0, :]
    x0 = closest(x, 0)
    y = closest(Y[:, 0], y_slice)
    ax6.plot(x[x0:], analytic_intensity[x0:, y], linestyle='--', label=f'Analytic Fresnel Intensity at y={y_slice} only positive x')
    ax6.plot(x[x0:], numerical_intensity[x0:, y], label=f'Numeric Fresnel Intensity at y={y_slice} only positive x')
    print(f'area under numerical y slice - {np.sum(numerical_intensity[x0:, y])}')
    print(f'area under analytic y slice - {np.sum(analytic_intensity[x0:, y])}')

    sigma_of_analytic = find_sigma(x[x0:], analytic_intensity[x0:, y])
    print(f'sigma of analytic - {sigma_of_analytic}')

    sigma_of_numeric = find_sigma(x[x0:], numerical_intensity[x0:, y])
    print(f'sigma of numeric - {sigma_of_numeric}')

    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.legend()
    ax6.set_title(f'Numeric vs Analytic sliced difference')

    plt.show()

fresnel_approx_of_gaussian_2d()