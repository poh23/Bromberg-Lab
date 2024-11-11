import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from Numerical_Project.utils.Fresnel.Fresnel_2D.fresnel_approximation_2d import fresnel_approximation_2d
from Numerical_Project.utils.Fresnel.Fresnel_2D.real_propagation_2d import real_propagation_2d
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap

# setting path
sys.path.append('../../../')

matplotlib.use('TkAgg')

def propagate_circ(square_width=3e-3, num_samples=1024, radius = 50e-6):
    lamda = 532e-9
    d = 5e-2

    # Generate spatial grid
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.where((X**2 + Y**2 <= radius**2), 1, 0)

    # Create a single figure for all subplots
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Subplot 1: Plane Wave
    plt.subplot(2, 2, 1)
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1))**2
    ax1 = XY_2d_heatmap(ax1, fig, X1, Y1, F, 'Initial Plane Wave Intensity')

    # Subplot 2: Numerical propagation computed exactly
    X, Y, G = real_propagation_2d(func, x, x, d, wl=lamda)
    numerical_exact_intensity = np.abs(G) ** 2
    ax2 = XY_2d_heatmap(ax2, fig, X, Y, numerical_exact_intensity, f'Numerical Exact Propagation at d = {d}')

    # Subplot 3: Numerical propagation computed using fresnel approximation
    X1, Y1, H = fresnel_approximation_2d(func, x, x, d, lamda=lamda)
    numerical_intensity = np.abs(H) ** 2
    ax3 = XY_2d_heatmap(ax3, fig, X1, Y1, numerical_intensity, f'Numerical Fresnel Approximation at d = {d}')

    # Subplot 4: difference between exact and numeric
    analytic_numeric_Diff = np.abs(numerical_exact_intensity - numerical_intensity)
    ax4 = XY_2d_heatmap(ax4, fig, X, Y, analytic_numeric_Diff, 'Exact vs Numeric difference', cmap='hot')


    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

propagate_circ()