import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from Numerical_Project.utils.Fresnel.Fresnel_2D.fresnel_approximation_2d import fresnel_approximation_2d
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap

# setting path
sys.path.append('../../../')

matplotlib.use('TkAgg')

def propagation_of_rect(square_width=3e-3, num_samples=2**10, rect_width = 50e-6):
    d = 0.01
    lamda = 532e-9

    # Generate spatial grid
    x = np.linspace(-square_width / 2, square_width / 2, num_samples)
    func = lambda X, Y: np.where((np.abs(X) <= rect_width / 2) & (np.abs(Y) <= rect_width / 2), 1, 0)

    # Compute Fresnel approximation
    X, Y, G = fresnel_approximation_2d(func, x, x, d, lamda=lamda)

    # Create a single figure for all subplots
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Subplot 1: Plane Wave Intensity
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1))**2  # Use absolute value for visualization
    ax1 = XY_2d_heatmap(ax1, fig, X1, Y1, F, 'Plane Wave Intensity')

    # Subplot 2: Numerical Fresnel Approximation
    numerical_intensity = np.abs(G) ** 2  # Corrected intensity calculation
    ax2 = XY_2d_heatmap(ax2, fig, X, Y, numerical_intensity, f'Numerical Fresnel Approximation at d = {d}')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# Call the function
propagation_of_rect()