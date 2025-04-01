import numpy as np
import sys
from Numerical_Project.utils.split_step.SplitStep2d import SplitStep2d
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap
from Numerical_Project.utils.closest import closest
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# setting path
sys.path.append('../../../')
free_space_impedance = 376.73 # Ohm

def non_linear_propagation_of_2d_gaussian():

    # Measurements of the cuvettes in length, width, height
    # Thin Cuvette measures 1.25x1.25x4.5 cm on the outside
    # Wide Cuvette measures 5.25x1.25x4.5 cm on the outside
    L = 1.25e-2  # max distance because then the beam is too wide and there are edge effects
    N = 2 ** 9
    square_width = 1.25e-2
    laser_power = 5e-3  # 5mW
    sigma = 50e-6 # incident beam width - 50 micrometers

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    X, Y = np.meshgrid(x, x)
    A = 4/sigma * np.sqrt(laser_power * free_space_impedance / np.pi)
    init_envelope = A*np.exp(-(X ** 2 + Y ** 2) / sigma ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    # Linear Propagation
    split_step = SplitStep2d(kerr_coefficient=0)

    z, total_envelope_linear, total_energies_linear = split_step.propagate(L, x, x, init_envelope)
    Intensity_linear = np.abs(total_envelope_linear) ** 2
    axes[0] = XY_2d_heatmap(axes[0], fig, X, Y, Intensity_linear[0], 'Initial Intensity', cmap='viridis')
    axes[1] = XY_2d_heatmap(axes[1], fig, X, Y, Intensity_linear[-1], 'Final Intensity after linear propagation',
                            cmap='viridis')

    # Linear + Non-Linear Propagation
    kerr_coefficient = 1e-12 # when kerr_coefficient is larger than 1e-3 there isn't energy conservation
    split_step = SplitStep2d(kerr_coefficient=kerr_coefficient)
    z, total_envelope, total_energies = split_step.propagate(L, x, x, init_envelope)
    Intensity = np.abs(total_envelope) ** 2
    axes[2] = XY_2d_heatmap(axes[2], fig, X, Y, Intensity[-1], f'Propagation of Gaussian beam, n2 = {kerr_coefficient}',
                            cmap='viridis')


    print(f'initial energy before propagation - {total_energies_linear[0]}')
    print(f'final energy after linear propagation - {total_energies_linear[-1]}')
    print(
        f'final energy after non-linear (n2 = {split_step.kerr_coefficient}, total steps = {split_step.num_steps}) '
        f'propagation - {total_energies[-1]}')

    # Plot slices of the intensity at y=0
    y_slice = 0
    x = X[0, :]
    y = closest(Y[:, 0], y_slice)

    axes[3].plot(x, Intensity_linear[0][:, y], linestyle='--',
                 label=f'Initial Intensity at y={y_slice} before propagation')
    axes[3].plot(x, Intensity_linear[-1][:, y], linestyle='--',
                 label=f'Final Intensity at y={y_slice} after linear propagation')
    axes[3].plot(x, Intensity[-1][:, y], linestyle='--',
                 label=f'Final Intensity at y={y_slice} after nonlinear propagation, n2 = {kerr_coefficient}')

    axes[3].set_xlabel('X')
    axes[3].set_ylabel('Intensity')
    axes[3].set_title(f'Intensity at Z = {L}m')
    axes[3].legend()

    plt.show()


non_linear_propagation_of_2d_gaussian()
