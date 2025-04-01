import numpy as np
import sys
from Numerical_Project.utils.split_step.SplitStep2d import SplitStep2d
from Numerical_Project.utils.FFT.FFT_2D.fourier_transform_2d import fourier_transform_2d
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap, FFT_2d_heatmap
from Numerical_Project.utils.closest import closest
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

free_space_impedance = 376.73 # Ohm
def propagation_through_setup():
    # Measurements of the cuvettes in length, width, height
    # Thin Cuvette measures 1.25x1.25x4.5 cm on the outside
    # Wide Cuvette measures 5.25x1.25x4.5 cm on the outside
    L = 1.25e-2
    N = 2 ** 9
    square_width = 1.25e-3
    laser_power = 1.2e-3  # 1mW
    d = 0.2  # distance to screen - 20cm
    sigma = 50e-6  # incident beam width,

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    X, Y = np.meshgrid(x, x)
    A = 4 / sigma * np.sqrt(laser_power * free_space_impedance / np.pi)
    init_envelope = A * np.exp(-(X ** 2 + Y ** 2) / sigma ** 2)
    #init_envelope = A * np.where(X ** 2 + Y ** 2 <= sigma ** 2, 1, 0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax = XY_2d_heatmap(ax, fig, X, Y, init_envelope, 'Initial Intensity', cmap='viridis')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    # Propagation
    kerr_coefficient = -3e-10
    split_step = SplitStep2d(kerr_coefficient=kerr_coefficient, num_steps=10e4)

    z, total_envelope_linear, total_energies_linear = split_step.propagate(L, x, x, init_envelope)
    Intensity_linear = (np.abs(total_envelope_linear) ** 2)/A**2 * 100
    axes[0] = XY_2d_heatmap(axes[0], fig, X, Y, Intensity_linear[0], f'Initial Intensity',
                            cmap='viridis')

    axes[1] = XY_2d_heatmap(axes[1], fig, X, Y, Intensity_linear[-1], f'Final Intensity, n2 ={kerr_coefficient}', cmap='viridis')

    Vx, Vy, final_envelope_after_fourier = fourier_transform_2d(total_envelope_linear[-1], x, x)
    #update far field axes
    Intensity_after_fourier = ((split_step.lamda * d)**-2) * np.abs(final_envelope_after_fourier) ** 2
    X_1 = Vx * split_step.lamda * d
    Y_1 = Vy * split_step.lamda * d

    x1 = X_1[0, :]
    y1 = Y_1[:, 0]
    dx1 = x1[1] - x1[0]
    dy1 = y1[1] - y1[0]

    axes[2] = XY_2d_heatmap(axes[2], fig, X_1, Y_1, Intensity_after_fourier/A * 100, f'Intensity after fourier, n2 ={kerr_coefficient}', cmap='viridis')

    # print energy conservation
    print(f'initial energy before propagation - {total_energies_linear[0]}')
    print(f'final energy {total_energies_linear[-1]}')
    print(f'final energy out of initial energy percentage - {total_energies_linear[-1]/total_energies_linear[0]*100} %')
    print(f'energy after fourier - {np.sum(Intensity_after_fourier * dx1 * dy1)}')
    print(f'energy after fourier out of initial energy percentage - {np.sum(Intensity_after_fourier*dx1*dy1)/total_energies_linear[0]*100} %')

    # Plot slices of the intensity at y=0
    y_slice = 0
    x = X_1[0, :]
    y = closest(Y_1[:, 0], y_slice)

    axes[3].plot(x, Intensity_after_fourier[:, y]/A * 100, linestyle='--',
                 label=f'Intensity after fourier at y={y_slice}')

    axes[3].set_xlabel('Vx')
    axes[3].set_ylabel('Intensity (%)')
    axes[3].set_title(f'Intensity at Z = {L}m')
    axes[3].legend()

    plt.show()

propagation_through_setup()