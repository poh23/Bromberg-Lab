import numpy as np
import sys
from Numerical_Project.utils.split_step.SplitStep1d import SplitStep1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# setting path
sys.path.append('../../../')
def non_linear_propagation_of_gaussian():
    L = 3e-1 # max distance because then the beam is too wide and there are edge effects
    N = 2**10
    square_width = 3e-3
    sigma = 200e-6
    split_step = SplitStep1d(kerr_coefficient=0)
    split_step.num_steps = 1000
    split_step.data_save_rate = 1

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    f = -15
    init_envelope_f = np.exp(-x ** 2 / sigma ** 2) * np.exp( - (1j * split_step.k / 2 * f) * x ** 2)  # Gaussian beam with a phase front

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    z, total_envelope_linear, total_energies_linear = split_step.propagate(L, x, init_envelope_f)
    Intensity_linear = np.abs(total_envelope_linear)**2
    ax1 = split_step.graph_propagation(ax1, fig, x, z, Intensity_linear, f'Propagation of Gaussian beam with lens phase f ={f}, n2 = 0')

    init_envelope = np.exp(-x ** 2 / sigma ** 2)
    kerr_coefficient = 0 # when kerr_coefficient is larger than 1e-3 there isn't energy conservation
    split_step = SplitStep1d(kerr_coefficient=kerr_coefficient)
    split_step.num_steps = 1000
    split_step.data_save_rate = 1
    z, total_envelope, total_energies = split_step.propagate(L, x, init_envelope)
    Intensity = np.abs(total_envelope) ** 2
    ax2 = split_step.graph_propagation(ax2, fig, x, z, Intensity, f'Propagation of Gaussian beam, n2 = {kerr_coefficient}')

    print(f'initial energy before propagation - {total_energies_linear[0]}')
    print(f'final energy after linear propagation - {total_energies_linear[-1]}')
    print(f'final energy after non-linear (n2 = {split_step.kerr_coefficient}, num steps = {split_step.num_steps}) propagation - {total_energies[-1]}')

    ax3.plot(x, np.abs(total_envelope_linear[-1])**2, label=f'Final Intensity, f = {f}')
    ax3.plot(x, np.abs(total_envelope[-1])**2, label=f'Final Intensity, n2 = {kerr_coefficient}')
    ax3.plot(x, np.abs(init_envelope)**2, label='Initial Intensity')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Intensity')
    ax3.set_title(f'Intensity at Z = {L}m')
    ax3.legend()

    plt.show()


non_linear_propagation_of_gaussian()