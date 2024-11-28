import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from Numerical_Project.utils.split_step.SplitStep1d import SplitStep1d
def propagate_sech():
    kerr_coefficient = 1e-3  # when kerr_coefficient is larger than 1e-3 there isn't energy conservation
    split_step = SplitStep1d(kerr_coefficient=kerr_coefficient)
    split_step.num_steps = 1e5

    L = 9e-2 # max distance because then the beam is too wide and there are edge effects
    N = 2**10
    square_width = 3e-3
    sigma = 50e-6
    A0 = np.sqrt(2 * split_step.free_space_impedance / ((split_step.k * sigma) ** 2 * kerr_coefficient))

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = A0 / np.cosh(x / sigma)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    z, total_envelope, total_energies = split_step.propagate(L, x, init_envelope)
    Intensity = np.abs(total_envelope)**2
    ax1 = split_step.graph_propagation(ax1, fig, x, z, Intensity, f'Propagation of Sech beam, n2 = {kerr_coefficient}')

    print(f'Initial energy: {total_energies[0]}')
    print(f'Final energy: {total_energies[-1]}')

    ax2.plot(x, np.abs(total_envelope[-1])**2, label=f'Final Intensity')
    ax2.plot(x, np.abs(init_envelope)**2, label='Initial Intensity')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Intensity at Z = {L}m')
    ax2.legend()

    plt.show()
    print('Propagation of Sech beam done')


propagate_sech()