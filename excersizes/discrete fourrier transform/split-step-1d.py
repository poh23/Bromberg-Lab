import numpy as np
from numeral_fresnel import fresnel_approximation_1d
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Constants
kerr_coefficient = 0 # The order of magnitude of the coefficient n2 (in units of cm^2/W) is 10e-l6 to 10e-l4 in glasses, 10e-l4 to 10e-7 in doped glasses, 10e-10 to 10e-8 in organic materials, and 10e-10 to 1o-2 in semiconductors.
refractive_index = 1.46 # Silica
free_space_impedance = 376.73 # Ohm
lamda = 532e-9
k0 = 2 * np.pi / lamda
k = k0 * refractive_index

step_size = 1e-5

def non_linear_propagation_part(current_envelope, half_step_size):
    updated_envelope = current_envelope - 0.5j * half_step_size * k * np.abs(current_envelope)**2 * current_envelope * kerr_coefficient / free_space_impedance
    return updated_envelope

def split_step_fourier_transform(L, x, init_envelope):
    num_steps = int(L // step_size)
    half_step_size = step_size / 2.0
    curr_envelope = np.array(init_envelope)
    total_envelope = [init_envelope]
    total_energies = [np.sum(np.abs(init_envelope)**2)]
    for i in range(num_steps):
        # Linear fresnel propagation
        fresnel_propagated_step = fresnel_approximation_1d(curr_envelope, x, half_step_size, lamda)[1]
        total_energies.append(np.sum(np.abs(fresnel_propagated_step)**2))
        curr_envelope = np.array(fresnel_propagated_step).copy()
        # Non-linear propagation
        non_linear_propagated_step = non_linear_propagation_part(curr_envelope, half_step_size)
        curr_envelope = np.array(non_linear_propagated_step).copy()

        total_envelope.append(curr_envelope)

    return total_envelope, total_energies

def graph_split_step_fourier_transform(ax, fig, X, Z, Intensity, title):
    c = ax.pcolormesh(X, Z, Intensity, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(title)
    return ax

def propagate_gaussian():
    L = 3e-1 # max distance because then the beam is too wide and there are edge effects
    N = 2**10
    square_width = 3e-3
    sigma = 50e-6

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = np.exp(-x ** 2 / sigma ** 2)
    z = np.arange(0, L, step_size)
    X, Z = np.meshgrid(x, z)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    total_envelope_linear = split_step_fourier_transform(L, x, init_envelope)
    Intensity_linear = np.abs(total_envelope_linear)**2
    ax1 = graph_split_step_fourier_transform(ax1, fig, X, Z, Intensity_linear, 'Propagation of Gaussian beam, n2 = 0')

    global kerr_coefficient
    kerr_coefficient = 1e-3 # when kerr_coefficient is larger than 1e-3 there isn't energy conservation
    total_envelope = split_step_fourier_transform(L, x, init_envelope)
    Intensity = np.abs(total_envelope) ** 2
    ax2 = graph_split_step_fourier_transform(ax2, fig, X, Z, Intensity, f'Propagation of Gaussian beam, n2 = {kerr_coefficient}')

    ax3.plot(x, np.abs(total_envelope_linear[-1])**2, label='Final Intensity, n2 = 0')
    ax3.plot(x, np.abs(total_envelope[-1])**2, label=f'Final Intensity, n2 = {kerr_coefficient}')
    ax3.plot(x, np.abs(init_envelope)**2, label='Initial Intensity')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Intensity')
    ax3.set_title(f'Intensity at Z = {L}m')
    ax3.legend()

    plt.show()


# propagate_gaussian()

def propagate_sech():
    global kerr_coefficient
    kerr_coefficient = 1e-3  # when kerr_coefficient is larger than 1e-3 there isn't energy conservation

    L = 3e-1 # max distance because then the beam is too wide and there are edge effects
    N = 2**10
    square_width = 3e-3
    sigma = 50e-6
    A1 = np.sqrt(2 * free_space_impedance / ((k * sigma) ** 2 * kerr_coefficient))
    A0 = 1.21 # np.sqrt(2 * free_space_impedance / ((k*sigma)**2 * kerr_coefficient))

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = A0 / np.cosh(x / sigma)
    z = np.arange(0, L, step_size)
    X, Z = np.meshgrid(x, z)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    total_envelope, total_energies = split_step_fourier_transform(L, x, init_envelope)
    Intensity = np.abs(total_envelope)**2
    ax1 = graph_split_step_fourier_transform(ax1, fig, X, Z, Intensity, f'Propagation of Sech beam, n2 = {kerr_coefficient}')

    print(f'Initial energy: {total_energies[0]}')
    print(f'Final energy: {total_energies[-1]}')

    ax2.plot(x, np.abs(total_envelope[-1])**2, label=f'Final Intensity')
    ax2.plot(x, np.abs(init_envelope)**2, label='Initial Intensity')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Intensity at Z = {L}m')
    ax2.legend()

    plt.show()


propagate_sech()

