import numpy as np
from OPTnumeral_fresnel import fresnel_approximation_1d
from matplotlib import pyplot as plt
import numba
import time
from datetime import datetime

# Constants
kerr_coefficient = 1e-3
refractive_index = 1.49
free_space_impedance = 376.73
effective_impedance = free_space_impedance / refractive_index
lamda = 532e-9
k0 = 2 * np.pi / lamda
k = k0 * refractive_index
step_size = 1e-5

@numba.njit
def non_linear_propagation_part(current_envelope, half_step_size):
    updated_envelope = current_envelope - 0.5j * half_step_size * k * np.abs(
        current_envelope) ** 2 * current_envelope * kerr_coefficient / free_space_impedance
    return updated_envelope

def split_step_fourier_transform(L, x, init_envelope):
    num_steps = int(L // step_size)
    half_step_size = step_size / 2.0
    curr_envelope = np.array(init_envelope, dtype=np.complex128)
    total_envelope = np.zeros((num_steps, len(init_envelope)), dtype=np.complex128)
    total_energies = np.zeros(num_steps)

    for i in range(num_steps):
        # Linear fresnel propagation
        fresnel_propagated_step = fresnel_approximation_1d(curr_envelope, x, half_step_size, lamda)[1]
        total_energies[i] = np.sum(np.abs(fresnel_propagated_step) ** 2)
        curr_envelope = fresnel_propagated_step.copy()

        # Non-linear propagation - numba optimized function
        non_linear_propagated_step = non_linear_propagation_part(curr_envelope, half_step_size)
        curr_envelope = non_linear_propagated_step.copy()

        total_envelope[i, :] = curr_envelope

    return total_envelope, total_energies

def graph_split_step_fourier_transform(ax, fig, X, Z, Intensity, title):
    X = X[:-1, :]
    Z = Z[:-1, :]
    c = ax.pcolormesh(X, Z, Intensity, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(title)
    return ax

def propagate_sech():
    start_time = time.time()  # Start timer
    L = 3e-1
    N = 2 ** 8
    square_width = 3e-3
    sigma = 50e-6
    A0 = np.sqrt(2 * free_space_impedance * refractive_index / ((k * sigma) ** 2 * kerr_coefficient))
    print(A0)

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = A0 / (np.cosh(x / sigma))
    z = np.arange(0, L, step_size)
    X, Z = np.meshgrid(x, z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    total_envelope, total_energies = split_step_fourier_transform(L, x, init_envelope)
    Intensity = np.abs(total_envelope) ** 2
    ax1 = graph_split_step_fourier_transform(ax1, fig, X, Z, Intensity,
                                             f'Propagation of Sech beam, n2 = {kerr_coefficient}')

    print(f'Initial energy: {total_energies[0]}')
    print(f'Final energy: {total_energies[-1]}')

    ax2.plot(x, np.abs(total_envelope[-1]) ** 2, label='Final Intensity')
    ax2.plot(x, np.abs(init_envelope) ** 2, label='Initial Intensity')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Intensity at Z = {L}m')
    ax2.legend()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    current_datetime = datetime.now()
    date_time = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    plt.show()
    plt.savefig(fr"C:\Users\ronen\OneDrive\Desktop\HW Y3 S1\Bromberg\{date_time}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

propagate_sech()
