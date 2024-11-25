import numpy as np
import sys
from Numerical_Project.utils.Fresnel.Fresnel_1D.fresnel_approximation_1d import fresnel_approximation_1d

# setting path
sys.path.append('../../../')
class SplitStep1d:
    def __init__(self, kerr_coefficient, lamda=532e-9):
        self.lamda = lamda
        self.kerr_coefficient = kerr_coefficient  # The order of magnitude of the coefficient n2 (in units of cm^2/W) is 10e-l6 to 10e-l4 in glasses, 10e-l4 to 10e-7 in doped glasses, 10e-10 to 10e-8 in organic materials, and 10e-10 to 1o-2 in semiconductors.
        self.refractive_index = 1.46  # Silica
        self.free_space_impedance = 376.73  # Ohm
        self.k0 = 2 * np.pi / lamda
        self.k = self.k0 * self.refractive_index
        self.num_steps = 1e5
        self.data_save_rate = 200  # Save data every N steps

    def non_linear_propagation_part(self, current_envelope, half_step_size):
        updated_envelope = current_envelope - 0.5j * half_step_size * self.k * np.abs(
            current_envelope) ** 2 * current_envelope * self.kerr_coefficient / self.free_space_impedance
        return updated_envelope

    def propagate(self, L, x, init_envelope):
        step_size = L / self.num_steps
        half_step_size = step_size / 2.0
        curr_envelope = np.array(init_envelope)
        total_envelope = [init_envelope]
        total_energies = [np.sum(np.abs(init_envelope) ** 2)]
        curr_z = 0
        z = [0]
        num_steps_int = int(self.num_steps)
        for i in range(num_steps_int):

            # Linear fresnel propagation
            fresnel_propagated_step = fresnel_approximation_1d(curr_envelope, x, half_step_size, self.lamda)[1]
            curr_envelope = np.array(fresnel_propagated_step).copy()
            total_energies.append(np.sum(np.abs(fresnel_propagated_step) ** 2))

            # Non-linear propagation
            non_linear_propagated_step = self.non_linear_propagation_part(curr_envelope, half_step_size)
            curr_envelope = np.array(non_linear_propagated_step).copy()

            curr_z += step_size

            if i % self.data_save_rate == 0:
                total_envelope.append(curr_envelope)
                z.append(curr_z)

        return z, total_envelope, total_energies

    def graph_propagation(self, ax, fig, x, z, Intensity, title):
        X, Z = np.meshgrid(x, z)

        print(f'X shape: {X.shape}')

        c = ax.pcolormesh(X, Z, Intensity, cmap='viridis', shading='auto')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(title)
        return ax


