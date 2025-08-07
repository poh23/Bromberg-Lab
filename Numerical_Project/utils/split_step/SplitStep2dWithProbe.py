import numpy as np
from Numerical_Project.utils.Fresnel.Fresnel_2D.fresnel_approximation_2d import fresnel_approximation_2d
from Numerical_Project.utils.split_step.SplitStep2d import SplitStep2d   # wherever you defined it
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap
import matplotlib.pyplot as plt


class SplitStep2dWithProbe(SplitStep2d):
    def __init__(self, *args, init_probe=None, probe_lambda=632, **kwargs):
        """
        All the same args/kwargs as the base class,
        plus an optional 'init_probe' field which
        you can pass to .propagate() below.
        """
        super().__init__(*args, **kwargs)
        self._init_probe = init_probe  # store it for propagate()
        self._probe_lambda = probe_lambda
        self.theta = 0

    def probe_propagation_part(self, pump_envelope, probe_envelope, probe_k0, step_size):
        updated_envelope = probe_envelope - 0.5j * step_size * probe_envelope * probe_k0 * np.abs(
            pump_envelope) ** 2 * self.kerr_coefficient / self.free_space_impedance
        #print(f'small parameter: {np.abs(np.sum(0.5 * step_size * self.k * np.abs(pump_envelope) ** 2 * self.kerr_coefficient / self.free_space_impedance))}')
        return updated_envelope

    def probe_propagation_part_w_tri_phase(self, pump_envelope, probe_envelope, probe_k0, step_size,Y):
        updated_envelope = probe_envelope * (1 - 0.5j * step_size * probe_k0 * np.abs(
            pump_envelope) ** 2 * self.kerr_coefficient / self.free_space_impedance + 1j*probe_k0*np.tan(self.theta) * step_size * Y)
        #print(f'small parameter: {np.abs(np.sum(0.5 * step_size * self.k * np.abs(pump_envelope) ** 2 * self.kerr_coefficient / self.free_space_impedance))}')
        return updated_envelope

    def propagate(self, L, x, y, init_envelope, tri_phase=False):
        """
        Same signature as SplitStep2d.propagate!

        - init_envelope: always your pump's initial field.
        - self._init_probe: if not None, we also carry a probe.
        """
        # Run exactly the base‐class pump‐only propagation,
        # *but* we lift out the loop so we can interleave probe.
        dz = L / int(self.num_steps)
        pump = init_envelope.copy()

        # If user supplied a probe at init,
        # make a working copy here.  Otherwise None.
        probe = None
        probe_lamda = self._probe_lambda if self._probe_lambda is not None else self.lamda
        probe_k0 = 2 * np.pi / probe_lamda
        if self._init_probe is not None:
            probe = self._init_probe.copy().astype(np.complex128)

        if tri_phase:
            y1 = y + max(y)
            X, Y = np.meshgrid(x, y1)  # Create a meshgrid for the coordinates

        pump_energy = []
        probe_energy = []

        # single loop, exactly like base class
        for i in range(int(self.num_steps)):
            # --- pump: linear Fresnel step ---
            _, _, pump = fresnel_approximation_2d(
                pump, x, y, dz, self.refractive_index, self.lamda)

            # --- pump: Kerr nonlinearity ---
            pump = self.non_linear_propagation_part(pump, dz)

            # record pump energy if you like
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            pump_energy.append(np.sum(np.abs(pump) ** 2) * dx * dy)

            if probe is not None:
                # 2) phase‐kick probe
                if tri_phase:
                    probe = self.probe_propagation_part_w_tri_phase(pump, probe, probe_k0, dz, Y)
                else:
                    probe = self.probe_propagation_part(pump, probe, probe_k0, dz)

                # 3) linear Fresnel on probe
                _, _, probe = fresnel_approximation_2d(
                    probe, x, y, dz, self.refractive_index, probe_lamda)

                # record probe energy
                probe_energy.append(np.sum(np.abs(probe) ** 2) * dx * dy)

            if i%self.data_save_rate == 0 or i == int(self.num_steps) - 1:
                # save every N steps, or last step
                if probe is not None:
                    print(f'loop {i} of {int(self.num_steps)}, pump and probe')

        # return exactly the same dict‐like structure shape
        out = {
            'z': np.linspace(0, L, int(self.num_steps) + 1),
            'pump_final': pump,
            'pump_energy': np.array(pump_energy),
        }
        if probe is not None:
            out['probe_final'] = probe
            out['probe_energy'] = np.array(probe_energy)
        return out
