# =============================
# Imports and Configuration
# =============================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
#matplotlib.use('TkAgg')

from Numerical_Project.utils.split_step.SplitStep2dWithProbe import SplitStep2dWithProbe
from Numerical_Project.utils.FFT.FFT_2D.fourier_transform_2d import fourier_transform_2d
from Numerical_Project.utils.heatmap_generator import XY_2d_heatmap

free_space_impedance = 376.73 # Ohm

# =============================
# Utility Functions
# =============================

def shadowgraph_from_probe(probe_stack: list,
                           L: float, dx: float, dy: float):
    """
    Take the final probe intensity and approximate
    a shadowgram contrast via ∇²(N) → ΔI/I₀.
    Here we just do ΔI/I₀ ≈ (I_final - I₀)/I₀.
    """
    I0 = np.abs(probe_stack[0]) ** 2
    I_end = np.abs(probe_stack[-1]) ** 2
    contrast = (I_end - I0) / I0
    # Or, if you prefer the Laplacian‐of‐δn route, compute δn_line =
    # sum_k delta_n[:,:,k] * dz, then do ∇²xd+dy², etc.
    return contrast

def get_pump_intensity(N, square_width: float, pump_laser_power: float):
    # 1) Load your image (using PIL just for I/O)
    img = Image.open('1.15A.png').convert('L')  # grayscale
    Iccd = np.array(img, dtype=float)  # shape (H, W)
    H, W = Iccd.shape

    # 2) Build a linear interpolator on the original pixel grid
    #    We index pixels as y=0..H-1, x=0..W-1
    orig_y = np.arange(H)
    orig_x = np.arange(W)
    interp = RegularGridInterpolator(
        (orig_y, orig_x), Iccd,
        method='linear',  # bilinear
        bounds_error=False,
        fill_value=0
    )

    # 3) Make your new N×N grid of “virtual pixel” coordinates
    # If you want to preserve physical width L_phys:
    #  Y goes from 0..H-1 maps to y_phys in [-L/2 .. +L/2], etc.
    new_y = np.linspace(0, H - 1, N)
    new_x = np.linspace(0, W - 1, N)
    YY, XX = np.meshgrid(new_y, new_x)

    # 4) Interpolate
    points = np.stack([YY.ravel(), XX.ravel()], axis=-1)  # shape (N*N, 2)
    I_resized = interp(points).reshape(N, N)

    # 5) Now normalize & form your complex probe envelope
    dx = square_width / N  # your physical pixel pitch
    I_resized *= (pump_laser_power / (dx * dx)) / I_resized.sum()
    target = (N / 2, N / 2)
    x = 110
    y = 87
    shift_x = int(round(target[1] - x))
    shift_y = int(round(target[0] - y))

    # === 4) Shift via np.roll (integer shift) ===
    I_centered = np.roll(I_resized, shift=(shift_y, shift_x), axis=(0, 1))
    return np.sqrt(I_centered).astype(np.complex128)

def get_far_field(ax, fig, lamda, d, final_propogated_enevelope, coords, title, cmap='viridis'):
    """
    Plot the far-field intensity distribution of the envelope after Fourier transform.
    """
    Vx, Vy, final_envelope_after_fourier = fourier_transform_2d(final_propogated_enevelope, *coords)
    Intensity_after_fourier = ((lamda * d) ** -2) * np.abs(final_envelope_after_fourier) ** 2
    X_1 = Vx * lamda * d
    Y_1 = Vy * lamda * d
    x1 = X_1[0, :]
    y1 = Y_1[:, 0]
    dx1 = x1[1] - x1[0]
    dy1 = y1[1] - y1[0]

    XY_2d_heatmap(ax, fig, X_1, Y_1, Intensity_after_fourier, title, cmap=cmap)

    return dx1, dy1, Intensity_after_fourier


# =============================
# Core Simulation/Analysis Functions
# =============================
def propagation_through_setup(kerr_coefficient, coords, d, L, init_pump_envelope, init_probe_envelope, probe_lambda):
    """
    Simulate the propagation of a pump and probe beam through a nonlinear medium and visualize results.
    """
    X, Y = np.meshgrid(coords[0], coords[1])  # Create a meshgrid for the coordinates
    # Visualization: Initial pump and probe
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes = axes.ravel()
    XY_2d_heatmap(axes[0], fig, X, Y, np.abs(init_pump_envelope), 'Initial Intensity', cmap='viridis')
    XY_2d_heatmap(axes[1], fig, X, Y, init_probe_envelope, 'Initial Probe Intensity', cmap='viridis')
    plt.show()

    # Propagation
    split_step = SplitStep2dWithProbe(kerr_coefficient=kerr_coefficient, num_steps=1e4, init_probe=init_probe_envelope, probe_lambda=probe_lambda)
    post_propogation = split_step.propagate(L, *coords, init_pump_envelope)
    z = post_propogation['z']
    final_intensity_pump = (np.abs(post_propogation['pump_final']) ** 2) * 100
    final_intensity_probe = (np.abs(post_propogation['probe_final']) ** 2) * 100


    # Visualization: Final pump and probe, and their far-field
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    XY_2d_heatmap(axes[0], fig, X, Y, final_intensity_pump, f'Final Intensity Pump, n2 ={kerr_coefficient}', cmap='viridis')
    XY_2d_heatmap(axes[1], fig, X, Y, final_intensity_probe, f'Final Intensity Probe', cmap='viridis')


    # update far field axes
    dx1, dy1, Intensity_after_fourier = get_far_field(axes[2], fig, split_step.lamda, d, post_propogation['pump_final'], coords, f'Intensity of pump after fourier, n2 ={kerr_coefficient}')

    # Print energy conservation for pump
    print('energy conservation of pump:')
    print(f'initial energy before propagation - {post_propogation["pump_energy"][0]}')
    print(f'final energy {post_propogation["pump_energy"][-1]}')
    print(f'final energy out of initial energy percentage - {post_propogation["pump_energy"][-1]/post_propogation["pump_energy"][0]*100} %')
    print(f'energy after fourier - {np.sum(Intensity_after_fourier * dx1 * dy1)}')
    print(f'energy after fourier out of initial energy percentage - {np.sum(Intensity_after_fourier*dx1*dy1)/post_propogation["pump_energy"][0]*100} %')

    # Plot fourier of probe envelope
    dx2, dy2, probe_intensity_after_fourier = get_far_field(axes[3], fig, probe_lambda, d, post_propogation['probe_final'], coords, f'Intensity of pump after fourier, n2 ={kerr_coefficient}')

    # Print energy conservation for probe
    print('energy conservation of probe:')
    print(f'initial energy before propagation - {post_propogation["probe_energy"][0]}')
    print(f'final energy {post_propogation["probe_energy"][-1]}')
    print(f'final energy out of initial energy percentage - {post_propogation["probe_energy"][-1] / post_propogation["probe_energy"][0] * 100} %')
    print(f'energy after fourier - {np.sum(probe_intensity_after_fourier * dx2 * dy2)}')
    print(
        f'energy after fourier out of initial energy percentage - {np.sum(probe_intensity_after_fourier * dx2 * dy2) / post_propogation["probe_energy"][0] * 100} %')
    plt.show()

# =============================
# Main Execution Block
# =============================
if __name__ == "__main__":

    # Thin Cuvette measures 1.25x1.25x4.5 cm on the outside in set up
    L = 1.25e-2 # length of the cuvette in meters
    N = 2 ** 8 # resolution of propagating envelopes, usually n=2**9
    square_width = 1.25e-3
    d = 0.2  # distance to screen - 20cm
    kerr_coefficient = -3e-10  # m^2/W

    laser_power = 1.2e1
    init_pump_envelope = get_pump_intensity(N, square_width, laser_power)  # load the pump intensity from the image

    probe_amplitude = 2.5e4
    probe_lambda = 632e-9  # probe wavelength in meters
    sigma_probe = 2e-4  # probe beam width
    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    x0 = 0
    y0 = 0
    # Create a meshgrid for the probe envelope
    X, Y = np.meshgrid(x, x)  # Create a meshgrid for the coordinates
    init_probe_envelope = probe_amplitude*np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (sigma_probe ** 2))

    propagation_through_setup(kerr_coefficient, (x, x), d, init_pump_envelope, init_probe_envelope, probe_lambda)