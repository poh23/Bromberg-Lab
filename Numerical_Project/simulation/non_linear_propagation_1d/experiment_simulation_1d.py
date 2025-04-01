import numpy as np
import sys
from Numerical_Project.utils.split_step.SplitStep1d import SplitStep1d
from Numerical_Project.utils.Fresnel.Fresnel_1D.fresnel_approximation_1d import fresnel_approximation_1d
from Numerical_Project.utils.FFT.FFT_1D.fourier_transform_1d import fourier_transform_1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# setting path
sys.path.append('../../../')
def propagation_through_setup():
    cuvette_length = 1.25e-3 # 1.25 cm
    cuvette_width = 1.25e-2 # 1.25 cm
    L = 0.05 # 20 cm
    N = 2 ** 10
    square_width = 3e-3
    # this is with static gaussian beam, without propagation in z and power
    sigma = 50e-6 # 50 micrometers
    split_step = SplitStep1d(kerr_coefficient=-3e-2)
    split_step.data_save_rate = 50
    split_step.num_steps = 5000
    # we want the multiplaction on n2 and I yo be bigger than pi
    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = 4*np.exp(-x ** 2 / sigma ** 2)

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    z, total_envelope_linear, total_energies_linear = split_step.propagate(cuvette_length, x, init_envelope)
    Intensity_linear = np.abs(total_envelope_linear) ** 2
    ax1 = split_step.graph_propagation(ax1, fig, x, z, Intensity_linear, f'Propagation of Gaussian beam right after cuvette, n2 = {split_step.kerr_coefficient}')

    ax2.plot(x, Intensity_linear[-1], label=f'Final Intensity w/o fourier after cuvette, n2 = {split_step.kerr_coefficient}')

    nu, final_envelope_after_fourier = fourier_transform_1d(total_envelope_linear[-1], x)
    Intensity_after_fourier = np.abs(final_envelope_after_fourier) ** 2
    ax3.title.set_text(f'Intensity after fourier after cuvette, n2 = {split_step.kerr_coefficient}')
    ax3.plot(nu, Intensity_after_fourier, label=f'Final Intensity w fourier after cuvette, n2 = {split_step.kerr_coefficient}')

    #X, final_envelope_after_fresnel = fresnel_approximation_1d(x, total_envelope_linear[-1], d=L, lamda=split_step.lamda)
    #Intensity_after_fresnel = np.abs(final_envelope_after_fresnel) ** 2
    ax4.plot(x,np.angle(total_envelope_linear[-1]), label=f'Final Intensity w fresnel after cuvette, L = {L}m, n2 = {split_step.kerr_coefficient}')

    plt.show()

propagation_through_setup()