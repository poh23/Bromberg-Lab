#todo: graph for each n2 step size between L*10^-5 and L*10^-1, graph time ran, graph energy conservation and graph final intensity
#run for n2 = 1e-6, 1e-5, 1e-4, 1e-3, 5e-3

import numpy as np
import sys
from Numerical_Project.utils.split_step.SplitStep1d import SplitStep1d
from Numerical_Project.utils.Fresnel.Fresnel_1D.fresnel_approximation_1d import fresnel_approximation_1d
from Numerical_Project.utils.closest import closest
import matplotlib.pyplot as plt
import matplotlib
import time
import mpld3
matplotlib.use('TkAgg')

# setting path
sys.path.append('../../../')
def non_linear_propagation_of_gaussian(n2, num_steps, init_envelope, x, L):

    split_step = SplitStep1d(kerr_coefficient=n2)
    split_step.num_steps = num_steps

    start_time = time.time()
    z, total_envelope, total_energies = split_step.propagate(L, x, init_envelope)
    end_time = time.time()
    execution_time = end_time - start_time

    Intensity = np.abs(total_envelope) ** 2

    final_envelope = Intensity[-1]

    energy_conservation = total_energies[-1] - total_energies[0]

    return final_envelope, energy_conservation, execution_time


def create_graph_for_different_step_sizes(n2_values):
    #num_steps = [10, 100, 1000, 5000, 10000]
    num_steps = [2,3,4,100,200]

    L = 2e-1  # max distance because then the beam is too wide and there are edge effects
    N = 2 ** 10
    square_width = 3e-3
    sigma = 50e-6

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = np.exp(-x ** 2 / sigma ** 2)

    final_intensities = []
    energy_conservation = []
    execution_time = []

    fig = plt.figure(figsize=(15, 5))

    plt.plot(0, max(init_envelope))

    for n2 in n2_values:
        max_intensities = []
        for num_step in num_steps:
            print(f"Running for {num_step} steps")
            final_intensity, energy, time_ran = non_linear_propagation_of_gaussian(n2, num_step, init_envelope, x, L)
            final_intensities.append(final_intensity)
            energy_conservation.append(energy)
            execution_time.append(time_ran)


            max_intensity = np.max(final_intensity)
            max_intensities.append(max_intensity)

        plt.plot(num_steps, max_intensities, label=f'n2 = {n2}')
        plt.legend()
        print(f"Graph for n2 = {n2} created")


    plt.xlabel('Number of steps')
    plt.ylabel('Final Max Intensity')
    plt.legend()


    plt.tight_layout()
    plt.show()
    #mpld3.save_html(fig, f'../../graphs/interactive_graph_n2_{n2}.html')

def create_graph_for_different_step_sizes_part_2(n2):
    num_steps = [2,3,5,100,200]

    L = 3e-1  # max distance because then the beam is too wide and there are edge effects
    N = 2 ** 10
    square_width = 3e-3
    sigma = 50e-6

    x = np.linspace(-0.5 * square_width, 0.5 * square_width, N)
    init_envelope = np.exp(-x ** 2 / sigma ** 2)

    final_intensities = []
    energy_conservation = []
    execution_time = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    axes[0].plot(x, init_envelope, label=f'initial Intensity')

    X, G = fresnel_approximation_1d(init_envelope, x, L/2, 532e-9)
    axes[0].plot(X, np.abs(G)**2, label=f'Fresnel Intensity')

    for num_step in num_steps:
        print(f"Running for {num_step} steps")
        final_intensity, energy, time_ran = non_linear_propagation_of_gaussian(n2, num_step, init_envelope, x, L)
        final_intensities.append(final_intensity)
        energy_conservation.append(energy)
        execution_time.append(time_ran)
        axes[0].plot(x, final_intensity, label=f'Number of steps = {num_step}')
        axes[1].scatter(num_step, energy, label=f'Number of steps = {num_step}')
        axes[2].scatter(num_step, time_ran, label=f'Number of steps = {num_step}')


    axes[0].set_xlabel('Number of steps')
    axes[0].set_ylabel('Final Intensity')
    axes[0].set_title(f'Final Intensity when n2 = {n2}')
    axes[0].legend()

    axes[1].set_xlabel('Number of steps')
    axes[1].set_ylabel('Energy difference')
    axes[1].set_title(f'Energy Conservation when n2 = {n2}')
    axes[1].legend()

    axes[2].set_xlabel('Number of steps')
    axes[2].set_ylabel('Execution Time')
    axes[2].set_title(f'Execution Time when n2 = {n2}')
    axes[2].legend()

    plt.tight_layout()
    #mpld3.save_html(fig, f'../../graphs/interactive_graph_n2_{n2}.html')
    plt.show()
    print(f"Graph for n2 = {n2} created")

# n2_step_sizes = [1e-6, 1e-5, 1e-4, 1e-3]
n2_values = [0]
create_graph_for_different_step_sizes_part_2(0)
