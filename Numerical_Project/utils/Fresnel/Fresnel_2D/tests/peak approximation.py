# Correcting the missing import and re-running the code
import math
import matplotlib.pyplot as plt
import numpy as np

# Define constants
L = 5  # Change as needed
N = 1  # Change as needed
l = 3e-2  # Length parameter

# Initialize range for dz
dz_values = np.logspace(math.log(l) * 2, math.log(l), 1000)  # 100 points from 1e-6 to 1e-3
final_amplitudes = []


# Iterate over dz values
for dz in dz_values:
    amplitude = 1.0
    num_iterations = int(l / dz)
    for _ in range(num_iterations):
        amplitude *= 1 - L * dz + N * dz ** 2
    final_amplitudes.append(amplitude)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(dz_values, final_amplitudes, marker='o', linestyle='-', markersize=4)
plt.xscale('log')  # Logarithmic scale for dz
plt.xlabel('dz (log scale)')
plt.ylabel('Final Amplitude')
plt.title('Final Amplitude vs. dz')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
