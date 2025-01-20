import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
k = 10  # thermal diffusivity
J_beam_heat_flux = 1.0  # beam heat flux
w_0 = 1.0  # beam radius

# Spatial and temporal parameters
r_max = 20.0  # maximum radius
dr = 0.05  # spatial step size
dt = 0.0001  # time step size
t_max = 50  # maximum time
r = np.arange(dr, r_max, dr)  # radial grid (excluding r=0 to avoid singularity)
t_steps = int(t_max / dt)  # number of time steps

# Initial condition
T = np.zeros_like(r)  # initial temperature profile

# Source term
source = J_beam_heat_flux * np.exp(-2 * r**2 / w_0**2)

# Stability condition for explicit method
if dt > dr**2 / (2 * k):
    raise ValueError("Time step is too large for stability! Reduce dt or increase dr.")

# Parameters for stopping condition
t_check = 0.1  # time after which to start checking for stability
check_index = int(t_check / dt)  # corresponding time step index
tolerance = 1e-8  # tolerance for determining stability

# Store temperature profiles for specific times
T_history = []  # Store temperature profiles at different times
T = np.zeros_like(r)  # initial temperature profile

time_snapshots = np.linspace(0, t_max, 100)  # number of time values
snapshot_indices = (time_snapshots / dt).astype(int)  # Corresponding indices

# Time-stepping loop with stopping condition
for t in range(t_steps):
    T_new = np.zeros_like(T)
    for i in range(1, len(r) - 1):
        dT_dr = (T[i + 1] - T[i - 1]) / (2 * dr)
        d2T_dr2 = (T[i + 1] - 2 * T[i] + T[i - 1]) / dr**2
        T_new[i] = T[i] + dt * (k * (d2T_dr2 + dT_dr / r[i]) + source[i])

    # Boundary conditions
    T_new[0] = T_new[1]  # Neumann BC at r=0
    T_new[-1] = 0  # Dirichlet BC at r=r_max

    # Check for stability after t_check
    # if t >= check_index:
    #     if np.all(np.abs(T_new - T) < tolerance):
    #         print(f"Stable state reached at t = {t * dt:.4f} seconds")
    #         break

    # Update temperature
    T = T_new

    # Save snapshots at specified times
    if t in snapshot_indices:
        T_history.append(T.copy())

# Define Gaussian function for fitting
def bigaussian(x, a_1, w_1, a_2, w_2):
    return a_1 * np.exp(- 2 * (x ** 2) / (w_1 ** 2)) + a_2 * np.exp(- 2 * (x ** 2) / (w_2 ** 2))

# Prepare for 3D plotting (Temperature)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each time snapshot for Temperature
for i, T_snapshot in enumerate(T_history):
    if i % 5 == 0:
        try:
            # Fit a Gaussian to the snapshot
            popt, _ = curve_fit(bigaussian, r, T_snapshot, p0=[np.max(T_snapshot), 1, 1, 1], bounds=([0, 0.1, 0, 0.1], [J_beam_heat_flux, r_max, J_beam_heat_flux, r_max]), maxfev=3000)
            print(popt)
            fitted_curve = bigaussian(r, *popt)

            # Calculate R^2 for the fit
            residuals = T_snapshot - fitted_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((T_snapshot - np.mean(T_snapshot))**2)
            r_squared = 1 - (ss_res / ss_tot)
            # Plot the snapshot
            ax.plot(r, [time_snapshots[i]] * len(r), T_snapshot,
                    label=f't={time_snapshots[i]:.2f}, R^2={r_squared:.4f}')
        except RuntimeError:
            print(f"Gaussian fit failed for snapshot at t={time_snapshots[i]:.2f}")
            ax.plot(r, [time_snapshots[i]] * len(r), T_snapshot, label=f't={time_snapshots[i]:.2f}, Fit Failed')
    else:
        # Plot the snapshot
        ax.plot(r, [time_snapshots[i]] * len(r), T_snapshot)

# Label axes
ax.set_xlabel("Radius (r)")
ax.set_ylabel("Time (t)")
ax.set_zlabel("Temperature (T)")
ax.set_title("Temperature vs Radius for Different Times")

# Show plot
plt.legend()
plt.show()

# Prepare for 3D plotting (Density)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each time snapshot for Density
for i, T_snapshot in enumerate(T_history):
    density_snapshot = -8.461834e-4 * (T_snapshot + 22) + 0.8063372  # Calculate density profile
    if i % 5 == 0:
        # Plot the snapshot
        ax.plot(r, [time_snapshots[i]] * len(r), density_snapshot, label=f't={time_snapshots[i]:.2f}')
    else:
        # Plot the snapshot
        ax.plot(r, [time_snapshots[i]] * len(r), density_snapshot)

# Label axes
ax.set_xlabel("Radius (r)")
ax.set_ylabel("Time (t)")
ax.set_zlabel("Density (g/cm³)")
ax.set_title("Density vs Radius for Different Times")

# Show plot
plt.legend()
plt.show()
