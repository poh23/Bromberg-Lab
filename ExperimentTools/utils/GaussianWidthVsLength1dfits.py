import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit


# Define a 1D Gaussian function
def gaussian_1d(x, x0, sigma, amplitude, offset):
    return offset + np.abs(amplitude) * np.exp(-2 * ((x - x0) / sigma) ** 2)


# Function to fit a 1D Gaussian to a profile and calculate the waist
def fit_1d_gaussian(profile, show_fit=False):
    x = np.arange(len(profile))
    initial_guess = [len(profile) // 2, len(profile) // 4, np.max(profile), np.min(profile)]

    try:
        popt, _ = curve_fit(gaussian_1d, x, profile, p0=initial_guess)

        if show_fit:
            # Plot the profile and the Gaussian fit
            plt.figure(figsize=(8, 6))
            plt.plot(x, profile, label="Data", color="blue")
            plt.plot(x, gaussian_1d(x, *popt), label="Gaussian Fit", color="red", linestyle="--")
            plt.xlabel("Position")
            plt.ylabel("Intensity")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Extract the beam waist (sigma) and other parameters
        x0, sigma, amplitude, offset = popt
        return sigma, amplitude, x0
    except RuntimeError:
        print("1D Gaussian fitting failed.")
        return None, None, None


# Function to process an image and fit 1D Gaussians along x=0 and y=0 slices
def analyze_slices(image, show_fit=False):
    # Find the centers of the Gaussian along both axes (x and y)
    center_x = np.argmax(np.max(image, axis=0))  # Get the center along x
    center_y = np.argmax(np.max(image, axis=1)) # Get the center along y

    # Slice along the fitted x center (vertical slice)
    profile_x = image[:, int(center_x)]

    # Slice along the fitted y center (horizontal slice)
    profile_y = image[int(center_y), :]

    # Fit 1D Gaussians to both slices
    sigma_x, amplitude_x, x0_x = fit_1d_gaussian(profile_x, show_fit)
    sigma_y, amplitude_y, x0_y = fit_1d_gaussian(profile_y, show_fit)

    return sigma_x, sigma_y

def beam_waist_model(z, w0, z0):
    wavelength_micrometer = 532 / 1000
    n = 1
    z = z
    z0_micrometer = 1000 * z0
    zR_micrometer = np.pi * np.power(w0, 2) * n / wavelength_micrometer
    return w0 * np.sqrt(1 + ((z * 1000 - z0_micrometer) / zR_micrometer) ** 2)


# Update process_folder_and_plot to include analysis for each slice
def process_folder_and_plot(folder_path, wavelength_nm=532):
    distances = []
    sigmas_x = []
    sigmas_y = []

    # Sort image files by propagation distance (assuming filenames encode distance)
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".tif")],
        key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))),
    )

    for file in image_files:
        # Extract the propagation distance from the filename
        distance = int(''.join(filter(str.isdigit, file.split('.')[0]))) * 10  # Convert cm to mm

        # Load the image
        image = Image.open(os.path.join(folder_path, file))
        image_data = np.array(image)

        # Analyze slices (x=0 and y=0)
        sigma_x, sigma_y = analyze_slices(image_data, show_fit=True)

        if sigma_x is not None and sigma_y is not None:
            # Convert sigma from pixels to micrometers
            sigma_x_metric = sigma_x * 4.8  # Convert to micrometers
            sigma_y_metric = sigma_y * 4.8  # Convert to micrometers

            sigmas_x.append(sigma_x_metric)
            sigmas_y.append(sigma_y_metric)
            distances.append(distance)

    # Calculate beam waist and Rayleigh range for each slice (x=0 and y=0)
    sigmas_x = np.array(sigmas_x)
    sigmas_y = np.array(sigmas_y)
    distances = np.array(distances)

    wavelength_micrometer = 532 / 1000

    # Fit the beam waist model for both slices
    try:
        # Fit for x direction (horizontal slice)
        popt_x, _ = curve_fit(beam_waist_model, distances, sigmas_x, p0=[min(sigmas_x), np.mean(distances)])
        w0_x, z0_x = popt_x
        # Fit for y direction (vertical slice)
        popt_y, _ = curve_fit(beam_waist_model, distances, sigmas_y, p0=[min(sigmas_y), np.mean(distances)])
        w0_y, z0_y = popt_y

        print(f"Results for x slice:")
        print(f"  Beam waist (w0_x): {w0_x:.2f} μm")
        print(f"  Focal position (z0_x): {z0_x:.2f} mm")

        print(f"Results for y slice:")
        print(f"  Beam waist (w0_y): {w0_y:.2f} μm")
        print(f"  Focal position (z0_y): {z0_y:.2f} mm")

        # Fit the asymptote to calculate divergence
        far_field_mask = distances > 400  # > (z0_mm + zR_mm * 1.5)  # Use points far beyond the Rayleigh range
        print(f"Far-field mask: {far_field_mask}")  # Debugging
        print(distances[far_field_mask])
        divergence_fit = np.polyfit(distances[far_field_mask], sigmas_x[far_field_mask] / distances[far_field_mask], 1)
        divergence_rad = np.abs(divergence_fit[0])  # Ensure divergence is positive
        divergence_mrad = divergence_rad * 1000  # Convert to milliradians

        # Calculate beam quality factor M^2
        m_squared = (divergence_rad * np.pi * w0_x) / wavelength_micrometer
        theta_0 = 1 / ((np.pi * w0_x) / wavelength_micrometer)
        print(f"Theta_0: {theta_0:.2f} rad")

        # Create numerical details for the background text
        fit_details = (
            f"$w_0={w0_x:.2f} \mu m$\n"
            # f"$z_0={z0_mm:.2f}$ mm\n"
            # f"$z_R={zR_mm:.2f}$ mm\n"
            f"$\\theta={divergence_mrad:.2f}$ mrad\n"
            f"$M^2={m_squared:.2f}$"
        )

        print(f"Fit parameters:")
        print(f"  Beam waist (w0): {w0_x:.2f} μm")
        # print(f"  Focal position (z0): {z0_mm:.2f} mm")
        # print(f"  Rayleigh range (zR): {zR_mm:.2f} mm")
        print(f"  Divergence (asymptote): {divergence_mrad:.2f} mrad")
        print(f"  Beam quality factor (M^2): {m_squared:.2f}")

    except RuntimeError:
        print("Beam waist fitting failed.")
        popt_x, popt_y = None, None

    # Plot Sigma vs. Distance for both slices
    plt.figure(figsize=(8, 6))
    plt.scatter(distances, sigmas_x, label="Sigma (x slice)", color="blue")
    plt.scatter(distances, sigmas_y, label="Sigma (y slice)", color="green")

    if popt_x is not None:
        z_fit = np.linspace(min(distances), max(distances), 500)
        sigma_fit_x = beam_waist_model(z_fit, *popt_x)
        plt.plot(z_fit, sigma_fit_x, label="Beam Waist Fit (x slice)", color="red", linestyle="--")

    if popt_y is not None:
        sigma_fit_y = beam_waist_model(z_fit, *popt_y)
        plt.plot(z_fit, sigma_fit_y, label="Beam Waist Fit (y slice)", color="orange", linestyle="--")

    plt.title("Beam Waist vs. Propagation Distance")
    plt.xlabel("Propagation Distance (mm)")
    plt.ylabel("Sigma (μm)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage with a wavelength of 532 nm
process_folder_and_plot(r"C:\Users\OWNER\Desktop\Eyal and Maya\BeamWidthMeasurements\folder", wavelength_nm=532)
