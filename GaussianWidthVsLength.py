import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit

from PIL import Image


# Define a 2D Gaussian function
def gaussian_2d(coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
    x, y = coords
    return (
        offset + amplitude * np.exp(
            -(2 * ((x - x0) ** 2) / (sigma_x ** 2) + (2 * (y - y0) ** 2) / (sigma_y ** 2))
        )
    ).ravel()


# Function to fit a 2D Gaussian to an image and visualize the fit
def fit_gaussian(image, show_fit=False, filename=None):

    # Generate x and y coordinate grids
    y, x = np.indices(image.shape)

    # Initial guesses for Gaussian parameters
    initial_guess = (
        image.shape[1] // 2,  # x0 (center x)
        image.shape[0] // 2,  # y0 (center y)
        image.shape[1] // 2,  # sigma_x (width x)
        image.shape[0] // 2,  # sigma_y (width y)
        np.max(image),  # amplitude
        np.min(image),  # offset
    )

    # Fit the Gaussian model
    try:
        popt, _ = curve_fit(
            gaussian_2d, (x, y), image.ravel(), p0=initial_guess
        )

        if show_fit:
            # Recreate the Gaussian model with the fit parameters
            fit_image = gaussian_2d((x, y), *popt).reshape(image.shape)

            # Plot the original image and the Gaussian fit
            plt.figure(figsize=(12, 6))

            # Original image
            plt.subplot(1, 2, 1)
            plt.title(f"Original Image - {filename if filename else 'N/A'}")
            plt.imshow(image, cmap="gray")
            plt.colorbar(label="Pixel Intensity")
            plt.xlabel("X Pixels")
            plt.ylabel("Y Pixels")

            # Gaussian fit
            plt.subplot(1, 2, 2)
            plt.title("Gaussian Fit")
            plt.imshow(fit_image, cmap="gray")
            plt.colorbar(label="Fitted Intensity")
            plt.xlabel("X Pixels")
            plt.ylabel("Y Pixels")

            plt.tight_layout()
            plt.show()

        return popt
    except RuntimeError:
        print(f"Gaussian fitting failed for this image: {filename if filename else 'Unknown'}")
        return None


# Add Gaussian beam waist fitting
def beam_waist_model(z, w0, z0):
    wavelength_micrometer = 532 / 1000
    n = 1
    z = z
    z0_micrometer = 1000 * z0
    zR_micrometer = np.pi * np.power(w0, 2) * n / wavelength_micrometer
    return w0 * np.sqrt(1 + ((z * 1000 - z0_micrometer) / zR_micrometer) ** 2)


from matplotlib.offsetbox import AnchoredText
# Update process_folder_and_plot to include fit visualization
def process_folder_and_plot(folder_path, wavelength_nm=532):
    distances = []
    sigmas = []

    # Sort image files by propagation distance (assuming filenames encode distance)
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".tif")],
        key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))),
    )

    for file in image_files:
        # Extract the propagation distance from the filename
        distance = int(''.join(filter(str.isdigit, file.split('.')[0]))) * 25.4  # Convert inches to millimeters

        # Load the image
        image = Image.open(os.path.join(folder_path, file))
        image_data = np.array(image)

        # Fit a 2D Gaussian and visualize the fit
        popt = fit_gaussian(image_data, show_fit=False, filename=file)
        if popt is not None:
            sigma_x, sigma_y = popt[2], popt[3]
            # Convert sigma from pixels to micrometers
            sigma_x_metric = sigma_x * 4.8  # Convert to micrometers
            sigma_y_metric = sigma_y * 4.8  # Convert to micrometers
            avg_sigma = (np.abs(sigma_x_metric) + np.abs(sigma_y_metric)) / 2
            distances.append(distance)
            sigmas.append(avg_sigma)



    # Fit the beam waist model
    distances = np.array(distances)
    try:
        popt, _ = curve_fit(beam_waist_model, distances, sigmas, p0=[min(sigmas), np.mean(distances)])
        w0, z0 = popt
        n = 1
        wavelength_micrometer = wavelength_nm / 1000
        zR_mm = (np.pi * np.square(w0) * n / wavelength_micrometer) / 1000  # Rayleigh range in mm
        print(zR_mm)
        z0_mm = z0  # Focal position in mm
        print(z0_mm)
        wavelength_mm = wavelength_micrometer / 1000  # Convert wavelength to mm

        # convert things to arrays
        distances = np.array(distances, dtype=np.float64)
        sigmas = np.array(sigmas, dtype=np.float64)

        # Fit the asymptote to calculate divergence
        far_field_mask = distances > (z0_mm + zR_mm * 1.5)  # Use points far beyond the Rayleigh range
        print(f"Far-field mask: {far_field_mask}")  # Debugging
        print(distances[far_field_mask])
        divergence_fit = np.polyfit(distances[far_field_mask], sigmas[far_field_mask] / distances[far_field_mask], 1)
        divergence_rad = np.abs(divergence_fit[0])  # Ensure divergence is positive
        divergence_mrad = divergence_rad * 1000  # Convert to milliradians

        # Calculate beam quality factor M^2
        m_squared = (divergence_rad * np.pi * w0) / wavelength_micrometer

        # Create numerical details for the background text
        fit_details = (
            f"$w_0={w0:.2f} \mu m$\n"
            f"$z_0={z0_mm:.2f}$ mm\n"
            f"$z_R={zR_mm:.2f}$ mm\n"
            f"$\\theta={divergence_mrad:.2f}$ mrad\n"
            f"$M^2={m_squared:.2f}$"
        )

        print(f"Fit parameters:")
        print(f"  Beam waist (w0): {w0:.2f} μm")
        print(f"  Focal position (z0): {z0_mm:.2f} mm")
        print(f"  Rayleigh range (zR): {zR_mm:.2f} mm")
        print(f"  Divergence (asymptote): {divergence_mrad:.2f} mrad")
        print(f"  Beam quality factor (M^2): {m_squared:.2f}")
    except RuntimeError:
        print("Beam waist fitting failed.")
        popt = None

    # Plot Sigma vs. Distance with Fit
    plt.figure(figsize=(8, 6))
    plt.scatter(distances, sigmas, label="Measured Data", color="blue")
    if popt is not None:
        z_fit = np.linspace(min(distances), max(distances), 500)
        sigma_fit = beam_waist_model(z_fit, *popt)
        fit_line, = plt.plot(z_fit, sigma_fit, label="Beam Waist Fit", color="red")

        # Add legend for the fit line
        plt.legend(handles=[fit_line], loc="upper left", frameon=False)

        # Add background text for numerical details
        anchored_text = AnchoredText(fit_details, loc="upper center", prop=dict(size=10), frameon=True)
        anchored_text.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
        anchored_text.patch.set_alpha(0.8)  # Set background transparency
        plt.gca().add_artist(anchored_text)

    plt.title("Beam Waist vs. Propagation Distance")
    plt.xlabel("Propagation Distance (mm)")
    plt.ylabel("Sigma (μm)")
    plt.grid(True)
    plt.show()



# Example usage with a wavelength of 532 nm
process_folder_and_plot(r"C:\Users\ronen\OneDrive\Desktop\HW Y3 S1\Bromberg\Preliminary Experiments\manual vimba pics", wavelength_nm=532)




