import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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