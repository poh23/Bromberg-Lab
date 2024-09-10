import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, fftfreq, ifft2


# Define 2D Gaussian
def gaussian_2d(x, y, a=1, b=0, c=1):
    gaussian = a * np.exp(-((x - b) ** 2 + (y - b) ** 2) / (2 * c ** 2))
    return gaussian


# Define 2D cosine modulation
def envelope_function_2d(x, y, d_x, d_y):
    envelope_cos = np.cos(d_x * x) * np.cos(d_y * y)
    return envelope_cos


# Analytical Fourier Transform of a 2D Gaussian with cosine modulation
def analytical_ft_gaussian_with_cosine_2d(k_x, k_y, a=1, b=0, c=1, d_x=1, d_y=1):
    # Fourier Transform of the unmodulated Gaussian
    gaussian_ft = a * c ** 2 * (2 * np.pi) * np.exp(-2 * (np.pi ** 2) * c ** 2 * (k_x ** 2 + k_y ** 2))
    d_x = d_x
    d_y = d_y
    # Fourier Transform with cosine modulation shifts
    shift1 = np.exp(-2 * (np.pi ** 2) * c ** 2 * ((k_x - d_x) ** 2 + (k_y - d_y) ** 2))
    shift2 = np.exp(-2 * (np.pi ** 2) * c ** 2 * ((k_x + d_x) ** 2 + (k_y - d_y) ** 2))
    shift3 = np.exp(-2 * (np.pi ** 2) * c ** 2 * ((k_x - d_x) ** 2 + (k_y + d_y) ** 2))
    shift4 = np.exp(-2 * (np.pi ** 2) * c ** 2 * ((k_x + d_x) ** 2 + (k_y + d_y) ** 2))

    # Combine the shifted components
    combined_ft = 0.25 * (shift1 + shift2 + shift3 + shift4)

    # Return the final Fourier Transform
    return gaussian_ft * combined_ft


# Perform numerical 2D Fourier Transform
def numerical_ft_2d(f):
    f_hat = fft2(f)
    f_hat = fftshift(f_hat)  # Shift zero frequency to the center
    f_hat /= f.shape[0] * f.shape[1]  # Normalize by the number of samples in both dimensions
    return f_hat


# Perform numerical 2D Inverse Fourier Transform
def numerical_ifft_2d(f_hat):
    f_hat = fftshift(f_hat)  # Shift back the zero frequency
    f_reconstructed = ifft2(f_hat)
    f_reconstructed *= f_hat.shape[0] * f_hat.shape[1]  # Reverse normalization
    return np.real(f_reconstructed)


# Normalize the Fourier Transform
def normalize_ft(ft):
    max_value = np.max(np.abs(ft))
    return ft / max_value if max_value != 0 else ft


# Plot the results
def plot_expressions_2d(x, y, f, f_reconstructed, k_x, k_y, f_analytical, f_numerical, title):
    f_analytical_normalized = normalize_ft(np.abs(f_analytical))
    f_numerical_normalized = normalize_ft(np.abs(f_numerical))
    difference_ft = np.abs(f_analytical_normalized - f_numerical_normalized)
    difference_reconstruction = np.abs(f - f_reconstructed)

    plt.figure(figsize=(16, 8))

    # Plot original function
    plt.subplot(2, 3, 1)
    original_contour = plt.contourf(x, y, f, cmap='viridis')
    plt.colorbar(original_contour)
    plt.title('Original 2D Function with Envelope')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot analytical Fourier Transform
    plt.subplot(2, 3, 2)
    analytical_contour = plt.contourf(k_x, k_y, f_analytical_normalized, cmap='viridis')
    plt.colorbar(analytical_contour)
    plt.title('Analytical FT')
    plt.xlabel('k_x')
    plt.ylabel('k_y')

    # Plot numerical Fourier Transform
    plt.subplot(2, 3, 3)
    numerical_contour = plt.contourf(k_x, k_y, f_numerical_normalized, cmap='plasma')
    plt.colorbar(numerical_contour)
    plt.title('Numerical FT')
    plt.xlabel('k_x')
    plt.ylabel('k_y')

    # Plot difference between analytical and numerical FTs
    plt.subplot(2, 3, 4)
    difference_contour = plt.contourf(k_x, k_y, difference_ft, cmap='inferno')
    plt.colorbar(difference_contour)
    plt.title('Difference (Analytical - Numerical)')
    plt.xlabel('k_x')
    plt.ylabel('k_y')

    # Plot inverse-transformed function (reconstructed)
    plt.subplot(2, 3, 5)
    reconstructed_contour = plt.contourf(x, y, f_reconstructed, cmap='viridis')
    plt.colorbar(reconstructed_contour)
    plt.title('Reconstructed (Inverse FFT)')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot difference between original and reconstructed function
    plt.subplot(2, 3, 6)
    difference_reconstruction_contour = plt.contourf(x, y, difference_reconstruction, cmap='inferno')
    plt.colorbar(difference_reconstruction_contour)
    plt.title('Difference (Original - Reconstructed)')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    gaussian_amplitude = 1
    gaussian_mean = 0
    gaussian_std = 0.6
    cos_freq_x = 10
    cos_freq_y = 2

    x = np.linspace(-100, 100, 5000)
    y = np.linspace(-100, 100, 5000)
    X, Y = np.meshgrid(x, y)

    f_original = gaussian_2d(X, Y, gaussian_amplitude, gaussian_mean * (2 * np.pi), gaussian_std * (2 * np.pi))
    envelope_cos = envelope_function_2d(X, Y, cos_freq_x, cos_freq_y)
    f = f_original * envelope_cos  # Apply the 2D envelope function

    k_x = fftfreq(len(x), x[1] - x[0])
    k_y = fftfreq(len(y), y[1] - y[0])
    k_x, k_y = np.meshgrid(k_x, k_y)
    k_x = fftshift(k_x) * (2 * np.pi)
    k_y = fftshift(k_y) * (2 * np.pi)

    # Perform analytical and numerical Fourier Transforms
    f_analytical = analytical_ft_gaussian_with_cosine_2d(k_x / 2, k_y / 2, gaussian_amplitude, gaussian_mean * np.sqrt(2), gaussian_std * np.sqrt(2), cos_freq_x, cos_freq_y)
    f_numerical = numerical_ft_2d(f)

    # Perform the inverse Fourier Transform on the numerical result
    f_reconstructed = numerical_ifft_2d(f_numerical)

    # Plot everything
    plot_expressions_2d(x, y, f, f_reconstructed, k_x, k_y, f_analytical, f_numerical,
                        "2D Gaussian Fourier Transform with Cosine Envelope")


if __name__ == "__main__":
    main()
