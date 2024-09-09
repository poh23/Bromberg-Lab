import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, fftfreq


# Define 2D Gaussian
def gaussian_2d(x, y, a=1, b=0, c=1):
    return a * np.exp(-((x - b) ** 2 + (y - b) ** 2) / (2 * c ** 2))


# Define 2D cosine modulation
def envelope_function_2d(x, y, d_x, d_y):
    return np.cos(d_x * x) * np.cos(d_y * y)


# Analytical Fourier Transform of a 2D Gaussian with cosine modulation
def analytical_ft_gaussian_with_cosine_2d(k_x, k_y, a=1, b=0, c=1, d_x=1, d_y=1):
    # Original 2D Gaussian Fourier Transform (centered at zero frequency)
    original_ft = a * c ** 2 * (2 * np.pi) * np.exp(-2 * (np.pi ** 2) * c ** 2 * (k_x ** 2 + k_y ** 2))

    # Shifted Fourier Transforms due to cosine modulation in both x and y directions
    shifted_ft_x_pos = np.exp(-2 * (np.pi ** 2) * c ** 2 * ((k_x - d_x) ** 2 + k_y ** 2))
    shifted_ft_x_neg = np.exp(-2 * (np.pi ** 2) * c ** 2 * ((k_x + d_x) ** 2 + k_y ** 2))

    shifted_ft_y_pos = np.exp(-2 * (np.pi ** 2) * c ** 2 * (k_x ** 2 + (k_y - d_y) ** 2))
    shifted_ft_y_neg = np.exp(-2 * (np.pi ** 2) * c ** 2 * (k_x ** 2 + (k_y + d_y) ** 2))

    # Combine shifts along both x and y directions
    combined_ft = 0.25 * (shifted_ft_x_pos + shifted_ft_x_neg + shifted_ft_y_pos + shifted_ft_y_neg)

    # Return the final Fourier Transform including the modulation
    return original_ft * combined_ft


# Perform numerical 2D Fourier Transform
def numerical_ft_2d(f):
    f_hat = fft2(f)
    f_hat = fftshift(f_hat)  # Shift zero frequency to the center
    return f_hat


# Normalize the Fourier Transform
def normalize_ft(ft):
    max_value = np.max(np.abs(ft))
    if max_value == 0:
        return ft  # Avoid division by zero
    else:
        return ft / max_value


# Plot the results
def plot_expressions_2d(x, y, f, k_x, k_y, f_analytical, f_numerical, title):
    # Normalize the Fourier Transforms
    f_analytical_normalized = normalize_ft(np.abs(f_analytical))
    f_numerical_normalized = normalize_ft(np.abs(f_numerical))

    # Compute the absolute difference between the analytical and numerical FTs
    difference = np.abs(f_analytical_normalized - f_numerical_normalized)

    plt.figure(figsize=(16, 8))

    # Plot original function
    plt.subplot(2, 2, 1)
    original_contour = plt.contourf(x, y, f, cmap='viridis')
    plt.colorbar(original_contour)
    plt.title('Original 2D Function with Envelope')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot analytical Fourier Transform
    plt.subplot(2, 2, 2)
    analytical_contour = plt.contourf(k_x, k_y, f_analytical_normalized, cmap='viridis')
    plt.colorbar(analytical_contour)
    plt.title('Analytical FT (Normalized)')
    plt.xlabel('k_x')
    plt.ylabel('k_y')

    # Plot numerical Fourier Transform
    plt.subplot(2, 2, 3)
    numerical_contour = plt.contourf(k_x, k_y, f_numerical_normalized, cmap='plasma')
    plt.colorbar(numerical_contour)
    plt.title('Numerical FT (Normalized)')
    plt.xlabel('k_x')
    plt.ylabel('k_y')

    # Plot difference between analytical and numerical FTs
    plt.subplot(2, 2, 4)
    difference_contour = plt.contourf(k_x, k_y, difference, cmap='inferno')
    plt.colorbar(difference_contour)
    plt.title('Difference (Analytical - Numerical)')
    plt.xlabel('k_x')
    plt.ylabel('k_y')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # User inputs for the parameters
    a = float(input("Enter parameter 'a' (amplitude): "))
    b = float(input("Enter parameter 'b' (mean): "))
    c = float(input("Enter parameter 'c' (standard deviation): "))
    d_x = float(input("Enter parameter 'd_x' for the envelope function cos(d_x * x): "))
    d_y = float(input("Enter parameter 'd_y' for the envelope function cos(d_y * y): "))

    # Define grid
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)

    # Create 2D Gaussian modulated by the envelope function
    f_original = gaussian_2d(X, Y, a, b, c)
    envelope = envelope_function_2d(X, Y, d_x, d_y)
    f = f_original * envelope  # Apply the 2D envelope function

    # Frequency domain setup
    k_x = fftfreq(len(x), x[1] - x[0])
    k_y = fftfreq(len(y), y[1] - y[0])
    k_x, k_y = np.meshgrid(k_x, k_y)
    k_x = fftshift(k_x)
    k_y = fftshift(k_y)

    # Perform analytical and numerical 2D Fourier Transforms
    f_analytical = analytical_ft_gaussian_with_cosine_2d(k_x, k_y, a, b, c, d_x, d_y)
    f_numerical = numerical_ft_2d(f)

    # Plot the results
    plot_expressions_2d(x, y, f, k_x, k_y, f_analytical, f_numerical, "2D Gaussian Fourier Transform with Envelope")


if __name__ == "__main__":
    main()
