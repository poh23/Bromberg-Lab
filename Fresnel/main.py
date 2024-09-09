import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift


def gaussian(x, a=1, b=0, c=1):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def envelope_function(x, d):
    return np.cos(d * x)


def analytical_ft_gaussian_with_cosine(k, a=1, b=0, c=1, d=1):
    # Original Fourier transform of Gaussian
    original_ft = a * c * np.sqrt(2 * np.pi) * np.exp(-2 * (np.pi * c * k) ** 2)
    # Apply the modulation by cosine in the frequency domain
    return 0.5 * (np.roll(original_ft, int(d / ((k[1] - k[0]) * (2 * np.pi)))) + np.roll(original_ft, -int(d / ((k[1] - k[0]) * (2 * np.pi)))))


def numerical_ft(x, f):
    n = len(x)
    dx = x[1] - x[0]
    window = np.hanning(n)
    f_windowed = f * window
    k = fftfreq(n, d=dx)
    k = fftshift(k) * (2 * np.pi)
    f_hat = fft(f_windowed)
    f_hat = fftshift(f_hat)
    return k, f_hat


def normalize_ft(ft):
    max_value = np.max(np.abs(ft))
    if max_value == 0:
        return ft  # Avoid division by zero
    else:
        return ft / max_value


def plot_expressions(x, f, k, f_analytical, f_numerical, title):
    # Normalize the Fourier Transforms
    f_analytical_normalized = normalize_ft(np.abs(f_analytical))
    f_numerical_normalized = normalize_ft(np.abs(f_numerical))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, f, label='Original Function with Envelope')
    plt.title('Original Expression with Envelope')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k, f_analytical_normalized, label='Analytical FT')
    plt.plot(k, f_numerical_normalized, 'r--', label='Numerical FT')
    plt.title('Fourier Transform (Normalized)')
    plt.xlabel('k')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    a = float(input("Enter parameter 'a' (amplitude): "))
    b = float(input("Enter parameter 'b' (mean): "))
    c = float(input("Enter parameter 'c' (standard deviation): "))
    d = float(input("Enter parameter 'd' for the envelope function cos(dx): "))

    x = np.linspace(-100, 100, 20000)

    f_original = gaussian(x, a, b, c)
    envelope = envelope_function(x, d)
    f = f_original * envelope  # Apply the envelope function

    k = fftfreq(len(x), x[1] - x[0])
    k = fftshift(k)
    f_analytical = analytical_ft_gaussian_with_cosine(k, a, b, c, d)
    k, f_numerical = numerical_ft(x, f)

    plot_expressions(x, f, k, f_analytical, f_numerical, "Gaussian Fourier Transform with Envelope")


if __name__ == "__main__":
    main()
