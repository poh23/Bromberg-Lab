import fft_funcs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.special import j1
import matplotlib.gridspec as gridspec

matplotlib.use('TkAgg')


# todo: 1. fourier of circ and compare to slice of jinc
def ft_and_ift_of_rect():
    N = 2048

    # Define the function f(x)
    f = lambda x: np.where((x >= -0.5) & (x <= 0.5), 1, 0)
    x = np.linspace(-1, 1, N)
    plt.figure(1)
    plt.plot(x, f(x))

    k, g = fft_funcs.fourier_transform_1d(f, x)
    plt.figure(2)
    plt.plot(k, np.real(g), label='Numerical')
    plt.plot(k, np.sinc(k), linestyle='--', label='Analytic')
    plt.xlim(-30, 30)
    plt.legend()

    y, h = fft_funcs.inverse_fourier_transform_1d(g, k)

    plt.figure(3)
    plt.plot(y, np.real(h), label='Numerical transform')
    plt.plot(x, f(x), linestyle='--', label='Analytical')
    plt.legend()

    plt.show()


def ft_of_gaussian():
    N = 5000
    mean = (2*np.pi)/5
    # mean = 0
    sigma = 1

    # Define the function f(x)
    f = lambda x: np.exp(-(x-mean)**2/(sigma**2))# *np.cos(2*np.pi*5*x)
    x = np.linspace(-50, 50, N, endpoint=False)
    dx = x[1] - x[0]
    plt.subplot(2, 3, 1)
    plt.plot(x, f(x))

    nu, g = fft_funcs.fourier_transform_1d(f, x)
    plt.subplot(2, 2, 2)
    plt.plot(nu, np.real(g), label='Numerical Real Part')
    # analytic_g = 0.5 * sigma*np.sqrt(np.pi) * np.exp(-((nu-5)**2)*(sigma**2)*np.pi**2)*np.exp(-1j*2*np.pi*nu*mean) + 0.5 * sigma*np.sqrt(np.pi) * np.exp(-((nu+5)**2)*(sigma**2)*np.pi**2)*np.exp(-1j*2*np.pi*nu*mean)
    analytic_g = sigma*np.sqrt(np.pi) * np.exp(-(nu**2)*(sigma**2)*np.pi**2)*np.exp(-1j*2*np.pi*nu*mean)
    plt.plot(nu, np.real(analytic_g), linestyle='--', label='Analytic')
    plt.legend()
    # plt.xlim(-10, 10)

    plt.subplot(2, 2, 3)
    plt.plot(nu, np.imag(g), label='Numerical Imaginary Part')
    plt.plot(nu, np.imag(analytic_g), linestyle='--', label='Analytic Imaginary Part')
    plt.legend()

    y, h = fft_funcs.inverse_fourier_transform_1d(g, nu)
    plt.subplot(2, 2, 4)
    plt.plot(y, np.real(h), label='Numerical Inverse real part')
    plt.plot(x, f(x), linestyle='--', label='Original Function')
    plt.legend()


    print(f'energy before 1d FFT - {np.sum(np.abs(f(x))**2)}')
    print(f'energy after 1d FFT - {1/(N*(dx**2)) * np.sum(np.abs(g) ** 2)}')
    print(f'energy after 1d IFFT - {np.sum(np.abs(h) ** 2)}')


    plt.show()

# ft_of_gaussian()

def ift_of_1d_gaussian():
    N = 4096
    mean = 0
    sigma = 20e3
    lamda = 532e-9
    d = 4.5e-3

    # Define the function f(x)
    nu = np.linspace(-1.5e-3, 1.5e-3, N, endpoint=False)
    g = sigma*np.sqrt(np.pi) * np.exp(-(nu**2)*(sigma**2)*np.pi**2)*np.exp(-1j*2*np.pi*nu*mean)
    H0 = np.exp(-1j * (2 * np.pi / lamda) * d)
    H = H0 * np.exp(1j * np.pi * lamda * d * nu ** 2)

    dnu = nu[1] - nu[0]

    y, h = fft_funcs.inverse_fourier_transform_1d(g*H, nu)
    h_analytic =np.exp(-(y-mean)**2/(sigma**2)) * np.exp(-1j * (2 * np.pi / lamda) * d)

    plt.figure(figsize=(12,12))
    plt.subplot(2, 1, 1)
    plt.plot(nu, np.abs(g)**2, label='Numerical Real Part')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y, np.abs(h)**2, label='Numerical Inverse real part')
    plt.plot(y, np.abs(h_analytic)**2, linestyle='--', label='Analytic Inverse real part')
    plt.legend()

    print(f'energy before 1d IFFT - {np.sum(np.abs(g)**2)}')
    print(f'energy after multiplication by transfer func - {np.sum(np.abs(g*H) ** 2)}')
    print(f'energy after 1d IFFT - {1/(N*(dnu**2))*np.sum(np.abs(h) ** 2)}')

    plt.show()

ift_of_1d_gaussian()

def ft_of_delta():
    N = 5000

    # Define the function f(x)
    f = lambda x: np.where((x >= -0.001) & (x <= 0.001), 1000, 0)
    x = np.linspace(-1, 1, N)
    plt.figure(1)
    plt.plot(x, f(x))

    k, g = fft_funcs.fourier_transform_1d(f, x)
    plt.figure(2)
    plt.plot(k, np.imag(g), label='Numerical')
    num_g = np.exp(-1j*k)
    plt.plot(k, np.imag(num_g), linestyle='--', label='Analytic')
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()


## 2D

## 2D rect
def analytical_fourier_transform_rect(VX, VY, width, height):
    # Compute the analytical solution of the 2D Fourier Transform of a rectangle
    sinc_x = 0.5 * np.sinc(width * (VX-1)) + 0.5 * np.sinc(width * (VX+1))
    sinc_y = np.sinc(height * VY)
    F = width * height * sinc_x * sinc_y  # Adjusted based on the new definition

    return F


def plot_comparison(func, x, y, width=1, height=1):
    # Perform the 2D Fourier transform
    VX, VY, G = fft_funcs.fourier_transform_2d_continuous(func, x, y)

    # Compute the analytical Fourier transform
    analytical_intensity = np.abs(analytical_fourier_transform_rect(VX, VY, width, height))

    # Perform the inverse 2D Fourier transform
    X, Y, reconstructed = fft_funcs.inverse_fourier_transform_2d(G, VX[:, 0], VY[0, :])

    # Plotting the numerical Fourier transform intensity
    plt.figure(figsize=(18, 6))

    # Numerical Solution Plot
    plt.subplot(1, 4, 1)
    plt.imshow(np.abs(G), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('Frequency $\\nu_x$')
    plt.ylabel('Frequency $\\nu_y$')
    plt.title('Numerical 2D Fourier Transform Intensity')

    # Analytical Solution Plot
    plt.subplot(1, 4, 2)
    plt.imshow(analytical_intensity, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('Frequency $\\nu_x$')
    plt.ylabel('Frequency $\\nu_y$')
    plt.title('Analytical 2D Fourier Transform Intensity')

    # Starting Function Plot
    plt.subplot(1, 4, 3)
    plt.imshow(func(X,Y), extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Amplitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Function')

    # Reconstructed Function Plot
    plt.subplot(1, 4, 4)
    plt.imshow(np.real(reconstructed), extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Amplitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Reconstructed Function from Inverse 2D FFT')

    plt.tight_layout()
    plt.show()

def fft2_of_rect_continuous():
    width = 1
    height = 1
    # Define the x and y ranges
    x = np.linspace(-5, 5, 256)
    y = np.linspace(-5, 5, 256)

    # Define the rectangular function correctly using X and Y grids
    func = lambda X, Y: np.where((np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2), 1, 0) * np.cos(2*np.pi*X)

    # Plot the Fourier transform, analytical solution, and reconstruction
    plot_comparison(func, x, y, width, height)

# Call the function to execute
# fft2_of_rect_continuous()

def ft_of_2d_gaussian(square_width=10, num_samples=500, sigma=0.6):
    mean_x = 1
    # The more sigma is smaller than the square width, the difference between the intensities is smaller
    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    func = lambda X, Y: np.exp(-((X-mean_x) ** 2 + Y ** 2) / sigma ** 2) * np.cos(20 * np.pi * X) * np.cos(4 * np.pi * Y)

    # Compute the numerical Fourier transform
    VX, VY, FFT = fft_funcs.fourier_transform_2d_continuous(func, x, x)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 12))

    # Subplot 1: Gaussian Aperture
    plt.subplot(2, 3, 1)
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1)) ** 2
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Aperture')

    # Subplot 2: Numerical Fourier Transform
    numerical_intensity = np.abs(FFT) ** 2
    plt.subplot(2, 3, 2)
    plt.imshow(numerical_intensity, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Numerical Fourier Transform in 2D')

    # Subplot 3: Analytical Fourier Transform
    G = 0.25 * (sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY+2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY+2)**2)))
    plt.subplot(2, 3, 3)
    plt.imshow(np.real(G) ** 2, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Analytic Fourier Transform in 2D')

    # Subplot 4: Difference between Numerical and Analytical
    difference = np.abs(numerical_intensity - np.abs(G) ** 2)
    plt.subplot(2, 3, 4)
    plt.imshow(difference, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='hot')
    plt.colorbar(label='Difference Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Difference: Numerical vs. Analytical')

    plt.subplot(2, 3, 5)
    plt.imshow(np.imag(FFT), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    plt.colorbar(label='Imaginary part of Numerical FFT')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Imaginary part of Numerical FFT')

    plt.subplot(2, 3, 6)
    vx = VX[0, :]
    vy = fft_funcs.closest(VY[:, 0], 10)
    plt.plot(vx, np.imag(FFT[:, vy]), label='f_hat Imaginary part at vy=10')
    plt.plot(vx, np.imag(G[:, vy]), linestyle='--', label='Analytic f_hat Imag part at vy=10')
    plt.legend()

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()


# Call the function
# ft_of_2d_gaussian()

## Choose an even number of samples!!
def ft_and_ift_2d_gaussian(square_width=20, num_samples=700, sigma=1):
    # The more sigma is smaller than the square width, the difference between the intensities is smaller
    # Generate Gaussian Aperture
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    func = lambda X, Y: np.exp(-(X ** 2 + Y ** 2) / sigma ** 2) * np.cos(4 * np.pi * X) * np.cos(2 * np.pi * Y)

    # Compute the numerical Fourier transform
    VX, VY, FFT = fft_funcs.fourier_transform_2d_continuous(func, x, x)

    vx = VX[0, :]
    vy = VY[:, 0]
    # Compute the numerical inverse Fourier transform
    IX, IY, IFFT = fft_funcs.inverse_fourier_transform_2d(FFT, vx, vy)

    # Create a single figure for all subplots
    plt.figure(figsize=(12, 12))

    # Subplot 1: Gaussian Aperture
    plt.subplot(1, 3, 1)
    X1, Y1 = np.meshgrid(x, x)
    F = np.abs(func(X1, Y1)) ** 2
    print(f'energy before FFT - {np.sum(F)}')
    plt.imshow(F, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Aperture')

    dx = x[1] - x[0]
    numerical_intensity_normalization = dx ** 4 * num_samples ** 2
    print(f'energy after FFT - {(np.sum(np.abs(FFT)**2)/numerical_intensity_normalization)}')

    # Subplot 2: Numerical Inverse Fourier Transform on Fourier transform
    numerical_intensity = np.abs(IFFT) ** 2
    print(f'energy after inverse FFT - {np.sum(numerical_intensity)}')
    print(f'energy after divided by energy before = {np.sum(np.abs(FFT)**2)/np.sum(F)}')
    plt.subplot(1, 3, 2)
    plt.imshow(numerical_intensity, extent=[IX.min(), IX.max(), IY.min(), IY.max()],
               origin='lower', cmap='gray')
    plt.colorbar(label='Intensity')
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Numerical Inverse Fourier Transform in 2D')

    # Subplot 4: Difference between Numerical and Analytical
    difference = np.abs(numerical_intensity - F)
    plt.subplot(1, 3, 3)
    plt.imshow(difference, extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
               origin='lower', cmap='hot')
    plt.colorbar(label='Difference Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Difference: Original vs. Numerical IFT of FT')

    # Show all subplots in one figure
    plt.tight_layout()
    plt.show()

# Call the function
# ft_and_ift_2d_gaussian()

def ft_gaussian_w_wo_windowing(square_width=10, num_samples=1000, sigma=1):
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    func = lambda X, Y: np.exp(-(X ** 2 + Y ** 2) / sigma ** 2) * np.cos(10 * 2 * np.pi * X) * np.cos(2 * 2 * np.pi * Y)

    X, Y = np.meshgrid(x, x)
    f = func(X, Y)

    # Create Hanning windows for both dimensions
    hanning_x = np.hanning(len(x))
    hanning_y = np.hanning(len(x))
    hanning_2d = np.outer(hanning_x, hanning_y)  # Create the 2D Hanning window

    # Apply the 2D Hanning window to the function
    f_windowed = f * hanning_2d

    # Compute the numerical Fourier transform Windowed and not Windowed
    VX, VY, g_orig = fft_funcs.fourier_transform_2d_continuous(func, x, x, window=False)
    VX, VY, g = fft_funcs.fourier_transform_2d_continuous(func, x, x, window=True)

    plt.figure(figsize=(6, 12))

    # Subplot 1: f
    plt.subplot(2, 3, 1)
    plt.imshow(f, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original f')

    # Subplot 2: f_windowed
    plt.subplot(2, 3, 2)
    plt.imshow(f_windowed, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Windowed f')

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(f_windowed - f), extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Diff of f and windowed f')

    # Subplot 4: g original
    plt.subplot(2, 3, 4)
    plt.imshow(np.abs(g_orig), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Original g')

    # Subplot 5: g windowed
    plt.subplot(2, 3, 5)
    plt.imshow(np.abs(g), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Windowed g')

    plt.subplot(2, 3, 6)
    plt.imshow(np.abs(g - g_orig), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Diff of g and windowed g')

    # generate graph of slices
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(x, f[:,np.where(x == 0)[0]], label='F at y=0')
    plt.plot(x, f_windowed[:,np.where(x == 0)[0]], linestyle='--', label='Windowed F at y=0')

    plt.subplot(2, 1, 2)

    vx = VX[0, :]
    vy = fft_funcs.closest(VY[:, 0], 10)
    # plt.plot(VX[:,0], np.abs(g_orig[closest(VY[0,:], 10), :])**2, label='f_hat at vx=10')
    plt.plot(vx, np.log(np.abs(g_orig[:, vy]) ** 2), label='f_hat at vy=10')
    # plt.plot(VX[:,0], np.abs(g[closest(VY[0,:], 10), :])**2, linestyle='--', label='Windowed f_hat at vx=10')
    # Subplot 3: Analytical Fourier Transform
    g_analytic = 0.25 * (sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY-2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX-10)**2 + (VY+2)**2)) +
                sigma**2 * np.pi * np.exp(-np.pi**2 * sigma**2 * ((VX+10)**2 + (VY+2)**2)))
    plt.plot(vx, np.log(np.abs(g_analytic[:, vy]) ** 2), linestyle='--', label='Analytic f_hat at vx=10')

    plt.legend()
    plt.show()


# gaussian_w_wo_windowing()

def ft_and_ift_2d_circ(square_width=10, num_samples=1000, r=0.5):
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    func = lambda X, Y: np.where(np.sqrt(X**2+Y**2) <= r, 1, 0)

    X, Y = np.meshgrid(x, x)
    f = func(X, Y)
    print(f'Energy before fourier is: {np.sum(np.abs(f) ** 2)}')

    # Compute the numerical Fourier transform Windowed and not Windowed
    VX, VY, FFT = fft_funcs.fourier_transform_2d_continuous(func, x, x, window=False)
    print(f'Energy after fourier is: {np.sum(np.abs(FFT) ** 2)}')

    plt.figure(figsize=(14,21))
    gs = gridspec.GridSpec(3, 2)
    # Subplot 1: f
    plt.subplot(gs[0,0:1])
    plt.imshow(f, extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('F')


    # Subplot 2: FFT
    plt.subplot(gs[0,1:2])
    plt.imshow(np.abs(FFT)**2, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Numerical FFT')

    # Subplot 3: Analytical FFT
    plt.subplot(gs[1,0:1])
    I = np.pi * sombrero(VX,VY, r)
    plt.imshow(np.abs(I)**2, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Analytical FFT')

    # Subplot 4: Analytical FFT vs. Numerical FFT diff
    plt.subplot(gs[1,1:2])
    I = np.pi * sombrero(VX, VY, r)
    plt.imshow(np.abs(np.abs(I) ** 2 - np.abs(FFT)**2), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Analytical FFT vs. Numerical FFT Diff')

    # subplot 5:
    vx = VX[0,:]
    vy = fft_funcs.closest(VY[:,0], 0)
    plt.subplot(gs[2,0:2])
    plt.plot(vx, np.log(np.abs(FFT[:, vy]) ** 2), label='Numerical FFT at vy=0')
    plt.plot(vx, np.log(np.abs(I[:, vy]) ** 2), linestyle='--', label='Analytical f_hat at vy=0')
    plt.title('Slice of Numeric vs. Analytic FFT')
    plt.legend()
    plt.show()

def sombrero(X,Y, r):
    return 2*(r**2)*j1(2 * np.pi * np.sqrt(X**2 + Y**2) * r)/(2 * r * np.pi * np.sqrt(X**2 + Y**2))

# ft_and_ift_2d_circ(r=0.5)
