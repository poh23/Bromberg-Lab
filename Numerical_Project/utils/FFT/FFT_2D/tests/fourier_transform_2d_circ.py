import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from scipy.special import j1
import matplotlib.gridspec as gridspec

# setting path
sys.path.append('../../../')
from Numerical_Project.utils.FFT.FFT_2D.fourier_transform_2d import fourier_transform_2d
from Numerical_Project.utils.closest import closest

matplotlib.use('TkAgg')
def ft_and_ift_2d_circ(square_width=10, num_samples=1000, r=0.5):
    x = np.linspace(-square_width / 2, square_width / 2, num_samples, endpoint=False)
    X, Y = np.meshgrid(x, x)
    f = np.where(np.sqrt(X**2+Y**2) <= r, 1, 0)


    print(f'Energy before fourier is: {np.sum(np.abs(f) ** 2)}')

    # Compute the numerical Fourier transform Windowed and not Windowed
    VX, VY, FFT = fourier_transform_2d(f, x, x)
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


    # Subplot 2: FFT_1D
    plt.subplot(gs[0,1:2])
    plt.imshow(np.abs(FFT)**2, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Numerical FFT_1D')

    # Subplot 3: Analytical FFT_1D
    plt.subplot(gs[1,0:1])
    I = np.pi * sombrero(VX,VY, r)
    plt.imshow(np.abs(I)**2, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Analytical FFT_1D')

    # Subplot 4: Analytical FFT_1D vs. Numerical FFT_1D diff
    plt.subplot(gs[1,1:2])
    I = np.pi * sombrero(VX, VY, r)
    plt.imshow(np.abs(np.abs(I) ** 2 - np.abs(FFT)**2), extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap='viridis')
    cbar = plt.colorbar(label='Intensity')
    cbar.ax.locator_params(nbins=16)
    plt.xlabel('VX')
    plt.ylabel('VY')
    plt.title('Analytical FFT_1D vs. Numerical FFT_1D Diff')

    # subplot 5:
    vx = VX[0,:]
    vy = closest(VY[:,0], 0)
    plt.subplot(gs[2,0:2])
    plt.plot(vx, np.log(np.abs(FFT[:, vy]) ** 2), label='Numerical FFT_1D at vy=0')
    plt.plot(vx, np.log(np.abs(I[:, vy]) ** 2), linestyle='--', label='Analytical f_hat at vy=0')
    plt.title('Slice of Numeric vs. Analytic FFT_1D')
    plt.legend()
    plt.show()

def sombrero(X,Y, r):
    return 2*(r**2)*j1(2 * np.pi * np.sqrt(X**2 + Y**2) * r)/(2 * r * np.pi * np.sqrt(X**2 + Y**2))

ft_and_ift_2d_circ(r=0.5)