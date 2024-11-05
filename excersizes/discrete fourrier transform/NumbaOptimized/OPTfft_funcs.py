from scipy.fft import fft, fftfreq, ifft, fftshift, fft2, ifftshift, ifft2
import numpy as np
import numba

@numba.njit
def closest(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx

def fourier_transform_1d(f, x):
    x0, dx = x[0], x[1] - x[0]
    G = fftshift(fft(fftshift(f)))
    nu = fftshift(fftfreq(f.size, dx))
    G *= dx
    return nu, G

def inverse_fourier_transform_1d(func, v):
    dv = v[1] - v[0]
    func_unshifted = ifftshift(func)
    f_reconstructed = ifft(func_unshifted) * len(v) * dv
    x = np.linspace(-0.5 / dv, 0.5 / dv, len(v))
    f_reconstructed = fftshift(f_reconstructed)
    return x, f_reconstructed

def fourier_transform_2d_continuous(func, x, y, window=False):
    x0, dx = x[0], x[1] - x[0]
    y0, dy = y[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    f = func(X, Y)

    if window:
        hanning_x = np.hanning(len(x))
        hanning_y = np.hanning(len(y))
        hanning_2d = np.outer(hanning_x, hanning_y)
        f = f * hanning_2d

    g = fft2(fftshift(f))
    vx = fftshift(fftfreq(f.shape[0], dx))
    vy = fftshift(fftfreq(f.shape[1], dy))
    VX, VY = np.meshgrid(vx, vy)
    g = fftshift(g) * dx * dy
    return VX, VY, g

def inverse_fourier_transform_2d(func, vx, vy):
    dvx = vx[1] - vx[0]
    dvy = vy[1] - vy[0]
    func_unshifted = ifftshift(ifftshift(func, axes=0), axes=1)
    f_reconstructed = ifft2(func_unshifted) * len(vx) * len(vy) * dvx * dvy
    x = np.linspace(-0.5 / dvx, 0.5 / dvx, len(vx))
    y = np.linspace(-0.5 / dvy, 0.5 / dvy, len(vy))
    x, y = np.meshgrid(x, y)
    f_reconstructed = fftshift(fftshift(f_reconstructed, axes=0), axes=1)
    return x, y, f_reconstructed
