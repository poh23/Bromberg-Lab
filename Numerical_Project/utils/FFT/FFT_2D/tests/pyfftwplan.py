import pyfftw
import numpy as np
import pickle

# Example sizes for 2D FFT
size = (1024, 1024)

# Generate plans for 2D FFT sizes
a = pyfftw.empty_aligned(size, dtype='complex128')

ar, ai = np.random.randn(2, 1024, 1024)
a[:] = ar + (1j * ai)


fft_object = pyfftw.builders.fft(a)
b = fft_object()

# Save wisdom to file
wisdom = pyfftw.export_wisdom()
# Save to a file
with open("../../../../simulation/non_linear_propagation_2d/fft_wisdom_2d.pkl", "wb") as f:
    pickle.dump(wisdom, f)