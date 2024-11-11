import numpy as np
def find_sigma(x, y):
    # find index of y closest to 1/e^2 of max value
    max_val = np.max(y)
    y_closest_idx = (np.abs(y - max_val/np.e**2)).argmin()
    sigma = x[y_closest_idx]
    return sigma