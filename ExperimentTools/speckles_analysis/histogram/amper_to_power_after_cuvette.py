import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_power_vs_current_with_fits(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter data up to 7A for fitting
    mask = df['A'] <= 7
    x_fit = df.loc[mask, 'A']
    y_fit = df.loc[mask, 'power (W) (take2)']

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Plot data with error bars
    plt.errorbar(df['A'], df['power (W) (take2)'],
                 yerr=df['+-'],
                 fmt='o',
                 capsize=5,
                 capthick=1,
                 markersize=8,
                 color='blue',
                 label='Experimental Data')

    # Generate smooth x values for plotting fits
    x_smooth = np.linspace(0, 7, 100)

    # Colors for different polynomial fits
    colors = ['red', 'green', 'purple', 'orange']

    # Fit polynomials of different degrees
    for degree, color in zip(range(2, 6), colors):
        # Fit polynomial
        coeffs = np.polyfit(x_fit, y_fit, degree)
        y_smooth = np.polyval(coeffs, x_smooth)

        # Create polynomial equation string
        eq = f'y = '
        for i, coef in enumerate(coeffs):
            power = degree - i
            if power > 1:
                eq += f'{coef:.4f}x^{power} + '
            elif power == 1:
                eq += f'{coef:.4f}x + '
            else:
                eq += f'{coef:.4f}'

        # Plot the fit
        plt.plot(x_smooth, y_smooth, '--',
                 color=color,
                 label=f'Degree {degree}: {eq}')
        print(f'Polynomial fit (degree {degree}): {eq}')

    # Customize the plot
    plt.xlabel('Current (A)')
    plt.ylabel('Power (W)')
    plt.title('Power vs Current with Polynomial Fits')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add minor grid lines
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.minorticks_on()

    plt.tight_layout()
    plt.show()
