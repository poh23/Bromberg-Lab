import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from PIL import Image

# --- Helper Functions ---
def load_image_gray(img_path):
    """Load an image as grayscale uint8 numpy array."""
    return np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

def amper_to_power(amperes):
    """Convert amperes to power in watts using a parabolic conversion."""
    if amperes <= 7:
        return -0.0001*amperes**3 + 0.0061*amperes**2 + -0.0109*amperes + 0.0050
    else:
        # after 7A, the conversion is static
        return -0.0001*7**3 + 0.0061*7**2 + -0.0109*7 + 0.0050

def get_circle_mask(h, w, cx, cy, r):
    """Return a boolean mask for a circle of radius r centered at (cx, cy) in an image of shape (h, w)."""
    y, x = np.ogrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

def get_normalized_pixels(img_gray, cx, cy, r):
    """Return normalized pixel values within a circle mask."""
    h, w = img_gray.shape
    mask = get_circle_mask(h, w, cx, cy, r)
    pixel_values = img_gray[mask].flatten()
    avg_value = np.mean(pixel_values)
    normalized_values = pixel_values / avg_value
    return normalized_values

def compute_histogram(normalized_values, bin_edges):
    """Compute histogram counts for normalized values using given bin edges."""
    counts, _ = np.histogram(normalized_values, bins=bin_edges)
    return counts

def find_global_min_max(folders, all_files, r, cx, cy):
    """Find global min and max of normalized pixel values across all images for binning."""
    min_val, max_val = float('inf'), float('-inf')
    for fname in all_files:
        for folder in folders:
            img_path = os.path.join(folder, fname)
            if not os.path.isfile(img_path):
                continue
            try:
                img_gray = load_image_gray(img_path)
                h, w = img_gray.shape
                cx_ = w // 2 if cx is None else cx
                cy_ = h // 2 if cy is None else cy
                normalized_values = get_normalized_pixels(img_gray, cx_, cy_, r)
                min_val = min(min_val, normalized_values.min())
                max_val = max(max_val, normalized_values.max())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return min_val, max_val

# --- Main Functions (refactored to use helpers) ---
def create_histograms_and_scatter(directory, r, cx=None, cy=None):
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' not found. Please check the path.")
        return
    png_files = [fname for fname in os.listdir(directory) if fname.lower().endswith('.png')]
    if not png_files:
        print(f"No PNG files found in directory '{directory}'.")
        return
    # Store histogram data for line plot
    all_hist = []  # (fname, bin_centers, pdf)
    for fname in png_files:
        img_path = os.path.join(directory, fname)
        try:
            img_gray = load_image_gray(img_path)
            h, w = img_gray.shape
            cx_ = w // 2 if cx is None else cx
            cy_ = h // 2 if cy is None else cy
            # show image + circle
            plt.figure(figsize=(6, 6))
            plt.imshow(img_gray, cmap='gray')
            plt.title(f'Image: {fname}')
            plt.axis('off')
            circle = plt.Circle((cx_, cy_), r, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.show()
            normalized_values = get_normalized_pixels(img_gray, cx_, cy_, r)
            counts, bins = np.histogram(
                normalized_values, bins=30, range=(normalized_values.min(), normalized_values.max())
            )
            pdf = counts / counts.sum()
            bin_centers = (bins[:-1] + bins[1:]) / 2
            all_hist.append((fname, bin_centers, pdf))
            plt.figure(figsize=(8, 5))
            plt.bar(bins[:-1], pdf, width=(bins[1] - bins[0]), color='blue', alpha=0.7, label='Normalized Histogram (PDF)')
            amperes = fname.split('A')[0]
            power = amper_to_power(float(amperes))
            plt.title(f'Histogram of Normalized Pixel Values (r={r} px): laser power:{power:.4f} W')
            plt.xlabel('Pixel Value / Mean Pixel Value')
            plt.ylabel('Fraction of Pixels (PDF)')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, which='both', axis='y')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    # Combined line plot for all files
    plt.figure(figsize=(10, 6))
    for fname, bin_centers, pdf in all_hist:
        amperes = fname.split('A')[0]
        power = amper_to_power(float(amperes))
        plt.plot(bin_centers, pdf, alpha=0.7, label=f'Power: {power:.4f} W')
    plt.xlabel('Pixel Value / Mean Pixel Value')
    plt.ylabel('Fraction of Pixels (PDF)')
    plt.title(f'All Normalized Histograms (r={r} px)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', axis='y')
    plt.tight_layout()
    plt.show()

def create_summed_histogram_by_filename(folders, r, cx=None, cy=None, bins=30, show_filenames=None):
    """
    For each unique filename across the given folders, computes the normalized histogram for that image in each folder (within a circle of radius r),
    sums the histograms for all folders that contain that file, and plots the combined histogram (PDF) for each filename.
    All histograms use the same bin edges, determined from all images.
    If show_filenames is provided (list of filenames), only those are shown in the combined scatter plot.
    """
    all_files = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        all_files.update([fname for fname in os.listdir(folder) if fname.lower().endswith('.png')])
    if not all_files:
        print("No PNG files found in any folder.")
        return
    min_val, max_val = find_global_min_max(folders, all_files, r, cx, cy)
    if min_val == float('inf') or max_val == float('-inf'):
        print("No valid images found for histogram range.")
        return
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    all_histograms = []  # To store (fname, bin_centers, pdf)
    for fname in sorted(all_files):
        summed_counts = None
        for folder in folders:
            img_path = os.path.join(folder, fname)
            if not os.path.isfile(img_path):
                continue
            try:
                img_gray = load_image_gray(img_path)
                h, w = img_gray.shape
                cx_ = w // 2 if cx is None else cx
                cy_ = h // 2 if cy is None else cy
                normalized_values = get_normalized_pixels(img_gray, cx_, cy_, r)
                counts = compute_histogram(normalized_values, bin_edges)
                if summed_counts is None:
                    summed_counts = counts.astype(float)
                else:
                    summed_counts += counts
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        if summed_counts is not None:
            pdf = summed_counts / summed_counts.sum()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            all_histograms.append((fname, bin_centers, pdf))
            plt.figure(figsize=(8, 5))
            plt.bar(bin_edges[:-1], pdf, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=fname)
            amperes = fname.split('A')[0]
            power = amper_to_power(float(amperes))
            plt.title(f'Summed Normalized Histogram for File: Laser Power: {power} W (r={r} px)')
            plt.xlabel('Pixel Value / Mean Pixel Value')
            plt.ylabel('Fraction of Pixels (PDF)')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, which='both', axis='y')
            plt.tight_layout()
            plt.show()
    if all_histograms:
        plt.figure(figsize=(10, 6))
        for fname, bin_centers, pdf in all_histograms:
            if show_filenames is None or fname in show_filenames:
                amperes = fname.split('A')[0]
                power = amper_to_power(float(amperes))
                plt.plot(bin_centers, pdf, alpha=0.7, label=f'Power: {power:.4f} W')
        plt.xlabel('Pixel Value / Mean Pixel Value')
        plt.ylabel('Fraction of Pixels (PDF)')
        plt.title(f'All Summed Normalized Histograms (r={r} px)')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', axis='y')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set the directory containing the PNGs
    png_dir = "0.5angle_before_cuvette_1"
    create_histograms_and_scatter(png_dir, r=200, cy=400)
