# --- Imports ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, label
import matplotlib.patches as patches

# --- Utility Functions ---
def safe_float(val, default=0.0):
    """Safely convert a value to float, with a default if conversion fails."""
    try:
        return float(val)
    except Exception:
        return default


def load_and_threshold_image(filepath, threshold=0.258):
    """Load an image and set all pixel values below threshold to 0."""
    img = plt.imread(filepath)
    # Convert color images to grayscale for thresholding
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.copy()
    img[img < threshold] = 0
    return img

# --- Image Analysis Functions ---
def find_blob_center(img_array):
    """
    Find the center of the blob in the image.
    If use_com is True, use center of mass; otherwise, use centroid of the largest thresholded blob.
    Returns (cy, cx), center_type.
    """
    threshold = img_array.mean() + img_array.std()
    binary_img = img_array > threshold
    labeled, num_features = label(binary_img)
    # If labeled is not a numpy array, treat as no features
    if not isinstance(labeled, np.ndarray) or num_features == 0:
        return (img_array.shape[0] // 2, img_array.shape[1] // 2), 'none'
    sizes = np.bincount(labeled.ravel().astype(np.intp))
    sizes[0] = 0
    largest_label = sizes.argmax()
    blob_mask = labeled == largest_label
    cy, cx = center_of_mass(blob_mask)
    return (cy, cx)

# --- Data Processing Functions ---
def load_data(csv_path):
    """Load the CSV and fill down missing laser power values."""
    df = pd.read_csv(csv_path)
    if 'laser power(A)' in df.columns:
        df['laser power(A)'] = df['laser power(A)'].fillna(method='ffill')
    return df

def find_images(df, pics_dir):
    """For each row, check if the corresponding image exists and store the result in new columns."""
    image_exists = []
    image_filenames = []
    for idx, row in df.iterrows():
        power = row['laser power(A)']
        medium = row['Medium']
        if pd.isna(power) or pd.isna(medium):
            image_exists.append(False)
            image_filenames.append(None)
            continue
        medium_clean = str(medium).replace(' ', '_').replace(':', '-')
        try:
            power_float = float(power)
            if power_float.is_integer():
                power_str = f"{int(power_float)}A"
            else:
                power_str = f"{power_float}A"
        except Exception:
            power_str = f"{power}A"
        filename = f"{medium_clean}_{power_str}.png"
        filepath = os.path.join(pics_dir, filename)
        exists = os.path.exists(filepath)
        image_exists.append(exists)
        image_filenames.append(filename if exists else None)
    df['Image Exists'] = image_exists
    df['Image Filename'] = image_filenames
    return df

def compute_transmittance(df, pics_dir, radius, show_circle=False, use_com_for=None):
    """
    For each row with a PNG, compute transmittance using a per-row factor if factor_col is given, else a constant factor.
    Uses regular center of mass for (medium, power) in use_com_for, else centroid of thresholded blob.
    """
    transmittance = [None] * len(df)
    for idx, row in df.iterrows():
        filename = row.get('Image Filename')
        if not row.get('Image Exists', False) or not filename:
            continue
        filepath = os.path.join(pics_dir, filename)
        try:
            img_array = load_and_threshold_image(filepath)
            if img_array.ndim == 3:
                img_array = img_array.mean(axis=2)
        except Exception:
            continue
        (cy, cx) = find_blob_center(img_array)
        y, x = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        sum_in_circle = img_array[mask].sum()
        total_sum = img_array.sum()
        transmittance[idx] = sum_in_circle / total_sum if total_sum > 0 else 0
        if show_circle:
            fig, ax = plt.subplots()
            ax.imshow(img_array, cmap='gray')
            circ = patches.Circle((cx, cy), radius, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(circ)
            ax.plot(cx, cy, 'bo')
            ax.set_title(f"{filename} aperture center: ({cx:.1f},{cy:.1f}), r={radius}")
            plt.show()
    colname = f'Transmittance (radius={radius})'
    new_df = df.copy()
    new_df[colname] = transmittance
    return new_df

# --- Plotting Functions ---
def plot_transmittance(df, trans_col, r, medium_filter=None):
    """Plot transmittance vs. Power after cuvette (mW) for each medium, or only one if medium_filter is set."""
    plt.figure(figsize=(8,6))
    mediums = df['Medium'].unique() if medium_filter is None else [medium_filter]
    colors = plt.cm.get_cmap('rainbow', len(mediums))
    for i, medium in enumerate(mediums):
        msk = (df['Medium'] == medium) & df[trans_col].notna()
        x = df.loc[msk, 'Power after cuvette (mW)'].astype(float)
        y = df.loc[msk, trans_col]
        if y.isna().all():  # Skip if all transmittance values are NaN
            continue
        plt.scatter(
            x,
            y,
            label=medium,
            color=colors(i)
        )
        # Add a line between points (sorted by x)
        sorted_idx = np.argsort(x)
        plt.plot(
            x.iloc[sorted_idx],
            y.iloc[sorted_idx],
            color=colors(i),
            alpha=0.7
        )
    plt.xlabel('Power after cuvette (mW)')
    plt.ylabel(f'Transmittance')
    plt.title(f'Transmittance vs. Power after cuvette (aperture radius={r})')
    plt.legend(title='Medium')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Main Function ---
def main():
    """Main function to load data, compute factors, transmittance, and plot results."""
    csv_path = os.path.join(os.path.dirname(__file__), 'power data - take5.csv')
    pics_dir = os.path.join(os.path.dirname(__file__), 'pics3')
    r = 185
    df = load_data(csv_path)
    df = find_images(df, pics_dir)
    df = compute_transmittance(df, pics_dir, r, show_circle=True)
    print("DataFrame with transmittance values:")
    #print(df)
    print(df.to_string(index=False))
    trans_col = f'Transmittance (radius={r})'
    plot_transmittance(df, trans_col, r)
    # To plot only one medium, uncomment and set the medium name:
    # plot_transmittance(df, trans_col, r, medium_filter='1:500 blue')

if __name__ == '__main__':
    main()
