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
def find_blob_center(img_array, use_com=False):
    """
    Find the center of the blob in the image.
    If use_com is True, use center of mass; otherwise, use centroid of the largest thresholded blob.
    Returns (cy, cx), center_type.
    """
    if use_com:
        cy, cx = center_of_mass(img_array)
        center_type = 'COM'
    else:
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
        center_type = 'centroid'
    return (cy, cx), center_type

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

def compute_conversion_factors_df(df, pics_dir):
    """
    Return a DataFrame with conversion factors for the first available image for each unique medium.
    """
    records = []
    seen_mediums = set()
    for idx, row in df.iterrows():
        medium = row.get('Medium')
        if medium in seen_mediums:
            continue
        if not row.get('Image Exists', False):
            continue
        filename = row.get('Image Filename')
        if not filename:
            continue
        filepath = os.path.join(pics_dir, filename)
        try:
            img_array = load_and_threshold_image(filepath)
            pixel_sum = img_array.sum()
        except Exception:
            continue
        exposure = safe_float(row.get('exposure (microsecs)'))
        post_power = safe_float(row.get('Power after cuvette (mW)'))
        nd_value = safe_float(row.get('ND'))
        if exposure == 0 or post_power == 0:
            continue
        R_pixels = pixel_sum * (10 ** nd_value) / exposure
        factor = post_power / R_pixels
        records.append({
            'Medium': medium,
            'Power': row.get('laser power(A)'),
            'Image Filename': filename,
            'Conversion Factor': factor
        })
        seen_mediums.add(medium)
    return pd.DataFrame(records)

def compute_transmittance(df, pics_dir, radius, factor=None, factor_col=None, show_circle=False, use_com_for=None):
    """
    For each row with a PNG, compute transmittance using a per-row factor if factor_col is given, else a constant factor.
    Uses regular center of mass for (medium, power) in use_com_for, else centroid of thresholded blob.
    """
    if use_com_for is None:
        use_com_for = [('1:5 blue', 8.5),('1:2 red', 8.5),('1:50 blue', 8.5),('1:5 blue', 7),('1:5 blue', 5.5),('1:5 blue', 4),('1:50 blue', 7),('1:50 blue', 5.5),('1:2 red', 5.5)]
    transmittance = [None] * len(df)
    for idx, row in df.iterrows():
        filename = row.get('Image Filename')
        if not row.get('Image Exists', False) or not filename:
            continue
        medium = row.get('Medium')
        power_float = safe_float(row.get('laser power(A)'))
        use_com = (medium, power_float) in use_com_for
        filepath = os.path.join(pics_dir, filename)
        try:
            img_array = load_and_threshold_image(filepath)
            if img_array.ndim == 3:
                img_array = img_array.mean(axis=2)
        except Exception:
            continue
        (cy, cx), center_type = find_blob_center(img_array, use_com=use_com)
        y, x = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        nd_value = safe_float(row.get('ND'))
        exposure = safe_float(row.get('exposure (microsecs)'), 1)
        sum_in_circle = img_array[mask].sum()
        R_pixels_circle = sum_in_circle * (10 ** nd_value) / exposure
        R_pixels = img_array.sum() * (10 ** nd_value) / exposure
        print(f"Image: {filename}, Medium: {medium}, Power: {power_float}, Center: ({cx:.1f}, {cy:.1f}), R_pixels_circle: {R_pixels_circle:.2f}, R_pixels: {R_pixels:.2f}, Nd: {nd_value}, Exposure: {exposure}")
        post_power = safe_float(row.get('Power after cuvette (mW)'))
        row_factor = row[factor_col] if factor_col and factor_col in row else factor
        if row_factor == 0 or post_power == 0:
            continue
        #transmittance[idx] = R_pixels_circle * row_factor / post_power
        transmittance[idx] = R_pixels_circle / R_pixels
        print('Transmittance:', transmittance[idx],'row factor:', row_factor, 'post power:', post_power)
        print('picture power:', R_pixels * row_factor)
        if show_circle:
            fig, ax = plt.subplots()
            ax.imshow(img_array, cmap='gray')
            circ = patches.Circle((cx, cy), radius, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(circ)
            ax.plot(cx, cy, 'bo')
            ax.set_title(f"{filename}\n{center_type} @ ({cx:.1f},{cy:.1f}), r={radius}")
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
    plt.ylabel(f'Transmittance (radius={r})')
    plt.title(f'Transmittance vs. Power after cuvette (radius={r})')
    plt.legend(title='Medium')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_colormap_from_file(filename, radius, cx, cy):
    """
    Display the image from the given filename using only two colors:
    - One color for pixel values > 0.258
    - Another color for pixel values <= 0.258
    Overlays a circle and prints pixel sums as before.
    """
    from matplotlib.colors import ListedColormap
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        img = load_and_threshold_image(filename)
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        # Create a binary mask for >0.258 and <=0.258
        binary_img = np.where(img > 0.258, 1, 0)
        cmap = ListedColormap(['blue', 'red'])  # 0: blue (<=0.258), 1: red (>0.258)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(binary_img, cmap=cmap, vmin=0, vmax=1)
        circ = patches.Circle((cx, cy), radius, edgecolor='green', facecolor='none', linewidth=2)
        plt.gca().add_patch(circ)
        plt.title(f"Binary Colormap (>0.258=red, <=0.258=blue): {os.path.basename(filename)}")
        plt.axis('off')
        cbar = plt.colorbar(im, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['<=0.258', '>0.258'])
        plt.show()
    except Exception as e:
        print(f"Error loading or displaying image: {e}")

# --- Main Function ---
def main():
    """Main function to load data, compute factors, transmittance, and plot results."""
    csv_path = os.path.join(os.path.dirname(__file__), 'power data - take5.csv')
    pics_dir = os.path.join(os.path.dirname(__file__), 'pics3')
    r = 185
    df = load_data(csv_path)
    df = find_images(df, pics_dir)
    factors_df = compute_conversion_factors_df(df, pics_dir)
    print("Conversion Factors DataFrame:", factors_df)
    if not factors_df.empty:
        df = df.merge(factors_df[['Medium', 'Conversion Factor']], on='Medium', how='left')
    else:
        df['Conversion Factor'] = 1.0
    df = compute_transmittance(df, pics_dir, r, factor_col='Conversion Factor', show_circle=True)
    print(df)
    trans_col = f'Transmittance (radius={r})'
    plot_transmittance(df, trans_col, r)
    # To plot only one medium, uncomment and set the medium name:
    # plot_transmittance(df, trans_col, r, medium_filter='1:500 blue')

if __name__ == '__main__':
    main()
