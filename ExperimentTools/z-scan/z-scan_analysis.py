import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
from ExperimentTools.utils.fit_gaussian_width.gaussian_fit_utils import fit_gaussian

matplotlib.use('TkAgg')

# --- Helper Functions ---
def calculate_phase_shift(circ_avgs, radius_factor):
    T_min = min(circ_avgs)
    T_max = max(circ_avgs)
    diff_T = T_max - T_min
    diff_phi = 2.463 * (1-radius_factor)**(-0.25) * diff_T
    return diff_phi

def estimate_gaussian_radius(img_array):
    """
    Estimate the radius of a Gaussian distribution in an image by fitting a 2D Gaussian.
    Args:
        img_array: Numpy array of the image
    Returns:
        tuple: (center, average sigma)
    """
    # If it's an RGB image, convert to grayscale for analysis
    if len(img_array.shape) > 2:
        gray_img = np.mean(img_array[:, :, :3], axis=2)
    else:
        gray_img = img_array

    # Remove background/baseline before fitting
    baseline = np.min(gray_img)
    data = gray_img - baseline

    # Use the generic fit_gaussian_2d utility (no offset)
    popt = fit_gaussian(data, show_fit=True)
    if popt is not None:
        x0, y0, sigma_x, sigma_y, amplitude, offset = popt
        print(f"Fitted Gaussian parameters: sigma_x={sigma_x:.2f}, sigma_y={sigma_y:.2f}")
        avg_radius = (np.abs(sigma_x) + np.abs(sigma_y)) / 2
        gauss_center = (int(x0), int(y0))
        return gauss_center, avg_radius
    else:
        print(f"Warning: Gaussian fitting failed. Using default radius.")
        height, width = gray_img.shape
        return (width // 2, height // 2), min(width, height) / 8  # Default center and radius if fitting fails


def create_circular_mask(height, width, center=None, radius=None):
    """
    Create a circular mask for an image.

    Args:
        height, width: Dimensions of the image
        center: Tuple of (x, y) coordinates for the center of the circle
        radius: Radius of the circle

    Returns:
        2D boolean numpy array with True inside the circle
    """
    if center is None:
        center = (width // 2, height // 2)

    if radius is None:
        radius = min(width, height) // 2

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def find_image_files(folder_path):
    """Get sorted list of image files with their distances."""
    pattern = re.compile(r'Dist(\d+)cm\.png', re.IGNORECASE)
    image_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            match = pattern.match(filename)
            if match:
                distance = int(match.group(1))
                image_files.append((distance, filename))

    image_files.sort()
    return image_files


def process_first_image(folder_path, image_files):
    """Process the first image to get Gaussian parameters."""
    if not image_files:
        return None, None

    first_distance, first_filename = image_files[0]
    first_img_path = os.path.join(folder_path, first_filename)
    first_img = Image.open(first_img_path)
    first_img_array = np.array(first_img)

    gaussian_center, gaussian_radius = estimate_gaussian_radius(first_img_array)
    print(f"Estimated Gaussian radius in first image: {gaussian_radius:.2f} pixels")
    return gaussian_center, gaussian_radius


def analyze_single_image(img_path, gaussian_center, sampling_radius):
    """Analyze a single image and return its measurements."""
    img = Image.open(img_path)
    img_array = np.array(img)
    height, width = img_array.shape

    circular_mask = create_circular_mask(height, width, gaussian_center, sampling_radius)
    masked_region = img_array[circular_mask]

    circle_avg = np.mean(masked_region)
    total_sum = np.sum(img_array)

    return circle_avg, total_sum


def normalize_measurements(circle_avgs, total_sums):
    """Normalize the measurements for comparison."""
    # Normalize circle averages
    circle_ref_line = min(circle_avgs[0], circle_avgs[-1]) + np.abs(circle_avgs[0] - circle_avgs[-1]) / 2
    norm_circle_avgs = [avg / circle_ref_line for avg in circle_avgs]

    # Normalize total sums
    first_sum, last_sum = total_sums[0], total_sums[-1]
    min_sum = min(first_sum, last_sum)
    diff_sums = np.abs(int(first_sum) - int(last_sum))
    total_ref_line = min_sum + diff_sums
    norm_total_sums = [sum_val / total_ref_line for sum_val in total_sums]

    return norm_circle_avgs, norm_total_sums

# ---- plotting helper functions ----

def plot_results(distances, circle_avgs, total_sums, sampling_radius, folder_name):
    """
    Create and save transmittance and consolidated transmittance and absorbance plots for the analysis results.

    Args:
        distances (list): List of distances
        circle_avgs (list): List of circular region average values
        total_sums (list): List of total sum values
        sampling_radius (float): Radius of the sampling circle used
        folder_name (str): Name of the folder being analyzed for plot titles
    """
    # Plot circular region average vs distance - closed aperture measurement
    plt.figure(figsize=(10, 6))
    plt.plot(distances, circle_avgs, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Distance (cm)')
    plt.ylabel('Circular Region Average Pixel Value')
    plt.title(f'{folder_name}A: Average Circular Region (r={sampling_radius:.2f} Transmittance) vs. Distance')
    plt.grid(True)
    plt.savefig(f'mes_graphs/{folder_name}/{folder_name}A_circle_avg_vs_distance.png')
    plt.show()


    # Additional plot: close and open aperture on same graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances, circle_avgs, 'o-', color='blue', linewidth=2, markersize=8,
             label='Circular Region Avg Transmittance')
    plt.plot(distances, total_sums, 's-', color='red', linewidth=2, markersize=8, label='Total Sum (normalized)')
    plt.xlabel('Distance (cm)')
    plt.ylabel('Normalized Value')
    plt.title(f'{folder_name}A: Transmittance & Normalized Intensity vs. Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'mes_graphs/{folder_name}/{folder_name}A_normalized_comparison.png')
    plt.show()

def create_consolidated_graph(root_folder_path, radius_factor=0.4):
    """
        Creates a consolidated graph of the transmittance graphs for the give folders with measurements across different z distances.
    """
    folder_paths = ['laser_1.8', 'laser_2.5', 'laser_3.5', 'laser_4.5']

    plt.figure(figsize=(10, 6))
    for folder in folder_paths:
        folder_path = os.path.join(root_folder_path, folder)
        distances, circle_avgs, total_sums, sampling_radius, folder_name, gaussian_center = analyze_circular_region_vs_distance(
            folder_path,
            radius_factor)

        plt.plot(distances, circle_avgs, 'o-', label=folder, linewidth=2, markersize=8)

    plt.xlabel('Distance (cm)')
    plt.ylabel('Normalized Circular Region Average Pixel Value')
    plt.legend()
    plt.title('Consolidated Graph of Circular Region Average vs Distance')
    plt.grid(True)
    plt.savefig('mes_graphs/consolidated_graph.png')
    plt.show()

def generate_diff_phi_vs_I0_graph(folder_path, radius_factor=0.4):
    """
    Generate a graph of the initial phase shift (phi0) vs. intensity (I0).

    Args:
        folder_path (str): Path to the folder containing images
        radius_factor (float): Factor to multiply by the Gaussian radius

    Returns:
        None
    """
    folder_paths = ['laser_1.8', 'laser_2.5', 'laser_3.5', 'laser_4.5']
    ## Define the laser power values for ampere
    power_dict = {
        'laser_1.8': 2.5,
        'laser_2.5': 4.3,
        'laser_3.5': 11.3,
        'laser_4.5': 17
    }
    diff_phi_list = []
    for folder in folder_paths:
        diff_phi = get_phi0(folder_path + '/' + folder, plot=True, radius_factor=radius_factor)
        diff_phi_list.append(diff_phi)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(power_dict.values(), diff_phi_list, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Laser Power (mW)')
    plt.ylabel('Initial Phase Shift (phi0)')
    plt.title('Initial Phase Shift vs. Laser Power')
    plt.grid(True)
    plt.savefig('mes_graphs/diff_phi_vs_I0.png')
    plt.show()

def graph_first_image(first_img_path, sampling_radius=0.4):
    first_img = Image.open(first_img_path)
    first_img_array = np.array(first_img)

    # Create figure for visualization
    plt.figure(figsize=(8, 8))

    # Display the image
    plt.imshow(first_img_array, cmap='gray')

    # Create and plot the circle
    circle = plt.Circle(gaussian_center, sampling_radius, fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(circle)

    plt.title(f"{folder_name}: Sampling Circle (r={sampling_radius:.2f} pixels)")
    plt.axis('off')
    plt.savefig(f'mes_graphs/{folder_name}/{folder_name}A_sampling_circle_visualization.png')
    plt.show()

# --- Main Analysis Functions ---

def analyze_circular_region_vs_distance(folder_path, radius_factor=0.4):
    """
    Analyze images using a circular region in the center where the radius is based on
    the first image's Gaussian distribution.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    image_files = find_image_files(folder_path)

    if not image_files:
        print("No matching image files found!")
        return [], [], [], 0, folder_name

    gaussian_center, gaussian_radius = process_first_image(folder_path, image_files)
    sampling_radius = np.sqrt(radius_factor) * gaussian_radius
    print(f"Using sampling circle radius: {sampling_radius:.2f} pixels (√({radius_factor} × {gaussian_radius:.2f}))")

    # Analyze all images
    circle_avg_data = {}
    total_sum_data = {}

    for distance, filename in image_files:
        img_path = os.path.join(folder_path, filename)
        circle_avg, total_sum = analyze_single_image(img_path, gaussian_center, sampling_radius)

        circle_avg_data[distance] = circle_avg
        total_sum_data[distance] = total_sum
        print(f"Processed {filename}: Distance={distance}cm, Circle Avg={circle_avg:.2f}, Total Sum={total_sum:.2f}")

    # Sort and normalize the data
    sorted_distances = sorted(circle_avg_data.keys())
    sorted_circle_avgs = [circle_avg_data[d] for d in sorted_distances]
    sorted_total_sums = [total_sum_data[d] for d in sorted_distances]

    norm_circle_avgs, norm_total_sums = normalize_measurements(sorted_circle_avgs, sorted_total_sums)

    return sorted_distances, norm_circle_avgs, norm_total_sums, sampling_radius, folder_name, gaussian_center

def get_phi0(folder_path, plot=True, radius_factor=0.4):
    """
    Get the initial phase shift (phi0) from the first image in the folder.

    Args:
        folder_path (str): Path to the folder containing images

    Returns:
        float: Initial phase shift
    """
    # Run the analysis
    distances, circle_avgs, total_sums, sampling_radius, folder_name, gaussian_center = analyze_circular_region_vs_distance(
        folder_path,
        radius_factor)

    # Create plots
    if plot:
        plot_results(distances, circle_avgs, total_sums, sampling_radius, folder_name)

    # Print the results
    print("\nResults:")
    print(f"Using circular sampling region with radius: {sampling_radius:.2f} pixels")
    for i, (dist, circle_avg, total_sum) in enumerate(zip(distances, circle_avgs, total_sums)):
        print(f"Distance: {dist}cm - Circle Avg: {circle_avg:.2f} - Total Sum: {total_sum:.2f}")

    diff_phi = calculate_phase_shift(circle_avgs, radius_factor)

    # Visualize the circular region on the first image (optional)
    first_img_path = os.path.join(folder_path, f"Dist{min(distances)}cm.png")
    if os.path.exists(first_img_path) and plot:
        graph_first_image(first_img_path, sampling_radius)

    return diff_phi

if __name__ == "__main__":

    folder_path = "./laser_2.98"
    radius_factor = 0.4 # Adjust as needed
    sorted_distances, sorted_circle_avgs, sorted_total_sums, sampling_radius, folder_name, gaussian_center = analyze_circular_region_vs_distance(folder_path, radius_factor)

    ## if we want to seethe size of the aperture circle in the first image
    first_img_path = os.path.join(folder_path, f"Dist{min(sorted_distances)}cm.png")
    if os.path.exists(first_img_path):
        graph_first_image(first_img_path, sampling_radius)

    plot_results(sorted_distances, sorted_circle_avgs, sorted_total_sums, sampling_radius, folder_name)

