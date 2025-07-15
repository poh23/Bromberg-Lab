import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use('TkAgg')

def plot_results(distances, circle_avgs, total_sums, sampling_radius, folder_name):
    """
    Create and save plots for the analysis results.

    Args:
        distances (list): List of distances
        circle_avgs (list): List of circular region average values
        total_sums (list): List of total sum values
        sampling_radius (float): Radius of the sampling circle used
        folder_name (str): Name of the folder being analyzed for plot titles
    """
    # Plot circular region average vs distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, circle_avgs, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Distance (cm)')
    plt.ylabel('Circular Region Average Pixel Value')
    plt.title(f'{folder_name}A: Average Circular Region (r={sampling_radius:.2f} Transmittance) vs. Distance')
    plt.grid(True)
    plt.savefig(f'mes_graphs/{folder_name}/{folder_name}A_circle_avg_vs_distance.png')
    plt.show()


    # Additional plot: Normalized values for easier comparison of trends
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


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y):
    """
    2D Gaussian function for fitting (without offset).

    Args:
        xy: Array of x and y coordinates
        amplitude: Peak height
        x0, y0: Center coordinates
        sigma_x, sigma_y: Standard deviations in x and y directions
    """
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = amplitude * np.exp(-(
            2 * (x - x0) ** 2 / (sigma_x ** 2) +
            2 * (y - y0) ** 2 / (sigma_y ** 2)
    ))
    return gauss.ravel()
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
        float: Estimated radius (average of sigma_x and sigma_y)
    """
    # If it's an RGB image, convert to grayscale for analysis
    if len(img_array.shape) > 2:
        gray_img = np.mean(img_array[:, :, :3], axis=2)
    else:
        gray_img = img_array

    # Create coordinate grids
    y, x = np.indices(gray_img.shape)

    # Initial guess for parameters - assume Gaussian is roughly centered
    height, width = gray_img.shape
    amplitude_guess = np.max(gray_img)
    x0_guess = width // 2
    y0_guess = height // 2
    sigma_guess = min(width, height) / 8  # Initial guess for sigma

    # Remove background/baseline before fitting
    # This helps with fitting a Gaussian without offset
    baseline = np.min(gray_img)
    data = gray_img - baseline

    # Flatten the image and coordinates for curve_fit
    data_flat = data.ravel()
    xy = np.vstack((x.ravel(), y.ravel()))

    try:
        # Try to fit the 2D Gaussian (without offset parameter)
        popt, _ = curve_fit(
            gaussian_2d,
            xy,
            data_flat,
            p0=[amplitude_guess, x0_guess, y0_guess, sigma_guess, sigma_guess],
            bounds=([0, 0, 0, 0, 0],
                    [np.inf, width * 2, height * 2, width, height]),
            maxfev=10000
        )

        # Extract the fitted parameters
        _, guass_center_x, guass_center_y, sigma_x, sigma_y = popt

        print(f"Fitted Gaussian parameters: sigma_x={sigma_x:.2f}, sigma_y={sigma_y:.2f}")

        # Return the average sigma as the radius
        avg_radius = (sigma_x + sigma_y) / 2
        gauss_center = (int(guass_center_x), int(guass_center_y))
        return gauss_center, avg_radius

    except (RuntimeError, ValueError) as e:
        print(f"Warning: Gaussian fitting failed. Using default radius. Error: {e}")
        return min(width, height) / 8  # Default radius if fitting fails


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


def analyze_circular_region_vs_distance(folder_path, radius_factor=0.4):
    """
    Analyze images using a circular region in the center where the radius is based on
    the first image's Gaussian distribution.

    Args:
        folder_path (str): Path to the folder containing images
        radius_factor (float): Factor to multiply by the Gaussian radius
    """
    # Get the folder name for plot titles
    folder_name = os.path.basename(os.path.normpath(folder_path))

    # Regular expression to extract distance from filenames like "Dist10cm.png"
    pattern = re.compile(r'(\d+)cm\.png', re.IGNORECASE)

    # Find all matching image files and sort by distance
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            match = pattern.match(filename)
            if match:
                distance = int(match.group(1))
                image_files.append((distance, filename))

    # Sort files by distance
    image_files.sort()

    if not image_files:
        print("No matching image files found!")
        return [], [], [], 0, folder_name

    # Process the first image to get the Gaussian radius
    first_distance, first_filename = image_files[0]
    first_img_path = os.path.join(folder_path, first_filename)
    first_img = Image.open(first_img_path)
    first_img_array = np.array(first_img)

    # Estimate the Gaussian radius from the first image
    gaussian_center, gaussian_radius = estimate_gaussian_radius(first_img_array)
    print(f"Estimated Gaussian radius in first image: {gaussian_radius:.2f} pixels")

    # Calculate the sampling circle radius
    sampling_radius = np.sqrt(radius_factor) * gaussian_radius
    print(f"Using sampling circle radius: {sampling_radius:.2f} pixels (√({radius_factor} × {gaussian_radius:.2f}))")

    # Dictionaries to store analysis results
    circle_avg_data = {}  # For circular region average
    total_sum_data = {}  # For sum of all pixels

    # Analyze all images
    for distance, filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        img_array = np.array(img)

        # Get image dimensions
        height, width = img_array.shape

        # Create circular mask centered in the image
        circular_mask = create_circular_mask(height, width, gaussian_center, sampling_radius)

        # Apply mask and calculate average (handling grayscale images)
        masked_region = img_array[circular_mask]
        circle_avg = np.mean(masked_region)
        total_sum = np.sum(img_array)

        # Store results
        circle_avg_data[distance] = circle_avg
        total_sum_data[distance] = total_sum

        print(f"Processed {filename}: Distance={distance}cm, Circle Avg={circle_avg:.2f}, Total Sum={total_sum:.2f}")

    # Sort the data by distance
    sorted_distances = sorted(circle_avg_data.keys())

    sorted_circle_avgs = [circle_avg_data[d] for d in sorted_distances]
    circle_ref_line = min(sorted_circle_avgs[0], sorted_circle_avgs[-1]) + np.abs(sorted_circle_avgs[0] - sorted_circle_avgs[-1]) / 2
    sorted_circle_avgs = sorted_circle_avgs / circle_ref_line

    sorted_total_sums = [total_sum_data[d] for d in sorted_distances]
    first_sum = sorted_total_sums[0]
    last_sum = sorted_total_sums[-1]
    min_sum = min(first_sum, last_sum)
    diff_sums = np.abs(int(first_sum) - int(last_sum))
    total_ref_line = min_sum + diff_sums
    sorted_total_sums = sorted_total_sums / total_ref_line

    return sorted_distances, sorted_circle_avgs, sorted_total_sums, sampling_radius, folder_name, gaussian_center

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
        # Load the image
        first_img = Image.open(first_img_path)
        first_img_array = np.array(first_img)

        # Create figure for visualization
        plt.figure(figsize=(8, 8))

        # Display the image
        plt.imshow(first_img_array, cmap='gray')

        # Create and plot the circle
        height, width = first_img_array.shape[:2]
        circle = plt.Circle(gaussian_center, sampling_radius, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(circle)

        plt.title(f"{folder_name}: Sampling Circle (r={sampling_radius:.2f} pixels)")
        plt.axis('off')
        plt.savefig(f'mes_graphs/{folder_name}/{folder_name}A_sampling_circle_visualization.png')
        plt.show()

    return diff_phi

def create_consolidated_graph(root_folder_path, radius_factor=0.4):
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
    # Placeholder for future implementation
    folder_paths = ['laser_1.8', 'laser_2.5', 'laser_3.5', 'laser_4.5']
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

if __name__ == "__main__":
    # Example usage
    folder_path = "./laser_3.99_take2"
    radius_factor = 0.4 # Adjust as needed
    sorted_distances, sorted_circle_avgs, sorted_total_sums, sampling_radius, folder_name, gaussian_center =analyze_circular_region_vs_distance(folder_path, radius_factor)
    first_img_path = os.path.join(folder_path, f"{min(sorted_distances)}cm.png")
    if os.path.exists(first_img_path):
        # Load the image
        first_img = Image.open(first_img_path)
        first_img_array = np.array(first_img)

        # Create figure for visualization
        plt.figure(figsize=(8, 8))

        # Display the image
        plt.imshow(first_img_array, cmap='gray')

        # Create and plot the circle
        height, width = first_img_array.shape[:2]
        circle = plt.Circle(gaussian_center, sampling_radius, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(circle)

        plt.title(f"{folder_name}: Sampling Circle (r={sampling_radius:.2f} pixels)")
        plt.axis('off')
        plt.savefig(f'mes_graphs/{folder_name}/{folder_name}A_sampling_circle_visualization.png')
        plt.show()
    plot_results(sorted_distances, sorted_circle_avgs, sorted_total_sums, sampling_radius, folder_name)


# todo: I0 vs. phi -> see if linear and extract n2
#todo: measure stability of laser: I0 vs. time