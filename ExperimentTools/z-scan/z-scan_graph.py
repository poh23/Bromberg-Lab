import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')

def plot_results(distances, center_avgs, total_sums, laser_power):
    """
    Create and save plots for the analysis results.

    Args:
        distances (list): List of distances
        center_avgs (list): List of center region average values
        total_sums (list): List of total sum values
    """
    # Plot center average vs distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, center_avgs, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Distance (cm)')
    plt.ylabel(f'Center Region Average Pixel Value')
    plt.title(f'Average Center Region Pixel Value vs. Distance - {laser_power} A laser')
    plt.grid(True)
    plt.savefig(f'./mes_graphs/center_avg_vs_distance_{laser_power}A.png')
    plt.show()

    # Plot total sum vs distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, total_sums, 'o-', color='red', linewidth=2, markersize=8)
    plt.xlabel('Distance (cm)')
    plt.ylabel('Total Sum of Pixel Values')
    plt.title(f'Sum of All Pixel Values vs. Distance - {laser_power} A laser')
    plt.grid(True)
    #plt.show()
    plt.savefig(f'mes_graphs/total_sum_vs_distance_{laser_power}A.png')

    # Plot both on same figure with two y-axes for comparison
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Left y-axis for center average
    color = 'blue'
    ax1.set_xlabel('Distance (cm)')
    ax1.set_ylabel('Center Region Average', color=color)
    ax1.plot(distances, center_avgs, 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)

    # Right y-axis for total sum
    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Total Sum of Pixels', color=color)
    ax2.plot(distances, total_sums, 's-', color=color, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f'Pixel Values vs. Distance - Comparison - {laser_power} A laser')
    plt.grid(True)
    plt.savefig(f'./mes_graphs/comparison_plot_{laser_power}A.png')
    plt.show()


def analyze_images_vs_distance(folder_path, region_size=20):
    """
    Analyze how pixel values change across images taken at different distances.

    Args:
        folder_path (str): Path to the folder containing images
        region_size (int): Size of the square region in the center to analyze (region_size x region_size)
    """
    # Regular expression to extract distance from filenames like "Dist10cm.png"
    pattern = re.compile(r'Dist(\d+)cm\.png', re.IGNORECASE)

    # Dictionaries to store distances and corresponding values
    center_avg_data = {}  # For center region average
    total_sum_data = {}  # For sum of all pixels

    # List all png files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            # Try to extract distance from filename
            match = pattern.match(filename)
            if match:
                distance = int(match.group(1))  # Extract distance value

                # Open the image
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)

                # Convert to numpy array for easier handling
                img_array = np.array(img)

                # Get image dimensions
                height, width = img_array.shape[:2]

                # Calculate center coordinates
                center_x = width // 2
                center_y = height // 2

                # Calculate region boundaries (ensuring they're within image bounds)
                half_region = region_size // 2
                start_x = max(0, center_x - half_region)
                end_x = min(width, center_x + half_region)
                start_y = max(0, center_y - half_region)
                end_y = min(height, center_y + half_region)

                # Extract the center region
                region = img_array[start_y:end_y, start_x:end_x]

                # Calculate average pixel value for center region
                if len(img_array.shape) > 2:
                    # For RGB/RGBA images, average across all pixels and RGB channels
                    center_avg = np.mean(region[:, :, :3])  # Averaging only RGB (ignoring alpha if present)

                    # Calculate sum of all pixels (across all channels)
                    total_sum = np.sum(img_array[:, :, :3])
                else:
                    # For grayscale images
                    center_avg = np.mean(region)
                    total_sum = np.sum(img_array)

                # Store the distance and values
                center_avg_data[distance] = center_avg
                total_sum_data[distance] = total_sum

                print(
                    f"Processed {filename}: Distance={distance}cm, Center Avg={center_avg:.2f}, Total Sum={total_sum:.2f}")

    # Sort the data by distance
    sorted_distances = sorted(center_avg_data.keys())
    sorted_center_avgs = [center_avg_data[d] for d in sorted_distances]
    sorted_total_sums = [total_sum_data[d] for d in sorted_distances]
    ref_line = min(sorted_center_avgs[0], sorted_center_avgs[-1]) + np.abs(sorted_center_avgs[0] - sorted_center_avgs[-1]) / 2
    sorted_center_avgs = sorted_center_avgs/ref_line

    return sorted_distances, sorted_center_avgs, sorted_total_sums


if __name__ == "__main__":
    # Set the folder path containing the images
    folder_path = "laser_4.5"  # Replace with your actual folder path

    # Set the size of the center region to analyze (e.g., 20x20 pixels)
    region_size = 8  # Adjust this value based on your image size and needs

    # Run the analysis
    distances, center_avgs, total_sums = analyze_images_vs_distance(folder_path, region_size)

    # Create plots
    laser_power = folder_path[-3:]  # Assuming the folder name contains the laser power
    plot_results(distances, center_avgs, total_sums, laser_power)

