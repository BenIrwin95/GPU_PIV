import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS # Import TAGS for explicit tag numbers
import os

def gaussian_2d(x, y, sigma):
    """
    Calculates the value of a 2D Gaussian function.
    Assumes center at (0,0).
    """
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

def generate_piv_test_image(width, height, num_particles, particle_size=3, particle_sigma=1.0, output_filename="piv_test_image_frame1.tif"):
    """
    Generates a synthetic TIFF image with randomly placed particles for PIV testing.
    Particles have a Gaussian intensity profile.

    Args:
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        num_particles (int): Number of particles to generate.
        particle_size (int): Defines the square region around the particle center where intensity is calculated.
                             Must be an odd number for a clear center.
        particle_sigma (float): Standard deviation for the Gaussian intensity profile.
                                Controls how sharp or spread out the particle appears.
        output_filename (str): Name of the output TIFF file for the first frame.

    Returns:
        tuple: (np.ndarray, np.ndarray) A tuple containing two NumPy arrays:
               - x_coords (np.ndarray): Original x-coordinates of particle centers.
               - y_coords (np.ndarray): Original y-coordinates of particle centers.
    """
    if particle_size % 2 == 0:
        print("Warning: particle_size should ideally be an odd number for a centered particle. Adjusting to next odd number.")
        particle_size += 1

    image_data = np.zeros((height, width), dtype=np.uint8)
    half_particle_size = particle_size // 2

    print(f"Generating {output_filename} ({width}x{height}) with {num_particles} Gaussian particles...")

    x_coords = np.random.randint(half_particle_size, width - half_particle_size, num_particles)
    y_coords = np.random.randint(half_particle_size, height - half_particle_size, num_particles)

    # Pre-calculate Gaussian kernel for a single particle to optimize
    x_grid = np.arange(-half_particle_size, half_particle_size + 1)
    y_grid = np.arange(-half_particle_size, half_particle_size + 1)
    # Create 2D grids for x and y coordinates relative to particle center
    X, Y = np.meshgrid(x_grid, y_grid)

    # Calculate Gaussian values for the particle kernel
    # Scale by 255 to get max intensity at the center
    particle_kernel = (gaussian_2d(X, Y, particle_sigma) * 255).astype(np.uint8)

    for i in range(num_particles):
        center_x = x_coords[i]
        center_y = y_coords[i]

        x_start = center_x - half_particle_size
        x_end = center_x + half_particle_size + 1
        y_start = center_y - half_particle_size
        y_end = center_y + half_particle_size + 1

        # Add the Gaussian particle kernel to the image data
        # Use np.maximum to avoid overwriting brighter pixels if particles overlap,
        # and to ensure values don't exceed 255.
        # This assumes the background is 0, so adding directly works too,
        # but max is safer for overlapping particles.
        image_data[y_start:y_end, x_start:x_end] = np.maximum(
            image_data[y_start:y_end, x_start:x_end],
            particle_kernel
        )

    img = Image.fromarray(image_data, mode='L')

    # Explicitly set TIFF metadata for single-channel grayscale
    # Use direct integer tag numbers as keys for tiffinfo
    tiff_tags = {
        262: 1, # PhotometricInterpretation: 1 (BlackIsZero)
        277: 1, # SamplesPerPixel: 1
        258: 8  # BitsPerSample: 8 (for uint8 data)
    }

    try:
        img.save(output_filename, compression="tiff_deflate", tiffinfo=tiff_tags)
        print(f"Successfully generated '{output_filename}'")
    except Exception as e:
        print(f"Error saving image: {e}")

    return x_coords, y_coords

def generate_displaced_image(width, height, original_x_coords, original_y_coords,
                             particle_size=3, particle_sigma=1.0, output_filename="piv_test_image_frame2.tif"):
    """
    Generates a second synthetic TIFF image with particles displaced according to a velocity field.
    Particles have a Gaussian intensity profile.

    Args:
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        original_x_coords (np.ndarray): Original x-coordinates of particle centers.
        original_y_coords (np.ndarray): Original y-coordinates of particle centers.
        particle_size (int): Size of the square region for particle drawing.
        particle_sigma (float): Standard deviation for the Gaussian intensity profile.
        output_filename (str): Name of the output TIFF file for the second frame.
    """
    if particle_size % 2 == 0:
        particle_size += 1 # Ensure odd size for consistency

    image_data = np.zeros((height, width), dtype=np.uint8)
    half_particle_size = particle_size // 2

    print(f"Generating {output_filename} ({width}x{height}) with displaced Gaussian particles...")

    displaced_x_coords = np.copy(original_x_coords)
    displaced_y_coords = np.copy(original_y_coords)

    # Pre-calculate Gaussian kernel for a single particle to optimize
    x_grid = np.arange(-half_particle_size, half_particle_size + 1)
    y_grid = np.arange(-half_particle_size, half_particle_size + 1)
    X, Y = np.meshgrid(x_grid, y_grid)
    particle_kernel = (gaussian_2d(X, Y, particle_sigma) * 255).astype(np.uint8)

    # Apply the velocity field: U = y^2 (displacement in x-direction)
    # Assuming V = 0 (no displacement in y-direction)
    for i in range(len(original_x_coords)):
        displacement_x = 16*(original_y_coords[i]/2023)**2
        displacement_y = 0

        displaced_x_coords[i] = original_x_coords[i] + displacement_x
        displaced_y_coords[i] = original_y_coords[i] + displacement_y

        # Clip coordinates to ensure particles stay within image bounds
        x_min_bound = half_particle_size
        x_max_bound = width - half_particle_size - 1
        y_min_bound = half_particle_size
        y_max_bound = height - half_particle_size - 1

        if (x_min_bound <= displaced_x_coords[i] <= x_max_bound and
            y_min_bound <= displaced_y_coords[i] <= y_max_bound):

            center_x = int(round(displaced_x_coords[i]))
            center_y = int(round(displaced_y_coords[i]))

            x_start = center_x - half_particle_size
            x_end = center_x + half_particle_size + 1
            y_start = center_y - half_particle_size
            y_end = center_y + half_particle_size + 1

            # Ensure slices are within image bounds (should be handled by clipping center_x/y)
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(width, x_end)
            y_end = min(height, y_end)

            # Add the Gaussian particle kernel to the image data
            image_data[y_start:y_end, x_start:x_end] = np.maximum(
                image_data[y_start:y_end, x_start:x_end],
                particle_kernel
            )

    img = Image.fromarray(image_data, mode='L')

    # Explicitly set TIFF metadata for single-channel grayscale
    tiff_tags = {
        262: 1, # PhotometricInterpretation: 1 (BlackIsZero)
        277: 1, # SamplesPerPixel: 1
        258: 8  # BitsPerSample: 8 (for uint8 data)
    }

    try:
        img.save(output_filename, compression="tiff_deflate", tiffinfo=tiff_tags)
        print(f"Successfully generated '{output_filename}'")
    except Exception as e:
        print(f"Error saving image: {e}")

# --- User-defined parameters ---
IMAGE_WIDTH = 2023
IMAGE_HEIGHT = 2023
NUM_PARTICLES = 40000
PARTICLE_SIZE = 7 # Increased particle size to better show Gaussian effect
PARTICLE_SIGMA = 1.5 # Standard deviation for Gaussian (adjust to change spread)

# --- Generate the images ---
if __name__ == "__main__":
    # Generate the first frame and get original particle positions
    original_x, original_y = generate_piv_test_image(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_PARTICLES, PARTICLE_SIZE, PARTICLE_SIGMA, "piv_test_image_frame1.tif")

    # Generate the second frame with displaced particles
    generate_displaced_image(IMAGE_WIDTH, IMAGE_HEIGHT, original_x, original_y, PARTICLE_SIZE, PARTICLE_SIGMA, "piv_test_image_frame2.tif")
