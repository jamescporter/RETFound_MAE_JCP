import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from multiprocessing import Manager, cpu_count
import cv2  # OpenCV for image resizing and Laplacian calculation
import logging
import gc

# ================================
# Configurable Parameters
# ================================

# Path to save the log file
LOG_FILE_PATH = 'processing.log'

# Paths to checkpoint files
STATS_CSV_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_image_statistics_checkpoint.csv'
PROCESSED_IMAGES_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_processed_images.txt'

# Root folder containing all images
ROOT_FOLDER = r"D:\2024_PHD-DATA_CFP-imgs_cropped"

# Number of workers (all cores but 2)
NUM_WORKERS = max(cpu_count() - 2, 1)

# Image resizing parameters (for GLCM calculations)
RESIZE_DIMENSIONS = (512, 512)

# Frequency of checkpoint saves (every N images)
CHECKPOINT_FREQUENCY = 1000

# ================================
# Setup Logging
# ================================

logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================
# Main Processing Function
# ================================

def process_image(subdir, file):
    filename = os.path.splitext(file)[0]
    img_path = os.path.join(subdir, file)

    try:
        # Load image
        img = np.array(Image.open(img_path))
        if img.ndim == 2:  # Grayscale image
            img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel grayscale

        # Calculate individual statistics
        median = np.median(img, axis=(0, 1))
        min_pixel = img.min(axis=(0, 1))
        max_pixel = img.max(axis=(0, 1))

        img_flat = img.reshape(-1, 3)
        img_skewness = skew(img_flat, axis=0)
        img_kurtosis = kurtosis(img_flat, axis=0)
        img_entropy = shannon_entropy(img)

        # Convert to grayscale for GLCM and additional statistics
        img_gray = np.mean(img, axis=2).astype(np.uint8)
        img_gray_resized = cv2.resize(img_gray, RESIZE_DIMENSIONS)

        # GLCM calculations
        glcm = graycomatrix(img_gray_resized, distances=[1], angles=[0], symmetric=True, normed=True)
        img_contrast = graycoprops(glcm, 'contrast')[0, 0]
        img_energy = graycoprops(glcm, 'energy')[0, 0]
        img_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Calculate RMS contrast
        rms_contrast = np.sqrt(np.mean((img_gray - np.mean(img_gray)) ** 2))

        # Calculate image sharpness using the variance of the Laplacian
        sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        # Clean up memory
        del img, img_flat, img_gray, img_gray_resized, glcm

        # Return the filename and statistics for this image
        return filename, {
            'median': median,
            'min_pixel': min_pixel,
            'max_pixel': max_pixel,
            'skewness': img_skewness,
            'kurtosis': img_kurtosis,
            'entropy': img_entropy,
            'contrast': img_contrast,
            'energy': img_energy,
            'homogeneity': img_homogeneity,
            'rms_contrast': rms_contrast,
            'sharpness': sharpness
        }, img.sum(axis=(0, 1)), (img ** 2).sum(axis=(0, 1)), img.shape[0] * img.shape[1]

    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None

# ================================
# Main Program Execution
# ================================

def main():
    logging.info("Processing started.")
    # Load previously processed data if exists
    if os.path.exists(STATS_CSV_PATH):
        stats_df = pd.read_csv(STATS_CSV_PATH, index_col='filename')
        processed_images = set(stats_df.index)
    else:
        stats_df = pd.DataFrame(columns=[
            'median', 'min_pixel', 'max_pixel', 'skewness',
            'kurtosis', 'entropy', 'contrast', 'energy', 'homogeneity',
            'rms_contrast', 'sharpness'
        ])
        processed_images = set()

    # Load list of processed images if exists
    if os.path.exists(PROCESSED_IMAGES_PATH):
        with open(PROCESSED_IMAGES_PATH, 'r') as f:
            processed_images.update(f.read().splitlines())

    # Initialize accumulators for global mean and std dev
    sum_pixels = np.zeros(3)  # Assuming RGB images
    sum_squared_pixels = np.zeros(3)
    total_pixels = 0

    # Use a Manager to share the processed images set between processes
    with Manager() as manager:
        processed_images_dict = manager.dict({img: None for img in processed_images})

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            total_images = sum(len(files) for _, _, files in os.walk(ROOT_FOLDER) if any(f.endswith('.png') for f in files))
            first_1000_time_start = time.time()

            for subdir, _, files in os.walk(ROOT_FOLDER):
                for file in files:
                    if file.endswith('.png'):
                        filename = os.path.splitext(file)[0]
                        if filename not in processed_images_dict:
                            futures.append(executor.submit(process_image, subdir, file))

            processed_count = 0
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is None:
                    continue  # Skip failed processing

                filename, stats, img_sum, img_squared_sum, img_pixel_count = result

                # Update global accumulators
                sum_pixels += img_sum
                sum_squared_pixels += img_squared_sum
                total_pixels += img_pixel_count

                # Update DataFrame with the statistics
                stats_df.loc[filename] = stats

                # Mark image as processed
                processed_images_dict[filename] = None
                processed_count += 1

                # Estimate time after processing the first CHECKPOINT_FREQUENCY images
                if processed_count == CHECKPOINT_FREQUENCY:
                    first_1000_time_end = time.time()
                    time_per_image = (first_1000_time_end - first_1000_time_start) / CHECKPOINT_FREQUENCY
                    remaining_images = total_images - processed_count
                    estimated_total_time = remaining_images * time_per_image

                    logging.info(f"First {CHECKPOINT_FREQUENCY} imgs took: {first_1000_time_end - first_1000_time_start:.2f} seconds.")
                    logging.info(f"Estimated remaining time: {estimated_total_time / 3600:.2f} hours.")

                # Periodically save progress and run garbage collection every CHECKPOINT_FREQUENCY images
                if processed_count % CHECKPOINT_FREQUENCY == 0:
                    stats_df.to_csv(STATS_CSV_PATH)
                    with open(PROCESSED_IMAGES_PATH, 'w') as f:
                        f.write("\n".join(list(processed_images_dict.keys())))
                    gc.collect()  # Run garbage collection less frequently
                    logging.info(f"Checkpoint saved after processing {processed_count} images.")

        # Final save after all futures complete
        stats_df.to_csv(STATS_CSV_PATH)
        with open(PROCESSED_IMAGES_PATH, 'w') as f:
            f.write("\n".join(list(processed_images_dict.keys())))

    # Calculate global mean and standard deviation
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)
    std_dev = np.sqrt(variance)

    # Output global results
    print(f"Global Mean: {mean}")
    print(f"Global Standard Deviation: {std_dev}")
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
