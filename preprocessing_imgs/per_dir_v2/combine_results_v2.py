import pandas as pd
import numpy as np
import glob
from numba import njit

# Load all global stats files
stats_files = glob.glob("D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\*_global_stats.csv")

# Initialize global accumulators
global_sum_pixels = np.zeros(3, dtype=np.float64)
global_sum_squares = np.zeros(3, dtype=np.float64)
global_min_pixel = np.full(3, 255, dtype=np.float64)
global_max_pixel = np.zeros(3, dtype=np.float64)
global_total_pixels = 0

# Process each stats file
for file in stats_files:
    stats = pd.read_csv(file)

    global_sum_pixels += stats.iloc[0]['global_sum_pixels']
    global_sum_squares += stats.iloc[0]['global_sum_squares']
    global_min_pixel = np.minimum(global_min_pixel, stats.iloc[0]['global_min_pixel'])
    global_max_pixel = np.maximum(global_max_pixel, stats.iloc[0]['global_max_pixel'])
    global_total_pixels += stats.iloc[0]['global_total_pixels']


# Calculate the final global statistics
@njit
def compute_final_stats(sum_pixels, sum_squares, total_pixels):
    global_mean = sum_pixels / total_pixels
    global_variance = (sum_squares / total_pixels) - (global_mean ** 2)
    global_std_dev = np.sqrt(global_variance)
    return global_mean, global_variance, global_std_dev


global_mean, global_variance, global_std_dev = compute_final_stats(global_sum_pixels, global_sum_squares,
                                                                   global_total_pixels)

# Combine results into a DataFrame
global_stats = pd.DataFrame({
    'global_mean': global_mean,
    'global_variance': global_variance,
    'global_std_dev': global_std_dev,
    'global_min_pixel': global_min_pixel,
    'global_max_pixel': global_max_pixel
})

print(global_stats)

# Combine the old statistics
old_stats_files = glob.glob("D:\\2024_PHD-DATA_CFP_PROCESSING\\*image_statistics_checkpoint.csv")
all_old_stats = pd.concat([pd.read_csv(f, index_col=0) for f in old_stats_files])

# Split multi-value columns into separate columns
for column in ['median', 'min_pixel', 'max_pixel', 'skewness', 'kurtosis']:
    all_old_stats[[f"{column}_R", f"{column}_G", f"{column}_B"]] = all_old_stats[column].str.strip('[]').str.split(
        expand=True).astype(float)
    all_old_stats.drop(columns=[column], inplace=True)

# Save the combined results
all_old_stats.to_csv('D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\final_image_statistics.csv')
