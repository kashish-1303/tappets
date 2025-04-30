import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import cv2

def normalize_ts(ts, min_val=-1, max_val=1):
    """
    Normalize time series data to specified range
    
    Args:
        ts (np.ndarray): Time series data
        min_val (float): Minimum value after normalization
        max_val (float): Maximum value after normalization
    
    Returns:
        np.ndarray: Normalized time series
    """
    ts_min, ts_max = np.min(ts), np.max(ts)
    if ts_min == ts_max:
        return np.zeros_like(ts)
    return min_val + ((ts - ts_min) * (max_val - min_val)) / (ts_max - ts_min)

def create_gasf(ts, image_size=224):
    """
    Create Gramian Angular Summation Field from time series
    
    Args:
        ts (np.ndarray): Time series data
        image_size (int): Size of the output image (square)
    
    Returns:
        np.ndarray: GASF image
    """
    # Normalize the time series to [-1, 1]
    normalized_ts = normalize_ts(ts, -1, 1)
    
    # If necessary, resize the normalized time series to match image_size
    if len(normalized_ts) != image_size:
        indices = np.linspace(0, len(normalized_ts) - 1, image_size).astype(int)
        normalized_ts = normalized_ts[indices]
    
    # Calculate GASF matrix
    # GASF(i,j) = cos(phi_i + phi_j) = x_i*x_j - sqrt(1-x_i^2)*sqrt(1-x_j^2)
    sqrt_ts = np.sqrt(1 - np.square(normalized_ts))
    gasf = np.outer(normalized_ts, normalized_ts) - np.outer(sqrt_ts, sqrt_ts)
    
    # Normalize to [0, 1] for image
    gasf_normalized = (gasf + 1) / 2
    
    # Convert to 8-bit grayscale
    gasf_image = (gasf_normalized * 255).astype(np.uint8)
    
    return gasf_image

def create_gadf(ts, image_size=224):
    """
    Create Gramian Angular Difference Field from time series
    
    Args:
        ts (np.ndarray): Time series data
        image_size (int): Size of the output image (square)
    
    Returns:
        np.ndarray: GADF image
    """
    # Normalize the time series to [-1, 1]
    normalized_ts = normalize_ts(ts, -1, 1)
    
    # If necessary, resize the normalized time series to match image_size
    if len(normalized_ts) != image_size:
        indices = np.linspace(0, len(normalized_ts) - 1, image_size).astype(int)
        normalized_ts = normalized_ts[indices]
    
    # Calculate GADF matrix
    # GADF(i,j) = sin(phi_i - phi_j) = x_j*sqrt(1-x_i^2) - x_i*sqrt(1-x_j^2)
    sqrt_ts = np.sqrt(1 - np.square(normalized_ts))
    gadf = np.outer(sqrt_ts, normalized_ts) - np.outer(normalized_ts, sqrt_ts)
    
    # Normalize to [0, 1] for image
    gadf_normalized = (gadf + 1) / 2
    
    # Convert to 8-bit grayscale
    gadf_image = (gadf_normalized * 255).astype(np.uint8)
    
    return gadf_image

def create_mtf(ts, image_size=224, n_bins=10):
    """
    Create Markov Transition Field from time series
    
    Args:
        ts (np.ndarray): Time series data
        image_size (int): Size of the output image (square)
        n_bins (int): Number of quantile bins
    
    Returns:
        np.ndarray: MTF image
    """
    # Normalize time series (optional but helps with binning)
    normalized_ts = normalize_ts(ts)
    
    # If necessary, resize the normalized time series to match image_size
    if len(normalized_ts) != image_size:
        indices = np.linspace(0, len(normalized_ts) - 1, image_size).astype(int)
        normalized_ts = normalized_ts[indices]
    
    # Create quantile bins and assign each point to a bin
    bins = np.linspace(np.min(normalized_ts), np.max(normalized_ts), n_bins + 1)
    binned_ts = np.digitize(normalized_ts, bins) - 1
    binned_ts[binned_ts >= n_bins] = n_bins - 1  # Fix any edge cases
    
    # Create the Markov transition matrix (Q × Q)
    w = np.zeros((n_bins, n_bins))
    for i in range(len(binned_ts) - 1):
        w[binned_ts[i], binned_ts[i + 1]] += 1
    
    # Normalize rows (avoid division by zero)
    row_sums = w.sum(axis=1)
    w_norm = np.zeros_like(w, dtype=float)
    for i in range(n_bins):
        if row_sums[i] > 0:
            w_norm[i, :] = w[i, :] / row_sums[i]
    
    # Create the Markov Transition Field using the matrix
    mtf = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            mtf[i, j] = w_norm[binned_ts[i], binned_ts[j]]
    
    # Convert to 8-bit grayscale
    mtf_image = (mtf * 255).astype(np.uint8)
    
    return mtf_image

def create_rp(ts, image_size=224, epsilon=None, percentage=10):
    """
    Create Recurrence Plot from time series
    
    Args:
        ts (np.ndarray): Time series data
        image_size (int): Size of the output image (square)
        epsilon (float): Threshold (if None, calculated from percentage)
        percentage (float): Percentage of the maximum distance to use as threshold
    
    Returns:
        np.ndarray: RP image
    """
    # Normalize time series (optional)
    normalized_ts = normalize_ts(ts)
    
    # If necessary, resize the normalized time series to match image_size
    if len(normalized_ts) != image_size:
        indices = np.linspace(0, len(normalized_ts) - 1, image_size).astype(int)
        normalized_ts = normalized_ts[indices]
    
    # Calculate pairwise distances
    distances = squareform(pdist(normalized_ts.reshape(-1, 1), 'euclidean'))
    
    # Determine epsilon threshold if not provided
    if epsilon is None:
        epsilon = np.percentile(distances, percentage)
    
    # Create recurrence plot matrix
    rp = (distances <= epsilon).astype(np.uint8) * 255
    
    return rp

def save_image(img_array, file_path):
    """
    Save numpy array as image
    
    Args:
        img_array (np.ndarray): Image as numpy array
        file_path (str): Path to save the image
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Ensure image is properly scaled
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image and save
    img = Image.fromarray(img_array)
    img.save(file_path)

def encode_all_methods(ts_path, output_dir, image_size=224):
    """
    Apply all four encoding methods to a time series file
    
    Args:
        ts_path (str): Path to time series data file (.npy)
        output_dir (str): Directory to save encoded images
        image_size (int): Size of the output images
    
    Returns:
        dict: Paths to saved images
    """
    # Load time series
    ts = np.load(ts_path)
    
    # Create base filename
    base_filename = os.path.splitext(os.path.basename(ts_path))[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply all encoding methods
    gasf_img = create_gasf(ts, image_size)
    gadf_img = create_gadf(ts, image_size)
    mtf_img = create_mtf(ts, image_size)
    rp_img = create_rp(ts, image_size)
    
    # Save images
    gasf_path = os.path.join(output_dir, f"{base_filename}_gasf.png")
    gadf_path = os.path.join(output_dir, f"{base_filename}_gadf.png")
    mtf_path = os.path.join(output_dir, f"{base_filename}_mtf.png")
    rp_path = os.path.join(output_dir, f"{base_filename}_rp.png")
    
    save_image(gasf_img, gasf_path)
    save_image(gadf_img, gadf_path)
    save_image(mtf_img, mtf_path)
    save_image(rp_img, rp_path)
    
    return {
        "gasf": gasf_path,
        "gadf": gadf_path,
        "mtf": mtf_path,
        "rp": rp_path
    }

def batch_encode_time_series(input_dir, output_dir, image_size=224):
    """
    Batch process all time series files in a directory
    
    Args:
        input_dir (str): Directory containing time series .npy files
        output_dir (str): Directory to save encoded images
        image_size (int): Size of the output images
    """
    # Get all .npy files
    ts_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    for ts_path in tqdm(ts_files, desc="Encoding time series"):
        encode_all_methods(ts_path, output_dir, image_size)
    
    print(f"Processed {len(ts_files)} time series files")

if __name__ == "__main__":
    # Example usage
    input_dir = "processed_data/normal"
    output_dir = "encoded_images/normal"
    
    batch_encode_time_series(input_dir, output_dir, image_size=224)