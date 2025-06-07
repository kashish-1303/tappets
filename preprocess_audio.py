# import numpy as np
# import librosa
# from scipy.fft import fft, ifft
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# def load_audio(file_path, sr=16000, duration=None):
#     try:
#         y, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
#         return y
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None

# def apply_low_pass_filter(signal, sr=16000, cutoff_freq=4000):
#     """
#     Apply low-pass filter to the signal using FFT
    
#     Args:
#         signal (np.ndarray): Input audio signal
#         sr (int): Sample rate of the signal
#         cutoff_freq (int): Cutoff frequency for the low-pass filter
    
#     Returns:
#         np.ndarray: Filtered signal
#     """
#     # Convert to frequency domain
#     freq_data = fft(signal)
    
#     # Calculate frequency bins
#     freq_bins = np.fft.fftfreq(len(signal), 1/sr)
    
#     # Create a mask for low-pass filter
#     mask = np.abs(freq_bins) <= cutoff_freq
    
#     # Apply the mask
#     filtered_freq_data = freq_data * mask
    
#     # Convert back to time domain
#     filtered_signal = np.real(ifft(filtered_freq_data))
    
#     return filtered_signal

# def extract_audio_segments(signal, sr=16000, segment_length=1.0, overlap=0.5):
#     """
#     Extract overlapping segments from the audio signal
    
#     Args:
#         signal (np.ndarray): Input audio signal
#         sr (int): Sample rate
#         segment_length (float): Length of each segment in seconds
#         overlap (float): Overlap between segments (0 to 1)
    
#     Returns:
#         list: List of signal segments
#     """
#     # Calculate samples
#     segment_samples = int(segment_length * sr)
#     hop_samples = int(segment_samples * (1 - overlap))
    
#     # Extract segments
#     segments = []
#     for i in range(0, len(signal) - segment_samples + 1, hop_samples):
#         segment = signal[i:i + segment_samples]
#         segments.append(segment)
    
#     return segments

# def process_audio_file(file_path, output_dir, sr=16000, segment_length=1.0, 
#                        overlap=0.5, cutoff_freq=4000):
#     """
#     Process a single audio file: load, filter, segment, and save segments
    
#     Args:
#         file_path (str): Path to audio file
#         output_dir (str): Directory to save processed segments
#         sr (int): Sample rate
#         segment_length (float): Length of each segment in seconds
#         overlap (float): Overlap between segments (0 to 1)
#         cutoff_freq (int): Cutoff frequency for low-pass filter
    
#     Returns:
#         int: Number of segments created
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load audio file
#     signal = load_audio(file_path, sr=sr)
    
#     # Apply low-pass filter
#     filtered_signal = apply_low_pass_filter(signal, sr=sr, cutoff_freq=cutoff_freq)
    
#     # Extract segments
#     segments = extract_audio_segments(filtered_signal, sr=sr, 
#                                      segment_length=segment_length, 
#                                      overlap=overlap)
    
#     # Save segments
#     filename = os.path.splitext(os.path.basename(file_path))[0]
#     for i, segment in enumerate(segments):
#         segment_path = os.path.join(output_dir, f"{filename}_segment_{i}.npy")
#         np.save(segment_path, segment)
    
#     return len(segments)

# def batch_process_audio_files(input_dir, output_dir, sr=16000, segment_length=1.0, 
#                              overlap=0.5, cutoff_freq=4000):
#     """
#     Process all audio files in a directory
    
#     Args:
#         input_dir (str): Directory containing audio files
#         output_dir (str): Directory to save processed segments
#         sr (int): Sample rate
#         segment_length (float): Length of each segment in seconds
#         overlap (float): Overlap between segments (0 to 1)
#         cutoff_freq (int): Cutoff frequency for low-pass filter
    
#     Returns:
#         int: Total number of segments created
#     """
#     # Get all audio files
#     audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
#                   if f.endswith(('.wav', '.mp3', '.flac'))]
    
#     total_segments = 0
    
#     # Process each file
#     for file_path in tqdm(audio_files, desc="Processing audio files"):
#         segments = process_audio_file(file_path, output_dir, sr=sr, 
#                                      segment_length=segment_length, 
#                                      overlap=overlap, cutoff_freq=cutoff_freq)
#         total_segments += segments
    
#     print(f"Total segments created: {total_segments}")
#     return total_segments

# if __name__ == "__main__":
#     # Process both normal and abnormal data
#     base_input_dir = "data/train"
#     base_output_dir = "processed_data"
    
#     # Process normal data
#     normal_input_dir = os.path.join(base_input_dir, "normal")
#     normal_output_dir = os.path.join(base_output_dir, "normal")
    
#     print("Processing normal audio files...")
#     batch_process_audio_files(normal_input_dir, normal_output_dir, 
#                              sr=16000, 
#                              segment_length=1.0, 
#                              overlap=0.5,
#                              cutoff_freq=4000)
    
#     # Process abnormal data
#     abnormal_input_dir = os.path.join(base_input_dir, "abnormal")
#     abnormal_output_dir = os.path.join(base_output_dir, "abnormal")
    
#     print("Processing abnormal audio files...")
#     batch_process_audio_files(abnormal_input_dir, abnormal_output_dir, 
#                              sr=16000, 
#                              segment_length=1.0, 
#                              overlap=0.5,
#                              cutoff_freq=4000)
# import numpy as np
# import librosa
# from scipy.fft import fft, ifft
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# def load_audio(file_path, sr=16000, duration=None):
#     try:
#         y, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
#         return y
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None

# def apply_low_pass_filter(signal, sr=16000, cutoff_freq=4000):
#     """
#     Apply low-pass filter to the signal using FFT
    
#     Args:
#         signal (np.ndarray): Input audio signal
#         sr (int): Sample rate of the signal
#         cutoff_freq (int): Cutoff frequency for the low-pass filter
    
#     Returns:
#         np.ndarray: Filtered signal
#     """
#     # Convert to frequency domain
#     freq_data = fft(signal)
    
#     # Calculate frequency bins
#     freq_bins = np.fft.fftfreq(len(signal), 1/sr)
    
#     # Create a mask for low-pass filter
#     mask = np.abs(freq_bins) <= cutoff_freq
    
#     # Apply the mask
#     filtered_freq_data = freq_data * mask
    
#     # Convert back to time domain
#     filtered_signal = np.real(ifft(filtered_freq_data))
    
#     return filtered_signal

# def extract_audio_segments(signal, sr=16000, segment_length=1.0, overlap=0.5):
#     """
#     Extract overlapping segments from the audio signal
    
#     Args:
#         signal (np.ndarray): Input audio signal
#         sr (int): Sample rate
#         segment_length (float): Length of each segment in seconds
#         overlap (float): Overlap between segments (0 to 1)
    
#     Returns:
#         list: List of signal segments
#     """
#     # Calculate samples
#     segment_samples = int(segment_length * sr)
#     hop_samples = int(segment_samples * (1 - overlap))
    
#     # Extract segments
#     segments = []
#     for i in range(0, len(signal) - segment_samples + 1, hop_samples):
#         segment = signal[i:i + segment_samples]
#         segments.append(segment)
    
#     return segments

# def process_audio_file(file_path, output_dir, sr=16000, segment_length=1.0, 
#                        overlap=0.5, cutoff_freq=4000):
#     """
#     Process a single audio file: load, filter, segment, and save segments
    
#     Args:
#         file_path (str): Path to audio file
#         output_dir (str): Directory to save processed segments
#         sr (int): Sample rate
#         segment_length (float): Length of each segment in seconds
#         overlap (float): Overlap between segments (0 to 1)
#         cutoff_freq (int): Cutoff frequency for low-pass filter
    
#     Returns:
#         int: Number of segments created
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load audio file
#     signal = load_audio(file_path, sr=sr)
    
#     # Apply low-pass filter
#     filtered_signal = apply_low_pass_filter(signal, sr=sr, cutoff_freq=cutoff_freq)
    
#     # Extract segments
#     segments = extract_audio_segments(filtered_signal, sr=sr, 
#                                      segment_length=segment_length, 
#                                      overlap=overlap)
    
#     # Save segments
#     filename = os.path.splitext(os.path.basename(file_path))[0]
#     for i, segment in enumerate(segments):
#         segment_path = os.path.join(output_dir, f"{filename}_segment_{i}.npy")
#         np.save(segment_path, segment)
    
#     return len(segments)

# def batch_process_audio_files(input_dir, output_dir, sr=16000, segment_length=1.0, 
#                              overlap=0.5, cutoff_freq=4000):
#     """
#     Process all audio files in a directory
    
#     Args:
#         input_dir (str): Directory containing audio files
#         output_dir (str): Directory to save processed segments
#         sr (int): Sample rate
#         segment_length (float): Length of each segment in seconds
#         overlap (float): Overlap between segments (0 to 1)
#         cutoff_freq (int): Cutoff frequency for low-pass filter
    
#     Returns:
#         int: Total number of segments created
#     """
#     # Get all audio files
#     audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
#                   if f.endswith(('.wav', '.mp3', '.flac'))]
    
#     total_segments = 0
    
#     # Process each file
#     for file_path in tqdm(audio_files, desc="Processing audio files"):
#         segments = process_audio_file(file_path, output_dir, sr=sr, 
#                                      segment_length=segment_length, 
#                                      overlap=overlap, cutoff_freq=cutoff_freq)
#         total_segments += segments
    
#     print(f"Total segments created: {total_segments}")
#     return total_segments

# if __name__ == "__main__":
#     # Process both normal and abnormal data
#     base_input_dir = "data/train"
#     base_output_dir = "processed_data"
    
#     # Process normal data
#     normal_input_dir = os.path.join(base_input_dir, "normal")
#     normal_output_dir = os.path.join(base_output_dir, "normal")
    
#     print("Processing normal audio files...")
#     batch_process_audio_files(normal_input_dir, normal_output_dir, 
#                              sr=16000, 
#                              segment_length=1.0, 
#                              overlap=0.5,
#                              cutoff_freq=4000)
    
#     # Process abnormal data
#     abnormal_input_dir = os.path.join(base_input_dir, "abnormal")
#     abnormal_output_dir = os.path.join(base_output_dir, "abnormal")
    
#     print("Processing abnormal audio files...")
#     batch_process_audio_files(abnormal_input_dir, abnormal_output_dir, 
#                              sr=16000, 
#                              segment_length=1.0, 
#                              overlap=0.5,
#                              cutoff_freq=4000)

import numpy as np
import librosa
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_audio(file_path, sr=16000, duration=None):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def apply_low_pass_filter(signal, sr=16000, cutoff_freq=4000):
    """
    Apply low-pass filter to the signal using FFT
    
    Args:
        signal (np.ndarray): Input audio signal
        sr (int): Sample rate of the signal
        cutoff_freq (int): Cutoff frequency for the low-pass filter
    
    Returns:
        np.ndarray: Filtered signal
    """
    # Convert to frequency domain
    freq_data = fft(signal)
    
    # Calculate frequency bins
    freq_bins = np.fft.fftfreq(len(signal), 1/sr)
    
    # Create a mask for low-pass filter
    mask = np.abs(freq_bins) <= cutoff_freq
    
    # Apply the mask
    filtered_freq_data = freq_data * mask
    
    # Convert back to time domain
    filtered_signal = np.real(ifft(filtered_freq_data))
    
    return filtered_signal

def extract_audio_segments(signal, sr=16000, segment_length=1.0, overlap=0.5):
    """
    Extract overlapping segments from the audio signal
    
    Args:
        signal (np.ndarray): Input audio signal
        sr (int): Sample rate
        segment_length (float): Length of each segment in seconds
        overlap (float): Overlap between segments (0 to 1)
    
    Returns:
        list: List of signal segments
    """
    # Calculate samples
    segment_samples = int(segment_length * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    
    # Extract segments
    segments = []
    for i in range(0, len(signal) - segment_samples + 1, hop_samples):
        segment = signal[i:i + segment_samples]
        segments.append(segment)
    
    return segments

def process_audio_file(file_path, output_dir, sr=16000, segment_length=1.0, 
                       overlap=0.5, cutoff_freq=4000):
    """Enhanced audio processing with better feature extraction"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    signal = load_audio(file_path, sr=sr)
    if signal is None:
        return 0
    
    # Apply low-pass filter
    filtered_signal = apply_low_pass_filter(signal, sr=sr, cutoff_freq=cutoff_freq)
    
    # Extract MFCC features instead of raw audio
    mfcc = librosa.feature.mfcc(y=filtered_signal, sr=sr, n_mfcc=13, 
                                hop_length=512, n_fft=2048)
    
    # Convert MFCC to time series (take mean across frequency bins)
    mfcc_time_series = np.mean(mfcc, axis=0)
    
    # Extract segments from MFCC time series
    segment_samples = int(segment_length * sr / 512)  # Adjust for hop_length
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    for i in range(0, len(mfcc_time_series) - segment_samples + 1, hop_samples):
        segment = mfcc_time_series[i:i + segment_samples]
        if len(segment) == segment_samples:  # Ensure consistent length
            segments.append(segment)
    
    # Save segments
    filename = os.path.splitext(os.path.basename(file_path))[0]
    for i, segment in enumerate(segments):
        segment_path = os.path.join(output_dir, f"{filename}_segment_{i}.npy")
        np.save(segment_path, segment)
    
    return len(segments)

def batch_process_audio_files(input_dir, output_dir, sr=16000, segment_length=1.0, 
                             overlap=0.5, cutoff_freq=4000):
    """
    Process all audio files in a directory
    
    Args:
        input_dir (str): Directory containing audio files
        output_dir (str): Directory to save processed segments
        sr (int): Sample rate
        segment_length (float): Length of each segment in seconds
        overlap (float): Overlap between segments (0 to 1)
        cutoff_freq (int): Cutoff frequency for low-pass filter
    
    Returns:
        int: Total number of segments created
    """
    # Get all audio files
    audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    total_segments = 0
    
    # Process each file
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        segments = process_audio_file(file_path, output_dir, sr=sr, 
                                     segment_length=segment_length, 
                                     overlap=overlap, cutoff_freq=cutoff_freq)
        total_segments += segments
    
    print(f"Total segments created: {total_segments}")
    return total_segments

if __name__ == "__main__":
    # Process both normal and abnormal data
    base_input_dir = "data/train"
    base_output_dir = "processed_data"
    
    # Process normal data
    normal_input_dir = os.path.join(base_input_dir, "normal")
    normal_output_dir = os.path.join(base_output_dir, "normal")
    
    print("Processing normal audio files...")
    batch_process_audio_files(normal_input_dir, normal_output_dir, 
                             sr=16000, 
                             segment_length=1.0, 
                             overlap=0.5,
                             cutoff_freq=4000)
    
    # Process abnormal data
    abnormal_input_dir = os.path.join(base_input_dir, "abnormal")
    abnormal_output_dir = os.path.join(base_output_dir, "abnormal")
    
    print("Processing abnormal audio files...")
    batch_process_audio_files(abnormal_input_dir, abnormal_output_dir, 
                             sr=16000, 
                             segment_length=1.0, 
                             overlap=0.5,
                             cutoff_freq=4000)
