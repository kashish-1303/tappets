import numpy as np
import librosa
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_audio(file_path, sr=16000, duration=None):
    """
    Load an audio file and return the signal
    
    Args:
        file_path (str): Path to audio file
        sr (int): Sample rate
        duration (float): Duration in seconds to load (None for full file)
    
    Returns:
        np.ndarray: Audio signal
    """
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    return y

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
    """
    Process a single audio file: load, filter, segment, and save segments
    
    Args:
        file_path (str): Path to audio file
        output_dir (str): Directory to save processed segments
        sr (int): Sample rate
        segment_length (float): Length of each segment in seconds
        overlap (float): Overlap between segments (0 to 1)
        cutoff_freq (int): Cutoff frequency for low-pass filter
    
    Returns:
        int: Number of segments created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    signal = load_audio(file_path, sr=sr)
    
    # Apply low-pass filter
    filtered_signal = apply_low_pass_filter(signal, sr=sr, cutoff_freq=cutoff_freq)
    
    # Extract segments
    segments = extract_audio_segments(filtered_signal, sr=sr, 
                                     segment_length=segment_length, 
                                     overlap=overlap)
    
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
    # Example usage
    input_dir = "path/to/MIMII/dataset/fan/id_00/normal"
    output_dir = "processed_data/normal"
    
    batch_process_audio_files(input_dir, output_dir, 
                             sr=16000, 
                             segment_length=1.0, 
                             overlap=0.5,
                             cutoff_freq=4000)
