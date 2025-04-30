import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.spatial.distance import cdist

def create_recurrence_plot(audio_path, sr=16000, img_size=224):
    try:
        # Load audio data
        y, sr = librosa.load(audio_path, sr=sr)

        # Compute recurrence matrix
        hop_length = 64  # You can adjust this parameter
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        recurrence_matrix = cdist(chromagram.T, chromagram.T, metric='cosine')

        # Threshold the recurrence matrix
        threshold = np.mean(recurrence_matrix) + 2 * np.std(recurrence_matrix)
        recurrence_matrix[recurrence_matrix > threshold] = 0

        # Resize the recurrence matrix to the desired image size
        rp_img = resize(recurrence_matrix, (img_size, img_size), anti_aliasing=True)

        # Convert to 8-bit grayscale image (0-255)
        rp_img = (rp_img * 255).astype(np.uint8)

        return rp_img
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Directory paths
data_dir = "fan"  # Path to your current dataset structure
output_dir = "rp_images/fan"  # Path to save RP images

# Create output directories
os.makedirs(os.path.join(output_dir, "train", "normal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "abnormal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "normal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "abnormal"), exist_ok=True)

# Process audio files and save RP images
for phase in ['train', 'test']:
    for label in ['normal', 'abnormal']:
        input_folder = os.path.join(data_dir, phase, label)
        output_folder = os.path.join(output_dir, phase, label)

        for filename in os.listdir(input_folder):
            if filename.endswith(".wav"):
                audio_path = os.path.join(input_folder, filename)
                rp_image = create_recurrence_plot(audio_path)

                if rp_image is not None:
                    plt.imsave(os.path.join(output_folder, f"{filename.replace('.wav', '.png')}"), rp_image, cmap='gray')

print("Recurrence Plot images created!")
