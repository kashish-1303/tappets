import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
current_dir = "fan"  # Path to your current dataset
output_dir = "rp_images/fan"  # Path to the desired output structure

# Create new directories
os.makedirs(os.path.join(output_dir, "train", "normal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "abnormal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "normal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "abnormal"), exist_ok=True)

# Collect all file paths
normal_files = []
abnormal_files = []

for machine_id in os.listdir(current_dir):
    machine_path = os.path.join(current_dir, machine_id)
    if os.path.isdir(machine_path):
        # Normal files
        normal_path = os.path.join(machine_path, "normal")
        if os.path.exists(normal_path):
            normal_files.extend([os.path.join(normal_path, f) for f in os.listdir(normal_path) if f.endswith(".wav")])

        # Abnormal files
        abnormal_path = os.path.join(machine_path, "abnormal")
        if os.path.exists(abnormal_path):
            abnormal_files.extend([os.path.join(abnormal_path, f) for f in os.listdir(abnormal_path) if f.endswith(".wav")])

# Split into training and testing sets (80% train, 20% test)
train_normal, test_normal = train_test_split(normal_files, test_size=0.2, random_state=42)
train_abnormal, test_abnormal = train_test_split(abnormal_files, test_size=0.2, random_state=42)

# Move files to the new structure
def move_files(file_list, dest_dir):
    for file in file_list:
        shutil.copy(file, dest_dir)

# Move normal files
move_files(train_normal, os.path.join(output_dir, "train", "normal"))
move_files(test_normal, os.path.join(output_dir, "test", "normal"))

# Move abnormal files
move_files(train_abnormal, os.path.join(output_dir, "train", "abnormal"))
move_files(test_abnormal, os.path.join(output_dir, "test", "abnormal"))

print("Dataset reorganization complete!")
