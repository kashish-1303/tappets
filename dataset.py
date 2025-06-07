# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from sklearn.model_selection import train_test_split

# class AnomalyDetectionDataset(Dataset):
#     """
#     Dataset for anomaly detection using encoded time series images
#     """
#     def __init__(self, image_paths, labels, transform=None):
#         """
#         Initialize the dataset
        
#         Args:
#             image_paths (list): List of paths to images
#             labels (list): List of labels (0 for normal, 1 for anomaly)
#             transform (callable, optional): Optional transform to be applied on an image
#         """
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform if transform is not None else transforms.ToTensor()
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')  # Convert to RGB for compatibility with CNN models
        
#         # Apply transformations
#         image = self.transform(image)
        
#         # Get label
#         label = self.labels[idx]
        
#         return image, label

# def create_dataset_from_directories(normal_dir, anomaly_dir, encoding_method='rp', 
#                                    test_size=0.25, random_state=42):
#     """
#     Create dataset from directories of normal and anomaly images
    
#     Args:
#         normal_dir (str): Directory containing normal encoded images
#         anomaly_dir (str): Directory containing anomaly encoded images
#         encoding_method (str): Image encoding method to use ('gasf', 'gadf', 'mtf', 'rp')
#         test_size (float): Proportion of the dataset to include in the test split
#         random_state (int): Random state for reproducibility
    
#     Returns:
#         tuple: (train_dataset, test_dataset)
#     """
#     # Get all images of the specified encoding method
#     normal_images = glob.glob(os.path.join(normal_dir, f"*_{encoding_method}.png"))
#     anomaly_images = glob.glob(os.path.join(anomaly_dir, f"*_{encoding_method}.png"))
    
#     # Create labels (0 for normal, 1 for anomaly)
#     normal_labels = [0] * len(normal_images)
#     anomaly_labels = [1] * len(anomaly_images)
    
#     # Combine data
#     all_images = normal_images + anomaly_images
#     all_labels = normal_labels + anomaly_labels
    
#     # Split into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         all_images, all_labels, test_size=test_size, 
#         random_state=random_state, stratify=all_labels
#     )
    
#     # Define transformations
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     # Create datasets
#     train_dataset = AnomalyDetectionDataset(X_train, y_train, transform=train_transform)
#     test_dataset = AnomalyDetectionDataset(X_test, y_test, transform=test_transform)
    
#     return train_dataset, test_dataset

# def get_data_loaders(train_dataset, test_dataset, batch_size=32, num_workers=4):
#     """
#     Create data loaders from datasets
    
#     Args:
#         train_dataset (Dataset): Training dataset, can be None if only test loader is needed
#         test_dataset (Dataset): Test dataset
#         batch_size (int): Batch size
#         num_workers (int): Number of worker threads for loading data
    
#     Returns:
#         tuple: (train_loader, test_loader) - train_loader will be None if train_dataset is None
#     """
#     train_loader = None
#     if train_dataset is not None:
#         train_loader = DataLoader(
#             train_dataset, 
#             batch_size=batch_size, 
#             shuffle=True, 
#             num_workers=0
#         )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )
    
#     return train_loader, test_loader

# if __name__ == "__main__":
#     # Example usage
#     normal_dir = "encoded_images/normal"
#     anomaly_dir = "encoded_images/abnormal"
    
#     # Create datasets using Recurrence Plot encoding
#     train_dataset, test_dataset = create_dataset_from_directories(
#         normal_dir, anomaly_dir, encoding_method='rp'
#     )
    
#     # Create data loaders
#     train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)
    
#     # Print some statistics
#     print(f"Training samples: {len(train_dataset)}")
#     print(f"Test samples: {len(test_dataset)}")
    
#     # Get a batch of data
#     for images, labels in train_loader:
#         print(f"Batch shape: {images.shape}")
#         print(f"Labels: {labels}")
#         break
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class AnomalyDetectionDataset(Dataset):
    """
    Dataset for anomaly detection using encoded time series images
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset
        
        Args:
            image_paths (list): List of paths to images
            labels (list): List of labels (0 for normal, 1 for anomaly)
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform is not None else transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for compatibility with CNN models
        
        # Apply transformations
        image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label

def create_dataset_from_directories(normal_dir, anomaly_dir, encoding_method='rp', 
                                   test_size=0.25, random_state=42):
    """
    Create dataset from directories of normal and anomaly images
    
    Args:
        normal_dir (str): Directory containing normal encoded images
        anomaly_dir (str): Directory containing anomaly encoded images
        encoding_method (str): Image encoding method to use ('gasf', 'gadf', 'mtf', 'rp')
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Get all images of the specified encoding method
    normal_images = glob.glob(os.path.join(normal_dir, f"*_{encoding_method}.png"))
    anomaly_images = glob.glob(os.path.join(anomaly_dir, f"*_{encoding_method}.png"))
    
    # Create labels (0 for normal, 1 for anomaly)
    normal_labels = [0] * len(normal_images)
    anomaly_labels = [1] * len(anomaly_images)
    
    # Combine data
    all_images = normal_images + anomaly_images
    all_labels = normal_labels + anomaly_labels
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=test_size, 
        random_state=random_state, stratify=all_labels
    )
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AnomalyDetectionDataset(X_train, y_train, transform=train_transform)
    test_dataset = AnomalyDetectionDataset(X_test, y_test, transform=test_transform)
    
    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create data loaders from datasets
    
    Args:
        train_dataset (Dataset): Training dataset, can be None if only test loader is needed
        test_dataset (Dataset): Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker threads for loading data
    
    Returns:
        tuple: (train_loader, test_loader) - train_loader will be None if train_dataset is None
    """
    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    normal_dir = "encoded_images/normal"
    anomaly_dir = "encoded_images/abnormal"
    
    # Create datasets using Recurrence Plot encoding
    train_dataset, test_dataset = create_dataset_from_directories(
        normal_dir, anomaly_dir, encoding_method='rp'
    )
    
    # Create data loaders
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)
    
    # Print some statistics
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get a batch of data
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break

