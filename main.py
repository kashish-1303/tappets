

#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
import platform
import numpy as np
import cv2
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# Import custom modules
from preprocess_audio import batch_process_audio_files
from image_encoding import batch_encode_time_series
from dataset import create_dataset_from_directories, get_data_loaders
from train_model import create_model, train_model, validate, plot_confusion_matrix
from gram_cam import batch_visualize_grad_cam, visualize_grad_cam

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Anomaly Detection using Time Series Imaging')
    
    # General settings
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'encode', 'train', 'evaluate', 'visualize', 'all'],
                        help='Operation mode: preprocess audio, encode time series, train model, evaluate model, visualize with GradCAM, or run all steps')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_mps', action='store_true', help='Use Apple Silicon GPU via MPS backend')
    
    # Data paths
    parser.add_argument('--normal_audio', type=str, default='data/normal', help='Path to normal audio files')
    parser.add_argument('--anomaly_audio', type=str, default='data/anomaly', help='Path to anomaly audio files')
    parser.add_argument('--processed_normal', type=str, default='processed_data/normal', help='Path to store processed normal data')
    parser.add_argument('--processed_anomaly', type=str, default='processed_data/abnormal', help='Path to store processed anomaly data')
    parser.add_argument('--encoded_normal', type=str, default='encoded_images/normal', help='Path to store encoded normal images')
    parser.add_argument('--encoded_anomaly', type=str, default='encoded_images/abnormal', help='Path to store encoded anomaly images')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    # Audio preprocessing parameters
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for audio preprocessing')
    parser.add_argument('--segment_length', type=float, default=1.0, help='Segment length in seconds')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between segments (0 to 1)')
    parser.add_argument('--cutoff_freq', type=int, default=4000, help='Cutoff frequency for low-pass filter')
    
    # Image encoding parameters
    parser.add_argument('--image_size', type=int, default=224, help='Size of encoded images')
    parser.add_argument('--encoding_method', type=str, default='rp', choices=['rp', 'gasf', 'gadf', 'mtf', 'all'],
                        help='Encoding method: rp (Recurrence Plot), gasf (Gramian Angular Summation Field), '
                             'gadf (Gramian Angular Difference Field), mtf (Markov Transition Field), or all')
    
    # Training parameters
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50', 'vgg16', 'densenet121'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    # Evaluation parameters
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint for evaluation')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # MPS doesn't have specific seeding requirements like CUDA
    # But we do want to ensure PyTorch operations are deterministic when possible

def get_device(gpu, use_mps=False):
    """Get the device to use"""
    # Auto-detect Apple Silicon and use MPS if available
    is_apple_silicon = (
        platform.system() == 'Darwin' and 
        platform.machine().startswith('arm')
    )
    
    # Check for MPS (Apple Silicon) 
    if (use_mps or is_apple_silicon) and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU via MPS")
        return device
    
    # Fall back to CUDA if specified and available
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(gpu)}")
        return device
    
    # Otherwise use CPU
    device = torch.device('cpu')
    print("Using CPU")
    return device

def preprocess_data(args):
    """Preprocess audio data"""
    print("\n=== Audio Preprocessing ===")
    
    # Create output directories if they don't exist
    os.makedirs(args.processed_normal, exist_ok=True)
    os.makedirs(args.processed_anomaly, exist_ok=True)
    
    # Process normal audio files
    print("Processing normal audio files...")
    normal_segments = batch_process_audio_files(
        args.normal_audio, 
        args.processed_normal,
        sr=args.sr,
        segment_length=args.segment_length,
        overlap=args.overlap,
        cutoff_freq=args.cutoff_freq
    )
    
    # Process anomaly audio files
    print("Processing anomaly audio files...")
    anomaly_segments = batch_process_audio_files(
        args.anomaly_audio, 
        args.processed_anomaly,
        sr=args.sr,
        segment_length=args.segment_length,
        overlap=args.overlap,
        cutoff_freq=args.cutoff_freq
    )
    
    print(f"Preprocessing complete. Created {normal_segments} normal segments and {anomaly_segments} anomaly segments.")

def encode_data(args):
    """Encode time series data as images"""
    print("\n=== Time Series Encoding ===")
    
    # Create output directories if they don't exist
    os.makedirs(args.encoded_normal, exist_ok=True)
    os.makedirs(args.encoded_anomaly, exist_ok=True)
    
    # Encode normal time series
    print("Encoding normal time series...")
    batch_encode_time_series(args.processed_normal, args.encoded_normal, image_size=args.image_size)
    
    # Encode anomaly time series
    print("Encoding anomaly time series...")
    batch_encode_time_series(args.processed_anomaly, args.encoded_anomaly, image_size=args.image_size)
    
    print("Encoding complete.")



# 1. Fix for main.py - modify the train function to ensure model stays on device
# Find this section in main.py and replace it with the following:

def train(args, device):
    """Train the model"""
    print("\n=== Model Training ===")
    
    # Define model name with encoding method
    full_model_name = f"{args.model_name}_{args.encoding_method}"
    model_save_dir = os.path.join(args.model_dir, full_model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create datasets and data loaders
    print("Creating datasets...")
    train_dataset, test_dataset = create_dataset_from_directories(
        args.encoded_normal,
        args.encoded_anomaly,
        encoding_method=args.encoding_method,
        test_size=0.25,
        random_state=args.seed
    )
    
    train_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=args.pretrained
    )
    
    # IMPORTANT: Move model to device BEFORE creating optimizer
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Train the model
    print("Starting training...")
    
    best_model_path = os.path.join(model_save_dir, f"{full_model_name}_best.pth")
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # MPS-specific handling
    is_mps = device.type == 'mps'
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print('-' * 10)
        
        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Move inputs and labels to the same device as model
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Verify model is on correct device before forward pass
            if next(model.parameters()).device != device:
                model = model.to(device)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                
                # For MPS device, explicitly move optimizer state to device if needed
                if is_mps:
                    # This ensures optimizer's internal state remains on the correct device
                    for param in model.parameters():
                        if param.grad is not None and param.grad.device != device:
                            param.grad = param.grad.to(device)
                
                optimizer.step()
                
                # Verify model is still on correct device after optimizer step
                if next(model.parameters()).device != device:
                    model = model.to(device)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.float() / total
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Validation phase - ensure model is on correct device
        model = model.to(device)  # Ensure model is on correct device before validation
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion, device)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Ensure model is on CPU before saving to avoid MPS-related issues
        model_cpu = model.to('cpu')
        
        # Save model checkpoint
        checkpoint_path = os.path.join(model_save_dir, f"{full_model_name}_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc
        }, checkpoint_path)
        
        # Move model back to original device
        model = model.to(device)
        
        # Check if this is the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_cpu.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc
            }, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Load best model for evaluation - handle MPS compatibility
    model_checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f"{full_model_name}_history.png"))
    plt.close()
    
    return model, test_loader, history

# 2. Also replace save_model function with a more MPS-friendly version:

def save_model(model, optimizer, epoch, loss, accuracy, save_path):
    """Save model with device compatibility for MPS"""
    # Get device of the model
    device = next(model.parameters()).device
    
    # Move model to CPU before saving to avoid MPS-related issues
    model_cpu = model.to('cpu')
    
    # Save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_cpu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, save_path)
    
    # Move model back to original device
    model = model.to(device)
    return model  # Return the model to ensure it's on the right device

# To implement these fixes, you need to make the following changes in your main.py file:

# 1. Replace the load_model function with our fixed version
# Look for this function around line 169 and replace it with:

def load_model(model_path, model_name, device):
    """Load model with device compatibility for MPS"""
    # Create the model architecture - always on CPU first
    model = create_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False
    )
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load state dict while model is on CPU
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Then move to target device after loading weights
    model = model.to(device)
    
    # Return model and checkpoint
    return model, checkpoint

# 2. Replace the visualize function with our MPS-compatible version
# Look for this function around line 440 and replace it with:

def visualize(args, device, model=None):
    """Generate GradCAM visualizations"""
    print("\n=== GradCAM Visualization ===")
    
    # Load model if not provided
    if model is None:
        # If model_path is not specified, use the default path
        if args.model_path is None:
            model_dir = os.path.join(args.model_dir, f"{args.model_name}_{args.encoding_method}")
            model_path = os.path.join(model_dir, f"{args.model_name}_{args.encoding_method}_best.pth")
        else:
            model_path = args.model_path
        
        # Load model using our fixed function
        print(f"Loading model from {model_path}...")
        model, _ = load_model(model_path, args.model_name, device)
        
        # Quick verification
        print(f"Model loaded successfully and moved to {next(model.parameters()).device}")
    
    # Define output directory for visualizations
    vis_dir = os.path.join(args.vis_dir, f"{args.model_name}_{args.encoding_method}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate GradCAM visualizations for normal samples
    print("Generating GradCAM visualizations for normal samples...")
    normal_vis_dir = os.path.join(vis_dir, "normal")
    os.makedirs(normal_vis_dir, exist_ok=True)
    
    # For MPS compatibility in GradCAM - always move to CPU for GradCAM
    # MPS has limited support for some operations needed by GradCAM
    if device.type == 'mps':
        print("Moving model to CPU for GradCAM compatibility")
        model = model.to('cpu')
        vis_device = torch.device('cpu')
    else:
        vis_device = device
    
    try:
        batch_visualize_grad_cam(
            model=model,
            image_dir=args.encoded_normal,
            output_dir=normal_vis_dir,
            pattern=f"*_{args.encoding_method}.png",
            target_layer_name="layer4",
            show=False,
            device=vis_device  # Use CPU for visualization
        )
        
        # Generate GradCAM visualizations for anomaly samples
        print("Generating GradCAM visualizations for anomaly samples...")
        anomaly_vis_dir = os.path.join(vis_dir, "anomaly")
        os.makedirs(anomaly_vis_dir, exist_ok=True)
        
        batch_visualize_grad_cam(
            model=model,
            image_dir=args.encoded_anomaly,
            output_dir=anomaly_vis_dir,
            pattern=f"*_{args.encoding_method}.png",
            target_layer_name="layer4",
            show=False,
            device=vis_device  # Use CPU for visualization
        )
    finally:
        # Move model back to original device if needed
        if device.type == 'mps' and vis_device.type == 'cpu':
            model = model.to(device)
            print(f"Model moved back to {device}")
    
    print(f"Visualization complete. Results saved to {vis_dir}")

# 3. You also need to make sure your evaluate function is compatible
# Look for this function around line 385 and make these changes:

def evaluate(args, device, model=None, test_loader=None):
    """Evaluate the model"""
    print("\n=== Model Evaluation ===")
    
    # Load model if not provided
    if model is None:
        # If model_path is not specified, use the default path
        if args.model_path is None:
            model_dir = os.path.join(args.model_dir, f"{args.model_name}_{args.encoding_method}")
            model_path = os.path.join(model_dir, f"{args.model_name}_{args.encoding_method}_best.pth")
        else:
            model_path = args.model_path
        
        # Load model using our fixed function
        print(f"Loading model from {model_path}...")
        model, _ = load_model(model_path, args.model_name, device)
        print(f"Model loaded successfully and moved to {next(model.parameters()).device}")
    
    # Create test data loader if not provided
    if test_loader is None:
        print("Creating test dataset...")
        _, test_dataset = create_dataset_from_directories(
            args.encoded_normal,
            args.encoded_anomaly,
            encoding_method=args.encoding_method,
            test_size=0.25,
            random_state=args.seed
        )
        
        _, test_loader = get_data_loaders(
            None,
            test_dataset,
            batch_size=args.batch_size,
            num_workers=4
        )
    
    # Evaluate the model
    print("Evaluating model...")
    criterion = nn.CrossEntropyLoss()
    _, accuracy, predictions, labels = validate(model, test_loader, criterion, device)
    
    # Print metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    
    # Plot confusion matrix
    vis_dir = os.path.join(args.vis_dir, f"{args.model_name}_{args.encoding_method}")
    os.makedirs(vis_dir, exist_ok=True)
    cm_path = os.path.join(vis_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, predictions, save_path=cm_path)
    
    print(f"Evaluation complete. Visualizations saved to {vis_dir}")
    return model

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device - pass the use_mps flag to prioritize MPS backend
    device = get_device(args.gpu, use_mps=args.use_mps)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Print system info
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.release()} on {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    else:
        print("MPS backend not available in this PyTorch version")
    print(f"Using device: {device}")
    print("=" * 30)
    
    # Run the requested operation
    if args.mode == 'preprocess' or args.mode == 'all':
        preprocess_data(args)
    
    if args.mode == 'encode' or args.mode == 'all':
        encode_data(args)
    
    model = None
    test_loader = None
    
    if args.mode == 'train' or args.mode == 'all':
        model, test_loader, history = train(args, device)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        model = evaluate(args, device, model, test_loader)
    
    if args.mode == 'visualize' or args.mode == 'all':
        visualize(args, device, model)
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()
