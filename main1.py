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
import cv2
import glob
from PIL import Image
import pandas as pd
from datetime import datetime
import torchvision.transforms as transforms
import json

# Import custom modules
from preprocess_audio import batch_process_audio_files
from image_encoding import batch_encode_time_series
from dataset import create_dataset_from_directories, get_data_loaders
from train_model import create_model, train_model, validate, plot_confusion_matrix
from gram_cam import batch_visualize_grad_cam, visualize_grad_cam

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Anomaly Detection using Time Series Imaging based on the paper: "Encoding Time Series as Images for Anomaly Detection in Manufacturing Processes Using Convolutional Neural Networks and Grad‑CAM"')
    
    # General settings
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'encode', 'train', 'evaluate', 'visualize', 'all', 'compare'],
                        help='Operation mode: preprocess audio, encode time series, train model, evaluate model, visualize with GradCAM, run all steps, or compare all models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_mps', action='store_true', help='Use Apple Silicon GPU via MPS backend')
    
    # Data paths
    parser.add_argument('--normal_audio', type=str, default='fan_reorg/train/normal', help='Path to normal audio files')
    parser.add_argument('--anomaly_audio', type=str, default='fan_reorg/train/abnormal', help='Path to anomaly audio files')
    parser.add_argument('--processed_normal', type=str, default='processed_data/normal', help='Path to store processed normal data')
    parser.add_argument('--processed_anomaly', type=str, default='processed_data/abnormal', help='Path to store processed anomaly data')
    parser.add_argument('--encoded_normal', type=str, default='encoded_images/normal', help='Path to store encoded normal images')
    parser.add_argument('--encoded_anomaly', type=str, default='encoded_images/abnormal', help='Path to store encoded anomaly images')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save experiment results')
    
    # Audio preprocessing parameters
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for audio preprocessing')
    parser.add_argument('--segment_length', type=float, default=1.0, help='Segment length in seconds')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between segments (0 to 1)')
    parser.add_argument('--cutoff_freq', type=int, default=4000, help='Cutoff frequency for low-pass filter')
    
    # Image encoding parameters
    parser.add_argument('--image_size', type=int, default=224, help='Size of encoded images')
    parser.add_argument('--encoding_method', type=str, default='all', choices=['rp', 'gasf', 'gadf', 'mtf', 'all'],
                        help='Encoding method: rp (Recurrence Plot), gasf (Gramian Angular Summation Field), '
                             'gadf (Gramian Angular Difference Field), mtf (Markov Transition Field), or all')
    
    # Training parameters
    parser.add_argument('--model_name', type=str, default='all', choices=['resnet50', 'vgg16', 'densenet121', 'all'],
                        help='Model architecture: resnet50, vgg16, densenet121, or all (to run all three)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(gpu, use_mps=False):
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

def encode_data(args, encoding_methods=None):
    """Encode time series data as images"""
    print("\n=== Time Series Encoding ===")
    
    # Create output directories if they don't exist
    os.makedirs(args.encoded_normal, exist_ok=True)
    os.makedirs(args.encoded_anomaly, exist_ok=True)
    
    # If encoding methods not specified, use the one from args
    if encoding_methods is None:
        if args.encoding_method == 'all':
            encoding_methods = ['rp', 'gasf', 'gadf', 'mtf']
        else:
            encoding_methods = [args.encoding_method]

    for method in encoding_methods:
        print(f"Using encoding method: {method}")
        
        # Create method-specific directories
        normal_dir = os.path.join(args.encoded_normal, method)
        anomaly_dir = os.path.join(args.encoded_anomaly, method)
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)
        
        # Override the encoding_method attribute temporarily
        original_method = args.encoding_method
        args.encoding_method = method
        
        # Encode normal time series
        print(f"Encoding normal time series with {method}...")
        batch_encode_time_series(args.processed_normal, normal_dir, method=method, image_size=args.image_size)
        
        # Encode anomaly time series
        print(f"Encoding anomaly time series with {method}...")
        batch_encode_time_series(args.processed_anomaly, anomaly_dir, method=method, image_size=args.image_size)
        
        # Restore the original encoding_method
        args.encoding_method = original_method
    
    print("Encoding complete.")

def train(args, device, model_name=None, encoding_method=None):
    """Train the model"""
    print("\n=== Model Training ===")
    
    # Use specified model name and encoding method or defaults from args
    if model_name is None:
        model_name = args.model_name
    
    if encoding_method is None:
        encoding_method = args.encoding_method
    
    # Define model name with encoding method
    full_model_name = f"{model_name}_{encoding_method}"
    model_save_dir = os.path.join(args.model_dir, full_model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Define paths for encoded images based on the encoding method
    if encoding_method != 'all':
        encoded_normal_dir = os.path.join(args.encoded_normal, encoding_method)
        encoded_anomaly_dir = os.path.join(args.encoded_anomaly, encoding_method)
    else:
        # For 'all', we'll use the base directories (combined approach would be implemented here)
        encoded_normal_dir = args.encoded_normal
        encoded_anomaly_dir = args.encoded_anomaly
    
    # Create datasets and data loaders
    print("Creating datasets...")
    try:
        train_dataset, test_dataset = create_dataset_from_directories(
            encoded_normal_dir,
            encoded_anomaly_dir,
            encoding_method=encoding_method,
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
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None, None
    
    # Create model
    print(f"Creating {model_name} model...")
    model = create_model(
        model_name=model_name,
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

def visualize(args, device, model=None, model_name=None, encoding_method=None):
    """Generate GradCAM visualizations"""
    print("\n=== GradCAM Visualization ===")
    
    # Use specified model name and encoding method or defaults from args
    if model_name is None:
        model_name = args.model_name
    
    if encoding_method is None:
        encoding_method = args.encoding_method
    
    # If 'all' was specified for model_name or encoding_method, we need to handle each combination
    model_names = [model_name] if model_name != 'all' else ['resnet50', 'vgg16', 'densenet121']
    encoding_methods = [encoding_method] if encoding_method != 'all' else ['rp', 'gasf', 'gadf', 'mtf']
    
    for m_name in model_names:
        for e_method in encoding_methods:
            print(f"\nVisualizing {m_name} with {e_method} encoding...")
            
            # Define paths based on the current model and encoding method
            full_model_name = f"{m_name}_{e_method}"
            
            # Load model if not provided
            if model is None or m_name != model_name or e_method != encoding_method:
                model_dir = os.path.join(args.model_dir, full_model_name)
                model_path = os.path.join(model_dir, f"{full_model_name}_best.pth")
                
                if not os.path.exists(model_path):
                    print(f"Model checkpoint not found at {model_path}. Skipping...")
                    continue
                
                # Load model using our fixed function
                print(f"Loading model from {model_path}...")
                try:
                    model, _ = load_model(model_path, m_name, device)
                    print(f"Model loaded successfully and moved to {next(model.parameters()).device}")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    continue
            
            # Define input directories for the current encoding method
            encoded_normal_dir = os.path.join(args.encoded_normal, e_method)
            encoded_anomaly_dir = os.path.join(args.encoded_anomaly, e_method)
            
            # Define output directory for visualizations
            vis_dir = os.path.join(args.vis_dir, full_model_name)
            os.makedirs(vis_dir, exist_ok=True)
            
            # For MPS compatibility in GradCAM - always move to CPU for GradCAM
            # MPS has limited support for some operations needed by GradCAM
            original_device = next(model.parameters()).device
            if device.type == 'mps':
                print("Moving model to CPU for GradCAM compatibility")
                model = model.to('cpu')
            
            try:
                # Generate GradCAM visualizations for normal samples
                print("Generating GradCAM visualizations for normal samples...")
                normal_vis_dir = os.path.join(vis_dir, "normal")
                os.makedirs(normal_vis_dir, exist_ok=True)
                
                # Determine the pattern based on encoding method
                pattern = f"*_{e_method}.png"
                
                # Determine target layer based on model architecture
                target_layer = None
                if m_name == "resnet50":
                    target_layer = "layer4"
                elif m_name == "vgg16":
                    target_layer = "features[-1]"  # Last convolution layer
                elif m_name == "densenet121":
                    target_layer = "features.denseblock4"  # Last dense block
                
                batch_visualize_grad_cam(
                    model=model,
                    image_dir=encoded_normal_dir,
                    output_dir=normal_vis_dir,
                    pattern=pattern,
                    target_layer_name=target_layer,
                    show=False
                )
                
                # Generate GradCAM visualizations for anomaly samples
                print("Generating GradCAM visualizations for anomaly samples...")
                anomaly_vis_dir = os.path.join(vis_dir, "anomaly")
                os.makedirs(anomaly_vis_dir, exist_ok=True)
                
                batch_visualize_grad_cam(
                    model=model,
                    image_dir=encoded_anomaly_dir,
                    output_dir=anomaly_vis_dir,
                    pattern=pattern,
                    target_layer_name=target_layer,
                    show=False
                )
            except Exception as e:
                print(f"Error during GradCAM visualization: {e}")
            finally:
                # Move model back to original device if needed
                if device.type == 'mps' and original_device.type == 'mps':
                    model = model.to(device)
                    print(f"Model moved back to {device}")
    
    print(f"Visualization complete. Results saved to {args.vis_dir}")

def evaluate(args, device, model=None, test_loader=None, model_name=None, encoding_method=None):
    """Evaluate the model"""
    print("\n=== Model Evaluation ===")
    
    # Use specified model name and encoding method or defaults from args
    if model_name is None:
        model_name = args.model_name
    
    if encoding_method is None:
        encoding_method = args.encoding_method
    
    # Define the full model name
    full_model_name = f"{model_name}_{encoding_method}"
    
    # Load model if not provided
    if model is None:
        # If model_path is not specified, use the default path
        if args.model_path is None:
            model_dir = os.path.join(args.model_dir, full_model_name)
            model_path = os.path.join(model_dir, f"{full_model_name}_best.pth")
        else:
            model_path = args.model_path
        
        # Load model using our fixed function
        print(f"Loading model from {model_path}...")
        try:
            model, _ = load_model(model_path, model_name, device)
            print(f"Model loaded successfully and moved to {next(model.parameters()).device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    # Create test data loader if not provided
    if test_loader is None:
        print("Creating test dataset...")
        # Define paths for encoded images based on the encoding method
        if encoding_method != 'all':
            encoded_normal_dir = os.path.join(args.encoded_normal, encoding_method)
            encoded_anomaly_dir = os.path.join(args.encoded_anomaly, encoding_method)
        else:
            # For 'all', we'll use the base directories
            encoded_normal_dir = args.encoded_normal
            encoded_anomaly_dir = args.encoded_anomaly
        
        try:
            _, test_dataset = create_dataset_from_directories(
                encoded_normal_dir,
                encoded_anomaly_dir,
                encoding_method=encoding_method,
                test_size=0.25,
                random_state=args.seed
            )
            
            _, test_loader = get_data_loaders(
                None,
                test_dataset,
                batch_size=args.batch_size,
                num_workers=4
            )
        except Exception as e:
            print(f"Error creating test dataset: {e}")
            return None
    
    # Evaluate the model
    print("Evaluating model...")
    criterion = nn.CrossEntropyLoss()
    val_loss, accuracy, predictions, labels = validate(model, test_loader, criterion, device)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
    
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Return metrics for comparison
    metrics = {
        'model': model_name,
        'encoding': encoding_method,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'val_loss': val_loss
    }
    
    # Plot confusion matrix
    vis_dir = os.path.join(args.vis_dir, full_model_name)
    os.makedirs(vis_dir, exist_ok=True)
    cm_path = os.path.join(vis_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, predictions, save_path=cm_path)
    
    print(f"Evaluation complete. Visualizations saved to {vis_dir}")
    return metrics

def compare_all_models(args, device):
    """Compare all models with all encoding methods"""
    print("\n=== Comparing All Models with All Encoding Methods ===")
    
    # Define lists of models and encoding methods
    model_names = ['resnet50', 'vgg16', 'densenet121']
    encoding_methods = ['rp', 'gasf', 'gadf', 'mtf']
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create a timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.results_dir, f"comparison_results_{timestamp}.csv")
    json_results_file = os.path.join(args.results_dir, f"comparison_results_{timestamp}.json")
    
    all_results = []
    
    # For each model and encoding method combination
    for model_name in model_names:
        for encoding_method in encoding_methods:
            print(f"\n=== Training and evaluating {model_name} with {encoding_method} encoding ===")
            
            # Train the model
            model, test_loader, history = train(args, device, model_name, encoding_method)
            
            if model is not None and test_loader is not None:
                # Evaluate the model
                metrics = evaluate(args, device, model, test_loader, model_name, encoding_method)
                
                if metrics is not None:
                    all_results.append(metrics)
                    
                    # Visualize with GradCAM
                    visualize(args, device, model, model_name, encoding_method)
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        
        # Save results to JSON
        with open(json_results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Print comparison table
        print("\n=== Model Comparison Results ===")
        print(results_df)
        
        # Find the best model
        best_model_idx = results_df['accuracy'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        
        print("\n=== Best Model ===")
        print(f"Model: {best_model['model']}")
        print(f"Encoding: {best_model['encoding']}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        print(f"Precision: {best_model['precision']:.4f}")
        print(f"Recall: {best_model['recall']:.4f}")
        print(f"F1 Score: {best_model['f1_score']:.4f}")
        
        return best_model
    else:
        print("No results to compare.")
        return None
    
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
    
    if args.mode == 'compare':
        compare_all_models(args, device)

    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()