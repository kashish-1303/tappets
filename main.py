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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(gpu):
    """Get the device to use"""
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu)}")
    else:
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
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Train the model
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        patience=args.early_stopping,
        save_dir=model_save_dir,
        model_name=full_model_name,
        
    )
    
    print(f"Training complete. Model saved to {model_save_dir}")
    return model, test_loader, history

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
        
        # Load model
        print(f"Loading model from {model_path}...")
        model = create_model(
            model_name=args.model_name,
            num_classes=2,
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
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
        
        # Load model
        print(f"Loading model from {model_path}...")
        model = create_model(
            model_name=args.model_name,
            num_classes=2,
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
    # Define output directory for visualizations
    vis_dir = os.path.join(args.vis_dir, f"{args.model_name}_{args.encoding_method}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate GradCAM visualizations for normal samples
    print("Generating GradCAM visualizations for normal samples...")
    normal_vis_dir = os.path.join(vis_dir, "normal")
    os.makedirs(normal_vis_dir, exist_ok=True)
    
    batch_visualize_grad_cam(
        model=model,
        image_dir=args.encoded_normal,
        output_dir=normal_vis_dir,
        pattern=f"*_{args.encoding_method}.png",
        target_layer_name="layer4",
        show=False
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
        show=False
    )
    
    print(f"Visualization complete. Results saved to {vis_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.gpu)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
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