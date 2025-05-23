import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import json
from datetime import datetime

# Import our modules
from preprocess_audio import batch_process_audio_files
from image_encoding import batch_encode_time_series
from dataset import create_dataset_from_directories, get_data_loaders
from train_model import create_model, train_model
from gram_cam import batch_visualize_grad_cam

class MIMIIAnomalyDetector:
    """
    Complete pipeline for MIMII anomaly detection using time-series to image conversion
    """
    
    def __init__(self, config):
        """
        Initialize the anomaly detector with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories for the pipeline"""
        dirs = [
            self.config['processed_data_dir'],
            self.config['encoded_images_dir'],
            self.config['models_dir'],
            self.config['visualizations_dir'],
            os.path.join(self.config['processed_data_dir'], 'normal'),
            os.path.join(self.config['processed_data_dir'], 'abnormal'),
            os.path.join(self.config['encoded_images_dir'], 'normal'),
            os.path.join(self.config['encoded_images_dir'], 'abnormal'),
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_preprocess_audio(self):
        """
        Step 1: Preprocess audio files according to paper methodology
        - Load audio files
        - Apply FFT transformation
        - Apply low-pass filtering (fc = 0.1 * sampling_rate)
        - Segment into fixed-length windows
        """
        print("=" * 60)
        print("STEP 1: PREPROCESSING AUDIO FILES")
        print("=" * 60)
        
        # Calculate cutoff frequency (10% of sampling rate as per paper)
        cutoff_freq = int(0.1 * self.config['sample_rate'])
        print(f"Applying low-pass filter with cutoff frequency: {cutoff_freq} Hz")
        
        # Process normal files
        normal_input = os.path.join(self.config['dataset_dir'], 'train','normal')
        normal_output = os.path.join(self.config['processed_data_dir'], 'normal')
        
        if os.path.exists(normal_input):
            print(f"Processing normal files from: {normal_input}")
            normal_segments = batch_process_audio_files(
                input_dir=normal_input,
                output_dir=normal_output,
                sr=self.config['sample_rate'],
                segment_length=self.config['segment_length'],
                overlap=self.config['overlap'],
                cutoff_freq=cutoff_freq
            )
            print(f"Created {normal_segments} normal segments")
        
        # Process abnormal files
        abnormal_input = os.path.join(self.config['dataset_dir'],'train', 'abnormal')
        abnormal_output = os.path.join(self.config['processed_data_dir'], 'abnormal')
        
        if os.path.exists(abnormal_input):
            print(f"Processing abnormal files from: {abnormal_input}")
            abnormal_segments = batch_process_audio_files(
                input_dir=abnormal_input,
                output_dir=abnormal_output,
                sr=self.config['sample_rate'],
                segment_length=self.config['segment_length'],
                overlap=self.config['overlap'],
                cutoff_freq=cutoff_freq
            )
            print(f"Created {abnormal_segments} abnormal segments")
        
        print("Step 1 completed successfully!")
    
    def step2_encode_images(self):
        """
        Step 2: Convert time-series to images using four encoding methods
        - Gramian Angular Summation Field (GASF)
        - Gramian Angular Difference Field (GADF)
        - Markov Transition Field (MTF)
        - Recurrence Plot (RP)
        """
        print("=" * 60)
        print("STEP 2: ENCODING TIME-SERIES TO IMAGES")
        print("=" * 60)
        
        # Encode normal segments
        normal_input = os.path.join(self.config['processed_data_dir'], 'normal')
        normal_output = os.path.join(self.config['encoded_images_dir'], 'normal')
        
        print(f"Encoding normal segments from: {normal_input}")
        batch_encode_time_series(
            input_dir=normal_input,
            output_dir=normal_output,
            image_size=self.config['image_size']
        )
        
        # Encode abnormal segments  
        abnormal_input = os.path.join(self.config['processed_data_dir'], 'abnormal')
        abnormal_output = os.path.join(self.config['encoded_images_dir'], 'abnormal')
        
        print(f"Encoding abnormal segments from: {abnormal_input}")
        batch_encode_time_series(
            input_dir=abnormal_input,
            output_dir=abnormal_output,
            image_size=self.config['image_size']
        )
        
        print("Step 2 completed successfully!")
    
    def step3_train_model(self, encoding_method='rp'):
        """
        Step 3: Train CNN model on encoded images
        Default to RP + ResNet50 combination (best performer from paper)
        
        Args:
            encoding_method (str): Encoding method to use ('gasf', 'gadf', 'mtf', 'rp')
        """
        print("=" * 60)
        print(f"STEP 3: TRAINING MODEL WITH {encoding_method.upper()} ENCODING")
        print("=" * 60)
        
        # Create datasets
        normal_dir = os.path.join(self.config['encoded_images_dir'], 'normal')
        abnormal_dir = os.path.join(self.config['encoded_images_dir'], 'abnormal')
        
        print("Creating datasets...")
        train_dataset, test_dataset = create_dataset_from_directories(
            normal_dir=normal_dir,
            anomaly_dir=abnormal_dir,
            encoding_method=encoding_method,
            test_size=0.25,  # 75% train, 25% test as per paper
            random_state=42
        )
        
        # Create data loaders with paper's specifications
        train_loader, test_loader = get_data_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=self.config['batch_size'],  # Paper uses 32
            num_workers=4
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create model (ResNet50 as per paper's best result)
        model = create_model(
            model_name=self.config['model_name'],
            num_classes=2,
            pretrained=False
        )
        model = model.to(self.device)
        
        # Setup training components with paper's hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate']  # Paper uses 1e-05
        )
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Train model
        model_name = f"{self.config['model_name']}_{encoding_method}"
        save_dir = os.path.join(self.config['models_dir'], model_name)
        
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            num_epochs=self.config['epochs'],  # Paper uses 20
            patience=self.config['patience'],
            save_dir=save_dir,
            model_name=model_name
        )
        
        # Save configuration and results
        results = {
            'config': self.config,
            'encoding_method': encoding_method,
            'model_name': self.config['model_name'],
            'final_metrics': history['final_metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Step 3 completed successfully!")
        return trained_model, save_dir
    
    def step4_generate_explanations(self, model, model_dir, encoding_method='rp'):
        """
        Step 4: Generate Grad-CAM explanations for model decisions
        
        Args:
            model (torch.nn.Module): Trained model
            model_dir (str): Directory containing the model
            encoding_method (str): Encoding method used
        """
        print("=" * 60)
        print("STEP 4: GENERATING GRAD-CAM EXPLANATIONS")
        print("=" * 60)
        
        # Create visualization directory
        viz_dir = os.path.join(self.config['visualizations_dir'], f"gradcam_{encoding_method}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate explanations for both normal and abnormal samples
        for label in ['normal', 'abnormal']:
            image_dir = os.path.join(self.config['encoded_images_dir'], label)
            output_dir = os.path.join(viz_dir, label)
            
            pattern = f"*_{encoding_method}.png"
            
            print(f"Generating Grad-CAM for {label} samples...")
            batch_visualize_grad_cam(
                model=model,
                image_dir=image_dir,
                output_dir=output_dir,
                pattern=pattern,
                target_layer_name='layer4',  # Last conv layer in ResNet
                show=False
            )
        
        print("Step 4 completed successfully!")
    
    def run_complete_pipeline(self, encoding_method='rp'):
        """
        Run the complete pipeline from raw audio to trained model with explanations
        
        Args:
            encoding_method (str): Encoding method to use ('gasf', 'gadf', 'mtf', 'rp')
        """
        print("STARTING COMPLETE MIMII ANOMALY DETECTION PIPELINE")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_time = datetime.now()
        
        # Step 1: Preprocess audio
        self.step1_preprocess_audio()
        
        # Step 2: Encode to images
        self.step2_encode_images()
        
        # Step 3: Train model
        trained_model, model_dir = self.step3_train_model(encoding_method)
        
        # Step 4: Generate explanations
        self.step4_generate_explanations(trained_model, model_dir, encoding_method)
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time}")
        print(f"Results saved in: {self.config['models_dir']}")
        print(f"Visualizations saved in: {self.config['visualizations_dir']}")
        print("=" * 60)

def load_config(config_path=None):
    """
    Load configuration from file or use defaults
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration following paper specifications
    default_config = {
        # Data paths
        'dataset_dir': 'data',  # MIMII fan dataset
        'processed_data_dir': 'processed_data',
        'encoded_images_dir': 'encoded_images',
        'models_dir': 'models',
        'visualizations_dir': 'visualizations',
        
        # Audio processing parameters
        'sample_rate': 16000,
        'segment_length': 1.0,  # 1 second segments
        'overlap': 0.5,  # 50% overlap
        
        # Image encoding parameters
        'image_size': 224,  # Standard input size for pretrained models
        
        # Training parameters (from paper)
        'model_name': 'resnet50',  # Best performing model in paper
        'batch_size': 32,  # Paper specification
        'learning_rate': 1e-05,  # Paper specification
        'epochs': 20,  # Paper specification
        'patience': 5,  # Early stopping patience
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='MIMII Anomaly Detection using Time Series to Image Conversion'
    )
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--encoding', type=str, default='rp', 
                       choices=['gasf', 'gadf', 'mtf', 'rp'],
                       help='Image encoding method (default: rp - best from paper)')
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', 'all'], 
                       default='all', help='Run specific step or all steps')
    parser.add_argument('--dataset-dir', type=str, help='Override dataset directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override dataset directory if provided
    if args.dataset_dir:
        config['dataset_dir'] = args.dataset_dir
    
    # Initialize detector
    detector = MIMIIAnomalyDetector(config)
    
    # Run specified steps
    if args.step == 'all':
        detector.run_complete_pipeline(args.encoding)
    elif args.step == '1':
        detector.step1_preprocess_audio()
    elif args.step == '2':
        detector.step2_encode_images()
    elif args.step == '3':
        trained_model, model_dir = detector.step3_train_model(args.encoding)
    elif args.step == '4':
        # For step 4, we need to load the trained model
        model_path = os.path.join(config['models_dir'], f"{config['model_name']}_{args.encoding}")
        if os.path.exists(model_path):
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)
            
            checkpoint_path = os.path.join(model_path, f"{config['model_name']}_{args.encoding}_best.pth")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            detector.step4_generate_explanations(model, model_path, args.encoding)
        else:
            print(f"Model not found at {model_path}. Please run step 3 first.")

if __name__ == "__main__":
    main()
