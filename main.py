import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import json
import pandas as pd
from datetime import datetime
import itertools

# Import our modules
from preprocess_audio import batch_process_audio_files
from image_encoding import batch_encode_time_series
from dataset import create_dataset_from_directories, get_data_loaders
from train_model import create_model, train_model
from gram_cam import batch_visualize_grad_cam

class AutomatedMIMIIAnomalyDetector:
    """
    Automated pipeline for MIMII anomaly detection testing all encoding-model combinations
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
        
        # Initialize results tracking
        self.all_results = []
        self.results_df = None
    
    def create_directories(self):
        """Create necessary directories for the pipeline"""
        dirs = [
            self.config['processed_data_dir'],
            self.config['encoded_images_dir'],
            self.config['models_dir'],
            self.config['visualizations_dir'],
            self.config['results_dir'],
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
        
        normal_segments = 0
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
        
        abnormal_segments = 0
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
        return normal_segments, abnormal_segments
    
    def step2_encode_images(self):
        """
        Step 2: Convert time-series to images using four encoding methods in separate folders
        """
        print("=" * 60)
        print("STEP 2: ENCODING TIME-SERIES TO IMAGES")
        print("=" * 60)
        
        encoding_methods = ['gasf', 'gadf', 'mtf', 'rp']
        
        for encoding in encoding_methods:
            print(f"\nProcessing {encoding.upper()} encoding...")
            
            # Create encoding-specific directories
            encoding_dir = os.path.join(self.config['encoded_images_dir'], encoding)
            normal_output = os.path.join(encoding_dir, 'normal')
            abnormal_output = os.path.join(encoding_dir, 'abnormal')
            
            os.makedirs(normal_output, exist_ok=True)
            os.makedirs(abnormal_output, exist_ok=True)
            
            # Encode normal segments
            normal_input = os.path.join(self.config['processed_data_dir'], 'normal')
            if os.path.exists(normal_input):
                batch_encode_time_series(
                    input_dir=normal_input,
                    output_dir=normal_output,
                    image_size=self.config['image_size'],
                    encoding_method=encoding  # Pass specific encoding
                )
            
            # Encode abnormal segments  
            abnormal_input = os.path.join(self.config['processed_data_dir'], 'abnormal')
            if os.path.exists(abnormal_input):
                batch_encode_time_series(
                    input_dir=abnormal_input,
                    output_dir=abnormal_output,
                    image_size=self.config['image_size'],
                    encoding_method=encoding  # Pass specific encoding
                )
        
        print("Step 2 completed successfully!")
        
    def step3_train_single_combination(self, encoding_method, model_name):
        """
        Train a single encoding-model combination
        
        Args:
            encoding_method (str): Encoding method ('gasf', 'gadf', 'mtf', 'rp')
            model_name (str): Model name ('resnet50', 'vgg16', 'densenet121')
        
        Returns:
            dict: Results for this combination
        """
        print("=" * 60)
        print(f"TRAINING: {model_name.upper()} + {encoding_method.upper()}")
        print("=" * 60)
        
        combination_start_time = datetime.now()
        
        # Create datasets from encoding-specific folders
        normal_dir = os.path.join(self.config['encoded_images_dir'], encoding_method, 'normal')
        abnormal_dir = os.path.join(self.config['encoded_images_dir'], encoding_method, 'abnormal')
                
        print("Creating datasets...")
        train_dataset, test_dataset = create_dataset_from_directories(
            normal_dir=normal_dir,
            anomaly_dir=abnormal_dir,
            encoding_method=encoding_method,
            test_size=0.25,  # 75% train, 25% test as per paper
            random_state=42
        )
        
        # Check if we have enough data
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            return {
                'encoding_method': encoding_method,
                'model_name': model_name,
                'status': 'failed',
                'error': 'No data found for this encoding method',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Create data loaders
        train_loader, test_loader = get_data_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        try:
            # Create model - IMPORTANT: pretrained=False for audio data
            model = create_model(
                model_name=model_name,
                num_classes=2,
                pretrained=False  # Critical for audio-derived images
            )
            model = model.to(self.device)
            
            # Setup training components
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.config['learning_rate']
            )
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
            
            # Create save directory for this combination
            combo_name = f"{model_name}_{encoding_method}"
            save_dir = os.path.join(self.config['models_dir'], combo_name)
            
            # Train model
            trained_model, history = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                num_epochs=self.config['epochs'],
                patience=self.config['patience'],
                save_dir=save_dir,
                model_name=combo_name
            )
            
            combination_end_time = datetime.now()
            training_time = combination_end_time - combination_start_time
            
            # Extract final metrics
            final_metrics = history['final_metrics']
            
            # Save detailed results for this combination
            combo_results = {
                'encoding_method': encoding_method,
                'model_name': model_name,
                'combination': combo_name,
                'status': 'success',
                'training_time': str(training_time),
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                'epochs_trained': len(history['train_losses']),
                'best_val_loss': min(history['val_losses']),
                'best_val_acc': max(history['val_accs']),
                'final_train_loss': history['train_losses'][-1],
                'final_val_loss': history['val_losses'][-1],
                'final_train_acc': history['train_accs'][-1],
                'final_val_acc': history['val_accs'][-1],
                'accuracy': final_metrics['accuracy'],
                'precision': final_metrics['precision'],
                'recall': final_metrics['recall'],
                'f1_score': final_metrics['f1_score'],
                'model_path': save_dir,
                'timestamp': combination_end_time.isoformat()
            }
            
            # Save individual combination results
            results_path = os.path.join(save_dir, 'detailed_results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'config': self.config,
                    'results': combo_results,
                    'training_history': history
                }, f, indent=2)
            
            print(f"✓ {combo_name}: Accuracy = {final_metrics['accuracy']:.4f}")
            
            return combo_results
            
        except Exception as e:
            print(f"✗ {combo_name}: Training failed - {str(e)}")
            return {
                'encoding_method': encoding_method,
                'model_name': model_name,
                'combination': f"{model_name}_{encoding_method}",
                'status': 'failed',
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def step3_train_all_combinations(self):
        """
        Train all encoding-model combinations
        """
        print("=" * 80)
        print("STEP 3: TRAINING ALL ENCODING-MODEL COMBINATIONS")
        print("=" * 80)
        
        # Define all combinations to test
        encoding_methods = ['gasf', 'gadf', 'mtf', 'rp']
        model_names = ['resnet50', 'vgg16', 'densenet121']
        
        # If user specified specific methods, use those instead
        if 'encoding_methods' in self.config:
            encoding_methods = self.config['encoding_methods']
        if 'model_names' in self.config:
            model_names = self.config['model_names']
        
        total_combinations = len(encoding_methods) * len(model_names)
        print(f"Testing {total_combinations} combinations:")
        
        for i, (encoding, model) in enumerate(itertools.product(encoding_methods, model_names)):
            print(f"  {i+1}. {model} + {encoding}")
        
        print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}")
        overall_start_time = datetime.now()
        
        # Train each combination
        for i, (encoding_method, model_name) in enumerate(itertools.product(encoding_methods, model_names)):
            print(f"\n{'='*20} COMBINATION {i+1}/{total_combinations} {'='*20}")
            
            result = self.step3_train_single_combination(encoding_method, model_name)
            self.all_results.append(result)
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        overall_end_time = datetime.now()
        total_time = overall_end_time - overall_start_time
        
        print(f"\n{'='*80}")
        print("ALL COMBINATIONS COMPLETED!")
        print(f"Total time: {total_time}")
        print(f"{'='*80}")
        
        # Create summary results
        self.create_results_summary()
    
    def create_results_summary(self):
        """
        Create comprehensive results summary
        """
        print("\nCreating results summary...")
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.all_results)
        
        # Save detailed CSV
        csv_path = os.path.join(self.config['results_dir'], 'all_results_detailed.csv')
        self.results_df.to_csv(csv_path, index=False)
        
        # Create summary statistics
        successful_results = self.results_df[self.results_df['status'] == 'success']
        
        if len(successful_results) > 0:
            # Summary by encoding method
            encoding_summary = successful_results.groupby('encoding_method').agg({
                'accuracy': ['mean', 'std', 'max'],
                'f1_score': ['mean', 'std', 'max'],
                'precision': ['mean', 'std', 'max'],
                'recall': ['mean', 'std', 'max']
            }).round(4)
            
            # Summary by model
            model_summary = successful_results.groupby('model_name').agg({
                'accuracy': ['mean', 'std', 'max'],
                'f1_score': ['mean', 'std', 'max'],
                'precision': ['mean', 'std', 'max'],
                'recall': ['mean', 'std', 'max']
            }).round(4)
            
            # Best combinations
            best_accuracy = successful_results.loc[successful_results['accuracy'].idxmax()]
            best_f1 = successful_results.loc[successful_results['f1_score'].idxmax()]
            
            # Create summary report
            summary_report = {
                'experiment_info': {
                    'total_combinations_tested': len(self.all_results),
                    'successful_combinations': len(successful_results),
                    'failed_combinations': len(self.all_results) - len(successful_results),
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config
                },
                'best_results': {
                    'best_accuracy': {
                        'combination': best_accuracy['combination'],
                        'encoding': best_accuracy['encoding_method'],
                        'model': best_accuracy['model_name'],
                        'accuracy': float(best_accuracy['accuracy']),
                        'f1_score': float(best_accuracy['f1_score']),
                        'precision': float(best_accuracy['precision']),
                        'recall': float(best_accuracy['recall'])
                    },
                    'best_f1_score': {
                        'combination': best_f1['combination'],
                        'encoding': best_f1['encoding_method'],
                        'model': best_f1['model_name'],
                        'accuracy': float(best_f1['accuracy']),
                        'f1_score': float(best_f1['f1_score']),
                        'precision': float(best_f1['precision']),
                        'recall': float(best_f1['recall'])
                    }
                },
                'summary_statistics': {
                    'by_encoding_method': encoding_summary.to_dict(),
                    'by_model': model_summary.to_dict()
                }
            }
            
            # Save summary report
            summary_path = os.path.join(self.config['results_dir'], 'summary_report.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_report, f, indent=2)
            
            # Print summary to console
            self.print_results_summary(successful_results, best_accuracy, best_f1)
            
        else:
            print("No successful training runs to summarize!")
    
    def print_results_summary(self, successful_results, best_accuracy, best_f1):
        """
        Print formatted results summary to console
        """
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"Total combinations tested: {len(self.all_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(self.all_results) - len(successful_results)}")
        
        print(f"\n{'Best Results:':^80}")
        print(f"{'-'*80}")
        print(f"🏆 BEST ACCURACY: {best_accuracy['combination']}")
        print(f"   Accuracy: {best_accuracy['accuracy']:.4f}")
        print(f"   F1-Score: {best_accuracy['f1_score']:.4f}")
        print(f"   Precision: {best_accuracy['precision']:.4f}")
        print(f"   Recall: {best_accuracy['recall']:.4f}")
        
        print(f"\n🎯 BEST F1-SCORE: {best_f1['combination']}")
        print(f"   Accuracy: {best_f1['accuracy']:.4f}")
        print(f"   F1-Score: {best_f1['f1_score']:.4f}")
        print(f"   Precision: {best_f1['precision']:.4f}")
        print(f"   Recall: {best_f1['recall']:.4f}")
        
        print(f"\n{'Top 5 Combinations by Accuracy:':^80}")
        print(f"{'-'*80}")
        top_5 = successful_results.nlargest(5, 'accuracy')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['combination']:20} - Acc: {row['accuracy']:.4f}, F1: {row['f1_score']:.4f}")
        
        print(f"\n{'Average Performance by Encoding Method:':^80}")
        print(f"{'-'*80}")
        for encoding in successful_results['encoding_method'].unique():
            subset = successful_results[successful_results['encoding_method'] == encoding]
            avg_acc = subset['accuracy'].mean()
            avg_f1 = subset['f1_score'].mean()
            print(f"{encoding.upper():8} - Avg Accuracy: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")
        
        print(f"\n{'Average Performance by Model:':^80}")
        print(f"{'-'*80}")
        for model in successful_results['model_name'].unique():
            subset = successful_results[successful_results['model_name'] == model]
            avg_acc = subset['accuracy'].mean()
            avg_f1 = subset['f1_score'].mean()
            print(f"{model:12} - Avg Accuracy: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")
        
        print(f"\n{'Files Saved:':^80}")
        print(f"{'-'*80}")
        print(f"📊 Detailed results: {os.path.join(self.config['results_dir'], 'all_results_detailed.csv')}")
        print(f"📋 Summary report: {os.path.join(self.config['results_dir'], 'summary_report.json')}")
        print(f"🗂  Model checkpoints: {self.config['models_dir']}/")
        
        print(f"\n{'='*80}")
    
    def run_complete_automated_pipeline(self):
        """
        Run the complete automated pipeline testing all combinations
        """
        print("STARTING AUTOMATED MIMII ANOMALY DETECTION PIPELINE")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        pipeline_start_time = datetime.now()
        
        # Step 1: Preprocess audio (only once)
        normal_segments, abnormal_segments = self.step1_preprocess_audio()
        
        # Check if we have enough data
        if normal_segments == 0 and abnormal_segments == 0:
            print("ERROR: No audio files processed. Check your dataset directory.")
            return
        
        # Step 2: Encode to images (only once, creates all 4 encodings)
        self.step2_encode_images()
        
        # Step 3: Train all combinations
        self.step3_train_all_combinations()
        
        pipeline_end_time = datetime.now()
        total_pipeline_time = pipeline_end_time - pipeline_start_time
        
        print(f"\n{'='*80}")
        print("🎉 AUTOMATED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total pipeline time: {total_pipeline_time}")
        print(f"Results available in: {self.config['results_dir']}")
        print(f"{'='*80}")

def load_config(config_path=None):
    """
    Load configuration from file or use defaults
    """
    # Enhanced default configuration
    default_config = {
        # Data paths
        'dataset_dir': 'data',
        'processed_data_dir': 'processed_data',
        'encoded_images_dir': 'encoded_images',
        'models_dir': 'models',
        'visualizations_dir': 'visualizations',
        'results_dir': 'results',  # New directory for consolidated results
        
        # Audio processing parameters
        'sample_rate': 16000,
        'segment_length': 1.0,
        'overlap': 0.5,
        
        # Image encoding parameters
        'image_size': 224,
        
        # Training parameters - optimized for audio data
        'batch_size': 16,  # Reduced for stability
        'learning_rate': 1e-4,  # Slightly higher for training from scratch
        'epochs': 6,  # More epochs since we're not using pretrained
        'patience': 4,  # More patience for audio data
        
        # Experiment configuration
        'encoding_methods': ['gasf', 'gadf', 'mtf', 'rp'],  # All methods
        'model_names': ['resnet50', 'vgg16', 'densenet121'],  # All models
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config

def main():
    """Main function with enhanced command line interface"""
    parser = argparse.ArgumentParser(
        description='Automated MIMII Anomaly Detection - Test All Combinations'
    )
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset-dir', type=str, help='Override dataset directory')
    parser.add_argument('--encodings', nargs='+', 
                       choices=['gasf', 'gadf', 'mtf', 'rp'],
                       help='Specific encoding methods to test (default: all)')
    parser.add_argument('--models', nargs='+',
                       choices=['resnet50', 'vgg16', 'densenet121'],
                       help='Specific models to test (default: all)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with RP + ResNet50 only')
    # Add these lines after the existing parser arguments:
    parser.add_argument('--manual', action='store_true',
                    help='Manual mode - run individual steps')
    parser.add_argument('--encoding', type=str, 
                    choices=['gasf', 'gadf', 'mtf', 'rp'],
                    help='Single encoding method to train')
    parser.add_argument('--model', type=str,
                    choices=['resnet50', 'vgg16', 'densenet121'],
                    help='Single model to train')
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                    help='Run specific step only (1=preprocess, 2=encode, 3=train)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.manual or args.step or (args.encoding and args.model):
        detector = AutomatedMIMIIAnomalyDetector(config)
        
        if args.step == 1:
            detector.step1_preprocess_audio()
        elif args.step == 2:
            detector.step2_encode_images()
        elif args.step == 3 or (args.encoding and args.model):
            if args.encoding and args.model:
                # Train single combination
                result = detector.step3_train_single_combination(args.encoding, args.model)
                # Save individual result
                result_file = f"{args.encoding}_{args.model}_results.json"
                result_path = os.path.join(config['results_dir'], result_file)
                os.makedirs(config['results_dir'], exist_ok=True)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {result_path}")
            else:
                detector.step3_train_all_combinations()
        else:
            print("Manual mode: Use --step 1/2/3 or --encoding + --model")
    else:
        # Original automated pipeline
        detector = AutomatedMIMIIAnomalyDetector(config)
        detector.run_complete_automated_pipeline()

if __name__ == "__main__":
    main()
