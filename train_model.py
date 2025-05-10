import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import create_dataset_from_directories, get_data_loaders

def create_model(model_name='resnet50', num_classes=2, pretrained=True):
    """
    Create a CNN model with pretrained weights
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: The model
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model (torch.nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
    
    Returns:
        float: Average loss for this epoch
    """
    model.train()
    running_loss = 0.0
    total_samples = len(dataloader.dataset)
    processed_samples = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        processed_samples += batch_size
        
        # Display progress
        progress = (i + 1) / len(dataloader) * 100
        samples_progress = processed_samples / total_samples * 100
        if (i + 1) % max(1, len(dataloader) // 10) == 0:  # Update every ~10% of the batches
            print(f"Batch {i+1}/{len(dataloader)} ({progress:.1f}%) - Samples: {processed_samples}/{total_samples} ({samples_progress:.1f}%) - Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / total_samples
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model (torch.nn.Module): The model to validate
        dataloader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to validate on
    
    Returns:
        tuple: (validation loss, accuracy, predictions, labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = len(dataloader.dataset)
    processed_samples = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            processed_samples += batch_size
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Display progress
            progress = (i + 1) / len(dataloader) * 100
            samples_progress = processed_samples / total_samples * 100
            if (i + 1) % max(1, len(dataloader) // 5) == 0:  # Update every ~20% of the batches
                print(f"Batch {i+1}/{len(dataloader)} ({progress:.1f}%) - Samples: {processed_samples}/{total_samples} ({samples_progress:.1f}%) - Loss: {loss.item():.4f}")
    
    # Calculate metrics
    val_loss = running_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Print validation summary
    print(f"Validation complete - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return val_loss, accuracy, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training history
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        train_accs (list): List of training accuracies
        val_accs (list): List of validation accuracies
        save_path (str): Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def train_model(model, train_loader, test_loader, criterion, optimizer, 
               scheduler, device, num_epochs=20, patience=5, 
               save_dir='models', model_name='resnet50_rp'):
    """
    Train the model
    
    Args:
        model (torch.nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        device (torch.device): Device to train on
        num_epochs (int): Number of epochs to train for
        patience (int): Early stopping patience
        save_dir (str): Directory to save the model
        model_name (str): Name of the model for saving
    
    Returns:
        tuple: (trained model, training history)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize variables
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    counter = 0
    
    # History for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    
    # Get total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training model with {total_params:,} trainable parameters")
    
    # Get total number of batches and samples
    total_train_samples = len(train_loader.dataset)
    total_train_batches = len(train_loader)
    total_val_samples = len(test_loader.dataset)
    
    print(f"Training on {total_train_samples} samples in {total_train_batches} batches")
    print(f"Validating on {total_val_samples} samples")
    
    # Start training
    start_time = time.time()
    print(f"\nStarting training at {time.strftime('%H:%M:%S')}")
    
    for epoch in range(num_epochs):
        epoch_str = f"Epoch {epoch+1}/{num_epochs}"
        separator = "=" * len(epoch_str)
        print(f"\n{separator}")
        print(epoch_str)
        print(separator)
        
        epoch_start_time = time.time()
        
        # Train for one epoch
        print(f"Training phase:")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        print(f"\nValidation phase:")
        val_loss, val_acc, val_preds, val_labels = validate(model, test_loader, criterion, device)
        
        # Calculate training accuracy
        model.eval()
        all_train_preds = []
        all_train_labels = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                _, preds = torch.max(model(inputs), 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.numpy())
        
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print summary for this epoch
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f} seconds")
        print(f"  Learning rate: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            improvement = (val_acc - best_val_acc) * 100
            print(f"  Validation accuracy improved by {improvement:.2f}%!")
            
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            counter = 0
            
            # Save the model
            model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, model_path)
            
            print(f"  Model saved to {model_path}")
        else:
            counter += 1
            print(f"  No improvement in validation accuracy for {counter} epochs")
        
        # Early stopping
        if counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f"\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    _, final_acc, final_preds, final_labels = validate(model, test_loader, criterion, device)
    
    # Calculate additional metrics
    precision = precision_score(final_labels, final_preds)
    recall = recall_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds)
    
    print("\nFinal Metrics:")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    cm_path = os.path.join(save_dir, f"{model_name}_cm.png")
    plot_confusion_matrix(final_labels, final_preds, save_path=cm_path)
    print(f"  Confusion matrix saved to {cm_path}")
    
    # Plot training history
    history_path = os.path.join(save_dir, f"{model_name}_history.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=history_path)
    print(f"  Training history plot saved to {history_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'epoch_times': epoch_times,
        'final_metrics': {
            'accuracy': final_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }
    
    print("\nEpoch Time Summary:")
    for i, epoch_time in enumerate(epoch_times):
        print(f"  Epoch {i+1}: {epoch_time:.2f} seconds")
    print(f"  Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds")
        
    return model, history
