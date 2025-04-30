import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms, models
import os

class GradCAM:
    """
    Class for creating Grad-CAM visualizations
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM
        
        Args:
            model (torch.nn.Module): The model to visualize
            target_layer (torch.nn.Module): The target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        
        # Register hooks
        self.register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize gradients and activations
        self.gradients = None
        self.activations = None
    
    def register_hooks(self):
        """
        Register forward and backward hooks
        """
        # Forward hook
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        # Save handles for removal later
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        """
        Remove the registered hooks
        """
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor (torch.Tensor): Input image tensor
            target_class (int, optional): Target class index. If None, uses the predicted class
        
        Returns:
            np.ndarray: Grad-CAM heatmap normalized to [0, 1]
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # One-hot encoding of the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weight the activation maps and sum
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Convert to numpy and squeeze dimensions
        cam = cam.cpu().detach().numpy()
        cam = np.squeeze(cam)
        
        return cam

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for resizing
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    
    return tensor

def overlay_cam_on_image(image_path, cam, output_path=None, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image
    
    Args:
        image_path (str): Path to the original image
        cam (np.ndarray): Grad-CAM heatmap
        output_path (str, optional): Path to save the overlaid image
        alpha (float): Transparency factor for overlay
    
    Returns:
        np.ndarray: Overlaid image
    """
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (cam.shape[1], cam.shape[0]))
    
    # Convert to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create heatmap from CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = heatmap * alpha + img * (1 - alpha)
    overlaid = np.uint8(overlaid)
    
    # Save if output path is specified
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    
    return overlaid

def visualize_grad_cam(model, image_path, target_layer_name='layer4', target_class=None, 
                      output_dir=None, show=True):
    """
    Visualize Grad-CAM for a given image and model
    
    Args:
        model (torch.nn.Module): The model to visualize
        image_path (str): Path to the image
        target_layer_name (str): Name of the target layer
        target_class (int, optional): Target class index. If None, uses the predicted class
        output_dir (str, optional): Directory to save visualizations
        show (bool): Whether to display the visualizations
    
    Returns:
        tuple: (cam, overlaid_image)
    """
    # Get target layer
    if isinstance(model, models.ResNet):
        if target_layer_name == 'layer4':
            target_layer = model.layer4[-1]
        elif target_layer_name == 'layer3':
            target_layer = model.layer3[-1]
        else:
            raise ValueError(f"Unsupported layer name: {target_layer_name}")
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor, target_class)
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    # Overlay CAM on image
    overlaid = overlay_cam_on_image(image_path, cam)
    
    # Save visualization if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_name)
        
        # Save CAM heatmap
        cam_path = os.path.join(output_dir, f"{base_name}_cam.png")
        plt.imsave(cam_path, cam, cmap='jet')
        
        # Save overlaid image
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        plt.imsave(overlay_path, overlaid)
    
    # Show visualization
    if show:
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(plt.imread(image_path))
        plt.title("Original Image")
        plt.axis('off')
        
        # CAM heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        # Overlaid image
        plt.subplot(1, 3, 3)
        plt.imshow(overlaid)
        plt.title("Overlaid Image")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return cam, overlaid

def batch_visualize_grad_cam(model, image_dir, output_dir, pattern='*_rp.png', 
                            target_layer_name='layer4', show=False):
    """
    Batch process Grad-CAM visualizations for all images in a directory
    
    Args:
        model (torch.nn.Module): The model to visualize
        image_dir (str): Directory containing images
        output_dir (str): Directory to save visualizations
        pattern (str): Glob pattern for image files
        target_layer_name (str): Name of the target layer
        show (bool): Whether to display the visualizations
    """
    import glob
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image paths
    image_paths = glob.glob(os.path.join(image_dir, pattern))
    
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        try:
            visualize_grad_cam(
                model=model,
                image_path=image_path,
                target_layer_name=target_layer_name,
                output_dir=output_dir,
                show=show
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    # Example usage
    model_path = "models/rp_resnet50/resnet50_rp_best.pth"
    image_path = "encoded_images/abnormal/sample_rp.png"
    output_dir = "visualizations/grad_cam"
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Visualize Grad-CAM
    visualize_grad_cam(
        model=model,
        image_path=image_path,
        output_dir=output_dir
    )