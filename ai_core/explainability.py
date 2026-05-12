import torch
from captum.attr import LayerGradCam
import numpy as np
import cv2
from PIL import Image
from .model import get_transform

def generate_gradcam_heatmap(model, image_path, target_class_idx, save_path, device="cpu"):
    """
    Generates a Grad-CAM heatmap for a specific disease class using Captum.
    """
    model.eval()
    model.to(device)
    
    # We target the last denseblock of DenseNet121 for Grad-CAM
    target_layer = model.densenet121.features.denseblock4
    
    layer_gc = LayerGradCam(model, target_layer)
    
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # Generate attribution heatmap
    attributions = layer_gc.attribute(input_tensor, target=target_class_idx)
    
    # Convert attributions to numpy and normalize
    heatmap = attributions.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    
    # Handle the case where the heatmap is entirely zero (e.g., target class not activated at all)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap onto original image
    original_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    cv2.imwrite(save_path, superimposed_img)
    return save_path

def generate_mock_bounding_box(image_path, save_path):
    """
    Mocks a lesion localization bounding box.
    In a full implementation, this would use YOLOv8 or U-Net predictions.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    h, w, _ = image.shape
    # Draw a mock box in the center-right (typical lung area)
    start_point = (int(w * 0.6), int(h * 0.4))
    end_point = (int(w * 0.8), int(h * 0.7))
    color = (0, 0, 255) # Red in BGR
    thickness = 2
    
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.putText(image, "Lesion Detected", (start_point[0], start_point[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
    cv2.imwrite(save_path, image)
    return save_path
