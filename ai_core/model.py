import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 14 classes as per NIH ChestX-ray14 dataset
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

class ChestXRayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ChestXRayModel, self).__init__()
        # Using DenseNet121 as requested in the tech stack
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.densenet121 = models.densenet121(weights=weights)
        
        # Replace the classifier for multi-label classification
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid() # Multi-label requires Sigmoid activation
        )

    def forward(self, x):
        return self.densenet121(x)

def get_transform():
    """
    Standard preprocessing for DenseNet121.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def predict(model, image_path, device="cpu"):
    """
    Run inference on a single image.
    """
    model.eval()
    model.to(device)
    
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    probabilities = output.squeeze().cpu().numpy()
    
    results = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    return results
