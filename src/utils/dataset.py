import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms

class ImageDataset(Dataset):
    """Custom Dataset for loading and preprocessing images"""
    def __init__(self, image_dir=None, transform=None, for_inference=False):
        self.transform = transform if transform is not None else self._default_transform()
        self.for_inference = for_inference
        
        # For training/validation
        self.image_paths = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}
        
        # Load dataset if directory is provided
        if image_dir and os.path.exists(image_dir):
            self._load_from_directory(image_dir)
    
    def _default_transform(self):
        """Default transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_from_directory(self, image_dir):
        """Load images from directory structure where folders are class names"""
        self.classes = [d for d in os.listdir(image_dir) 
                  if os.path.isdir(os.path.join(image_dir, d))]
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        for class_name in self.classes:
            class_dir = os.path.join(image_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def add_single_image(self, image):
        """Add a single PIL Image to the dataset (for inference)"""
        self.image = image
        self.for_inference = True
    
    def __len__(self):
        if self.for_inference and hasattr(self, 'image'):
            return 1
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.for_inference and hasattr(self, 'image'):
            # Return the single image for inference
            return self.transform(self.image)
            
        # Get image and label for training/validation
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def preprocess_image(image, size=(64, 64)):
    """Preprocess an image (PIL, numpy, or path) for model input"""
    if isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension 