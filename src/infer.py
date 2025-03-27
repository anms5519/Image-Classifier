import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from models.tiny_model import create_model
from utils.dataset import preprocess_image

def parse_args():
    parser = argparse.ArgumentParser(description='Classify images using trained model')
    parser.add_argument('--model_path', type=str, default='assets/model.pth', help='Path to trained model')
    parser.add_argument('--class_labels', type=str, default='assets/class_labels.txt', help='Path to class labels file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image for classification')
    return parser.parse_args()

def load_class_labels(labels_path):
    """Load class labels from file"""
    if not os.path.exists(labels_path):
        print(f"Warning: Class labels file not found at {labels_path}")
        return ["Class_" + str(i) for i in range(10)]  # Default class names
    
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class labels
    class_labels = load_class_labels(args.class_labels)
    print(f"Loaded {len(class_labels)} class labels")
    
    # Create model and load weights
    model = create_model(num_classes=len(class_labels))
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    try:
        # Preprocess image
        inputs = preprocess_image(args.image_path)
        inputs = inputs.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get prediction details
        prediction_idx = predicted.item()
        confidence_value = confidence.item() * 100
        class_name = class_labels[prediction_idx]
        
        # Display result
        print(f"\nPrediction: {class_name}")
        print(f"Confidence: {confidence_value:.2f}%")
        
        # Display image with prediction (optional)
        try:
            image = cv2.imread(args.image_path)
            cv2.putText(image, f"{class_name} ({confidence_value:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Prediction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image: {str(e)}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main() 