import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyModel(nn.Module):
    """
    A lightweight CNN model designed for image classification with minimal parameters
    """
    def __init__(self, num_classes=10):
        super(TinyModel, self).__init__()
        # Simple convolutional architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Input expected to be 3x64x64
        x = self.pool(F.relu(self.conv1(x)))  # 16x32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv3(x)))  # 64x8x8
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """Helper method for making predictions"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
def create_model(num_classes=10, pretrained=False):
    """Factory function to create and optionally load a pretrained model"""
    model = TinyModel(num_classes=num_classes)
    return model 