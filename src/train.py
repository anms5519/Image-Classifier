import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tiny_model import create_model
from src.utils.dataset import ImageDataset
from src.utils.training import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--output_dir', type=str, default='assets', help='Output directory for model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split (0-1)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for training')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = ImageDataset(args.data_dir, transform=train_transform)
    print(f"Loaded dataset with {len(full_dataset)} images and {len(full_dataset.classes)} classes")
    
    # Split dataset into training and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Reduced for compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=0,  # Reduced for compatibility
        pin_memory=True
    )
    
    # Create model
    model = create_model(num_classes=len(full_dataset.classes))
    
    # Create trainer
    trainer = Trainer(model, device=device)
    trainer.set_optimizer('adam', lr=args.lr, weight_decay=1e-5)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs")
    trainer.train(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=os.path.join(args.output_dir, 'model.pth')
    )
    
    print(f"Model saved to {os.path.join(args.output_dir, 'model.pth')}")
    
    # Save class labels
    with open(os.path.join(args.output_dir, 'class_labels.txt'), 'w') as f:
        for class_name in full_dataset.classes:
            f.write(f"{class_name}\n")
    
    print(f"Class labels saved to {os.path.join(args.output_dir, 'class_labels.txt')}")

if __name__ == "__main__":
    main() 