import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sample image data for testing')
    parser.add_argument('--output_dir', type=str, default='sample_data', help='Output directory for sample data')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per class')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    return parser.parse_args()

def generate_shape(shape_type, size, bg_color, fg_color):
    """Generate an image with a specific shape"""
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate dimensions based on image size
    margin = int(size * 0.2)  # 20% margin from edges
    
    if shape_type == 'circle':
        draw.ellipse((margin, margin, size - margin, size - margin), fill=fg_color)
    elif shape_type == 'square':
        draw.rectangle((margin, margin, size - margin, size - margin), fill=fg_color)
    elif shape_type == 'triangle':
        draw.polygon([
            (size // 2, margin),
            (margin, size - margin),
            (size - margin, size - margin)
        ], fill=fg_color)
    elif shape_type == 'star':
        # Simple 5-point star
        points = []
        cx, cy = size // 2, size // 2
        r_outer = size // 2 - margin
        r_inner = r_outer // 2
        
        for i in range(10):
            # Alternate between outer and inner radius
            radius = r_outer if i % 2 == 0 else r_inner
            angle = i * 36 * (np.pi / 180)  # 36 degrees in radians
            x = cx + radius * np.sin(angle)
            y = cy - radius * np.cos(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=fg_color)
    elif shape_type == 'letter':
        # Draw a random letter
        try:
            font = ImageFont.truetype("arial.ttf", size=size // 2)
        except IOError:
            font = ImageFont.load_default()
        
        letter = chr(random.randint(65, 90))  # Random uppercase letter (A-Z)
        
        # Handle different versions of PIL
        try:
            # For newer versions of Pillow
            try:
                left, top, right, bottom = draw.textbbox((0, 0), letter, font=font)
                text_width = right - left
                text_height = bottom - top
            except AttributeError:
                # For older versions of Pillow
                text_width, text_height = draw.textsize(letter, font=font)
        except Exception:
            # Fallback method
            text_width, text_height = size // 2, size // 2
        
        position = ((size - text_width) // 2, (size - text_height) // 2)
        draw.text(position, letter, fill=fg_color, font=font)
    
    return img

def generate_dataset(output_dir, num_samples, img_size):
    """Generate a dataset with different shape classes"""
    shapes = ['circle', 'square', 'triangle', 'star', 'letter']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class directories
    for shape in shapes:
        os.makedirs(os.path.join(output_dir, shape), exist_ok=True)
    
    # Generate images for each class
    for shape in shapes:
        for i in range(num_samples):
            # Random background and foreground colors
            bg_color = (
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255)
            )
            fg_color = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100)
            )
            
            # Create image
            img = generate_shape(shape, img_size, bg_color, fg_color)
            
            # Save image
            img_path = os.path.join(output_dir, shape, f"{shape}_{i+1}.png")
            img.save(img_path)
    
    print(f"Generated {num_samples} samples each for {len(shapes)} classes")
    print(f"Total: {num_samples * len(shapes)} images in {output_dir}")

def main():
    args = parse_args()
    generate_dataset(args.output_dir, args.num_samples, args.img_size)

if __name__ == "__main__":
    main() 