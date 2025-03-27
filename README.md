# Portable Image Classifier

A fully self-contained deep learning image classification project that compiles into a standalone Windows executable. This project embodies the DIY ethos by using only open-source components and creates a portable application that can run directly from a USB drive without installation.

## Features

- Real-time image classification from files or webcam
- Modern GUI interface with PyQt5
- Lightweight CNN architecture for efficient inference
- Training script for creating custom models with your own classes
- Single executable deployment for maximum portability

## Project Structure

```
├── assets/                 # Contains model and class label files
├── src/
│   ├── gui/                # GUI application code
│   ├── models/             # Deep learning model architecture
│   ├── utils/              # Utility functions
│   ├── main.py             # Entry point for the application
│   ├── train.py            # Script for training the model
│   └── infer.py            # Script for standalone inference
├── build_exe.py            # Script to build the executable
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/portable-image-classifier.git
cd portable-image-classifier
```

2. Create and activate a virtual environment (recommended):
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training a Custom Model

To train the model on your own dataset, organize your images in folders where each folder name is a class:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Then run the training script:

```
python src/train.py --data_dir path/to/dataset --epochs 20 --batch_size 32
```

This will save the trained model and class labels to the `assets` directory.

### Running the Application

To run the application without building an executable:

```
python src/main.py
```

### Standalone Inference

To classify a single image using the command line:

```
python src/infer.py --image_path path/to/image.jpg
```

### Building the Executable

To build a standalone executable:

```
python build_exe.py
```

This will create an executable in the `dist` directory that can be copied to a USB drive and run on any compatible Windows system.

## Customization Options

- **Model Architecture**: Edit `src/models/tiny_model.py` to modify the neural network architecture.
- **UI Appearance**: Customize the interface in `src/gui/app.py`.
- **Image Preprocessing**: Adjust preprocessing in `src/utils/dataset.py`.

## Technical Details

### Deep Learning Architecture

The model uses a lightweight CNN architecture with 3 convolutional layers followed by max pooling and fully connected layers. This balance of depth and parameter count allows for good accuracy while maintaining fast inference speed on CPUs.

### Portability Approach

The application is compiled into a single executable using PyInstaller, which bundles all dependencies including Python, PyTorch, and required libraries. This enables running the application on systems without any Python installation or additional setup.

## License

This project is open source and available under the MIT License.

## Credits

This project uses the following open-source libraries:
- PyTorch and torchvision
- OpenCV
- PyQt5
- NumPy
- Pillow
- PyInstaller 