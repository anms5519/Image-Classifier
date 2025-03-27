import os
import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QComboBox, QSlider, QProgressBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

# Add parent directory to path to import project modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.tiny_model import create_model
from src.utils.dataset import preprocess_image

# Default class labels for demo (can be overridden)
DEFAULT_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portable Image Classifier")
        self.setMinimumSize(800, 600)
        
        # Initialize model
        self.model = None
        self.class_labels = self.load_class_labels()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize UI
        self.init_ui()
        
        # For camera capture
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def load_class_labels(self):
        """Load class labels from file or use defaults"""
        labels_path = os.path.join(project_root, 'assets', 'class_labels.txt')
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return DEFAULT_CLASSES
    
    def resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        base_path = getattr(sys, '_MEIPASS', project_root)
        return os.path.join(base_path, relative_path)
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f5f5f5;")
        main_layout.addWidget(self.image_label)
        
        # Results display
        results_layout = QHBoxLayout()
        
        # Prediction label
        self.result_label = QLabel("Ready for classification")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        results_layout.addWidget(self.result_label)
        
        # Confidence bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        results_layout.addWidget(self.confidence_bar)
        
        main_layout.addLayout(results_layout)
        
        # Control buttons layout
        control_layout = QHBoxLayout()
        
        # Load image button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # Capture from camera button
        self.camera_btn = QPushButton("Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.camera_btn)
        
        # Classify button
        self.classify_btn = QPushButton("Classify")
        self.classify_btn.clicked.connect(self.classify_current_image)
        self.classify_btn.setEnabled(False)
        control_layout.addWidget(self.classify_btn)
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItem("Default Model")
        self.model_selector.currentIndexChanged.connect(self.load_model)
        control_layout.addWidget(self.model_selector)
        
        main_layout.addLayout(control_layout)
        
        # Load the default model
        self.load_model()
    
    def load_model(self):
        """Load the deep learning model"""
        try:
            # In a real app, we would have different models to choose from
            # For this demo, we just create a single model
            self.model = create_model(num_classes=len(self.class_labels))
            
            # Try to load a pre-trained model if available
            model_path = os.path.join(project_root, 'assets', 'model.pth')
            print(f"Looking for model at: {model_path}")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Pre-trained model loaded successfully!")
                QMessageBox.information(self, "Model Loaded", "Pre-trained model loaded successfully!")
            else:
                # Generate a dummy model for demo purposes
                print("No pre-trained model found. Created new model for demonstration.")
                QMessageBox.information(self, "Model Created", 
                                     "No pre-trained model found. Created new model for demonstration.")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Error Loading Model", f"Failed to load model: {str(e)}")
    
    def load_image(self):
        """Load an image from disk"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.display_image(file_path)
            self.classify_btn.setEnabled(True)
            
            # If camera is on, turn it off
            if self.camera is not None:
                self.toggle_camera()
    
    def display_image(self, image):
        """Display an image on the image_label"""
        if isinstance(image, str):
            # Load from file path
            pixmap = QPixmap(image)
            self.current_image_path = image
            self.current_frame = None
        else:
            # Convert OpenCV image (numpy array) to QPixmap
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.current_image_path = None
            self.current_frame = image
        
        # Resize to fit the label while keeping aspect ratio
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera is None:
            # Start camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera.")
                self.camera = None
                return
            
            self.timer.start(30)  # Update every 30ms (approx. 30 fps)
            self.camera_btn.setText("Stop Camera")
            self.classify_btn.setEnabled(True)
        else:
            # Stop camera
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_btn.setText("Camera")
            self.image_label.clear()
            self.classify_btn.setEnabled(False)
    
    def update_frame(self):
        """Update camera frame"""
        ret, frame = self.camera.read()
        if ret:
            self.display_image(frame)
    
    def classify_current_image(self):
        """Classify the currently displayed image"""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model loaded.")
            return
        
        if self.current_image_path is None and self.current_frame is None:
            QMessageBox.warning(self, "Warning", "No image to classify.")
            return
        
        try:
            # Preprocess image for the model
            if self.current_image_path:
                # From file
                inputs = preprocess_image(self.current_image_path)
            else:
                # From camera frame
                inputs = preprocess_image(self.current_frame)
            
            # Run inference
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get class name and confidence
            prediction_idx = predicted.item()
            confidence_value = confidence.item() * 100
            class_name = self.class_labels[prediction_idx]
            
            # Update UI
            self.result_label.setText(f"Predicted: {class_name}")
            self.confidence_bar.setValue(int(confidence_value))
            
        except Exception as e:
            print(f"Classification failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Classification failed: {str(e)}")
    
def run_app():
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_()) 