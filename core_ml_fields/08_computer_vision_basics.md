# Computer Vision Basics: Image Processing, CNNs, and Detection

*"Teaching machines to see and understand the visual world around us"*

---

## 📚 Table of Contents

1. [Introduction to Computer Vision](#introduction-to-computer-vision)
2. [Image Processing Fundamentals](#image-processing-fundamentals)
3. [Convolutional Neural Networks](#convolutional-neural-networks)
4. [Object Detection and Recognition](#object-detection-and-recognition)
5. [Real-World Applications](#real-world-applications)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction to Computer Vision

### What is Computer Vision?

Computer Vision is the field of artificial intelligence that enables machines to interpret and understand visual information from the world. It combines image processing, machine learning, and pattern recognition to extract meaningful information from images and videos.

### Historical Evolution

| Era | Key Development | Impact |
|-----|----------------|---------|
| **1960s** | First image processing algorithms | Foundation for digital image analysis |
| **1980s** | Edge detection and feature extraction | SIFT, SURF algorithms |
| **2000s** | Machine learning integration | SVM, AdaBoost for classification |
| **2012** | AlexNet breakthrough | Deep learning revolution in CV |
| **2015+** | CNN architectures | ResNet, Inception, EfficientNet |
| **2020+** | Vision Transformers | ViT, Swin Transformer |
| **2025** | Multimodal AI | Vision + Language understanding |

### Core Computer Vision Tasks

#### 1. **Image Classification**
- Categorize images into predefined classes
- Single-label and multi-label classification
- Fine-grained classification

#### 2. **Object Detection**
- Locate and classify objects in images
- Bounding box detection
- Instance segmentation

#### 3. **Image Segmentation**
- Pixel-level classification
- Semantic segmentation
- Instance segmentation

#### 4. **Feature Extraction**
- Edge detection
- Corner detection
- Texture analysis

---

## 🖼️ Image Processing Fundamentals

### Digital Image Representation

Images are represented as 2D or 3D arrays of pixel values:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.kernels = {
            'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            'gaussian': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        }
    
    def load_image(self, path):
        """Load and preprocess image"""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize_image(self, image, width=None, height=None):
        """Resize image maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if width and height:
            return cv2.resize(image, (width, height))
        elif width:
            ratio = width / w
            height = int(h * ratio)
            return cv2.resize(image, (width, height))
        elif height:
            ratio = height / h
            width = int(w * ratio)
            return cv2.resize(image, (width, height))
        
        return image
    
    def apply_filter(self, image, kernel_name):
        """Apply convolution filter to image"""
        if kernel_name not in self.kernels:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        
        kernel = self.kernels[kernel_name]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply convolution
        filtered = cv2.filter2D(gray, -1, kernel)
        return filtered
    
    def detect_edges(self, image, method='canny'):
        """Detect edges using various methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method == 'canny':
            edges = cv2.Canny(gray, 50, 150)
        elif method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.uint8(edges)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        return edges
    
    def histogram_equalization(self, image):
        """Enhance image contrast using histogram equalization"""
        if len(image.shape) == 3:
            # Convert to YUV for better color preservation
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            enhanced = cv2.equalizeHist(image)
        
        return enhanced

# Example usage
processor = ImageProcessor()

# Load and process image
# image = processor.load_image('sample.jpg')
# resized = processor.resize_image(image, width=224)
# edges = processor.detect_edges(resized, method='canny')
# enhanced = processor.histogram_equalization(resized)
```

### Advanced Image Processing

```python
class AdvancedImageProcessor:
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
    
    def extract_features(self, image):
        """Extract SIFT features from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio=0.75):
        """Match features between two images"""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def detect_corners(self, image, max_corners=100):
        """Detect corners using Harris corner detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
        return corners
    
    def morphological_operations(self, image, operation='erosion', kernel_size=3):
        """Apply morphological operations"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'erosion':
            result = cv2.erode(image, kernel, iterations=1)
        elif operation == 'dilation':
            result = cv2.dilate(image, kernel, iterations=1)
        elif operation == 'opening':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return result
```

---

## 🧠 Convolutional Neural Networks

### CNN Architecture Fundamentals

Convolutional Neural Networks are specifically designed for processing grid-like data such as images.

#### 1. **Convolution Layer**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = ConvLayer(3, 32, kernel_size=3)
        self.conv2 = ConvLayer(32, 64, kernel_size=3)
        self.conv3 = ConvLayer(64, 128, kernel_size=3)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.conv1(x))  # 32x32
        x = self.pool(self.conv2(x))  # 16x16
        x = self.pool(self.conv3(x))  # 8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

#### 2. **Advanced CNN Architectures**

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
```

#### 3. **Transfer Learning with Pre-trained Models**

```python
from torchvision import models
import torch.optim as optim

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet18'):
        super().__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            # Freeze early layers
            for param in list(self.backbone.parameters())[:-4]:
                param.requires_grad = False
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=True)
            # Freeze early layers
            for param in list(self.backbone.parameters())[:-4]:
                param.requires_grad = False
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def train_transfer_model(model, train_loader, val_loader, epochs=10):
    """Train transfer learning model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}: Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100*correct/total:.2f}%')
```

---

## 🎯 Object Detection and Recognition

### 1. **Sliding Window Detection**

```python
class SlidingWindowDetector:
    def __init__(self, classifier, window_size=(64, 64), stride=32):
        self.classifier = classifier
        self.window_size = window_size
        self.stride = stride
    
    def detect(self, image, threshold=0.5):
        """Detect objects using sliding window"""
        h, w = image.shape[:2]
        detections = []
        
        for y in range(0, h - self.window_size[1], self.stride):
            for x in range(0, w - self.window_size[0], self.stride):
                # Extract window
                window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                # Classify window
                prediction = self.classify_window(window)
                
                if prediction > threshold:
                    detections.append({
                        'bbox': (x, y, x + self.window_size[0], y + self.window_size[1]),
                        'confidence': prediction
                    })
        
        return detections
    
    def classify_window(self, window):
        """Classify a single window"""
        # Preprocess window
        window = cv2.resize(window, (64, 64))
        window = window / 255.0
        window = np.transpose(window, (2, 0, 1))
        window = np.expand_dims(window, 0)
        
        # Get prediction
        with torch.no_grad():
            window_tensor = torch.FloatTensor(window)
            output = self.classifier(window_tensor)
            probability = torch.softmax(output, dim=1)[0, 1].item()
        
        return probability
```

### 2. **YOLO-like Detection**

```python
class YOLOLikeDetector(nn.Module):
    def __init__(self, num_classes, grid_size=7, num_boxes=2):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Darknet-like backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(192, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, (5 + num_classes) * num_boxes, 1, 1)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        
        # Reshape to (batch, grid, grid, boxes, 5 + num_classes)
        batch_size = x.size(0)
        x = x.view(batch_size, self.grid_size, self.grid_size, 
                  self.num_boxes, 5 + self.num_classes)
        
        return x
    
    def decode_predictions(self, predictions, conf_threshold=0.5):
        """Decode YOLO predictions to bounding boxes"""
        batch_size = predictions.size(0)
        boxes = []
        
        for b in range(batch_size):
            batch_boxes = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for k in range(self.num_boxes):
                        pred = predictions[b, i, j, k]
                        
                        # Extract components
                        x, y, w, h = pred[:4]
                        confidence = pred[4]
                        class_probs = pred[5:]
                        
                        if confidence > conf_threshold:
                            # Convert to absolute coordinates
                            x_abs = (j + x) / self.grid_size
                            y_abs = (i + y) / self.grid_size
                            w_abs = w / self.grid_size
                            h_abs = h / self.grid_size
                            
                            # Get class
                            class_id = torch.argmax(class_probs).item()
                            class_conf = torch.max(class_probs).item()
                            
                            batch_boxes.append({
                                'bbox': (x_abs, y_abs, w_abs, h_abs),
                                'confidence': confidence.item(),
                                'class_id': class_id,
                                'class_confidence': class_conf
                            })
            
            boxes.append(batch_boxes)
        
        return boxes
```

---

## 🎯 Real-World Applications

### 1. **Face Detection and Recognition**

```python
class FaceDetector:
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image):
        """Detect faces in image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        return faces
    
    def extract_face_embeddings(self, image, faces):
        """Extract face embeddings for recognition"""
        embeddings = []
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (160, 160))
            
            # Normalize
            face_roi = face_roi / 255.0
            
            # Extract embedding (simplified)
            embedding = self.compute_face_embedding(face_roi)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_face_embedding(self, face_image):
        """Compute face embedding using a simple CNN"""
        # This would typically use a pre-trained face recognition model
        # For demonstration, we'll use a simple feature extraction
        face_tensor = torch.FloatTensor(face_image).unsqueeze(0)
        face_tensor = face_tensor.permute(0, 3, 1, 2)
        
        # Simple embedding network
        embedding_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
        with torch.no_grad():
            embedding = embedding_net(face_tensor)
        
        return embedding.squeeze().numpy()
```

### 2. **Medical Image Analysis**

```python
class MedicalImageAnalyzer:
    def __init__(self):
        self.segmentation_model = self.load_segmentation_model()
        self.classification_model = self.load_classification_model()
    
    def analyze_xray(self, image):
        """Analyze chest X-ray for abnormalities"""
        # Preprocess image
        processed = self.preprocess_medical_image(image)
        
        # Segment lungs
        lung_mask = self.segment_lungs(processed)
        
        # Classify abnormalities
        abnormalities = self.classify_abnormalities(processed, lung_mask)
        
        return {
            'lung_mask': lung_mask,
            'abnormalities': abnormalities,
            'confidence': self.compute_confidence(abnormalities)
        }
    
    def preprocess_medical_image(self, image):
        """Preprocess medical image"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Normalize
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(normalized)
        
        return enhanced
    
    def segment_lungs(self, image):
        """Segment lung regions"""
        # Threshold to get lung regions
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
```

---

## 🧪 Exercises and Projects

### Exercise 1: Implement Image Filtering

```python
def implement_custom_filter():
    """Implement a custom image filter"""
    # Create a custom kernel for edge enhancement
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    
    # Apply to image
    # filtered = cv2.filter2D(image, -1, kernel)
    pass
```

### Exercise 2: Build a Simple CNN

```python
def build_simple_cnn():
    """Build and train a simple CNN for image classification"""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model
```

### Project 1: Real-time Object Detection

Build a system that can:
- Detect objects in real-time video
- Track objects across frames
- Provide confidence scores
- Handle multiple object classes

### Project 2: Image Style Transfer

Create a system that can:
- Transfer artistic styles to photos
- Use neural style transfer
- Optimize for different styles
- Provide real-time preview

### Project 3: Medical Image Segmentation

Build a system that can:
- Segment organs in medical images
- Detect abnormalities
- Provide confidence maps
- Handle different imaging modalities

---

## 📖 Further Reading

### Essential Papers
- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"
- Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks"
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition"

### Books
- "Computer Vision: Algorithms and Applications" by Richard Szeliski
- "Deep Learning for Computer Vision" by Adrian Rosebrock

### Online Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Computer Vision Tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### Next Steps
- **[Computer Vision Advanced](09_computer_vision_advanced.md)**: Segmentation, GANs, 3D vision
- **[RL Basics](10_rl_basics.md)**: Reinforcement learning fundamentals
- **[ML Engineering](21_data_engineering.md)**: Production computer vision systems

---

## 🎯 Key Takeaways

1. **Image Processing**: Foundation for all computer vision tasks through filtering and enhancement
2. **Convolutional Networks**: Specialized architecture for processing grid-like data
3. **Object Detection**: Locating and classifying objects in images
4. **Transfer Learning**: Leveraging pre-trained models for new tasks
5. **Real-World Impact**: Powers autonomous vehicles, medical imaging, and surveillance
6. **2025 Relevance**: Integration with multimodal AI and edge computing

---

*"Computer vision is not just about seeing - it's about understanding the visual world."*

**Next: [Computer Vision Advanced](09_computer_vision_advanced.md) → Segmentation, GANs, and 3D vision** 