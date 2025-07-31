# Portfolio Building Templates

## 🎯 Overview
Templates and guides for building a professional ML portfolio that showcases your skills, projects, and expertise. This comprehensive guide provides ready-to-use templates and best practices for creating an impressive portfolio.

---

## 📁 GitHub Repository Templates

### Professional ML Project Structure
Standardized repository structure for ML projects that demonstrates best practices.

#### Complete ML Project Template

```markdown
# Project Name

## 🎯 Project Overview
Brief description of the project, problem it solves, and key achievements.

## 📊 Results Summary
- **Accuracy**: 95.2%
- **Performance**: 10x faster than baseline
- **Deployment**: Production-ready API
- **Impact**: 25% improvement in user engagement

## 🚀 Quick Start

### Prerequisites
```bash
python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/username/project-name.git
cd project-name
pip install -r requirements.txt
```

### Usage
```python
from src.model import MLModel

# Load model
model = MLModel.load('models/best_model.pkl')

# Make prediction
prediction = model.predict(input_data)
```

## 📁 Project Structure
```
project-name/
├── data/
│   ├── raw/                 # Original data files
│   ├── processed/           # Cleaned and processed data
│   └── external/            # External data sources
├── models/
│   ├── trained/             # Saved model files
│   └── checkpoints/         # Training checkpoints
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── docs/
│   ├── README.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
└── README.md
```

## 🔧 Configuration

### Environment Setup
```python
# config.py
import os
from pathlib import Path

class Config:
    # Data paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model paths
    MODELS_DIR = Path("models")
    TRAINED_MODELS_DIR = MODELS_DIR / "trained"
    
    # Training parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    
    # Model hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/training.log"
```

## 📊 Data Pipeline

### Data Loading and Preprocessing
```python
# src/data/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

class DataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train, validation, and test sets"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        # Second split: train vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config.VALIDATION_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data"""
        
        output_path = self.config.PROCESSED_DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if filename.endswith('.csv'):
            data.to_csv(output_path, index=False)
        elif filename.endswith('.parquet'):
            data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {filename}")
```

## 🏗️ Model Development

### Model Architecture and Training
```python
# src/models/model.py
import torch
import torch.nn as nn
from typing import Dict, Any
import joblib
import json

class MLModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def save(self, filepath: str, metadata: Dict[str, Any] = None):
        """Save model with metadata"""
        
        # Save model state
        torch.save(self.state_dict(), filepath)
        
        # Save metadata
        if metadata:
            metadata_path = filepath.replace('.pth', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, input_size: int, hidden_size: int, output_size: int):
        """Load model from file"""
        
        model = cls(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(filepath))
        return model

# src/models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
            
            # Record losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint('best_model.pth')
            
            # Log progress
            self.logger.info(
                f'Epoch {epoch+1}/{self.config.EPOCHS}: '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}'
            )
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        
        checkpoint_path = self.config.TRAINED_MODELS_DIR / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
        }, checkpoint_path)
        
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training history"""
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
```

## 📈 Evaluation Framework

### Comprehensive Model Evaluation
```python
# src/evaluation/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average='weighted')
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: list = None) -> None:
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report"""
        
        return classification_report(y_true, y_pred, output_dict=True)
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to file"""
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {filename}")

# src/evaluation/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

class VisualizationHelper:
    def __init__(self):
        self.style = 'seaborn-v0_8'
        plt.style.use(self.style)
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: List[float], 
                              top_n: int = 20):
        """Plot feature importance"""
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:top_n]]
        top_scores = [importance_scores[i] for i in sorted_idx[:top_n]]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                        class_names: List[str]):
        """Plot ROC curves for multi-class classification"""
        
        from sklearn.metrics import roc_curve
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            auc = roc_auc_score(y_true == i, y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot (if available)
        if 'train_accuracies' in history and 'val_accuracies' in history:
            ax2.plot(history['train_accuracies'], label='Train Accuracy')
            ax2.plot(history['val_accuracies'], label='Validation Accuracy')
            ax2.set_title('Training Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
```

## 🧪 Testing Framework

### Comprehensive Testing Suite
```python
# tests/test_models.py
import unittest
import torch
import numpy as np
from src.models.model import MLModel
from src.models.trainer import ModelTrainer

class TestMLModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.input_size = 10
        self.hidden_size = 64
        self.output_size = 3
        self.model = MLModel(self.input_size, self.hidden_size, self.output_size)
    
    def test_model_forward(self):
        """Test model forward pass"""
        batch_size = 32
        x = torch.randn(batch_size, self.input_size)
        
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_size))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create dummy metadata
        metadata = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'accuracy': 0.95
        }
        
        # Save model
        self.model.save('test_model.pth', metadata)
        
        # Load model
        loaded_model = MLModel.load('test_model.pth', 
                                  self.input_size, 
                                  self.hidden_size, 
                                  self.output_size)
        
        # Test that loaded model produces same output
        x = torch.randn(1, self.input_size)
        original_output = self.model(x)
        loaded_output = loaded_model(x)
        
        self.assertTrue(torch.allclose(original_output, loaded_output))
    
    def test_model_parameters(self):
        """Test model has expected number of parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        expected_params = (
            self.input_size * self.hidden_size + self.hidden_size +  # First layer
            self.hidden_size * (self.hidden_size // 2) + (self.hidden_size // 2) +  # Second layer
            (self.hidden_size // 2) * self.output_size + self.output_size  # Output layer
        )
        
        self.assertEqual(total_params, expected_params)

# tests/test_data.py
import unittest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from unittest.mock import Mock

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.TEST_SIZE = 0.2
        self.config.VALIDATION_SIZE = 0.2
        self.config.RANDOM_STATE = 42
        self.data_loader = DataLoader(self.config)
    
    def test_load_csv_data(self):
        """Test loading CSV data"""
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Save test data
        test_data.to_csv('test_data.csv', index=False)
        
        # Load data
        loaded_data = self.data_loader.load_data('test_data.csv')
        
        # Assertions
        self.assertEqual(loaded_data.shape, test_data.shape)
        self.assertTrue(loaded_data.equals(test_data))
    
    def test_data_split(self):
        """Test data splitting functionality"""
        # Create test data
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(X, y)
        
        # Assertions
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), len(y))
        
        # Check that splits are disjoint
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        self.assertTrue(train_indices.isdisjoint(val_indices))
        self.assertTrue(train_indices.isdisjoint(test_indices))
        self.assertTrue(val_indices.isdisjoint(test_indices))

if __name__ == '__main__':
    unittest.main()
```

## 📋 Requirements and Setup

### Project Dependencies
```txt
# requirements.txt
# Core ML libraries
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0

# Data processing
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Development tools
jupyter>=1.0.0
ipykernel>=6.0.0
black>=21.0.0
flake8>=3.9.0
pytest>=6.0.0

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=0.5.0

# Deployment
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Monitoring
mlflow>=1.20.0
wandb>=0.12.0
```

### Setup Script
```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-project-template",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive ML project template",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/ml-project-template",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
)
```

This comprehensive template provides a professional structure for ML projects with proper organization, testing, evaluation, and documentation. 