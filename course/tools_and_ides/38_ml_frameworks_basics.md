# ML Frameworks Basics: Scikit-learn, TensorFlow, PyTorch

*"Master the essential frameworks that power modern machine learning"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Scikit-learn Mastery](#scikit-learn-mastery)
3. [TensorFlow Fundamentals](#tensorflow-fundamentals)
4. [PyTorch Essentials](#pytorch-essentials)
5. [Framework Comparison](#framework-comparison)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction

Machine learning frameworks provide the tools and abstractions needed to build, train, and deploy ML models efficiently. Scikit-learn offers classical ML algorithms, TensorFlow provides deep learning capabilities with production deployment, and PyTorch offers dynamic computation graphs with research-friendly design.

### Framework Selection Guide

| Framework | Best For | Strengths | Use Cases |
|-----------|----------|-----------|-----------|
| **Scikit-learn** | Classical ML | Simple API, extensive algorithms | Data science, prototyping |
| **TensorFlow** | Production DL | Scalability, deployment | Large-scale production |
| **PyTorch** | Research DL | Dynamic graphs, flexibility | Research, rapid prototyping |

### 2025 Framework Trends

- **Unified APIs**: Cross-framework compatibility
- **AutoML Integration**: Automated model selection
- **Edge Deployment**: Mobile and IoT optimization
- **Federated Learning**: Distributed training support

---

## 🔬 Scikit-learn Mastery

### 1. Advanced Pipeline Construction

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

class AdvancedSklearnPipeline:
    """Advanced scikit-learn pipeline construction"""
    
    def __init__(self):
        self.pipeline = None
        self.best_params = None
    
    def create_comprehensive_pipeline(self, numeric_features, categorical_features):
        """Create comprehensive ML pipeline"""
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine features
        preprocessor = FeatureUnion(
            transformer_list=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Complete pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        return self.pipeline
    
    def hyperparameter_tuning(self, X, y, param_grid):
        """Advanced hyperparameter tuning"""
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, 
            scoring='f1_weighted', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        return grid_search.best_estimator_
    
    def feature_importance_analysis(self, model, feature_names):
        """Analyze feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None

# Usage
sklearn_pipeline = AdvancedSklearnPipeline()
```

### 2. Custom Estimators

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer"""
    
    def __init__(self, polynomial_degree=2):
        self.polynomial_degree = polynomial_degree
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer"""
        X = check_array(X)
        self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        """Transform the data"""
        X = check_array(X)
        
        # Create polynomial features
        features = []
        for degree in range(1, self.polynomial_degree + 1):
            if degree == 1:
                features.append(X)
            else:
                # Add polynomial combinations
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        features.append((X[:, i:i+1] * X[:, j:j+1]) ** (degree - 1))
        
        return np.hstack(features)

class CustomClassifier(BaseEstimator):
    """Custom classifier with sklearn interface"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit the classifier"""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Simple threshold-based classifier
        self.mean_values_ = np.mean(X, axis=0)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = check_array(X)
        scores = np.mean(X, axis=1)
        return (scores > self.threshold).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = check_array(X)
        scores = np.mean(X, axis=1)
        proba = np.zeros((X.shape[0], 2))
        proba[:, 1] = scores
        proba[:, 0] = 1 - scores
        return proba

# Usage
custom_engineer = CustomFeatureEngineer(polynomial_degree=2)
custom_classifier = CustomClassifier(threshold=0.5)
```

---

## 🔥 TensorFlow Fundamentals

### 1. Modern TensorFlow 2.x

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ModernTensorFlow:
    """Modern TensorFlow 2.x implementation"""
    
    def __init__(self):
        # Enable mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def create_advanced_model(self, input_shape, num_classes):
        """Create advanced neural network"""
        
        inputs = keras.Input(shape=input_shape)
        
        # Feature extraction layers
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with dropout
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def custom_training_loop(self, model, train_dataset, val_dataset, epochs=10):
        """Custom training loop with advanced features"""
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        
        # Metrics
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        
        # Learning rate scheduler
        lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)
                    loss_value = loss_fn(y_batch, logits)
                
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                train_acc_metric.update_state(y_batch, logits)
                
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}: Loss = {loss_value:.4f}")
            
            # Validation
            for x_batch, y_batch in val_dataset:
                val_logits = model(x_batch, training=False)
                val_acc_metric.update_state(y_batch, val_logits)
            
            print(f"Training accuracy: {train_acc_metric.result():.4f}")
            print(f"Validation accuracy: {val_acc_metric.result():.4f}")
            
            # Reset metrics
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()
    
    def model_interpretation(self, model, test_data):
        """Model interpretation using TensorFlow"""
        
        # Grad-CAM implementation
        def grad_cam(model, image, class_index):
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.layers[-2].output, model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_index]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            return heatmap
        
        return grad_cam(model, test_data, 0)

# Usage
tf_model = ModernTensorFlow()
```

### 2. TensorFlow Data Pipeline

```python
class TensorFlowDataPipeline:
    """Advanced TensorFlow data pipeline"""
    
    def __init__(self, batch_size=32, buffer_size=1000):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def create_efficient_dataset(self, data, labels):
        """Create efficient TensorFlow dataset"""
        
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
        # Optimize performance
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def data_augmentation(self, dataset):
        """Apply data augmentation"""
        
        augmentation_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1)
        ])
        
        def augment(image, label):
            image = augmentation_layer(image, training=True)
            return image, label
        
        return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Usage
tf_pipeline = TensorFlowDataPipeline()
```

---

## 🔥 PyTorch Essentials

### 1. Modern PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class AdvancedPyTorchModel(nn.Module):
    """Advanced PyTorch model implementation"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(AdvancedPyTorchModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract intermediate features"""
        return self.feature_extractor(x)

class PyTorchTrainer:
    """Advanced PyTorch training class"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def train(self, train_loader, val_loader, epochs=10):
        """Complete training loop"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

# Usage
model = AdvancedPyTorchModel(input_size=784, hidden_size=256, num_classes=10)
trainer = PyTorchTrainer(model)
```

### 2. Custom PyTorch Dataset

```python
class CustomPyTorchDataset(Dataset):
    """Custom PyTorch dataset"""
    
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class DataTransforms:
    """Data transformation utilities"""
    
    @staticmethod
    def normalize(data):
        """Normalize data"""
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / std
    
    @staticmethod
    def add_noise(data, noise_factor=0.1):
        """Add noise to data"""
        noise = torch.randn_like(data) * noise_factor
        return data + noise

# Usage
dataset = CustomPyTorchDataset(data, labels, transform=DataTransforms.normalize)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## ⚖️ Framework Comparison

### Performance Comparison

| Metric | Scikit-learn | TensorFlow | PyTorch |
|--------|--------------|------------|---------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Deployment** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Research** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Community** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Framework Selection Guidelines

```python
def select_framework(use_case, requirements):
    """Framework selection logic"""
    
    if use_case == "classical_ml":
        return "scikit-learn"
    elif use_case == "production_deep_learning":
        return "tensorflow"
    elif use_case == "research_prototyping":
        return "pytorch"
    elif use_case == "edge_deployment":
        return "tensorflow_lite"
    else:
        return "pytorch"  # Default for flexibility
```

---

## 🧪 Exercises and Projects

### Exercise 1: Scikit-learn Pipeline
1. Build comprehensive ML pipeline
2. Implement custom transformers
3. Perform hyperparameter tuning
4. Create feature importance analysis

### Exercise 2: TensorFlow Model
1. Create CNN for image classification
2. Implement custom training loop
3. Add data augmentation
4. Deploy model with TensorFlow Serving

### Exercise 3: PyTorch Research
1. Build custom neural network
2. Implement custom loss function
3. Create data loading pipeline
4. Add model interpretation

### Project: Multi-Framework ML System

**Objective**: Build ML system using multiple frameworks

**Requirements**:
- Scikit-learn for preprocessing
- TensorFlow for production model
- PyTorch for research experiments
- Unified evaluation metrics

**Deliverables**:
- Complete ML pipeline
- Model comparison framework
- Deployment strategy
- Performance benchmarks

---

## 📖 Further Reading

### Essential Resources

1. **Scikit-learn**
   - [User Guide](https://scikit-learn.org/stable/user_guide.html)
   - [API Reference](https://scikit-learn.org/stable/modules/classes.html)

2. **TensorFlow**
   - [Tutorials](https://www.tensorflow.org/tutorials)
   - [Guide](https://www.tensorflow.org/guide)

3. **PyTorch**
   - [Tutorials](https://pytorch.org/tutorials/)
   - [Documentation](https://pytorch.org/docs/)

### Advanced Topics

- **Model Optimization**: Quantization, pruning, distillation
- **Distributed Training**: Multi-GPU, multi-node training
- **Model Deployment**: Serving, edge deployment, mobile
- **AutoML**: Automated model selection and hyperparameter tuning

### 2025 Trends

- **Unified APIs**: Cross-framework compatibility
- **AutoML Integration**: Automated model development
- **Edge Computing**: Mobile and IoT optimization
- **Federated Learning**: Privacy-preserving distributed training

---

## 🎯 Key Takeaways

1. **Framework Selection**: Choose based on use case and requirements
2. **Performance Optimization**: Each framework has unique optimization strategies
3. **Deployment Considerations**: Production requirements influence framework choice
4. **Research Flexibility**: PyTorch excels in research and experimentation
5. **Ecosystem Integration**: Consider the broader tool ecosystem

---

*"The best framework is the one that gets your model into production."*

**Next: [ML Frameworks Advanced](tools_and_ides/36_ml_frameworks_advanced.md) → Hugging Face, ONNX, JAX**