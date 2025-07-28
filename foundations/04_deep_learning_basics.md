# Deep Learning Basics: Neural Networks and Beyond

*"From simple neurons to complex architectures that power modern AI"*

---

## 📚 Table of Contents

1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Activation Functions](#activation-functions)
4. [Backpropagation](#backpropagation)
5. [Optimizers](#optimizers)
6. [Regularization Techniques](#regularization-techniques)
7. [Implementation with PyTorch](#implementation-with-pytorch)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

---

## 🧠 Introduction to Neural Networks

### What is Deep Learning?

**Deep Learning** is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. Unlike traditional ML algorithms, deep learning can automatically learn hierarchical representations of data.

### Why Deep Learning?

| Aspect | Traditional ML | Deep Learning |
|--------|----------------|---------------|
| **Feature Engineering** | Manual feature extraction | Automatic feature learning |
| **Data Requirements** | Small to medium datasets | Large datasets |
| **Performance** | Good for simple patterns | Excellent for complex patterns |
| **Interpretability** | High | Lower (black box) |
| **Computational Cost** | Low | High |

### Historical Context

```
1943: McCulloch-Pitts Neuron
1957: Perceptron (Rosenblatt)
1969: Perceptron limitations (Minsky & Papert)
1986: Backpropagation revival (Rumelhart et al.)
2006: Deep Learning renaissance (Hinton)
2012: ImageNet breakthrough (Krizhevsky)
2015: ResNet (He et al.)
2017: Transformer architecture
2020: GPT-3 and large language models
2024: Multimodal AI revolution
```

---

## 🏗️ Neural Network Architecture

### Basic Structure

A neural network consists of:
1. **Input Layer**: Receives data
2. **Hidden Layers**: Process information
3. **Output Layer**: Produces predictions

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
    ↓              ↓                ↓              ↓
   x₁            h₁₁              h₂₁            y₁
   x₂            h₁₂              h₂₂            y₂
   x₃            h₁₃              h₂₃            y₃
```

### Mathematical Foundation

For a single layer with inputs **x** and weights **W**:

```
z = W^T x + b
a = f(z)
```

Where:
- **z**: Weighted sum (pre-activation)
- **W**: Weight matrix
- **b**: Bias vector
- **f**: Activation function
- **a**: Activation (output)

### Forward Propagation

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize neural network
        
        Args:
            layer_sizes (list): List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization for better training
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((layer_sizes[i + 1], 1))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X (np.array): Input data of shape (features, samples)
        
        Returns:
            list: Activations for each layer
        """
        activations = [X]
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            
            # Apply activation function (sigmoid for hidden layers)
            if i == len(self.weights) - 1:
                # Output layer - no activation for regression
                a = z
            else:
                # Hidden layers - sigmoid activation
                a = self.sigmoid(z)
            
            activations.append(a)
        
        return activations

# Example usage
layer_sizes = [2, 3, 1]  # 2 inputs, 3 hidden neurons, 1 output
nn = SimpleNeuralNetwork(layer_sizes)

# Test forward propagation
X = np.array([[1, 2], [3, 4]])  # 2 features, 2 samples
activations = nn.forward(X)
print(f"Output shape: {activations[-1].shape}")
```

---

## ⚡ Activation Functions

### Why Activation Functions?

Without activation functions, neural networks would be limited to linear transformations, regardless of depth. Activation functions introduce non-linearity, enabling networks to learn complex patterns.

### Common Activation Functions

#### 1. Sigmoid Function

```python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

# Plot sigmoid
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = sigmoid(x)
dy = sigmoid_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, dy)
plt.title('Sigmoid Derivative')
plt.xlabel('x')
plt.ylabel('σ\'(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Pros**: Smooth, bounded output (0,1)
**Cons**: Vanishing gradient problem, not zero-centered

#### 2. ReLU (Rectified Linear Unit)

```python
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return np.where(x > 0, 1, 0)

# Plot ReLU
x = np.linspace(-5, 5, 100)
y = relu(x)
dy = relu_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, dy)
plt.title('ReLU Derivative')
plt.xlabel('x')
plt.ylabel('ReLU\'(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Pros**: Computationally efficient, helps with vanishing gradients
**Cons**: Dying ReLU problem, not bounded

#### 3. Tanh (Hyperbolic Tangent)

```python
def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2

# Plot tanh
x = np.linspace(-5, 5, 100)
y = tanh(x)
dy = tanh_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, dy)
plt.title('Tanh Derivative')
plt.xlabel('x')
plt.ylabel('tanh\'(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Pros**: Zero-centered, bounded (-1,1)
**Cons**: Still has vanishing gradient problem

#### 4. Leaky ReLU

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where(x > 0, 1, alpha)

# Plot Leaky ReLU
x = np.linspace(-5, 5, 100)
y = leaky_relu(x)
dy = leaky_relu_derivative(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Leaky ReLU Function')
plt.xlabel('x')
plt.ylabel('LeakyReLU(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, dy)
plt.title('Leaky ReLU Derivative')
plt.xlabel('x')
plt.ylabel('LeakyReLU\'(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Pros**: Prevents dying ReLU, computationally efficient
**Cons**: Requires tuning of alpha parameter

### Activation Function Comparison

| Function | Range | Pros | Cons |
|----------|-------|------|------|
| **Sigmoid** | (0,1) | Smooth, bounded | Vanishing gradient |
| **ReLU** | [0,∞) | Efficient, no vanishing gradient | Dying ReLU |
| **Tanh** | (-1,1) | Zero-centered, bounded | Vanishing gradient |
| **Leaky ReLU** | (-∞,∞) | Prevents dying ReLU | Requires tuning |

---

## 🔄 Backpropagation

### The Learning Algorithm

Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to the network parameters.

### Mathematical Foundation

For a neural network with L layers:

**Forward Pass**:
```
a⁽⁰⁾ = x
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f(z⁽ˡ⁾)
```

**Backward Pass**:
```
δ⁽ᴸ⁾ = ∇ₐJ ⊙ f'(z⁽ᴸ⁾)
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)
```

**Gradients**:
```
∂J/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
∂J/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### Implementation

```python
class NeuralNetworkWithBackprop:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.01
            b = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer - no activation for regression
                a = z
            else:
                # Hidden layers
                a = self.sigmoid(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def compute_loss(self, y_pred, y_true):
        """Mean squared error loss"""
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X, y, activations, z_values):
        """Backpropagation"""
        m = X.shape[1]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = activations[-1] - y
        
        # Backpropagate through layers
        for l in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW[l] = np.dot(delta, activations[l].T) / m
            db[l] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Compute error for previous layer
            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * self.sigmoid_derivative(z_values[l - 1])
        
        return dW, db
    
    def update_parameters(self, dW, db, learning_rate):
        """Update weights and biases"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
    
    def train(self, X, y, epochs, learning_rate):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(activations[-1], y)
            losses.append(loss)
            
            # Backward pass
            dW, db = self.backward(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(dW, db, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

# Example: XOR problem
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

# Create and train network
nn = NeuralNetworkWithBackprop([2, 4, 1])
losses = nn.train(X, y, epochs=1000, learning_rate=0.1)

# Test predictions
activations, _ = nn.forward(X)
predictions = activations[-1]
print("\nPredictions:")
for i in range(X.shape[1]):
    print(f"Input: {X[:, i]}, Predicted: {predictions[0, i]:.3f}, Actual: {y[0, i]}")
```

---

## 🎯 Optimizers

### Gradient Descent Variants

#### 1. Stochastic Gradient Descent (SGD)

```python
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """Update parameters using SGD"""
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad
```

#### 2. Adam Optimizer

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, params, grads):
        """Update parameters using Adam"""
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

#### 3. RMSprop

```python
class RMSprop:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.v = None
    
    def update(self, params, grads):
        """Update parameters using RMSprop"""
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update moving average of squared gradients
            self.v[i] = self.rho * self.v[i] + (1 - self.rho) * (grad ** 2)
            
            # Update parameters
            param -= self.learning_rate * grad / (np.sqrt(self.v[i]) + self.epsilon)
```

### Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| **SGD** | Simple, memory efficient | Slow convergence, sensitive to learning rate | Simple problems |
| **Adam** | Fast convergence, adaptive learning rate | More memory usage | Most problems |
| **RMSprop** | Good for non-convex problems | Requires tuning | RNNs, non-convex |

---

## 🛡️ Regularization Techniques

### 1. L1 and L2 Regularization

```python
def l2_regularization(weights, lambda_reg=0.01):
    """L2 regularization penalty"""
    penalty = 0
    for w in weights:
        penalty += np.sum(w ** 2)
    return lambda_reg * penalty / 2

def l1_regularization(weights, lambda_reg=0.01):
    """L1 regularization penalty"""
    penalty = 0
    for w in weights:
        penalty += np.sum(np.abs(w))
    return lambda_reg * penalty
```

### 2. Dropout

```python
def dropout_mask(shape, dropout_rate=0.5):
    """Create dropout mask"""
    mask = np.random.binomial(1, 1 - dropout_rate, shape) / (1 - dropout_rate)
    return mask

def apply_dropout(activations, dropout_rate=0.5, training=True):
    """Apply dropout to activations"""
    if training and dropout_rate > 0:
        mask = dropout_mask(activations.shape, dropout_rate)
        return activations * mask
    return activations
```

### 3. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

---

## 🔧 Implementation with PyTorch

### Basic Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN(input_size=2, hidden_size=4, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training data (XOR problem)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Test predictions
with torch.no_grad():
    predictions = model(X)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i].numpy()}, Predicted: {predictions[i].item():.3f}, Actual: {y[i].item()}")
```

### Convolutional Neural Network

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create CNN model
cnn_model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
```

---

## 🧪 Exercises and Projects

### Exercise 1: Build a Neural Network from Scratch

```python
# TODO: Implement a complete neural network class
# Features: Multiple layers, different activation functions, mini-batch training

class CompleteNeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize neural network
        
        Args:
            layer_sizes (list): List of layer sizes
            activation (str): Activation function ('relu', 'sigmoid', 'tanh')
        """
        pass
    
    def forward(self, X):
        """Forward propagation"""
        pass
    
    def backward(self, X, y):
        """Backward propagation"""
        pass
    
    def train(self, X, y, epochs, batch_size=32):
        """Train with mini-batch gradient descent"""
        pass
```

### Exercise 2: Image Classification with CNN

```python
# TODO: Build an image classifier using CNN
# Dataset: MNIST or CIFAR-10
# Features: Convolutional layers, pooling, dropout, batch normalization

def build_image_classifier():
    """
    Build a complete image classification system
    
    Steps:
    1. Load and preprocess image data
    2. Design CNN architecture
    3. Implement training loop with validation
    4. Add data augmentation
    5. Evaluate and visualize results
    """
    pass
```

### Exercise 3: Text Classification with RNN

```python
# TODO: Build a text classifier using RNN/LSTM
# Dataset: Sentiment analysis or text classification
# Features: Embedding layer, LSTM, attention mechanism

def build_text_classifier():
    """
    Build a complete text classification system
    
    Steps:
    1. Load and preprocess text data
    2. Create word embeddings
    3. Design RNN/LSTM architecture
    4. Implement attention mechanism
    5. Train and evaluate model
    """
    pass
```

---

## 📖 Further Reading

### Essential Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Hands-On Machine Learning" by Aurélien Géron

### Key Papers
- "Backpropagation Through Time" by Rumelhart et al. (1986)
- "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al. (2012)
- "Attention Is All You Need" by Vaswani et al. (2017)

### Online Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official tutorials
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) - Official tutorials
- [fast.ai](https://www.fast.ai/) - Practical deep learning

### Next Steps
- **[Deep Learning Advanced](05_deep_learning_advanced.md)**: Modern architectures and techniques
- **[NLP Fundamentals](core_ml_fields/06_nlp_fundamentals.md)**: Natural language processing
- **[Computer Vision Basics](core_ml_fields/08_computer_vision_basics.md)**: Image processing

---

## 🎯 Key Takeaways

1. **Neural Networks**: Multi-layer perceptrons with non-linear activations
2. **Backpropagation**: Algorithm for computing gradients efficiently
3. **Activation Functions**: Introduce non-linearity (ReLU most popular)
4. **Optimizers**: Adam generally works well for most problems
5. **Regularization**: Prevents overfitting (dropout, L2, early stopping)
6. **Frameworks**: PyTorch and TensorFlow for production systems

---

*"Deep learning is not just about stacking layers—it's about understanding the mathematical foundations and choosing the right architecture for your problem."*

**Next: [Deep Learning Advanced](05_deep_learning_advanced.md) → Modern architectures and cutting-edge techniques** 