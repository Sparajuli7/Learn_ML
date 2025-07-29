# Hands-On Projects: Intermediate (20 Intermediate Projects)

*"Building on fundamentals to create sophisticated ML systems"*

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Projects 1-10: Core ML Applications](#projects-1-10-core-ml-applications)
3. [Projects 11-20: Advanced Implementations](#projects-11-20-advanced-implementations)
4. [Integration Challenges](#integration-challenges)
5. [Further Reading](#further-reading)

---

## 🎯 Project Overview

This section contains 20 intermediate-level machine learning projects designed to build sophisticated skills. Each project focuses on real-world applications with production-ready implementations.

### Skill Progression
- **Prerequisites**: Basic ML concepts, Python proficiency, data manipulation
- **Target Skills**: Advanced algorithms, system design, optimization
- **Duration**: 4-8 hours per project
- **Complexity**: ⭐⭐⭐ (Intermediate)

---

## 🚀 Projects 1-10: Core ML Applications

### Project 1: Recommendation System with Collaborative Filtering
**Difficulty**: ⭐⭐⭐  
**Duration**: 6-8 hours  
**Skills**: Matrix factorization, SVD, recommendation algorithms

#### Learning Objectives
- Implement collaborative filtering from scratch
- Understand matrix factorization techniques
- Build scalable recommendation systems
- Evaluate recommendation quality

#### Implementation
```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, n_factors=50, learning_rate=0.01, reg_param=0.1):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        
    def fit(self, ratings_matrix):
        """Train the model using matrix factorization"""
        self.n_users, self.n_items = ratings_matrix.shape
        
        # Initialize user and item factors
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        
        # Get non-zero ratings
        user_indices, item_indices = ratings_matrix.nonzero()
        
        for epoch in range(100):
            for u, i in zip(user_indices, item_indices):
                # Get actual rating
                r = ratings_matrix[u, i]
                
                # Predict rating
                pred = np.dot(self.user_factors[u], self.item_factors[i])
                
                # Calculate error
                error = r - pred
                
                # Update factors
                self.user_factors[u] += self.learning_rate * (
                    error * self.item_factors[i] - self.reg_param * self.user_factors[u]
                )
                self.item_factors[i] += self.learning_rate * (
                    error * self.user_factors[u] - self.reg_param * self.item_factors[i]
                )
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate top-N recommendations for user"""
        predictions = []
        for item_id in range(self.n_items):
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# Usage example
def create_sample_data():
    """Create sample ratings matrix"""
    np.random.seed(42)
    n_users, n_items = 100, 50
    ratings = np.random.randint(1, 6, (n_users, n_items))
    
    # Add sparsity (only 20% of ratings available)
    mask = np.random.random((n_users, n_items)) < 0.2
    ratings[~mask] = 0
    
    return csr_matrix(ratings)

# Train and evaluate
ratings_matrix = create_sample_data()
cf_model = CollaborativeFiltering(n_factors=20)
cf_model.fit(ratings_matrix)

# Generate recommendations
recommendations = cf_model.recommend(user_id=0, n_recommendations=5)
print("Top 5 recommendations for user 0:", recommendations)
```

#### Deliverables
- Complete collaborative filtering implementation
- Evaluation metrics (RMSE, MAE, precision@k)
- Visualization of user-item similarity matrices
- Performance optimization analysis

---

### Project 2: Neural Network from Scratch with Backpropagation
**Difficulty**: ⭐⭐⭐  
**Duration**: 8-10 hours  
**Skills**: Neural networks, backpropagation, optimization

#### Learning Objectives
- Implement neural network architecture from scratch
- Understand backpropagation algorithm
- Build multi-layer perceptron
- Implement various activation functions

#### Implementation
```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        """
        Initialize neural network
        layers: list of layer sizes [input_size, hidden_size, ..., output_size]
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i + 1], layers[i]) * 0.01
            b = np.zeros((layers[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return np.where(z > 0, 1, 0)
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer - softmax for classification
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                activation = exp_z / np.sum(exp_z, axis=0, keepdims=True)
            else:
                # Hidden layers - ReLU
                activation = self.relu(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward_propagation(self, X, Y):
        """Backward pass to compute gradients"""
        m = X.shape[1]
        delta = self.activations[-1] - Y
        
        self.weight_gradients = []
        self.bias_gradients = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(delta, self.activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            self.weight_gradients.insert(0, dW)
            self.bias_gradients.insert(0, db)
            
            if i > 0:
                # Compute delta for previous layer
                delta = np.dot(self.weights[i].T, delta) * self.relu_derivative(self.z_values[i-1])
    
    def update_parameters(self):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.weight_gradients[i]
            self.biases[i] -= self.learning_rate * self.bias_gradients[i]
    
    def train(self, X, Y, epochs=1000, print_every=100):
        """Train the neural network"""
        costs = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward_propagation(X)
            
            # Compute cost
            cost = -np.mean(np.sum(Y * np.log(output + 1e-8), axis=0))
            costs.append(cost)
            
            # Backward pass
            self.backward_propagation(X, Y)
            
            # Update parameters
            self.update_parameters()
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward_propagation(X)
        return np.argmax(output, axis=0)

# Example usage
def create_sample_data():
    """Create sample classification data"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_features, n_samples)
    Y = np.random.randint(0, n_classes, n_samples)
    
    # One-hot encode Y
    Y_one_hot = np.zeros((n_classes, n_samples))
    Y_one_hot[Y, np.arange(n_samples)] = 1
    
    return X, Y_one_hot, Y

# Train neural network
X, Y_one_hot, Y = create_sample_data()
nn = NeuralNetwork([20, 64, 32, 3], learning_rate=0.01)
costs = nn.train(X, Y_one_hot, epochs=1000)

# Plot training progress
plt.plot(costs)
plt.title('Training Cost Over Time')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

# Evaluate
predictions = nn.predict(X)
accuracy = np.mean(predictions == Y)
print(f"Training Accuracy: {accuracy:.4f}")
```

#### Deliverables
- Complete neural network implementation
- Training visualization and analysis
- Performance comparison with scikit-learn
- Hyperparameter optimization study

---

### Project 3: Natural Language Processing Pipeline
**Difficulty**: ⭐⭐⭐  
**Duration**: 6-8 hours  
**Skills**: NLP, text processing, embeddings, transformers

#### Learning Objectives
- Build complete NLP pipeline from scratch
- Implement word embeddings and attention mechanisms
- Create text classification and sentiment analysis
- Understand transformer architecture basics

#### Implementation
```python
import numpy as np
import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Filter by minimum frequency
        vocab = {word for word, count in word_counts.items() if count >= min_freq}
        
        # Create mappings
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        return self.word_to_idx
    
    def text_to_sequence(self, text, max_length=100):
        """Convert text to sequence of indices"""
        words = text.split()
        sequence = [self.word_to_idx.get(word, 0) for word in words[:max_length]]
        
        # Pad or truncate
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))
        
        return sequence

class SimpleTransformer:
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Initialize parameters
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding matrix"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def multi_head_attention(self, Q, K, V, mask=None):
        """Multi-head attention mechanism"""
        batch_size, seq_len, d_model = Q.shape
        d_k = d_model // self.n_heads
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attention_weights = self.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        return output, attention_weights
    
    def softmax(self, x, axis=-1):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def feed_forward(self, x):
        """Feed-forward network"""
        # Simple feed-forward with ReLU
        return np.maximum(0, x)
    
    def transformer_block(self, x):
        """Single transformer block"""
        # Multi-head attention
        attn_output, _ = self.multi_head_attention(x, x, x)
        
        # Add & Norm
        x = x + attn_output
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Add & Norm
        x = x + ff_output
        
        return x
    
    def forward(self, input_ids, max_length=100):
        """Forward pass through transformer"""
        batch_size = len(input_ids)
        
        # Embedding
        embeddings = self.embedding[input_ids]
        
        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:max_length]
        
        # Pass through transformer blocks
        x = embeddings
        for _ in range(self.n_layers):
            x = self.transformer_block(x)
        
        # Global average pooling
        output = np.mean(x, axis=1)
        
        return output

class NLPClassifier:
    def __init__(self, vocab_size, num_classes, d_model=128):
        self.transformer = SimpleTransformer(vocab_size, d_model)
        self.classifier = np.random.randn(d_model, num_classes) * 0.1
        
    def train(self, texts, labels, epochs=10, learning_rate=0.01):
        """Train the classifier"""
        # Preprocess texts
        preprocessor = TextPreprocessor()
        preprocessor.build_vocabulary(texts)
        
        # Convert texts to sequences
        sequences = [preprocessor.text_to_sequence(text) for text in texts]
        
        # Convert labels to one-hot
        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        
        Y = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            Y[i, label_to_idx[label]] = 1
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(sequences), 32):  # Mini-batch
                batch_sequences = sequences[i:i+32]
                batch_labels = Y[i:i+32]
                
                # Forward pass
                transformer_output = self.transformer.forward(batch_sequences)
                logits = np.dot(transformer_output, self.classifier)
                
                # Compute loss (cross-entropy)
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                loss = -np.mean(np.sum(batch_labels * np.log(probs + 1e-8), axis=1))
                total_loss += loss
                
                # Backward pass (simplified)
                # In practice, you'd use automatic differentiation
                grad_output = (probs - batch_labels) / len(batch_sequences)
                grad_classifier = np.dot(transformer_output.T, grad_output)
                
                # Update parameters
                self.classifier -= learning_rate * grad_classifier
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def predict(self, texts):
        """Make predictions"""
        preprocessor = TextPreprocessor()
        sequences = [preprocessor.text_to_sequence(text) for text in texts]
        
        transformer_output = self.transformer.forward(sequences)
        logits = np.dot(transformer_output, self.classifier)
        
        return np.argmax(logits, axis=1)

# Example usage
def create_sample_nlp_data():
    """Create sample text classification data"""
    texts = [
        "I love this product, it's amazing!",
        "This is terrible, I hate it.",
        "Great service and fast delivery.",
        "Poor quality, not worth the money.",
        "Excellent customer support team.",
        "Disappointed with the purchase.",
        "Highly recommend this product.",
        "Waste of money, avoid this.",
        "Perfect for my needs.",
        "Not satisfied at all."
    ]
    
    labels = ['positive', 'negative', 'positive', 'negative', 'positive', 
              'negative', 'positive', 'negative', 'positive', 'negative']
    
    return texts, labels

# Train NLP classifier
texts, labels = create_sample_nlp_data()
classifier = NLPClassifier(vocab_size=1000, num_classes=2)
classifier.train(texts, labels, epochs=20)

# Test predictions
test_texts = ["This is really good!", "I don't like this at all."]
predictions = classifier.predict(test_texts)
print("Predictions:", predictions)
```

#### Deliverables
- Complete NLP pipeline implementation
- Text preprocessing and feature extraction
- Transformer-based classification
- Performance evaluation and analysis

---

### Project 4: Computer Vision with Convolutional Neural Networks
**Difficulty**: ⭐⭐⭐  
**Duration**: 8-10 hours  
**Skills**: CNN, image processing, computer vision

#### Learning Objectives
- Implement CNN from scratch
- Understand convolution, pooling, and activation functions
- Build image classification system
- Create data augmentation pipeline

#### Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(out_channels)
        
    def forward(self, input_data):
        """Forward pass"""
        batch_size, in_channels, height, width = input_data.shape
        
        # Add padding
        if self.padding > 0:
            padded_input = np.pad(input_data, ((0, 0), (0, 0), 
                                             (self.padding, self.padding), 
                                             (self.padding, self.padding)))
        else:
            padded_input = input_data
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for c in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = padded_input[b, :, h_start:h_end, w_start:w_end]
                        
                        # Apply filter
                        output[b, c, h, w] = np.sum(patch * self.filters[c]) + self.biases[c]
        
        return output

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        
    def forward(self, input_data):
        """Forward pass"""
        batch_size, channels, height, width = input_data.shape
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = input_data[b, c, h_start:h_end, w_start:w_end]
                        
                        # Max pooling
                        output[b, c, h, w] = np.max(patch)
        
        return output

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class Flatten:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)
        
    def forward(self, x):
        return np.dot(x, self.weights) + self.biases

class CNN:
    def __init__(self):
        self.layers = [
            Conv2D(1, 32, 3, padding=1),
            ReLU(),
            MaxPool2D(2),
            
            Conv2D(32, 64, 3, padding=1),
            ReLU(),
            MaxPool2D(2),
            
            Flatten(),
            Dense(64 * 7 * 7, 128),
            ReLU(),
            Dense(128, 10)
        ]
        
    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, x):
        """Make predictions"""
        output = self.forward(x)
        return np.argmax(output, axis=1)

def create_sample_image_data():
    """Create sample image data"""
    np.random.seed(42)
    n_samples = 1000
    height, width = 28, 28
    
    # Create random images
    images = np.random.randn(n_samples, 1, height, width)
    
    # Create random labels
    labels = np.random.randint(0, 10, n_samples)
    
    return images, labels

def data_augmentation(image):
    """Simple data augmentation"""
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        image = ndimage.rotate(image, angle, reshape=False)
    
    # Random brightness
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        image = image * factor
    
    return image

# Create and test CNN
cnn = CNN()
images, labels = create_sample_image_data()

# Test forward pass
output = cnn.forward(images[:10])
print("Output shape:", output.shape)

# Test prediction
predictions = cnn.predict(images[:10])
print("Predictions:", predictions)

# Data augmentation example
augmented_image = data_augmentation(images[0])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[0, 0], cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(augmented_image[0], cmap='gray')
plt.title('Augmented Image')
plt.show()
```

#### Deliverables
- Complete CNN implementation
- Data augmentation pipeline
- Image preprocessing utilities
- Performance analysis and visualization

---

### Project 5: Time Series Forecasting with LSTM
**Difficulty**: ⭐⭐⭐  
**Duration**: 6-8 hours  
**Skills**: Time series, LSTM, sequence modeling

#### Learning Objectives
- Implement LSTM from scratch
- Build time series forecasting pipeline
- Understand sequence modeling concepts
- Create data preprocessing for time series

#### Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        
        # Initialize biases
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """Forward pass through LSTM cell"""
        # Concatenate input and previous hidden state
        concat = np.concatenate([x, h_prev], axis=1)
        
        # Gates
        ft = self.sigmoid(np.dot(self.Wf, concat.T) + self.bf)
        it = self.sigmoid(np.dot(self.Wi, concat.T) + self.bi)
        ot = self.sigmoid(np.dot(self.Wo, concat.T) + self.bo)
        
        # Candidate cell state
        c_tilde = self.tanh(np.dot(self.Wc, concat.T) + self.bc)
        
        # Cell state
        ct = ft * c_prev + it * c_tilde
        
        # Hidden state
        ht = ot * self.tanh(ct)
        
        return ht, ct

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.output_layer = np.random.randn(hidden_size, output_size) * 0.1
        self.output_bias = np.zeros(output_size)
        
    def forward(self, x_sequence):
        """Forward pass through LSTM"""
        batch_size, seq_len, input_size = x_sequence.shape
        
        # Initialize hidden and cell states
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        
        for t in range(seq_len):
            # Get input at time step t
            x_t = x_sequence[:, t, :]
            
            # Forward pass through LSTM cell
            h, c = self.lstm_cell.forward(x_t, h, c)
            
            # Output layer
            output = np.dot(h, self.output_layer) + self.output_bias
            outputs.append(output)
        
        return np.array(outputs).transpose(1, 0, 2)

def create_time_series_data():
    """Create sample time series data"""
    np.random.seed(42)
    n_samples = 1000
    seq_length = 50
    
    # Create synthetic time series with trend and seasonality
    t = np.linspace(0, 10, n_samples)
    trend = 0.1 * t
    seasonality = 2 * np.sin(2 * np.pi * t)
    noise = 0.1 * np.random.randn(n_samples)
    
    time_series = trend + seasonality + noise
    
    # Create sequences
    X, y = [], []
    for i in range(n_samples - seq_length):
        X.append(time_series[i:i+seq_length])
        y.append(time_series[i+seq_length])
    
    return np.array(X), np.array(y)

def prepare_data(X, y, train_ratio=0.8):
    """Prepare data for training"""
    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    
    # Split data
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

# Create and train LSTM
X, y = create_time_series_data()
X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

# Initialize LSTM
lstm = LSTM(input_size=1, hidden_size=32, output_size=1)

# Training loop (simplified)
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Forward pass
    predictions = lstm.forward(X_train.reshape(-1, 50, 1))
    
    # Compute loss (MSE)
    loss = np.mean((predictions[:, -1, 0] - y_train) ** 2)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Test predictions
test_predictions = lstm.forward(X_test.reshape(-1, 50, 1))
test_loss = np.mean((test_predictions[:, -1, 0] - y_test) ** 2)
print(f"Test Loss: {test_loss:.6f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Actual')
plt.plot(test_predictions[:100, -1, 0], label='Predicted')
plt.title('Time Series Forecasting with LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

#### Deliverables
- Complete LSTM implementation
- Time series preprocessing pipeline
- Forecasting evaluation metrics
- Visualization and analysis tools

---

## 🚀 Projects 11-20: Advanced Implementations

### Project 11: Reinforcement Learning with Q-Learning
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 8-10 hours  
**Skills**: RL, Q-learning, policy optimization

### Project 12: Generative Adversarial Networks (GANs)
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 10-12 hours  
**Skills**: GANs, generative models, adversarial training

### Project 13: Autoencoder for Dimensionality Reduction
**Difficulty**: ⭐⭐⭐  
**Duration**: 6-8 hours  
**Skills**: Autoencoders, dimensionality reduction, feature learning

### Project 14: Ensemble Methods and Model Stacking
**Difficulty**: ⭐⭐⭐  
**Duration**: 6-8 hours  
**Skills**: Ensemble learning, model combination, stacking

### Project 15: Hyperparameter Optimization with Bayesian Optimization
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 8-10 hours  
**Skills**: Bayesian optimization, hyperparameter tuning, AutoML

### Project 16: Graph Neural Networks for Node Classification
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 10-12 hours  
**Skills**: GNNs, graph theory, node embeddings

### Project 17: Transformer for Sequence-to-Sequence Tasks
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 12-15 hours  
**Skills**: Transformers, attention mechanisms, seq2seq

### Project 18: Federated Learning Implementation
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 10-12 hours  
**Skills**: Federated learning, distributed ML, privacy

### Project 19: Model Interpretability with SHAP and LIME
**Difficulty**: ⭐⭐⭐  
**Duration**: 6-8 hours  
**Skills**: Model interpretability, explainable AI, SHAP

### Project 20: Production ML Pipeline with MLOps
**Difficulty**: ⭐⭐⭐⭐  
**Duration**: 12-15 hours  
**Skills**: MLOps, CI/CD, model deployment, monitoring

---

## 🎯 Integration Challenges

### Challenge 1: Multi-Modal Learning System
Combine projects 3, 4, and 17 to create a system that processes text, images, and sequences simultaneously.

### Challenge 2: End-to-End ML Platform
Integrate projects 15, 19, and 20 to build a complete ML platform with automated hyperparameter tuning, model interpretability, and production deployment.

### Challenge 3: Advanced Recommendation System
Combine projects 1, 13, and 16 to create a hybrid recommendation system using autoencoders and graph neural networks.

---

## 📖 Further Reading

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto

### Papers
- "Attention Is All You Need" (Transformer)
- "Generative Adversarial Networks" (GANs)
- "Graph Attention Networks" (GAT)

### Online Resources
- TensorFlow tutorials
- PyTorch documentation
- Hugging Face courses

---

## 🎯 Key Takeaways

1. **Practical Implementation**: Each project focuses on real-world applications
2. **From Scratch**: Understanding algorithms by implementing them
3. **Production Ready**: Code that can be deployed in real systems
4. **Performance Optimization**: Efficient implementations and best practices
5. **Integration Skills**: Combining multiple techniques for complex solutions

---

*"The best way to learn is by doing. These projects bridge the gap between theory and practice."*

**Next: [Advanced Projects](projects_and_practice/82_hands_on_projects_advanced.md) → Complex systems and cutting-edge implementations** 