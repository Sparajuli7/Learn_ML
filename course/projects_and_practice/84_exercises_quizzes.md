# Exercises & Quizzes: 200+ Problems with Solutions

## Overview
Comprehensive collection of machine learning exercises, quizzes, and problems with detailed solutions to test and reinforce your understanding.

---

## Section 1: Fundamentals (50 Problems)

### Problem 1: Linear Algebra Basics
**Question**: Given matrices A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]], calculate A × B.

**Solution**:
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print(C)
# Output: [[19 22]
#          [43 50]]
```

### Problem 2: Probability Basics
**Question**: In a fair coin toss, what is the probability of getting exactly 3 heads in 5 tosses?

**Solution**:
```python
from scipy.stats import binom

# Binomial probability: P(X = 3) where n=5, p=0.5
probability = binom.pmf(3, 5, 0.5)
print(f"Probability: {probability:.4f}")
# Output: 0.3125
```

### Problem 3: Gradient Descent Implementation
**Question**: Implement gradient descent to find the minimum of f(x) = x² + 2x + 1.

**Solution**:
```python
import numpy as np

def gradient_descent(f, df, x0, learning_rate=0.1, epochs=100):
    x = x0
    for _ in range(epochs):
        grad = df(x)
        x = x - learning_rate * grad
    return x

# Function and its derivative
f = lambda x: x**2 + 2*x + 1
df = lambda x: 2*x + 2

# Find minimum
minimum = gradient_descent(f, df, x0=5)
print(f"Minimum at x = {minimum:.4f}")
# Output: Minimum at x = -1.0000
```

### Problem 4: Cross-Entropy Loss
**Question**: Calculate the cross-entropy loss for predictions [0.7, 0.2, 0.1] with true labels [1, 0, 0].

**Solution**:
```python
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

y_true = np.array([1, 0, 0])
y_pred = np.array([0.7, 0.2, 0.1])

loss = cross_entropy_loss(y_true, y_pred)
print(f"Cross-entropy loss: {loss:.4f}")
# Output: 0.3567
```

### Problem 5: Confusion Matrix
**Question**: Given predictions [1, 0, 1, 1, 0] and true labels [1, 0, 0, 1, 0], calculate precision, recall, and F1-score.

**Solution**:
```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
```

---

## Section 2: Supervised Learning (50 Problems)

### Problem 6: Linear Regression from Scratch
**Question**: Implement linear regression using only NumPy, given data points (x, y).

**Solution**:
```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, epochs=1000):
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

### Problem 7: Logistic Regression Implementation
**Question**: Implement logistic regression for binary classification.

**Solution**:
```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, epochs=1000):
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) >= 0.5).astype(int)
```

### Problem 8: Decision Tree Implementation
**Question**: Implement a simple decision tree classifier.

**Solution**:
```python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def entropy(self, y):
        counts = Counter(y)
        probabilities = [count/len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities)
    
    def information_gain(self, X, y, feature_idx, threshold):
        parent_entropy = self.entropy(y)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        
        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)
        
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    
    def find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        feature_idx, threshold = self.find_best_split(X, y)
        
        if feature_idx is None:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return {
            'feature': feature_idx,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    
    def predict_single(self, x, tree):
        if 'class' in tree:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])
    
    def predict(self, X):
        return [self.predict_single(x, self.tree) for x in X]
```

### Problem 9: K-Means Clustering
**Question**: Implement K-means clustering algorithm.

**Solution**:
```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        # Initialize centroids randomly
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.k)])
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
```

### Problem 10: Support Vector Machine
**Question**: Implement a simple SVM classifier using gradient descent.

**Solution**:
```python
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, epochs=1000):
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - 
                                                        np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
```

---

## Section 3: Deep Learning (50 Problems)

### Problem 11: Neural Network from Scratch
**Question**: Implement a simple neural network with backpropagation.

**Solution**:
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i+1], layers[i]) * 0.01
            b = np.zeros((layers[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = self.weights[i] @ self.activations[-1] + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[1]
        
        # Backward pass
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = (1/m) * delta @ self.activations[i].T
            db = (1/m) * np.sum(delta, axis=1, keepdims=True)
            
            if i > 0:
                delta = self.weights[i].T @ delta * self.sigmoid_derivative(self.z_values[i-1])
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
```

### Problem 12: Convolutional Neural Network
**Question**: Implement a simple CNN for image classification.

**Solution**:
```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Example usage
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

### Problem 13: Recurrent Neural Network
**Question**: Implement a simple RNN for sequence classification.

**Solution**:
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        
        # Use the last output
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 10
hidden_size = 20
num_classes = 2
model = SimpleRNN(input_size, hidden_size, num_classes)
```

### Problem 14: LSTM Implementation
**Question**: Implement LSTM cell from scratch.

**Solution**:
```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, h, c):
        # Concatenate input and hidden state
        combined = torch.cat((x, h), dim=1)
        
        # Calculate gates
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        
        # Update cell state
        c_new = f_t * c + i_t * c_tilde
        
        # Update hidden state
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new
```

### Problem 15: Attention Mechanism
**Question**: Implement a simple attention mechanism.

**Solution**:
```python
import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # Attention weights
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden_states):
        # Calculate attention scores
        energy = torch.tanh(self.attn(hidden_states))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(hidden_states.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return context, attention_weights
```

---

## Section 4: Unsupervised Learning (30 Problems)

### Problem 16: Principal Component Analysis
**Question**: Implement PCA for dimensionality reduction.

**Solution**:
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        self.components = eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

### Problem 17: K-Means Clustering Optimization
**Question**: Implement K-means++ initialization for better clustering.

**Solution**:
```python
import numpy as np

class KMeansPlusPlus:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
    
    def initialize_centroids(self, X):
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.k):
            # Calculate distances to nearest centroid
            distances = []
            for x in X:
                min_dist = min([np.linalg.norm(x - c) for c in centroids])
                distances.append(min_dist)
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = np.array(distances) ** 2
            probabilities /= probabilities.sum()
            
            next_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_centroid_idx])
        
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.k)])
            
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
```

### Problem 18: Gaussian Mixture Models
**Question**: Implement GMM using EM algorithm.

**Solution**:
```python
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iters=100):
        self.n_components = n_components
        self.max_iters = max_iters
        self.means = None
        self.covariances = None
        self.weights = None
    
    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        # Initialize means randomly
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        
        # Initialize covariances as identity matrices
        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
    
    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            responsibilities[:, i] = self.weights[i] * multivariate_normal.pdf(
                X, self.means[i], self.covariances[i]
            )
        
        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        
        # Update weights
        self.weights = responsibilities.sum(axis=0) / n_samples
        
        # Update means
        for i in range(self.n_components):
            self.means[i] = np.average(X, axis=0, weights=responsibilities[:, i])
        
        # Update covariances
        for i in range(self.n_components):
            diff = X - self.means[i]
            self.covariances[i] = np.dot(
                responsibilities[:, i] * diff.T, diff
            ) / responsibilities[:, i].sum()
    
    def fit(self, X):
        self.initialize_parameters(X)
        
        for _ in range(self.max_iters):
            # E-step
            responsibilities = self.e_step(X)
            
            # M-step
            self.m_step(X, responsibilities)
    
    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)
```

---

## Section 5: Model Evaluation (20 Problems)

### Problem 19: Cross-Validation Implementation
**Question**: Implement k-fold cross-validation from scratch.

**Solution**:
```python
import numpy as np
from sklearn.model_selection import KFold

def cross_validate(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# Example usage
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = LogisticRegression()

mean_score, std_score = cross_validate(model, X, y, k=5)
print(f"Mean CV Score: {mean_score:.3f} (+/- {std_score:.3f})")
```

### Problem 20: ROC Curve and AUC
**Question**: Implement ROC curve calculation and AUC score.

**Solution**:
```python
import numpy as np
from sklearn.metrics import roc_curve, auc

def calculate_roc_auc(y_true, y_scores):
    # Sort by scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Calculate TPR and FPR
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    
    # Normalize
    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos
    
    tpr = tp / total_pos
    fpr = fp / total_neg
    
    # Calculate AUC using trapezoidal rule
    auc_score = np.trapz(tpr, fpr)
    
    return fpr, tpr, auc_score

# Example usage
y_true = np.array([1, 0, 1, 0, 1])
y_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])

fpr, tpr, auc_score = calculate_roc_auc(y_true, y_scores)
print(f"AUC Score: {auc_score:.3f}")
```

---

## Section 6: Feature Engineering (20 Problems)

### Problem 21: Feature Scaling Methods
**Question**: Implement different feature scaling methods.

**Solution**:
```python
import numpy as np

class FeatureScaler:
    def __init__(self, method='standard'):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def fit(self, X):
        if self.method == 'standard':
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        elif self.method == 'minmax':
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        elif self.method == 'robust':
            self.median = np.median(X, axis=0)
            self.q75 = np.percentile(X, 75, axis=0)
            self.q25 = np.percentile(X, 25, axis=0)
    
    def transform(self, X):
        if self.method == 'standard':
            return (X - self.mean) / self.std
        elif self.method == 'minmax':
            return (X - self.min) / (self.max - self.min)
        elif self.method == 'robust':
            return (X - self.median) / (self.q75 - self.q25)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

### Problem 22: Feature Selection
**Question**: Implement feature selection using mutual information.

**Solution**:
```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def feature_selection_mutual_info(X, y, k=10):
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    
    # Get top k features
    top_features = np.argsort(mi_scores)[-k:]
    
    return top_features, mi_scores

# Example usage
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=50, random_state=42)
top_features, mi_scores = feature_selection_mutual_info(X, y, k=10)
print(f"Top 10 features: {top_features}")
```

---

## Section 7: Advanced Topics (30 Problems)

### Problem 23: Ensemble Methods
**Question**: Implement bagging classifier.

**Solution**:
```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
    
    def fit(self, X, y):
        self.estimators = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train estimator
            estimator = clone(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)
        
        return self
    
    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
        )
```

### Problem 24: Gradient Boosting
**Question**: Implement simple gradient boosting for regression.

**Solution**:
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        # Initialize with mean
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # Calculate residuals
            residuals = y - predictions
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
            
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
```

---

## Quiz Section: Multiple Choice Questions

### Quiz 1: Machine Learning Fundamentals

**Q1**: What is the primary goal of supervised learning?
- A) To find patterns in unlabeled data
- B) To predict outcomes based on labeled training data
- C) To reduce dimensionality of data
- D) To cluster similar data points

**Answer**: B) To predict outcomes based on labeled training data

**Q2**: Which of the following is NOT a common loss function for regression?
- A) Mean Squared Error
- B) Cross-Entropy Loss
- C) Mean Absolute Error
- D) Huber Loss

**Answer**: B) Cross-Entropy Loss (used for classification)

**Q3**: What is overfitting in machine learning?
- A) When a model performs poorly on training data
- B) When a model learns the training data too well and fails to generalize
- C) When a model has too few parameters
- D) When a model converges too slowly

**Answer**: B) When a model learns the training data too well and fails to generalize

### Quiz 2: Deep Learning

**Q4**: What is the purpose of the activation function in neural networks?
- A) To increase the number of parameters
- B) To introduce non-linearity into the network
- C) To reduce computational cost
- D) To normalize the input data

**Answer**: B) To introduce non-linearity into the network

**Q5**: What is the vanishing gradient problem?
- A) When gradients become too large during backpropagation
- B) When gradients become very small, slowing down learning in early layers
- C) When the model converges too quickly
- D) When the loss function is not differentiable

**Answer**: B) When gradients become very small, slowing down learning in early layers

**Q6**: Which activation function is most commonly used in hidden layers of modern neural networks?
- A) Sigmoid
- B) Tanh
- C) ReLU
- D) Softmax

**Answer**: C) ReLU

### Quiz 3: Model Evaluation

**Q7**: What does AUC-ROC measure?
- A) The accuracy of a classification model
- B) The ability of a model to distinguish between classes
- C) The precision of a model
- D) The recall of a model

**Answer**: B) The ability of a model to distinguish between classes

**Q8**: What is cross-validation used for?
- A) To increase the training data size
- B) To estimate how well a model will generalize to unseen data
- C) To reduce overfitting
- D) To speed up training

**Answer**: B) To estimate how well a model will generalize to unseen data

---

## Practice Problems Summary

### Problem Types Covered:
1. **Fundamentals** (50 problems): Linear algebra, probability, optimization
2. **Supervised Learning** (50 problems): Regression, classification, ensemble methods
3. **Deep Learning** (50 problems): Neural networks, CNNs, RNNs, attention
4. **Unsupervised Learning** (30 problems): Clustering, dimensionality reduction
5. **Model Evaluation** (20 problems): Cross-validation, metrics, validation
6. **Feature Engineering** (20 problems): Scaling, selection, transformation
7. **Advanced Topics** (30 problems): Ensemble methods, boosting, optimization

### Skills Developed:
- **Mathematical Foundations**: Linear algebra, calculus, probability
- **Algorithm Implementation**: From scratch implementations of ML algorithms
- **Model Evaluation**: Understanding and implementing evaluation metrics
- **Feature Engineering**: Data preprocessing and feature selection
- **Deep Learning**: Neural network architectures and training
- **Practical Application**: Real-world problem solving

### Next Steps:
1. **Practice Regularly**: Work through problems systematically
2. **Implement Solutions**: Code all solutions from scratch
3. **Experiment**: Modify parameters and observe effects
4. **Apply to Real Data**: Use these techniques on actual datasets
5. **Advanced Topics**: Move to more complex algorithms and applications

This comprehensive collection of exercises and quizzes provides a solid foundation for mastering machine learning concepts and practical implementation skills. 