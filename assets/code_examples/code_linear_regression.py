"""
Example implementation of linear regression using numpy.
Demonstrates proper code organization and documentation.
"""

import numpy as np
from typing import Tuple

class LinearRegression:
    """Simple linear regression implementation."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """Initialize the model parameters.
        
        Args:
            learning_rate: Step size for gradient descent
            iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using gradient descent.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            y_pred = self._predict(X)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for given features.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias