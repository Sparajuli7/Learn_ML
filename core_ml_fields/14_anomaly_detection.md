# Anomaly Detection

## Overview
Anomaly detection identifies unusual patterns in data that may indicate errors, fraud, or other significant events. This guide covers statistical methods, machine learning approaches, and deep learning techniques for 2025.

## Table of Contents
1. [Anomaly Detection Fundamentals](#anomaly-detection-fundamentals)
2. [Statistical Methods](#statistical-methods)
3. [Machine Learning Approaches](#machine-learning-approaches)
4. [Deep Learning Methods](#deep-learning-methods)
5. [Time Series Anomaly Detection](#time-series-anomaly-detection)
6. [Production Systems](#production-systems)

## Anomaly Detection Fundamentals

### Basic Setup
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyDetector:
    def __init__(self, method: str = 'isolation_forest'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray):
        """Fit the anomaly detector"""
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        if self.method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif self.method == 'lof':
            self.model = LocalOutlierFactor(contamination=0.1, novelty=True)
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(kernel='rbf', nu=0.1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model.fit(data_scaled)
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        
        if self.method == 'lof':
            predictions = self.model.predict(data_scaled)
        else:
            predictions = self.model.predict(data_scaled)
        
        # Convert to binary (1 for normal, -1 for anomaly)
        return predictions
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        
        if hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(data_scaled)
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(data_scaled)
        else:
            raise ValueError("Model does not support scoring")
        
        return scores
```

## Statistical Methods

### Z-Score Method
```python
class ZScoreAnomalyDetector:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray):
        """Fit the Z-score detector"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using Z-score"""
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted yet")
        
        z_scores = np.abs((data - self.mean) / self.std)
        anomalies = np.any(z_scores > self.threshold, axis=1)
        
        return np.where(anomalies, -1, 1)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get Z-scores"""
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted yet")
        
        z_scores = np.abs((data - self.mean) / self.std)
        return np.max(z_scores, axis=1)

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.normal(5, 1, (50, 2))
data = np.vstack([normal_data, anomaly_data])

z_detector = ZScoreAnomalyDetector(threshold=3.0)
z_detector.fit(normal_data)
predictions = z_detector.predict(data)
scores = z_detector.score_samples(data)

print(f"Detected {np.sum(predictions == -1)} anomalies")
```

### IQR Method
```python
class IQRAnomalyDetector:
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.q1 = None
        self.q3 = None
        self.iqr = None
        
    def fit(self, data: np.ndarray):
        """Fit the IQR detector"""
        self.q1 = np.percentile(data, 25, axis=0)
        self.q3 = np.percentile(data, 75, axis=0)
        self.iqr = self.q3 - self.q1
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using IQR"""
        if self.q1 is None or self.q3 is None or self.iqr is None:
            raise ValueError("Model not fitted yet")
        
        lower_bound = self.q1 - self.factor * self.iqr
        upper_bound = self.q3 + self.factor * self.iqr
        
        anomalies = np.any((data < lower_bound) | (data > upper_bound), axis=1)
        
        return np.where(anomalies, -1, 1)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get IQR-based scores"""
        if self.q1 is None or self.q3 is None or self.iqr is None:
            raise ValueError("Model not fitted yet")
        
        lower_bound = self.q1 - self.factor * self.iqr
        upper_bound = self.q3 + self.factor * self.iqr
        
        # Calculate distance from bounds
        distances = np.maximum(lower_bound - data, data - upper_bound)
        scores = np.max(distances, axis=1)
        
        return scores

# Example usage
iqr_detector = IQRAnomalyDetector(factor=1.5)
iqr_detector.fit(normal_data)
predictions = iqr_detector.predict(data)
scores = iqr_detector.score_samples(data)

print(f"Detected {np.sum(predictions == -1)} anomalies")
```

## Machine Learning Approaches

### Isolation Forest
```python
class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        
    def fit(self, data: np.ndarray):
        """Fit isolation forest"""
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(data)
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(data)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.score_samples(data)

# Example usage
if_detector = IsolationForestDetector(contamination=0.1)
if_detector.fit(normal_data)
predictions = if_detector.predict(data)
scores = if_detector.score_samples(data)

print(f"Detected {np.sum(predictions == -1)} anomalies")
```

### Local Outlier Factor
```python
class LOFDetector:
    def __init__(self, contamination: float = 0.1, n_neighbors: int = 20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.model = None
        
    def fit(self, data: np.ndarray):
        """Fit LOF detector"""
        self.model = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            novelty=True
        )
        self.model.fit(data)
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(data)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get LOF scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.score_samples(data)

# Example usage
lof_detector = LOFDetector(contamination=0.1)
lof_detector.fit(normal_data)
predictions = lof_detector.predict(data)
scores = lof_detector.score_samples(data)

print(f"Detected {np.sum(predictions == -1)} anomalies")
```

## Deep Learning Methods

### Autoencoder for Anomaly Detection
```python
class AutoencoderAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input"""
        return self.encoder(x)
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode encoded representation"""
        return self.decoder(encoded)

class AutoencoderDetector:
    def __init__(self, encoding_dim: int = 32, threshold_percentile: float = 95):
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray, n_epochs: int = 100):
        """Fit autoencoder"""
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        # Create model
        self.model = AutoencoderAnomalyDetector(data.shape[1], self.encoding_dim)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.model(data_tensor)
            loss = criterion(reconstructed, data_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Calculate reconstruction error threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed)**2, dim=1)
            self.threshold = np.percentile(reconstruction_errors.numpy(), self.threshold_percentile)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.model is None or self.threshold is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed)**2, dim=1)
            
            anomalies = reconstruction_errors.numpy() > self.threshold
            return np.where(anomalies, -1, 1)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get reconstruction error scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed)**2, dim=1)
            
            return reconstruction_errors.numpy()

# Example usage
ae_detector = AutoencoderDetector(encoding_dim=16)
ae_detector.fit(normal_data, n_epochs=50)
predictions = ae_detector.predict(data)
scores = ae_detector.score_samples(data)

print(f"Detected {np.sum(predictions == -1)} anomalies")
```

### Variational Autoencoder for Anomaly Detection
```python
class VAEAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(64, encoding_dim)
        self.fc_logvar = nn.Linear(64, encoding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

class VAEDetector:
    def __init__(self, encoding_dim: int = 32, threshold_percentile: float = 95):
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray, n_epochs: int = 100):
        """Fit VAE"""
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        # Create model
        self.model = VAEAnomalyDetector(data.shape[1], self.encoding_dim)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = self.model(data_tensor)
            
            # Loss: reconstruction + KL divergence
            recon_loss = F.mse_loss(reconstructed, data_tensor, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Calculate reconstruction error threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed)**2, dim=1)
            self.threshold = np.percentile(reconstruction_errors.numpy(), self.threshold_percentile)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.model is None or self.threshold is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed)**2, dim=1)
            
            anomalies = reconstruction_errors.numpy() > self.threshold
            return np.where(anomalies, -1, 1)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get reconstruction error scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed)**2, dim=1)
            
            return reconstruction_errors.numpy()

# Example usage
vae_detector = VAEDetector(encoding_dim=16)
vae_detector.fit(normal_data, n_epochs=50)
predictions = vae_detector.predict(data)
scores = vae_detector.score_samples(data)

print(f"Detected {np.sum(predictions == -1)} anomalies")
```

## Time Series Anomaly Detection

### LSTM Autoencoder for Time Series
```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Linear layers
        self.encoder_linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_linear = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Encode
        encoded, (hidden, cell) = self.encoder_lstm(x)
        encoded = self.encoder_linear(encoded)
        
        # Decode
        decoded, _ = self.decoder_lstm(encoded, (hidden, cell))
        decoded = self.decoder_linear(decoded)
        
        return decoded

class TimeSeriesAnomalyDetector:
    def __init__(self, sequence_length: int = 10, threshold_percentile: float = 95):
        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, data: np.ndarray, n_epochs: int = 100):
        """Fit LSTM autoencoder"""
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        sequences = self._create_sequences(data_scaled)
        sequences_tensor = torch.FloatTensor(sequences)
        
        # Create model
        self.model = LSTMAutoencoder(data.shape[1], hidden_dim=32)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.model(sequences_tensor)
            loss = criterion(reconstructed, sequences_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Calculate reconstruction error threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(sequences_tensor)
            reconstruction_errors = torch.mean((sequences_tensor - reconstructed)**2, dim=(1, 2))
            self.threshold = np.percentile(reconstruction_errors.numpy(), self.threshold_percentile)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.model is None or self.threshold is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        sequences = self._create_sequences(data_scaled)
        sequences_tensor = torch.FloatTensor(sequences)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(sequences_tensor)
            reconstruction_errors = torch.mean((sequences_tensor - reconstructed)**2, dim=(1, 2))
            
            # Map back to original data points
            predictions = np.ones(len(data))
            for i, error in enumerate(reconstruction_errors.numpy()):
                if error > self.threshold:
                    # Mark all points in sequence as anomaly
                    for j in range(self.sequence_length):
                        if i + j < len(predictions):
                            predictions[i + j] = -1
            
            return predictions
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Get reconstruction error scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        data_scaled = self.scaler.transform(data)
        sequences = self._create_sequences(data_scaled)
        sequences_tensor = torch.FloatTensor(sequences)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(sequences_tensor)
            reconstruction_errors = torch.mean((sequences_tensor - reconstructed)**2, dim=(1, 2))
            
            # Map back to original data points
            scores = np.zeros(len(data))
            for i, error in enumerate(reconstruction_errors.numpy()):
                for j in range(self.sequence_length):
                    if i + j < len(scores):
                        scores[i + j] = max(scores[i + j], error)
            
            return scores

# Example usage
# Generate time series data
np.random.seed(42)
time_steps = 1000
normal_ts = np.cumsum(np.random.randn(time_steps)) + np.sin(np.linspace(0, 10*np.pi, time_steps))
anomaly_ts = normal_ts.copy()
anomaly_ts[500:520] += 10  # Add anomaly

ts_detector = TimeSeriesAnomalyDetector(sequence_length=20)
ts_detector.fit(normal_ts.reshape(-1, 1), n_epochs=50)
predictions = ts_detector.predict(anomaly_ts.reshape(-1, 1))
scores = ts_detector.score_samples(anomaly_ts.reshape(-1, 1))

print(f"Detected {np.sum(predictions == -1)} anomalies in time series")
```

## Production Systems

### Ensemble Anomaly Detector
```python
class EnsembleAnomalyDetector:
    def __init__(self, detectors: List[Any], weights: List[float] = None):
        self.detectors = detectors
        self.weights = weights if weights else [1.0] * len(detectors)
        
        if len(self.detectors) != len(self.weights):
            raise ValueError("Number of detectors must match number of weights")
    
    def fit(self, data: np.ndarray):
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(data)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Ensemble prediction"""
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(data)
            predictions.append(pred)
        
        # Weighted voting
        ensemble_pred = np.zeros(len(data))
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * (pred == -1).astype(float)
        
        # Threshold for final prediction
        threshold = sum(self.weights) / 2
        final_pred = np.where(ensemble_pred > threshold, -1, 1)
        
        return final_pred
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Ensemble scores"""
        scores = []
        for detector in self.detectors:
            if hasattr(detector, 'score_samples'):
                score = detector.score_samples(data)
                scores.append(score)
        
        if scores:
            # Average scores
            return np.mean(scores, axis=0)
        else:
            return np.zeros(len(data))

# Example usage
detectors = [
    IsolationForest(contamination=0.1),
    LocalOutlierFactor(contamination=0.1, novelty=True),
    OneClassSVM(kernel='rbf', nu=0.1)
]

ensemble = EnsembleAnomalyDetector(detectors, weights=[0.4, 0.3, 0.3])
ensemble.fit(normal_data)
predictions = ensemble.predict(data)
scores = ensemble.score_samples(data)

print(f"Ensemble detected {np.sum(predictions == -1)} anomalies")
```

## Conclusion

Anomaly detection provides powerful methods for identifying unusual patterns in data. Key areas include:

1. **Statistical Methods**: Z-score, IQR, and other statistical approaches
2. **Machine Learning**: Isolation Forest, LOF, and One-Class SVM
3. **Deep Learning**: Autoencoders, VAEs, and LSTM-based methods
4. **Time Series**: Specialized methods for temporal data
5. **Production Systems**: Ensemble methods and scalable deployments

The field continues to evolve with new methods for more accurate and interpretable anomaly detection.

## Resources

- [Anomaly Detection Survey](https://arxiv.org/abs/1901.03407)
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Local Outlier Factor](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf)
- [Deep Learning for Anomaly Detection](https://arxiv.org/abs/2001.04990)