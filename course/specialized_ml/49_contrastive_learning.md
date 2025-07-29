# Contrastive Learning

## Overview
Contrastive Learning enables AI systems to learn meaningful representations by pulling similar samples closer and pushing dissimilar samples apart. This guide covers modern contrastive techniques, loss functions, and production applications for 2025.

## Table of Contents
1. [Contrastive Learning Fundamentals](#contrastive-learning-fundamentals)
2. [Loss Functions](#loss-functions)
3. [Modern Contrastive Methods](#modern-contrastive-methods)
4. [Multi-Modal Contrastive Learning](#multi-modal-contrastive-learning)
5. [Production Systems](#production-systems)

## Contrastive Learning Fundamentals

### Basic Contrastive Framework
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np

class ContrastiveLearner:
    def __init__(self, encoder, projector, temperature=0.07):
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning"""
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to representation space
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def compute_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss"""
        batch_size = z1.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ContrastiveProjector(nn.Module):
    def __init__(self, feature_dim, projection_dim=128):
        super().__init__():
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)
```

## Loss Functions

### NT-Xent Loss
```python
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss implementation"""
        batch_size = z1.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
```

### InfoNCE Loss
```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query: torch.Tensor, positive: torch.Tensor, 
                negatives: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss implementation"""
        # Concatenate positive and negatives
        key = torch.cat([positive.unsqueeze(1), negatives], dim=1)
        
        # Compute similarities
        logits = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        
        # Labels are 0 (positive is first)
        labels = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss
```

## Modern Contrastive Methods

### SimCLR Implementation
```python
class SimCLR(nn.Module):
    def __init__(self, encoder, projector, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SimCLR forward pass"""
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to representation space
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def compute_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute SimCLR loss"""
        batch_size = z1.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
```

### BYOL Implementation
```python
class BYOL(nn.Module):
    def __init__(self, encoder, projector, predictor, target_encoder, 
                 target_projector, momentum=0.996):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.target_projector = target_projector
        self.momentum = momentum
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BYOL forward pass"""
        # Online network
        h1 = self.encoder(x1)
        z1 = self.projector(h1)
        p1 = self.predictor(z1)
        
        h2 = self.encoder(x2)
        z2 = self.projector(h2)
        p2 = self.predictor(z2)
        
        # Target network
        with torch.no_grad():
            h1_target = self.target_encoder(x1)
            z1_target = self.target_projector(h1_target)
            
            h2_target = self.target_encoder(x2)
            z2_target = self.target_projector(h2_target)
        
        return p1, p2, z1_target, z2_target
    
    def compute_loss(self, p1: torch.Tensor, p2: torch.Tensor, 
                    z1_target: torch.Tensor, z2_target: torch.Tensor) -> torch.Tensor:
        """Compute BYOL loss"""
        # Normalize target representations
        z1_target = F.normalize(z1_target, dim=1)
        z2_target = F.normalize(z2_target, dim=1)
        
        # Compute loss
        loss1 = F.mse_loss(p1, z2_target)
        loss2 = F.mse_loss(p2, z1_target)
        
        return (loss1 + loss2) / 2
    
    def update_target_network(self):
        """Update target network with momentum"""
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * param.data
        
        for target_param, param in zip(self.target_projector.parameters(), self.projector.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * param.data
```

## Multi-Modal Contrastive Learning

### CLIP-Style Implementation
```python
class MultiModalContrastiveLearner(nn.Module):
    def __init__(self, text_encoder, image_encoder, projection_dim=512, temperature=0.07):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.text_projection = nn.Linear(512, projection_dim)
        self.image_projection = nn.Linear(512, projection_dim)
        self.temperature = temperature
        
    def forward(self, text_inputs: Dict[str, torch.Tensor], 
                image_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multi-modal contrastive learning"""
        # Encode text
        text_features = self.text_encoder(**text_inputs)
        text_projections = self.text_projection(text_features)
        text_projections = F.normalize(text_projections, dim=1)
        
        # Encode images
        image_features = self.image_encoder(image_inputs)
        image_projections = self.image_projection(image_features)
        image_projections = F.normalize(image_projections, dim=1)
        
        return text_projections, image_projections
    
    def compute_loss(self, text_projections: torch.Tensor, 
                    image_projections: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        batch_size = text_projections.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_projections, image_projections.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=text_projections.device)
        
        # Compute loss in both directions
        text_to_image_loss = F.cross_entropy(similarity_matrix, labels)
        image_to_text_loss = F.cross_entropy(similarity_matrix.T, labels)
        
        return (text_to_image_loss + image_to_text_loss) / 2
```

## Production Systems

### Contrastive Learning Pipeline
```python
class ContrastiveLearningPipeline:
    def __init__(self, model, data_loader, validation_loader=None):
        self.model = model
        self.data_loader = data_loader
        self.validation_loader = validation_loader
        self.training_history = []
        
    def train(self, num_epochs: int, save_path: str = None):
        """Train contrastive learning model"""
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            self.model.train()
            for batch in self.data_loader:
                # Create positive pairs
                x1, x2 = self._create_positive_pairs(batch)
                
                # Forward pass
                z1, z2 = self.model(x1, x2)
                
                # Compute loss
                loss = self.model.compute_loss(z1, z2)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_history.append(avg_loss)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def _create_positive_pairs(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create positive pairs through augmentation"""
        # Simplified implementation
        return batch, batch
```

## Conclusion

Contrastive Learning has become a cornerstone of modern representation learning. Key areas include:

1. **Loss Functions**: NT-Xent, InfoNCE, and other contrastive losses
2. **Modern Methods**: SimCLR, BYOL, and other advanced techniques
3. **Multi-Modal Learning**: CLIP-style contrastive learning
4. **Production Applications**: Feature learning, transfer learning

The field continues to evolve with new techniques for more efficient and effective contrastive learning.

## Resources

- [SimCLR Paper](https://arxiv.org/abs/2002.05709)
- [BYOL Paper](https://arxiv.org/abs/2006.07733)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Contrastive Learning Survey](https://arxiv.org/abs/2002.05709) 