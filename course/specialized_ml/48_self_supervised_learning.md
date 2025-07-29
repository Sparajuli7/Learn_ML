# Self-Supervised Learning

## Overview
Self-Supervised Learning enables AI systems to learn meaningful representations from unlabeled data through carefully designed pretext tasks. This guide covers modern SSL techniques, contrastive learning, and production applications for 2025.

## Table of Contents
1. [Self-Supervised Learning Fundamentals](#self-supervised-learning-fundamentals)
2. [Pretext Tasks](#pretext-tasks)
3. [Contrastive Learning](#contrastive-learning)
4. [Masked Modeling](#masked-modeling)
5. [Multi-Modal SSL](#multi-modal-ssl)
6. [Production Systems](#production-systems)
7. [Advanced Applications](#advanced-applications)

## Self-Supervised Learning Fundamentals

### Basic SSL Framework
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np

class SelfSupervisedLearner:
    def __init__(self, encoder, projector, pretext_task):
        self.encoder = encoder
        self.projector = projector
        self.pretext_task = pretext_task
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(projector.parameters()),
            lr=0.001
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and projector"""
        features = self.encoder(x)
        projections = self.projector(features)
        return projections
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SSL loss based on pretext task"""
        return self.pretext_task.compute_loss(self, batch)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned features"""
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)
        return features

class SSLEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class SSLProjector(nn.Module):
    def __init__(self, feature_dim, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)
```

### Data Augmentation Pipeline
```python
class SSLDataAugmentation:
    def __init__(self, augmentation_strength=0.5):
        self.augmentation_strength = augmentation_strength
        
    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply image augmentations"""
        # Random crop
        if np.random.random() < 0.5:
            image = self._random_crop(image)
        
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
        
        # Color jittering
        if np.random.random() < 0.5:
            image = self._color_jitter(image)
        
        # Random grayscale
        if np.random.random() < 0.2:
            image = self._to_grayscale(image)
        
        return image
    
    def _random_crop(self, image: torch.Tensor) -> torch.Tensor:
        """Random crop with resize"""
        # Simplified implementation
        return image
    
    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jittering"""
        # Simplified implementation
        return image
    
    def _to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """Convert to grayscale"""
        # Simplified implementation
        return image
    
    def create_positive_pairs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create positive pairs through augmentation"""
        view1 = self.augment_image(x)
        view2 = self.augment_image(x)
        return view1, view2
```

## Pretext Tasks

### Rotation Prediction
```python
class RotationPredictionTask:
    def __init__(self, num_rotations=4):
        self.num_rotations = num_rotations
        self.rotation_classifier = nn.Linear(128, num_rotations)
        
    def create_rotation_task(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create rotation prediction task"""
        batch_size = x.shape[0]
        
        # Generate random rotations
        rotations = torch.randint(0, self.num_rotations, (batch_size,))
        
        # Apply rotations
        rotated_images = []
        for i in range(batch_size):
            angle = rotations[i] * 90  # 0, 90, 180, 270 degrees
            rotated = self._rotate_image(x[i], angle)
            rotated_images.append(rotated)
        
        rotated_batch = torch.stack(rotated_images)
        return rotated_batch, rotations
    
    def _rotate_image(self, image: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate image by given angle"""
        # Simplified rotation implementation
        return image
    
    def compute_loss(self, ssl_learner: SelfSupervisedLearner, 
                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute rotation prediction loss"""
        rotated_images, rotation_labels = batch['rotated_images'], batch['rotation_labels']
        
        # Forward pass
        features = ssl_learner.encoder(rotated_images)
        logits = self.rotation_classifier(features)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, rotation_labels)
        return loss
```

### Jigsaw Puzzle
```python
class JigsawPuzzleTask:
    def __init__(self, num_patches=9, num_permutations=100):
        self.num_patches = num_patches
        self.num_permutations = num_permutations
        self.permutation_classifier = nn.Linear(128, num_permutations)
        
    def create_jigsaw_task(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create jigsaw puzzle task"""
        batch_size = x.shape[0]
        
        # Generate random permutations
        permutations = torch.randint(0, self.num_permutations, (batch_size,))
        
        # Create jigsaw puzzles
        jigsaw_images = []
        for i in range(batch_size):
            permuted = self._permute_patches(x[i], permutations[i])
            jigsaw_images.append(permuted)
        
        jigsaw_batch = torch.stack(jigsaw_images)
        return jigsaw_batch, permutations
    
    def _permute_patches(self, image: torch.Tensor, permutation_id: int) -> torch.Tensor:
        """Permute image patches"""
        # Simplified patch permutation
        return image
    
    def compute_loss(self, ssl_learner: SelfSupervisedLearner, 
                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute jigsaw puzzle loss"""
        jigsaw_images, permutation_labels = batch['jigsaw_images'], batch['permutation_labels']
        
        # Forward pass
        features = ssl_learner.encoder(jigsaw_images)
        logits = self.permutation_classifier(features)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, permutation_labels)
        return loss
```

### Inpainting Task
```python
class InpaintingTask:
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
        self.inpainting_head = nn.Linear(128, 784)  # For 28x28 images
        
    def create_inpainting_task(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create inpainting task"""
        batch_size = x.shape[0]
        
        # Create masks
        masks = torch.rand(batch_size, x.shape[1]) < self.mask_ratio
        
        # Apply masks
        masked_images = x.clone()
        masked_images[masks] = 0  # Mask with zeros
        
        return masked_images, x, masks
    
    def compute_loss(self, ssl_learner: SelfSupervisedLearner, 
                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute inpainting loss"""
        masked_images = batch['masked_images']
        original_images = batch['original_images']
        masks = batch['masks']
        
        # Forward pass
        features = ssl_learner.encoder(masked_images)
        reconstructions = self.inpainting_head(features)
        
        # Compute loss only on masked regions
        loss = F.mse_loss(reconstructions[masks], original_images[masks])
        return loss
```

## Contrastive Learning

### SimCLR Implementation
```python
class SimCLR(nn.Module):
    def __init__(self, encoder, projector, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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

class SimCLRTrainer:
    def __init__(self, simclr_model, learning_rate=0.001):
        self.model = simclr_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.augmentation = SSLDataAugmentation()
        
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Create positive pairs
        x1, x2 = self.augmentation.create_positive_pairs(batch)
        
        # Forward pass
        z1, z2 = self.model(x1, x2)
        
        # Compute loss
        loss = self.model.compute_loss(z1, z2)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
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
        """Forward pass for BYOL"""
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

class BYOLTrainer:
    def __init__(self, byol_model, learning_rate=0.001):
        self.model = byol_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.augmentation = SSLDataAugmentation()
        
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Create positive pairs
        x1, x2 = self.augmentation.create_positive_pairs(batch)
        
        # Forward pass
        p1, p2, z1_target, z2_target = self.model(x1, x2)
        
        # Compute loss
        loss = self.model.compute_loss(p1, p2, z1_target, z2_target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.model.update_target_network()
        
        return loss.item()
```

## Masked Modeling

### Masked Autoencoder (MAE)
```python
class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with masking"""
        batch_size, channels, height, width = x.shape
        
        # Create random mask
        mask = torch.rand(batch_size, height * width) < self.mask_ratio
        mask = mask.view(batch_size, height, width)
        
        # Apply mask
        masked_x = x.clone()
        masked_x[:, :, mask] = 0
        
        # Encode
        features = self.encoder(masked_x)
        
        # Decode
        reconstructions = self.decoder(features)
        
        return reconstructions, x, mask
    
    def compute_loss(self, reconstructions: torch.Tensor, 
                    original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss on masked regions"""
        # Compute loss only on masked regions
        loss = F.mse_loss(reconstructions[:, :, mask], original[:, :, mask])
        return loss

class MAEEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=512, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class MAEDecoder(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=512, output_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim, 7, 7)),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)
```

### Masked Language Modeling
```python
class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, 
                 num_layers=6, num_heads=8, mask_ratio=0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(512, embedding_dim)  # Max sequence length
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with masking"""
        batch_size, seq_len = input_ids.shape
        
        # Create mask
        mask = torch.rand(batch_size, seq_len) < self.mask_ratio
        mask_token_id = self.vocab_size - 1  # Assuming mask token is last in vocab
        
        # Apply mask
        masked_ids = input_ids.clone()
        masked_ids[mask] = mask_token_id
        
        # Embeddings
        token_embeddings = self.token_embedding(masked_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Transformer
        hidden_states = self.transformer(embeddings)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits, input_ids, mask
    
    def compute_loss(self, logits: torch.Tensor, 
                    original_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked language modeling loss"""
        # Reshape for loss computation
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = original_ids.view(-1)
        mask_flat = mask.view(-1)
        
        # Compute loss only on masked tokens
        loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
        return loss
```

## Multi-Modal SSL

### CLIP-Style Contrastive Learning
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

class MultiModalTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train_step(self, text_batch: Dict[str, torch.Tensor], 
                   image_batch: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        text_projections, image_projections = self.model(text_batch, image_batch)
        
        # Compute loss
        loss = self.model.compute_loss(text_projections, image_projections)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Production Systems

### SSL Training Pipeline
```python
class SSLTrainingPipeline:
    def __init__(self, ssl_model, data_loader, validation_loader=None):
        self.model = ssl_model
        self.data_loader = data_loader
        self.validation_loader = validation_loader
        self.training_history = []
        
    def train(self, num_epochs: int, save_path: str = None):
        """Train SSL model"""
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            self.model.train()
            for batch in self.data_loader:
                loss = self.model.train_step(batch)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_history.append(avg_loss)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Validation
            if self.validation_loader is not None:
                val_loss = self._validate()
                print(f"Validation Loss = {val_loss:.4f}")
            
            # Save checkpoint
            if save_path and epoch % 10 == 0:
                self._save_checkpoint(save_path, epoch)
    
    def _validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.validation_loader:
                loss = self.model.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }
        torch.save(checkpoint, f"{path}_epoch_{epoch}.pt")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        return checkpoint['epoch']
```

### SSL Feature Extractor
```python
class SSLFeatureExtractor:
    def __init__(self, ssl_model, feature_dim=128):
        self.ssl_model = ssl_model
        self.feature_dim = feature_dim
        self.feature_cache = {}
        
    def extract_features(self, data_loader, cache_key: str = None) -> torch.Tensor:
        """Extract features from SSL model"""
        features_list = []
        
        self.ssl_model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    # Handle different batch formats
                    if 'x' in batch:
                        x = batch['x']
                    elif 'images' in batch:
                        x = batch['images']
                    else:
                        raise ValueError("Unknown batch format")
                else:
                    x = batch
                
                # Extract features
                batch_features = self.ssl_model.extract_features(x)
                features_list.append(batch_features)
        
        features = torch.cat(features_list, dim=0)
        
        # Cache features if requested
        if cache_key:
            self.feature_cache[cache_key] = features
        
        return features
    
    def get_cached_features(self, cache_key: str) -> torch.Tensor:
        """Get cached features"""
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        else:
            raise KeyError(f"Features not found for key: {cache_key}")
    
    def compute_feature_similarity(self, features1: torch.Tensor, 
                                 features2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between feature sets"""
        features1_norm = F.normalize(features1, dim=1)
        features2_norm = F.normalize(features2, dim=1)
        
        similarity = torch.matmul(features1_norm, features2_norm.T)
        return similarity
```

## Advanced Applications

### SSL for Anomaly Detection
```python
class SSLAnomalyDetector:
    def __init__(self, ssl_model, threshold_percentile=95):
        self.ssl_model = ssl_model
        self.threshold_percentile = threshold_percentile
        self.normal_features = None
        self.threshold = None
        
    def fit(self, normal_data_loader):
        """Fit anomaly detector on normal data"""
        # Extract features from normal data
        normal_features = self.ssl_model.extract_features(normal_data_loader)
        self.normal_features = normal_features
        
        # Compute reconstruction errors for normal data
        reconstruction_errors = self._compute_reconstruction_errors(normal_features)
        
        # Set threshold
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies"""
        # Extract features
        test_features = self.ssl_model.extract_features(data_loader)
        
        # Compute reconstruction errors
        reconstruction_errors = self._compute_reconstruction_errors(test_features)
        
        # Predict anomalies
        predictions = reconstruction_errors > self.threshold
        
        return predictions, reconstruction_errors
    
    def _compute_reconstruction_errors(self, features: torch.Tensor) -> np.ndarray:
        """Compute reconstruction errors"""
        # Simplified implementation - compute distance to nearest neighbor
        distances = []
        
        for i in range(features.shape[0]):
            feature = features[i].unsqueeze(0)
            distances_to_normal = torch.norm(self.normal_features - feature, dim=1)
            min_distance = torch.min(distances_to_normal).item()
            distances.append(min_distance)
        
        return np.array(distances)
```

### SSL for Transfer Learning
```python
class SSLTransferLearner:
    def __init__(self, ssl_model, target_task_model):
        self.ssl_model = ssl_model
        self.target_task_model = target_task_model
        
    def fine_tune(self, target_data_loader, num_epochs: int, 
                  learning_rate: float = 0.001) -> List[float]:
        """Fine-tune SSL model on target task"""
        # Freeze SSL encoder
        for param in self.ssl_model.encoder.parameters():
            param.requires_grad = False
        
        # Setup optimizer for target task
        optimizer = torch.optim.Adam(self.target_task_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in target_data_loader:
                x, y = batch['x'], batch['y']
                
                # Extract SSL features
                with torch.no_grad():
                    ssl_features = self.ssl_model.extract_features(x)
                
                # Forward pass through target task model
                outputs = self.target_task_model(ssl_features)
                loss = criterion(outputs, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            training_losses.append(avg_loss)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return training_losses
    
    def evaluate(self, test_data_loader) -> Dict[str, float]:
        """Evaluate fine-tuned model"""
        self.target_task_model.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_data_loader:
                x, y = batch['x'], batch['y']
                
                # Extract SSL features
                ssl_features = self.ssl_model.extract_features(x)
                
                # Forward pass
                outputs = self.target_task_model(ssl_features)
                predictions = outputs.argmax(dim=1)
                
                total_correct += (predictions == y).sum().item()
                total_samples += y.shape[0]
        
        accuracy = total_correct / total_samples
        
        return {'accuracy': accuracy}
```

## Conclusion

Self-Supervised Learning has revolutionized representation learning by enabling models to learn meaningful features from unlabeled data. Key areas include:

1. **Pretext Tasks**: Rotation prediction, jigsaw puzzles, inpainting
2. **Contrastive Learning**: SimCLR, BYOL, and other contrastive methods
3. **Masked Modeling**: MAE, masked language modeling
4. **Multi-Modal SSL**: CLIP-style contrastive learning
5. **Production Applications**: Anomaly detection, transfer learning

The field continues to evolve with new techniques for more efficient and effective self-supervised learning.

## Resources

- [Self-Supervised Learning Survey](https://arxiv.org/abs/2002.05709)
- [SimCLR Paper](https://arxiv.org/abs/2002.05709)
- [BYOL Paper](https://arxiv.org/abs/2006.07733)
- [MAE Paper](https://arxiv.org/abs/2111.06377)
- [CLIP Paper](https://arxiv.org/abs/2103.00020) 