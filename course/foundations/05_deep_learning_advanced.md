# Deep Learning Advanced: Modern Architectures & 2025 Techniques

*"From ResNet to Transformers: Mastering cutting-edge deep learning architectures"*

---

## 📚 Table of Contents

1. [Modern Neural Architectures](#modern-neural-architectures)
2. [Transfer Learning](#transfer-learning)
3. [Model Compression & Efficiency](#model-compression--efficiency)
4. [Advanced Training Techniques](#advanced-training-techniques)
5. [2025 Efficiency Trends](#2025-efficiency-trends)
6. [Implementation Examples](#implementation-examples)
7. [Exercises and Projects](#exercises-and-projects)
8. [Further Reading](#further-reading)

---

## 🏗️ Modern Neural Architectures

### 1. Residual Networks (ResNet)

**Problem**: Vanishing gradients in deep networks
**Solution**: Skip connections that allow gradients to flow directly

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # Skip connection
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
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

# Create ResNet-18
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])
```

### 2. Attention Mechanisms

**Core Idea**: Allow the model to focus on relevant parts of the input

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute attention scores"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights
```

### 3. Transformer Architecture

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def create_positional_encoding(self, max_seq_len, d_model):
        """Create positional encoding matrix"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len].to(x.device)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Final linear layer
        output = self.fc(x)
        
        return output
```

---

## 🔄 Transfer Learning

### Pre-trained Models

```python
import torchvision.models as models
from torchvision import transforms

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained ResNet
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Create model
model = TransferLearningModel(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=0.001)
```

### Fine-tuning Strategies

```python
class FineTuningScheduler:
    def __init__(self, model, learning_rates):
        self.model = model
        self.learning_rates = learning_rates
        self.current_epoch = 0
    
    def step(self):
        """Update learning rates based on epoch"""
        if self.current_epoch < len(self.learning_rates):
            lr = self.learning_rates[self.current_epoch]
            
            # Unfreeze more layers progressively
            if self.current_epoch == 5:
                for param in list(self.model.backbone.parameters())[-30:]:
                    param.requires_grad = True
            
            if self.current_epoch == 10:
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
            
            # Update optimizer learning rate
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.current_epoch += 1

# Progressive fine-tuning
learning_rates = [0.001] * 5 + [0.0001] * 5 + [0.00001] * 5
scheduler = FineTuningScheduler(model, learning_rates)
```

---

## 🗜️ Model Compression & Efficiency

### 1. Quantization

```python
class QuantizedModel(nn.Module):
    def __init__(self, original_model):
        super(QuantizedModel, self).__init__()
        self.original_model = original_model
    
    def quantize_model(self):
        """Quantize model to int8"""
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.original_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        return self.quantized_model
    
    def forward(self, x):
        return self.quantized_model(x)

# Usage
original_model = ResNet18()
quantized_model = QuantizedModel(original_model)
quantized_model.quantize_model()

# Compare model sizes
original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
print(f"Original size: {original_size / 1e6:.2f} MB")
print(f"Quantized size: {quantized_size / 1e6:.2f} MB")
```

### 2. Pruning

```python
class PrunedModel(nn.Module):
    def __init__(self, original_model, pruning_rate=0.3):
        super(PrunedModel, self).__init__()
        self.original_model = original_model
        self.pruning_rate = pruning_rate
    
    def prune_model(self):
        """Prune model weights"""
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                # Calculate threshold for pruning
                weights = module.weight.data
                threshold = torch.quantile(torch.abs(weights), self.pruning_rate)
                
                # Create mask
                mask = torch.abs(weights) > threshold
                module.weight.data = weights * mask
    
    def forward(self, x):
        return self.original_model(x)

# Usage
model = ResNet18()
pruned_model = PrunedModel(model, pruning_rate=0.5)
pruned_model.prune_model()
```

### 3. Knowledge Distillation

```python
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_output, teacher_output, labels):
        # Hard target loss
        hard_loss = self.ce_loss(student_output, labels)
        
        # Soft target loss
        soft_loss = self.kl_loss(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss

# Training with knowledge distillation
teacher_model = ResNet50(pretrained=True)
student_model = ResNet18()

distillation_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        teacher_output = teacher_model(data)
        student_output = student_model(data)
        
        # Compute loss
        loss = distillation_loss(student_output, teacher_output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🚀 Advanced Training Techniques

### 1. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, data, target):
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            output = self.model(data)
            loss = F.cross_entropy(output, target)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# Usage
trainer = MixedPrecisionTrainer(model, optimizer)
for batch_idx, (data, target) in enumerate(train_loader):
    loss = trainer.train_step(data, target)
```

### 2. Gradient Accumulation

```python
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()
    
    def train_step(self, data, target, step):
        # Forward pass
        with autocast():
            output = self.model(data)
            loss = F.cross_entropy(output, target) / self.accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps

# Usage
trainer = GradientAccumulationTrainer(model, optimizer, accumulation_steps=4)
for step, (data, target) in enumerate(train_loader):
    loss = trainer.train_step(data, target, step)
```

### 3. Learning Rate Scheduling

```python
class CosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        self.T_cur += 1
        
        if self.T_cur >= self.T_0:
            self.T_cur = 0
            self.T_0 *= self.T_mult
        
        # Cosine annealing
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.T_cur / self.T_0)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Usage
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        pass
    scheduler.step()
```

---

## ⚡ 2025 Efficiency Trends

### 1. Open-Weight Models

```python
class EfficientTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size):
        super(EfficientTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Efficient attention with linear complexity
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Feed-forward with gated linear units
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # Efficient self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

### 2. Model Parallelism

```python
class ModelParallelTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size):
        super(ModelParallelTransformer, self).__init__()
        
        # Distribute layers across devices
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerBlock(d_model, num_heads, d_model * 4)
            # Place every other layer on different device
            device = torch.device(f'cuda:{i % 2}')
            layer = layer.to(device)
            self.layers.append(layer)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Move input to layer's device
            device = next(layer.parameters()).device
            x = x.to(device)
            x = layer(x)
        return x
```

### 3. Dynamic Batching

```python
class DynamicBatchingDataset:
    def __init__(self, dataset, max_tokens=4096):
        self.dataset = dataset
        self.max_tokens = max_tokens
    
    def collate_fn(self, batch):
        # Sort by sequence length
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        
        # Create batches with similar lengths
        batches = []
        current_batch = []
        current_tokens = 0
        
        for item in batch:
            seq_len = len(item[0])
            if current_tokens + seq_len > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_tokens = seq_len
            else:
                current_batch.append(item)
                current_tokens += seq_len
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

# Usage
dataset = DynamicBatchingDataset(text_dataset)
dataloader = DataLoader(dataset, batch_sampler=dataset.collate_fn)
```

---

## 🧪 Exercises and Projects

### Exercise 1: Build a Vision Transformer

```python
# TODO: Implement Vision Transformer (ViT)
# Features: Patch embedding, positional encoding, transformer blocks

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, 
                 d_model=768, num_heads=12, num_layers=12):
        super(VisionTransformer, self).__init__()
        # Your implementation here
        pass
    
    def forward(self, x):
        # Your implementation here
        pass
```

### Exercise 2: Efficient Model Training

```python
# TODO: Implement efficient training pipeline
# Features: Mixed precision, gradient accumulation, dynamic batching

def efficient_training_pipeline():
    """
    Build an efficient training pipeline
    
    Steps:
    1. Implement mixed precision training
    2. Add gradient accumulation
    3. Use dynamic batching
    4. Implement model parallelism
    5. Add distributed training
    """
    pass
```

### Exercise 3: Model Compression Pipeline

```python
# TODO: Build comprehensive model compression
# Features: Quantization, pruning, knowledge distillation

def model_compression_pipeline():
    """
    Build a complete model compression pipeline
    
    Steps:
    1. Train teacher model
    2. Implement knowledge distillation
    3. Apply quantization
    4. Perform pruning
    5. Evaluate compressed model
    """
    pass
```

---

## 📖 Further Reading

### Essential Papers
- "Deep Residual Learning for Image Recognition" by He et al. (2016)
- "Attention Is All You Need" by Vaswani et al. (2017)
- "An Image is Worth 16x16 Words" by Dosovitskiy et al. (2021)

### Books
- "Transformers for Natural Language Processing" by Denis Rothman
- "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, Thomas Viehmann

### Online Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained models
- [Papers With Code](https://paperswithcode.com/) - Latest research implementations
- [PyTorch Lightning](https://lightning.ai/) - Advanced training framework

### Next Steps
- **[NLP Advanced](core_ml_fields/07_nlp_advanced.md)**: Modern language models
- **[Computer Vision Advanced](core_ml_fields/09_computer_vision_advanced.md)**: Advanced CV techniques
- **[Agentic AI Advanced](specialized_ml/17_agentic_ai_advanced.md)**: Autonomous AI systems

---

## 🎯 Key Takeaways

1. **Modern Architectures**: ResNet, Transformers, Vision Transformers
2. **Transfer Learning**: Leverage pre-trained models for new tasks
3. **Model Efficiency**: Quantization, pruning, knowledge distillation
4. **Advanced Training**: Mixed precision, gradient accumulation, scheduling
5. **2025 Trends**: Open-weight models, model parallelism, dynamic batching
6. **Best Practices**: Progressive fine-tuning, efficient attention, distributed training

---

*"The future of deep learning lies not just in bigger models, but in smarter, more efficient architectures that can do more with less."*

**Next: [NLP Fundamentals](core_ml_fields/06_nlp_fundamentals.md) → Natural language processing foundations** 