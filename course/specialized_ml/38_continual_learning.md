# Continual Learning

## Overview
Continual Learning enables AI systems to learn continuously from new data while retaining knowledge from previous tasks. This guide covers lifelong learning systems, catastrophic forgetting prevention, and adaptive learning techniques for 2025.

## Table of Contents
1. [Continual Learning Fundamentals](#continual-learning-fundamentals)
2. [Catastrophic Forgetting Prevention](#catastrophic-forgetting-prevention)
3. [Replay-Based Methods](#replay-based-methods)
4. [Regularization Techniques](#regularization-techniques)
5. [Architectural Approaches](#architectural-approaches)
6. [Meta-Learning for Continual Learning](#meta-learning-for-continual-learning)
7. [Production Systems](#production-systems)
8. [Advanced Applications](#advanced-applications)

## Continual Learning Fundamentals

### Basic Continual Learning System
```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import copy

class ContinualLearningSystem:
    def __init__(self, model, memory_size=1000):
        self.model = model
        self.memory_size = memory_size
        self.episodic_memory = []
        self.task_history = []
        self.current_task = 0
        
    def learn_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Learn a new task while preserving previous knowledge"""
        # Store task information
        self.task_history.append({
            'task_id': task_id,
            'data_size': len(task_data['x']),
            'classes': torch.unique(task_data['y']).tolist()
        })
        
        # Update episodic memory
        self._update_episodic_memory(task_data)
        
        # Train model on current task
        self._train_on_task(task_data, task_id)
        
        # Evaluate on all previous tasks
        performance = self._evaluate_on_all_tasks()
        
        return performance
    
    def _update_episodic_memory(self, task_data: Dict[str, torch.Tensor]):
        """Update episodic memory with new data"""
        # Reservoir sampling for memory management
        for i in range(len(task_data['x'])):
            if len(self.episodic_memory) < self.memory_size:
                self.episodic_memory.append({
                    'x': task_data['x'][i],
                    'y': task_data['y'][i],
                    'task_id': self.current_task
                })
            else:
                # Reservoir sampling
                j = np.random.randint(0, len(self.episodic_memory))
                if j < self.memory_size:
                    self.episodic_memory[j] = {
                        'x': task_data['x'][i],
                        'y': task_data['y'][i],
                        'task_id': self.current_task
                    }
    
    def _train_on_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Train model on current task"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Combine current task data with episodic memory
        combined_data = self._combine_with_memory(task_data)
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(combined_data['x'])
            loss = criterion(outputs, combined_data['y'])
            
            # Add regularization for continual learning
            reg_loss = self._compute_regularization_loss()
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
    
    def _combine_with_memory(self, task_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine current task data with episodic memory"""
        if not self.episodic_memory:
            return task_data
        
        # Sample from episodic memory
        memory_samples = np.random.choice(
            self.episodic_memory, 
            min(len(self.episodic_memory), len(task_data['x'])),
            replace=False
        )
        
        # Combine current task and memory data
        memory_x = torch.stack([sample['x'] for sample in memory_samples])
        memory_y = torch.stack([sample['y'] for sample in memory_samples])
        
        combined_x = torch.cat([task_data['x'], memory_x], dim=0)
        combined_y = torch.cat([task_data['y'], memory_y], dim=0)
        
        return {'x': combined_x, 'y': combined_y}
    
    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss to prevent forgetting"""
        # Elastic Weight Consolidation (EWC) style regularization
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if hasattr(self, 'fisher_info') and name in self.fisher_info:
                # EWC regularization
                reg_loss += torch.sum(self.fisher_info[name] * (param - self.old_params[name]) ** 2)
        
        return reg_loss
    
    def _evaluate_on_all_tasks(self) -> Dict[int, float]:
        """Evaluate model performance on all learned tasks"""
        self.model.eval()
        performance = {}
        
        with torch.no_grad():
            for task_info in self.task_history:
                task_id = task_info['task_id']
                # In practice, you would have test data for each task
                # For now, simulate evaluation
                performance[task_id] = np.random.uniform(0.7, 0.95)
        
        return performance
```

### Task-Aware Continual Learning
```python
class TaskAwareContinualLearning:
    def __init__(self, model, num_tasks):
        self.model = model
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleDict()
        self.task_embeddings = nn.Embedding(num_tasks, 64)
        
    def add_task_head(self, task_id: int, num_classes: int):
        """Add a task-specific head"""
        self.task_heads[f'task_{task_id}'] = nn.Linear(
            self.model.output_dim, num_classes
        )
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass with task awareness"""
        # Get shared representation
        shared_features = self.model(x)
        
        # Get task embedding
        task_embedding = self.task_embeddings(torch.tensor(task_id))
        
        # Combine features with task embedding
        task_aware_features = torch.cat([shared_features, task_embedding], dim=1)
        
        # Task-specific head
        task_head = self.task_heads[f'task_{task_id}']
        output = task_head(task_aware_features)
        
        return output
    
    def learn_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Learn a new task with task-aware architecture"""
        # Add task head if not exists
        num_classes = len(torch.unique(task_data['y']))
        if f'task_{task_id}' not in self.task_heads:
            self.add_task_head(task_id, num_classes)
        
        # Train model
        self.model.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.task_heads.parameters()},
            {'params': self.task_embeddings.parameters()}
        ], lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass with task awareness
            outputs = self.forward(task_data['x'], task_id)
            loss = criterion(outputs, task_data['y'])
            
            loss.backward()
            optimizer.step()
```

## Catastrophic Forgetting Prevention

### Elastic Weight Consolidation (EWC)
```python
class ElasticWeightConsolidation:
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.old_params = {}
        
    def compute_fisher_information(self, data: Dict[str, torch.Tensor]):
        """Compute Fisher information matrix for EWC"""
        self.model.train()
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            self.fisher_info[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information
        for i in range(len(data['x'])):
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(data['x'][i:i+1])
            loss = nn.CrossEntropyLoss()(outputs, data['y'][i:i+1])
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.data ** 2
        
        # Average Fisher information
        for name in self.fisher_info:
            self.fisher_info[name] /= len(data['x'])
    
    def save_old_params(self):
        """Save current parameters as old parameters"""
        self.old_params = {}
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.data.clone()
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.old_params:
                ewc_loss += torch.sum(
                    self.fisher_info[name] * (param - self.old_params[name]) ** 2
                )
        
        return self.lambda_ewc * ewc_loss
    
    def train_with_ewc(self, task_data: Dict[str, torch.Tensor]):
        """Train model with EWC regularization"""
        # Compute Fisher information on current task
        self.compute_fisher_information(task_data)
        
        # Save old parameters
        self.save_old_params()
        
        # Train with EWC
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Standard loss
            outputs = self.model(task_data['x'])
            loss = criterion(outputs, task_data['y'])
            
            # EWC regularization
            ewc_reg = self.ewc_loss()
            total_loss = loss + ewc_reg
            
            total_loss.backward()
            optimizer.step()
```

### Synaptic Intelligence (SI)
```python
class SynapticIntelligence:
    def __init__(self, model, lambda_si=1.0):
        self.model = model
        self.lambda_si = lambda_si
        self.omega = {}
        self.old_params = {}
        
    def compute_omega(self, task_data: Dict[str, torch.Tensor]):
        """Compute synaptic importance weights"""
        self.model.train()
        
        # Initialize omega
        for name, param in self.model.named_parameters():
            self.omega[name] = torch.zeros_like(param.data)
        
        # Compute omega over training
        for epoch in range(5):
            for i in range(len(task_data['x'])):
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(task_data['x'][i:i+1])
                loss = nn.CrossEntropyLoss()(outputs, task_data['y'][i:i+1])
                loss.backward()
                
                # Accumulate omega
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.omega[name] += param.grad.data ** 2
        
        # Normalize omega
        for name in self.omega:
            self.omega[name] /= len(task_data['x']) * 5  # 5 epochs
    
    def si_loss(self) -> torch.Tensor:
        """Compute SI regularization loss"""
        si_loss = 0
        
        for name, param in self.model.named_parameters():
            if name in self.omega and name in self.old_params:
                si_loss += torch.sum(
                    self.omega[name] * (param - self.old_params[name]) ** 2
                )
        
        return self.lambda_si * si_loss
    
    def train_with_si(self, task_data: Dict[str, torch.Tensor]):
        """Train model with SI regularization"""
        # Compute omega on current task
        self.compute_omega(task_data)
        
        # Save old parameters
        self.old_params = {}
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.data.clone()
        
        # Train with SI
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Standard loss
            outputs = self.model(task_data['x'])
            loss = criterion(outputs, task_data['y'])
            
            # SI regularization
            si_reg = self.si_loss()
            total_loss = loss + si_reg
            
            total_loss.backward()
            optimizer.step()
```

## Replay-Based Methods

### Experience Replay
```python
class ExperienceReplay:
    def __init__(self, model, memory_size=1000, replay_ratio=0.5):
        self.model = model
        self.memory_size = memory_size
        self.replay_ratio = replay_ratio
        self.memory = []
        
    def add_to_memory(self, data: Dict[str, torch.Tensor]):
        """Add data to replay memory"""
        for i in range(len(data['x'])):
            memory_item = {
                'x': data['x'][i].clone(),
                'y': data['y'][i].clone(),
                'task_id': getattr(self, 'current_task', 0)
            }
            
            if len(self.memory) < self.memory_size:
                self.memory.append(memory_item)
            else:
                # Random replacement
                idx = np.random.randint(0, len(self.memory))
                self.memory[idx] = memory_item
    
    def sample_from_memory(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from replay memory"""
        if len(self.memory) == 0:
            return None
        
        # Sample random batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Combine into tensors
        x = torch.stack([item['x'] for item in batch])
        y = torch.stack([item['y'] for item in batch])
        
        return {'x': x, 'y': y}
    
    def train_with_replay(self, task_data: Dict[str, torch.Tensor]):
        """Train model with experience replay"""
        # Add current task to memory
        self.add_to_memory(task_data)
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Current task loss
            outputs = self.model(task_data['x'])
            current_loss = criterion(outputs, task_data['y'])
            
            # Replay loss
            replay_data = self.sample_from_memory(len(task_data['x']))
            replay_loss = 0
            if replay_data is not None:
                replay_outputs = self.model(replay_data['x'])
                replay_loss = criterion(replay_outputs, replay_data['y'])
            
            # Combined loss
            total_loss = current_loss + self.replay_ratio * replay_loss
            
            total_loss.backward()
            optimizer.step()
```

### Generative Replay
```python
class GenerativeReplay:
    def __init__(self, model, generator, memory_size=1000):
        self.model = model
        self.generator = generator
        self.memory_size = memory_size
        self.real_memory = []
        self.generated_memory = []
        
    def train_generator(self, task_data: Dict[str, torch.Tensor]):
        """Train generator on current task data"""
        self.generator.train()
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(5):
            optimizer.zero_grad()
            
            # Generate fake data
            noise = torch.randn(len(task_data['x']), self.generator.latent_dim)
            fake_data = self.generator(noise)
            
            # Generator loss (try to fool discriminator)
            fake_outputs = self.model(fake_data)
            generator_loss = criterion(fake_outputs, task_data['y'])
            
            generator_loss.backward()
            optimizer.step()
    
    def generate_replay_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate replay data using trained generator"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.latent_dim)
            generated_data = self.generator(noise)
            
            # Use model to predict labels for generated data
            predicted_labels = self.model(generated_data).argmax(dim=1)
            
            return {
                'x': generated_data,
                'y': predicted_labels
            }
    
    def train_with_generative_replay(self, task_data: Dict[str, torch.Tensor]):
        """Train model with generative replay"""
        # Train generator on current task
        self.train_generator(task_data)
        
        # Generate replay data
        replay_data = self.generate_replay_data(len(task_data['x']))
        
        # Train model with both real and generated data
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Real data loss
            real_outputs = self.model(task_data['x'])
            real_loss = criterion(real_outputs, task_data['y'])
            
            # Generated data loss
            generated_outputs = self.model(replay_data['x'])
            generated_loss = criterion(generated_outputs, replay_data['y'])
            
            # Combined loss
            total_loss = real_loss + 0.5 * generated_loss
            
            total_loss.backward()
            optimizer.step()

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.generator(z)
```

## Regularization Techniques

### Learning Without Forgetting (LwF)
```python
class LearningWithoutForgetting:
    def __init__(self, model, temperature=2.0, alpha=0.5):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.old_model = None
        
    def save_old_model(self):
        """Save current model as old model"""
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
    
    def knowledge_distillation_loss(self, current_outputs: torch.Tensor, 
                                  old_outputs: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        # Soften probabilities
        current_probs = torch.softmax(current_outputs / self.temperature, dim=1)
        old_probs = torch.softmax(old_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(current_probs), old_probs
        )
        
        return kl_loss * (self.temperature ** 2)
    
    def train_with_lwf(self, task_data: Dict[str, torch.Tensor]):
        """Train model with Learning Without Forgetting"""
        if self.old_model is None:
            # First task, no old model
            self._standard_training(task_data)
        else:
            # Subsequent tasks, use LwF
            self._lwf_training(task_data)
        
        # Save current model as old model
        self.save_old_model()
    
    def _standard_training(self, task_data: Dict[str, torch.Tensor]):
        """Standard training for first task"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            outputs = self.model(task_data['x'])
            loss = criterion(outputs, task_data['y'])
            
            loss.backward()
            optimizer.step()
    
    def _lwf_training(self, task_data: Dict[str, torch.Tensor]):
        """Training with LwF for subsequent tasks"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Current task outputs
            current_outputs = self.model(task_data['x'])
            current_loss = criterion(current_outputs, task_data['y'])
            
            # Old task outputs (using old model)
            with torch.no_grad():
                old_outputs = self.old_model(task_data['x'])
            
            # Knowledge distillation loss
            distillation_loss = self.knowledge_distillation_loss(
                current_outputs, old_outputs
            )
            
            # Combined loss
            total_loss = current_loss + self.alpha * distillation_loss
            
            total_loss.backward()
            optimizer.step()
```

### Progressive Neural Networks
```python
class ProgressiveNeuralNetwork:
    def __init__(self, base_model, num_tasks):
        self.base_model = base_model
        self.num_tasks = num_tasks
        self.columns = [base_model]  # First column is base model
        self.task_heads = nn.ModuleDict()
        
    def add_column(self, task_id: int):
        """Add a new column for a new task"""
        # Create new column (copy of base model)
        new_column = copy.deepcopy(self.base_model)
        
        # Freeze previous columns
        for column in self.columns:
            for param in column.parameters():
                param.requires_grad = False
        
        # Add lateral connections from previous columns
        lateral_connections = nn.ModuleList()
        for prev_column in self.columns:
            lateral = nn.Linear(prev_column.output_dim, new_column.input_dim)
            lateral_connections.append(lateral)
        
        self.columns.append(new_column)
        
        # Add task head
        self.task_heads[f'task_{task_id}'] = nn.Linear(
            new_column.output_dim, 10  # Assuming 10 classes
        )
        
        return new_column, lateral_connections
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass through progressive network"""
        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not yet added to network")
        
        # Forward through base column
        current_output = self.columns[0](x)
        
        # Forward through subsequent columns with lateral connections
        for i in range(1, task_id + 1):
            # Lateral connection from previous column
            lateral_input = self.lateral_connections[i-1](current_output)
            
            # Forward through current column
            current_output = self.columns[i](lateral_input)
        
        # Task-specific head
        task_head = self.task_heads[f'task_{task_id}']
        output = task_head(current_output)
        
        return output
    
    def train_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Train network on new task"""
        # Add column if needed
        if task_id >= len(self.columns) - 1:
            self.add_column(task_id)
        
        # Train only the new column and task head
        trainable_params = list(self.columns[task_id].parameters()) + \
                          list(self.task_heads[f'task_{task_id}'].parameters())
        
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            outputs = self.forward(task_data['x'], task_id)
            loss = criterion(outputs, task_data['y'])
            
            loss.backward()
            optimizer.step()
```

## Architectural Approaches

### Dynamic Architecture Networks
```python
class DynamicArchitectureNetwork:
    def __init__(self, base_model, growth_rate=0.2):
        self.base_model = base_model
        self.growth_rate = growth_rate
        self.task_modules = nn.ModuleDict()
        self.task_id = 0
        
    def add_task_module(self, task_id: int, input_dim: int, output_dim: int):
        """Add a new task-specific module"""
        module = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * self.growth_rate)),
            nn.ReLU(),
            nn.Linear(int(input_dim * self.growth_rate), output_dim)
        )
        
        self.task_modules[f'task_{task_id}'] = module
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass with dynamic architecture"""
        # Base model features
        base_features = self.base_model(x)
        
        # Task-specific module
        if f'task_{task_id}' in self.task_modules:
            task_output = self.task_modules[f'task_{task_id}'](base_features)
            return task_output
        else:
            # Fallback to base model
            return base_features
    
    def learn_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Learn new task with dynamic architecture"""
        # Add task module if needed
        if f'task_{task_id}' not in self.task_modules:
            input_dim = self.base_model.output_dim
            output_dim = len(torch.unique(task_data['y']))
            self.add_task_module(task_id, input_dim, output_dim)
        
        # Train task module
        optimizer = torch.optim.Adam([
            {'params': self.base_model.parameters(), 'lr': 0.0001},  # Slow learning
            {'params': self.task_modules[f'task_{task_id}'].parameters(), 'lr': 0.001}
        ])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            outputs = self.forward(task_data['x'], task_id)
            loss = criterion(outputs, task_data['y'])
            
            loss.backward()
            optimizer.step()
```

### Continual Learning with Attention
```python
class ContinualLearningWithAttention:
    def __init__(self, model, num_tasks, attention_dim=64):
        self.model = model
        self.num_tasks = num_tasks
        self.attention_dim = attention_dim
        self.task_embeddings = nn.Embedding(num_tasks, attention_dim)
        self.attention_mechanism = nn.MultiheadAttention(attention_dim, num_heads=4)
        
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass with task-aware attention"""
        # Get base features
        base_features = self.model(x)
        
        # Get task embedding
        task_embedding = self.task_embeddings(torch.tensor(task_id))
        task_embedding = task_embedding.unsqueeze(0).expand(base_features.size(0), -1)
        
        # Apply attention mechanism
        attended_features, _ = self.attention_mechanism(
            base_features.unsqueeze(0),
            task_embedding.unsqueeze(0),
            task_embedding.unsqueeze(0)
        )
        
        return attended_features.squeeze(0)
    
    def learn_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Learn new task with attention mechanism"""
        # Train model with attention
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.task_embeddings.parameters()},
            {'params': self.attention_mechanism.parameters()}
        ], lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            outputs = self.forward(task_data['x'], task_id)
            loss = criterion(outputs, task_data['y'])
            
            loss.backward()
            optimizer.step()
```

## Meta-Learning for Continual Learning

### Model-Agnostic Meta-Learning (MAML) for Continual Learning
```python
class MAMLContinualLearning:
    def __init__(self, model, alpha=0.01, beta=0.001):
        self.model = model
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta    # Outer loop learning rate
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=beta)
        
    def inner_loop_update(self, task_data: Dict[str, torch.Tensor]) -> nn.Module:
        """Perform inner loop update (task-specific adaptation)"""
        # Create a copy of the model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.alpha)
        
        # Inner loop training
        for _ in range(5):  # Few-shot learning
            adapted_optimizer.zero_grad()
            
            outputs = adapted_model(task_data['x'])
            loss = nn.CrossEntropyLoss()(outputs, task_data['y'])
            
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model
    
    def meta_update(self, task_batch: List[Dict[str, torch.Tensor]]):
        """Perform meta-update across multiple tasks"""
        meta_loss = 0
        
        for task_data in task_batch:
            # Inner loop adaptation
            adapted_model = self.inner_loop_update(task_data)
            
            # Compute meta-loss on validation data
            # In practice, you would have separate validation data
            val_outputs = adapted_model(task_data['x'])  # Using same data for simplicity
            task_loss = nn.CrossEntropyLoss()(val_outputs, task_data['y'])
            meta_loss += task_loss
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
    
    def continual_learning_with_maml(self, task_stream: List[Dict[str, torch.Tensor]]):
        """Apply MAML for continual learning"""
        for i, task_data in enumerate(task_stream):
            # Adapt to current task
            adapted_model = self.inner_loop_update(task_data)
            
            # Use adapted model for current task
            self.model = adapted_model
            
            # Meta-update every few tasks
            if i % 5 == 0 and i > 0:
                task_batch = task_stream[max(0, i-5):i]
                self.meta_update(task_batch)
```

## Production Systems

### Continual Learning Platform
```python
class ContinualLearningPlatform:
    def __init__(self, base_model, memory_size=1000):
        self.base_model = base_model
        self.memory_size = memory_size
        self.episodic_memory = []
        self.task_performance = {}
        self.forgetting_metrics = {}
        
    def add_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Add a new task to the continual learning system"""
        # Update memory
        self._update_memory(task_data, task_id)
        
        # Train model
        self._train_on_task(task_data, task_id)
        
        # Evaluate performance
        performance = self._evaluate_performance(task_id)
        self.task_performance[task_id] = performance
        
        # Compute forgetting metrics
        forgetting = self._compute_forgetting_metrics()
        self.forgetting_metrics[task_id] = forgetting
        
        return {
            'task_id': task_id,
            'performance': performance,
            'forgetting': forgetting
        }
    
    def _update_memory(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Update episodic memory with new task data"""
        # Reservoir sampling
        for i in range(len(task_data['x'])):
            if len(self.episodic_memory) < self.memory_size:
                self.episodic_memory.append({
                    'x': task_data['x'][i].clone(),
                    'y': task_data['y'][i].clone(),
                    'task_id': task_id
                })
            else:
                # Random replacement
                idx = np.random.randint(0, len(self.episodic_memory))
                self.episodic_memory[idx] = {
                    'x': task_data['x'][i].clone(),
                    'y': task_data['y'][i].clone(),
                    'task_id': task_id
                }
    
    def _train_on_task(self, task_data: Dict[str, torch.Tensor], task_id: int):
        """Train model on new task"""
        # Combine with memory
        combined_data = self._combine_with_memory(task_data)
        
        # Train model
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            outputs = self.base_model(combined_data['x'])
            loss = criterion(outputs, combined_data['y'])
            
            loss.backward()
            optimizer.step()
    
    def _combine_with_memory(self, task_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine current task with episodic memory"""
        if not self.episodic_memory:
            return task_data
        
        # Sample from memory
        memory_samples = np.random.choice(
            self.episodic_memory,
            min(len(self.episodic_memory), len(task_data['x'])),
            replace=False
        )
        
        memory_x = torch.stack([sample['x'] for sample in memory_samples])
        memory_y = torch.stack([sample['y'] for sample in memory_samples])
        
        combined_x = torch.cat([task_data['x'], memory_x], dim=0)
        combined_y = torch.cat([task_data['y'], memory_y], dim=0)
        
        return {'x': combined_x, 'y': combined_y}
    
    def _evaluate_performance(self, task_id: int) -> float:
        """Evaluate performance on current task"""
        # In practice, you would have separate test data
        # For now, simulate evaluation
        return np.random.uniform(0.8, 0.95)
    
    def _compute_forgetting_metrics(self) -> Dict[str, float]:
        """Compute forgetting metrics"""
        if len(self.task_performance) < 2:
            return {'catastrophic_forgetting': 0.0}
        
        # Compute forgetting as performance drop
        forgetting = {}
        for task_id in self.task_performance:
            if task_id in self.task_performance:
                # Simplified forgetting computation
                forgetting[f'task_{task_id}_forgetting'] = np.random.uniform(0.0, 0.2)
        
        return forgetting
```

## Advanced Applications

### Continual Learning for Robotics
```python
class ContinualLearningRobotics:
    def __init__(self, robot_model, environment_simulator):
        self.robot_model = robot_model
        self.environment_simulator = environment_simulator
        self.skill_memory = []
        self.task_hierarchy = {}
        
    def learn_new_skill(self, skill_data: Dict[str, torch.Tensor], skill_name: str):
        """Learn a new robotic skill"""
        # Add skill to memory
        self.skill_memory.append({
            'name': skill_name,
            'data': skill_data,
            'model': copy.deepcopy(self.robot_model)
        })
        
        # Train model on new skill
        self._train_skill(skill_data, skill_name)
        
        # Update task hierarchy
        self._update_task_hierarchy(skill_name)
        
    def _train_skill(self, skill_data: Dict[str, torch.Tensor], skill_name: str):
        """Train robot model on new skill"""
        optimizer = torch.optim.Adam(self.robot_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # For continuous control
        
        for epoch in range(20):
            optimizer.zero_grad()
            
            # Predict actions
            predicted_actions = self.robot_model(skill_data['states'])
            loss = criterion(predicted_actions, skill_data['actions'])
            
            loss.backward()
            optimizer.step()
    
    def _update_task_hierarchy(self, skill_name: str):
        """Update hierarchical task structure"""
        # Analyze skill relationships
        # In practice, use more sophisticated analysis
        self.task_hierarchy[skill_name] = {
            'prerequisites': [],
            'dependencies': [],
            'complexity': np.random.uniform(0.1, 1.0)
        }
    
    def transfer_learning(self, source_skill: str, target_skill: str):
        """Transfer learning between skills"""
        if source_skill not in [skill['name'] for skill in self.skill_memory]:
            raise ValueError(f"Source skill {source_skill} not found")
        
        # Get source skill model
        source_model = next(skill['model'] for skill in self.skill_memory 
                          if skill['name'] == source_skill)
        
        # Initialize target skill with source knowledge
        self.robot_model.load_state_dict(source_model.state_dict())
        
        # Fine-tune for target skill
        target_data = self._get_skill_data(target_skill)
        self._train_skill(target_data, target_skill)
```

### Continual Learning for Natural Language Processing
```python
class ContinualLearningNLP:
    def __init__(self, language_model, vocabulary_manager):
        self.language_model = language_model
        self.vocabulary_manager = vocabulary_manager
        self.domain_adapters = nn.ModuleDict()
        self.knowledge_base = {}
        
    def learn_new_domain(self, domain_data: Dict[str, torch.Tensor], domain_name: str):
        """Learn a new domain while preserving previous knowledge"""
        # Create domain-specific adapter
        if domain_name not in self.domain_adapters:
            self.domain_adapters[domain_name] = nn.Linear(
                self.language_model.hidden_dim, self.language_model.hidden_dim
            )
        
        # Update vocabulary
        new_vocab = self._extract_vocabulary(domain_data)
        self.vocabulary_manager.add_vocabulary(new_vocab, domain_name)
        
        # Train model on new domain
        self._train_domain(domain_data, domain_name)
        
        # Update knowledge base
        self._update_knowledge_base(domain_data, domain_name)
        
    def _extract_vocabulary(self, domain_data: Dict[str, torch.Tensor]) -> List[str]:
        """Extract vocabulary from domain data"""
        # Simplified vocabulary extraction
        # In practice, use proper tokenization
        return ['new_word_1', 'new_word_2', 'new_word_3']
    
    def _train_domain(self, domain_data: Dict[str, torch.Tensor], domain_name: str):
        """Train model on new domain"""
        optimizer = torch.optim.Adam([
            {'params': self.language_model.parameters(), 'lr': 0.0001},
            {'params': self.domain_adapters[domain_name].parameters(), 'lr': 0.001}
        ])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass with domain adapter
            hidden_states = self.language_model(domain_data['input_ids'])
            domain_adapted = self.domain_adapters[domain_name](hidden_states)
            outputs = self.language_model.classifier(domain_adapted)
            
            loss = criterion(outputs, domain_data['labels'])
            
            loss.backward()
            optimizer.step()
    
    def _update_knowledge_base(self, domain_data: Dict[str, torch.Tensor], domain_name: str):
        """Update knowledge base with new domain information"""
        # Extract key information from domain
        self.knowledge_base[domain_name] = {
            'vocabulary_size': len(self._extract_vocabulary(domain_data)),
            'domain_specific_patterns': self._extract_patterns(domain_data),
            'learned_concepts': self._extract_concepts(domain_data)
        }
    
    def _extract_patterns(self, domain_data: Dict[str, torch.Tensor]) -> List[str]:
        """Extract domain-specific patterns"""
        # Simplified pattern extraction
        return ['pattern_1', 'pattern_2']
    
    def _extract_concepts(self, domain_data: Dict[str, torch.Tensor]) -> List[str]:
        """Extract learned concepts from domain"""
        # Simplified concept extraction
        return ['concept_1', 'concept_2']
```

## Conclusion

Continual Learning represents a crucial advancement in AI systems, enabling lifelong learning and adaptation. Key areas include:

1. **Catastrophic Forgetting Prevention**: EWC, SI, and other regularization techniques
2. **Replay-Based Methods**: Experience replay and generative replay
3. **Architectural Approaches**: Dynamic networks and progressive architectures
4. **Meta-Learning Integration**: MAML and other meta-learning approaches

The field is rapidly evolving with new techniques for memory management, knowledge preservation, and adaptive learning emerging regularly.

## Resources

- [Continual Learning Papers](https://arxiv.org/list/cs.LG/recent)
- [Avalanche Framework](https://avalanche.continualai.org/)
- [ContinualAI](https://www.continualai.org/)
- [Continual Learning Survey](https://arxiv.org/abs/2004.07211)
- [Catastrophic Forgetting](https://arxiv.org/abs/1612.00796) 