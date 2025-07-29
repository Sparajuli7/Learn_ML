# Model Training: Hyperparameter Tuning and Distributed Training

*"Training models efficiently and effectively: From single machines to distributed clusters"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Distributed Training](#distributed-training)
5. [Training Optimization](#training-optimization)
6. [Advanced Training Techniques](#advanced-training-techniques)
7. [Applications](#applications)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

---

## 🎯 Introduction

Model training is the core process of optimizing machine learning models to achieve the best possible performance. In 2025, with the increasing complexity of models and the need for efficient training, advanced techniques like distributed training and automated hyperparameter optimization have become essential.

### Why Model Training Matters in 2025

- **Model Complexity**: Large models require efficient training strategies
- **Resource Optimization**: Cost-effective training on cloud infrastructure
- **Time Efficiency**: Faster training with distributed computing
- **Automation**: Automated hyperparameter optimization
- **Scalability**: Training models that can handle massive datasets

### Training Challenges

1. **Computational Resources**: GPU/TPU utilization and memory management
2. **Hyperparameter Tuning**: Finding optimal model configurations
3. **Distributed Training**: Coordinating training across multiple nodes
4. **Overfitting**: Balancing model complexity and generalization
5. **Training Stability**: Ensuring consistent convergence

---

## 🧮 Mathematical Foundations

### Loss Functions

**Cross-Entropy Loss**:
```
L = -Σᵢ yᵢ log(ŷᵢ)
```

**Mean Squared Error**:
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

**Huber Loss** (robust to outliers):
```
L = { 0.5(y - ŷ)² if |y - ŷ| ≤ δ
    { δ(|y - ŷ| - 0.5δ) otherwise
```

### Optimization Algorithms

**Stochastic Gradient Descent**:
```
θ(t+1) = θ(t) - η∇L(θ(t))
```

**Adam Optimizer**:
```
m(t) = β₁m(t-1) + (1 - β₁)∇L(θ(t))
v(t) = β₂v(t-1) + (1 - β₂)(∇L(θ(t)))²
θ(t+1) = θ(t) - η × m(t) / (√v(t) + ε)
```

### Learning Rate Scheduling

**Exponential Decay**:
```
η(t) = η₀ × exp(-λt)
```

**Cosine Annealing**:
```
η(t) = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2
```

---

## 🔧 Hyperparameter Optimization

### Why This Matters
Hyperparameter optimization can significantly improve model performance and training efficiency.

### How It Works
1. Define hyperparameter search space
2. Use optimization algorithms to explore space
3. Evaluate configurations using cross-validation
4. Select best configuration

### Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import ray
from ray import tune

class HyperparameterOptimizer:
    """Comprehensive hyperparameter optimization toolkit"""
    
    def __init__(self, method='optuna', n_trials=100):
        """
        Initialize hyperparameter optimizer
        
        Args:
            method: 'grid', 'random', 'optuna', 'ray_tune'
            n_trials: Number of trials for optimization
        """
        self.method = method
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
    
    def optimize_grid_search(self, model, param_grid, X, y, cv=5):
        """Grid search optimization"""
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return grid_search
    
    def optimize_random_search(self, model, param_distributions, X, y, cv=5):
        """Random search optimization"""
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=self.n_trials,
            cv=cv, scoring='accuracy', n_jobs=-1, random_state=42
        )
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        return random_search
    
    def optimize_optuna(self, model_class, param_space, X, y, cv=5):
        """Optuna-based optimization"""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
            
            # Create model with sampled parameters
            model = model_class(**params)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return study
    
    def optimize_ray_tune(self, model_class, param_space, X, y, cv=5):
        """Ray Tune optimization"""
        
        def trainable(config):
            # Create model with config
            model = model_class(**config)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            # Report results
            tune.report(accuracy=scores.mean())
        
        # Run optimization
        analysis = tune.run(
            trainable,
            config=param_space,
            num_samples=self.n_trials,
            resources_per_trial={"cpu": 1}
        )
        
        self.best_params = analysis.best_config
        self.best_score = analysis.best_result['accuracy']
        
        return analysis

# Example usage
def demonstrate_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = (X.iloc[:, 0] + X.iloc[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Define parameter spaces
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Test different optimization methods
    optimizer = HyperparameterOptimizer(method='grid')
    
    # Grid search
    grid_result = optimizer.optimize_grid_search(
        RandomForestClassifier(random_state=42),
        rf_param_grid, X, y
    )
    
    print(f"Grid Search - Best score: {optimizer.best_score:.4f}")
    print(f"Best parameters: {optimizer.best_params}")
    
    # Random search
    optimizer.method = 'random'
    random_result = optimizer.optimize_random_search(
        RandomForestClassifier(random_state=42),
        rf_param_distributions, X, y
    )
    
    print(f"\nRandom Search - Best score: {optimizer.best_score:.4f}")
    print(f"Best parameters: {optimizer.best_params}")

# Run demonstration
demonstrate_hyperparameter_optimization()
```

### Advanced Optimization with Optuna

```python
class AdvancedOptunaOptimizer:
    """Advanced Optuna optimization with pruning and early stopping"""
    
    def __init__(self, n_trials=100, timeout=3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
    
    def optimize_with_pruning(self, model_class, param_space, X, y, cv=5):
        """Optimize with pruning for early stopping"""
        
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params(trial, param_space)
            
            # Create model
            model = model_class(**params)
            
            # Cross-validation with pruning
            from sklearn.model_selection import cross_val_score
            
            scores = []
            for i in range(cv):
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=i
                )
                
                # Train and evaluate
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
                
                # Report intermediate value for pruning
                trial.report(np.mean(scores), i)
                
                # Prune if necessary
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Create study with pruning
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        self.study = optuna.create_study(direction='maximize', pruner=pruner)
        
        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return self.study
    
    def _sample_params(self, trial, param_space):
        """Sample parameters from parameter space"""
        params = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
        
        return params
    
    def plot_optimization_history(self):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("No study available")
        
        optuna.visualization.plot_optimization_history(self.study)
        optuna.visualization.plot_param_importances(self.study)
        optuna.visualization.plot_parallel_coordinate(self.study)
```

---

## 🚀 Distributed Training

### Why This Matters
Distributed training enables training large models on multiple machines, reducing training time and enabling larger datasets.

### How It Works
1. Distribute data across multiple nodes
2. Train model in parallel
3. Synchronize gradients/parameters
4. Aggregate results

### Implementation

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import ray
from ray import train

class DistributedTrainer:
    """Distributed training framework"""
    
    def __init__(self, backend='nccl'):
        self.backend = backend
        self.world_size = None
        self.rank = None
    
    def setup_distributed(self, rank, world_size):
        """Setup distributed training"""
        self.rank = rank
        self.world_size = world_size
        
        # Initialize process group
        dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
        
        # Set device
        torch.cuda.set_device(rank)
    
    def train_distributed(self, model, train_dataset, val_dataset, 
                         batch_size=32, epochs=10, lr=0.001):
        """Train model using distributed data parallel"""
        
        # Create distributed sampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        
        # Move model to GPU and wrap with DDP
        model = model.cuda(self.rank)
        model = DDP(model, device_ids=[self.rank])
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)
            
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(self.rank), target.cuda(self.rank)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.cuda(self.rank), target.cuda(self.rank)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            # Print results (only on rank 0)
            if self.rank == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss/len(val_loader):.4f}, '
                      f'Val Acc: {100*correct/total:.2f}%')
    
    def cleanup(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()

# Ray-based distributed training
class RayDistributedTrainer:
    """Ray-based distributed training"""
    
    def __init__(self):
        self.trainer = None
    
    def train_with_ray(self, model_class, train_dataset, val_dataset, 
                       config, num_workers=4):
        """Train using Ray Train"""
        
        def train_func(config):
            # Setup model
            model = model_class(**config['model_params'])
            
            # Setup data
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            
            # Setup optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(config['epochs']):
                # Training
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                
                # Report metrics
                train.report({
                    'epoch': epoch,
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss / len(val_loader),
                    'val_accuracy': 100 * correct / total
                })
        
        # Run distributed training
        trainer = train.torch.prepare_model(train_func)
        result = trainer.run(config, num_workers=num_workers)
        
        return result
```

---

## ⚡ Training Optimization

### Why This Matters
Training optimization techniques can significantly improve training speed and model performance.

### Implementation

```python
class TrainingOptimizer:
    """Training optimization techniques"""
    
    def __init__(self):
        self.optimization_techniques = {}
    
    def implement_mixed_precision(self, model, optimizer):
        """Implement mixed precision training"""
        from torch.cuda.amp import GradScaler, autocast
        
        scaler = GradScaler()
        
        def train_step(data, target):
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            return loss
        
        return train_step, scaler
    
    def implement_gradient_accumulation(self, model, optimizer, accumulation_steps=4):
        """Implement gradient accumulation"""
        
        def train_step(data, target, step):
            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Scale loss
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            return loss
        
        return train_step
    
    def implement_learning_rate_scheduling(self, optimizer, scheduler_type='cosine'):
        """Implement learning rate scheduling"""
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            raise ValueError(f"Scheduler type {scheduler_type} not supported")
        
        return scheduler
    
    def implement_early_stopping(self, patience=10, min_delta=0.001):
        """Implement early stopping"""
        
        class EarlyStopping:
            def __init__(self, patience=10, min_delta=0.001):
                self.patience = patience
                self.min_delta = min_delta
                self.counter = 0
                self.best_loss = None
                self.early_stop = False
            
            def __call__(self, val_loss):
                if self.best_loss is None:
                    self.best_loss = val_loss
                elif val_loss > self.best_loss - self.min_delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_loss = val_loss
                    self.counter = 0
        
        return EarlyStopping(patience=patience, min_delta=min_delta)

# Example: Optimized training loop
def demonstrate_optimized_training():
    """Demonstrate optimized training techniques"""
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    # Setup optimizer and techniques
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training_optimizer = TrainingOptimizer()
    
    # Implement optimizations
    train_step, scaler = training_optimizer.implement_mixed_precision(model, optimizer)
    scheduler = training_optimizer.implement_learning_rate_scheduling(optimizer, 'cosine')
    early_stopping = training_optimizer.implement_early_stopping(patience=5)
    
    print("Optimized training setup complete")
    print(f"Mixed precision: {scaler is not None}")
    print(f"Learning rate scheduler: {scheduler}")
    print(f"Early stopping: {early_stopping}")

# Run demonstration
demonstrate_optimized_training()
```

---

## 🎯 Applications

### 1. **Large Language Model Training**

```python
class LLMTrainer:
    """Large Language Model training with optimizations"""
    
    def __init__(self, model_size='medium'):
        self.model_size = model_size
        self.optimization_config = {
            'mixed_precision': True,
            'gradient_accumulation': 4,
            'learning_rate_scheduling': 'cosine',
            'early_stopping': True
        }
    
    def train_language_model(self, model, train_dataset, val_dataset, config):
        """Train language model with optimizations"""
        
        # Setup distributed training
        trainer = DistributedTrainer()
        
        # Setup optimizations
        training_optimizer = TrainingOptimizer()
        
        # Implement all optimizations
        train_step, scaler = training_optimizer.implement_mixed_precision(model, config['optimizer'])
        scheduler = training_optimizer.implement_learning_rate_scheduling(config['optimizer'])
        early_stopping = training_optimizer.implement_early_stopping()
        
        # Training loop with optimizations
        for epoch in range(config['epochs']):
            # Training with optimizations
            train_loss = self._train_epoch_optimized(model, train_dataset, train_step, scaler)
            
            # Validation
            val_loss = self._validate_epoch(model, val_dataset)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def _train_epoch_optimized(self, model, dataset, train_step, scaler):
        """Optimized training epoch"""
        model.train()
        total_loss = 0.0
        
        for batch in dataset:
            loss = train_step(batch['input_ids'], batch['labels'])
            total_loss += loss.item()
        
        return total_loss / len(dataset)
    
    def _validate_epoch(self, model, dataset):
        """Validation epoch"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataset:
                output = model(batch['input_ids'])
                loss = nn.CrossEntropyLoss()(output, batch['labels'])
                total_loss += loss.item()
        
        return total_loss / len(dataset)
```

### 2. **Computer Vision Model Training**

```python
class VisionModelTrainer:
    """Computer Vision model training with optimizations"""
    
    def __init__(self):
        self.augmentation_pipeline = None
        self.optimization_config = {
            'mixed_precision': True,
            'gradient_accumulation': 2,
            'learning_rate_scheduling': 'step',
            'early_stopping': True
        }
    
    def setup_augmentation(self, config):
        """Setup data augmentation pipeline"""
        from torchvision import transforms
        
        self.augmentation_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def train_vision_model(self, model, train_dataset, val_dataset, config):
        """Train vision model with optimizations"""
        
        # Setup optimizations
        training_optimizer = TrainingOptimizer()
        
        # Implement optimizations
        train_step, scaler = training_optimizer.implement_mixed_precision(model, config['optimizer'])
        scheduler = training_optimizer.implement_learning_rate_scheduling(config['optimizer'], 'step')
        early_stopping = training_optimizer.implement_early_stopping()
        
        # Training loop
        for epoch in range(config['epochs']):
            # Training
            train_loss, train_acc = self._train_epoch_optimized(model, train_dataset, train_step, scaler)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(model, val_dataset)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    def _train_epoch_optimized(self, model, dataset, train_step, scaler):
        """Optimized training epoch for vision models"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataset):
            loss = train_step(data, target)
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(dataset), 100 * correct / total
    
    def _validate_epoch(self, model, dataset):
        """Validation epoch for vision models"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataset:
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(dataset), 100 * correct / total
```

---

## 🧪 Exercises and Projects

### Exercise 1: Implement Custom Optimizer

```python
# TODO: Implement custom optimizer
# 1. Create adaptive learning rate optimizer
# 2. Implement momentum and weight decay
# 3. Add gradient clipping
# 4. Compare with standard optimizers

def implement_custom_optimizer():
    # Your implementation here
    pass
```

### Exercise 2: Distributed Training Setup

```python
# TODO: Setup distributed training environment
# 1. Configure multi-GPU training
# 2. Implement data parallelism
# 3. Add model parallelism for large models
# 4. Monitor training metrics

def setup_distributed_training():
    # Your implementation here
    pass
```

### Quiz Questions

1. **Which hyperparameter optimization method is most efficient for high-dimensional spaces?**
   - A) Grid search
   - B) Random search
   - C) Bayesian optimization ✓
   - D) Genetic algorithms

2. **What is the main advantage of mixed precision training?**
   - A) Better accuracy
   - B) Faster training and lower memory usage ✓
   - C) More stable gradients
   - D) Easier implementation

3. **Which learning rate scheduler is best for long training runs?**
   - A) Step decay
   - B) Exponential decay
   - C) Cosine annealing ✓
   - D) Linear decay

### Advanced Project: Multi-Node Training System

```python
class MultiNodeTrainingSystem:
    """Multi-node training system with fault tolerance"""
    
    def __init__(self, nodes_config):
        self.nodes_config = nodes_config
        self.coordinator = None
        self.workers = []
    
    def setup_cluster(self):
        """Setup distributed cluster"""
        # Initialize Ray cluster
        ray.init()
        
        # Setup coordinator
        self.coordinator = self._create_coordinator()
        
        # Setup workers
        for node_config in self.nodes_config:
            worker = self._create_worker(node_config)
            self.workers.append(worker)
    
    def train_distributed(self, model_config, data_config):
        """Train model on distributed cluster"""
        # Distribute model across nodes
        distributed_model = self._distribute_model(model_config)
        
        # Distribute data across nodes
        distributed_data = self._distribute_data(data_config)
        
        # Start training
        training_futures = []
        for worker in self.workers:
            future = worker.train.remote(distributed_model, distributed_data)
            training_futures.append(future)
        
        # Monitor training
        results = ray.get(training_futures)
        
        return results
    
    def _create_coordinator(self):
        """Create training coordinator"""
        @ray.remote
        class Coordinator:
            def __init__(self):
                self.training_state = {}
                self.checkpoint_manager = None
            
            def coordinate_training(self, model_config, data_config):
                # Coordinate training across nodes
                pass
            
            def handle_failure(self, failed_node):
                # Handle node failures
                pass
        
        return Coordinator.remote()
    
    def _create_worker(self, node_config):
        """Create training worker"""
        @ray.remote(num_gpus=node_config.get('gpus', 1))
        class Worker:
            def __init__(self, node_id):
                self.node_id = node_id
                self.model = None
                self.optimizer = None
            
            def train(self, model_config, data_config):
                # Train model on this node
                pass
            
            def checkpoint(self):
                # Save model checkpoint
                pass
        
        return Worker.remote(node_config['node_id'])

# Project: Build multi-node training system
def build_multi_node_training_system():
    # 1. Design cluster architecture
    # 2. Implement fault tolerance
    # 3. Add monitoring and logging
    # 4. Test with large models
    # 5. Optimize for cost and performance
    pass
```

---

## 📖 Further Reading

### Essential Papers
- "Adam: A Method for Stochastic Optimization" by Kingma & Ba (2015)
- "Mixed Precision Training" by Micikevicius et al. (2018)
- "Distributed Training Strategies" by Li et al. (2020)

### Books
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Hands-On Machine Learning" by Aurélien Géron

### Online Resources
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Ray Documentation](https://docs.ray.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Next Steps
- **[Model Deployment](ml_engineering/28_deployment.md)**: Deploying trained models
- **[MLOps Basics](ml_engineering/29_mlops_basics.md)**: Production training pipelines
- **[Inference Optimization](ml_engineering/33_inference_optimization.md)**: Optimizing model inference

---

## 🎯 Key Takeaways

1. **Hyperparameter Optimization**: Critical for model performance
2. **Distributed Training**: Essential for large models and datasets
3. **Training Optimization**: Improves efficiency and stability
4. **Monitoring**: Track training progress and detect issues
5. **Automation**: Reduce manual intervention in training

---

*"Efficient training is the foundation of successful machine learning systems."*

**Next: [Evaluation & Testing](ml_engineering/27_evaluation_testing.md) → Metrics, A/B testing, and model debugging**