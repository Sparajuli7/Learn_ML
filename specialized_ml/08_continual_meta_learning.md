# Continual & Meta Learning

## Overview
Continual Learning enables AI systems to learn continuously from new data while retaining knowledge from previous tasks. Meta-Learning focuses on "learning to learn" - developing algorithms that can quickly adapt to new tasks with minimal data.

## Continual Learning

### 1. Catastrophic Forgetting Problem
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ContinualLearner:
    def __init__(self, model, memory_size=1000):
        self.model = model
        self.memory = []
        self.memory_size = memory_size
        self.task_count = 0
        
    def add_task(self, task_data, task_labels):
        """Add a new task to the learning system"""
        # Store some examples in memory
        self.update_memory(task_data, task_labels)
        
        # Train on new task while preserving old knowledge
        self.train_on_task(task_data, task_labels)
        
        self.task_count += 1
    
    def update_memory(self, data, labels):
        """Update episodic memory with new examples"""
        # Simple random sampling strategy
        indices = np.random.choice(len(data), min(len(data), 100), replace=False)
        
        for idx in indices:
            if len(self.memory) < self.memory_size:
                self.memory.append((data[idx], labels[idx]))
            else:
                # Replace random memory item
                replace_idx = np.random.randint(0, len(self.memory))
                self.memory[replace_idx] = (data[idx], labels[idx])
    
    def train_on_task(self, data, labels, epochs=10):
        """Train on new task while preserving old knowledge"""
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Train on new task
            optimizer.zero_grad()
            outputs = self.model(data)
            new_task_loss = criterion(outputs, labels)
            
            # Add memory replay loss
            if self.memory:
                memory_data, memory_labels = zip(*self.memory)
                memory_data = torch.stack(memory_data)
                memory_labels = torch.stack(memory_labels)
                
                memory_outputs = self.model(memory_data)
                memory_loss = criterion(memory_outputs, memory_labels)
                
                # Combine losses
                total_loss = new_task_loss + 0.5 * memory_loss
            else:
                total_loss = new_task_loss
            
            total_loss.backward()
            optimizer.step()
    
    def evaluate_on_all_tasks(self, task_datasets):
        """Evaluate performance on all learned tasks"""
        accuracies = []
        
        for task_id, (data, labels) in enumerate(task_datasets):
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                accuracies.append(accuracy)
        
        return accuracies

# Example continual learning scenario
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize continual learner
model = SimpleModel(input_size=10, hidden_size=50, num_classes=5)
learner = ContinualLearner(model)

# Simulate learning multiple tasks
for task_id in range(3):
    # Generate task-specific data
    task_data = torch.randn(100, 10)
    task_labels = torch.randint(0, 5, (100,))
    
    # Learn the task
    learner.add_task(task_data, task_labels)
    
    print(f"Learned task {task_id + 1}")
```

### 2. Elastic Weight Consolidation (EWC)
```python
class EWCContinualLearner:
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, data, labels):
        """Compute Fisher information matrix for current task"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Compute gradients for Fisher information
        for batch_data, batch_labels in self.create_batches(data, labels):
            optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Accumulate gradients squared
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.fisher_info:
                        self.fisher_info[name] = torch.zeros_like(param.data)
                    self.fisher_info[name] += param.grad.data ** 2
        
        # Average over batches
        num_batches = len(data) // 32 + 1
        for name in self.fisher_info:
            self.fisher_info[name] /= num_batches
    
    def store_optimal_params(self):
        """Store optimal parameters for current task"""
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
    
    def ewc_loss(self, outputs, labels):
        """Compute EWC loss"""
        criterion = nn.CrossEntropyLoss()
        task_loss = criterion(outputs, labels)
        
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                ewc_loss += torch.sum(fisher * (param - optimal) ** 2)
        
        return task_loss + self.lambda_ewc * ewc_loss
    
    def train_with_ewc(self, data, labels, epochs=10):
        """Train with EWC regularization"""
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            for batch_data, batch_labels in self.create_batches(data, labels):
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.ewc_loss(outputs, batch_labels)
                loss.backward()
                optimizer.step()
    
    def create_batches(self, data, labels, batch_size=32):
        """Create batches from data"""
        indices = torch.randperm(len(data))
        for i in range(0, len(data), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield data[batch_indices], labels[batch_indices]
    
    def learn_task(self, data, labels):
        """Learn a new task with EWC"""
        # Compute Fisher information for current task
        self.compute_fisher_information(data, labels)
        
        # Store optimal parameters
        self.store_optimal_params()
        
        # Train with EWC
        self.train_with_ewc(data, labels)

# Example EWC usage
ewc_learner = EWCContinualLearner(model)

for task_id in range(3):
    task_data = torch.randn(100, 10)
    task_labels = torch.randint(0, 5, (100,))
    
    ewc_learner.learn_task(task_data, task_labels)
    print(f"Learned task {task_id + 1} with EWC")
```

### 3. Progressive Neural Networks
```python
class ProgressiveNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ProgressiveNeuralNetwork, self).__init__()
        self.columns = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
    def add_column(self):
        """Add a new column for a new task"""
        column = self.create_column()
        self.columns.append(column)
        return len(self.columns) - 1
    
    def create_column(self):
        """Create a new column architecture"""
        column = nn.ModuleDict({
            'fc1': nn.Linear(self.input_size, self.hidden_size),
            'fc2': nn.Linear(self.hidden_size, self.hidden_size),
            'fc3': nn.Linear(self.hidden_size, self.num_classes)
        })
        return column
    
    def forward(self, x, task_id):
        """Forward pass with lateral connections"""
        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not found")
        
        column = self.columns[task_id]
        
        # First layer with lateral connections
        h1 = column['fc1'](x)
        if task_id > 0:
            # Add lateral connections from previous columns
            for prev_task in range(task_id):
                prev_column = self.columns[prev_task]
                lateral_input = prev_column['fc1'](x)
                h1 = h1 + lateral_input
        
        h1 = torch.relu(h1)
        
        # Second layer with lateral connections
        h2 = column['fc2'](h1)
        if task_id > 0:
            for prev_task in range(task_id):
                prev_column = self.columns[prev_task]
                lateral_input = prev_column['fc2'](h1)
                h2 = h2 + lateral_input
        
        h2 = torch.relu(h2)
        
        # Output layer
        output = column['fc3'](h2)
        
        return output

class ProgressiveLearner:
    def __init__(self, input_size, hidden_size, num_classes):
        self.network = ProgressiveNeuralNetwork(input_size, hidden_size, num_classes)
        self.task_mappings = {}
        
    def learn_task(self, task_id, data, labels, epochs=10):
        """Learn a new task"""
        if task_id not in self.task_mappings:
            self.task_mappings[task_id] = self.network.add_column()
        
        column_id = self.task_mappings[task_id]
        
        # Train the column for this task
        optimizer = optim.Adam(self.network.columns[column_id].parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.network(data, column_id)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    def predict(self, x, task_id):
        """Predict for a specific task"""
        if task_id not in self.task_mappings:
            raise ValueError(f"Task {task_id} not learned")
        
        column_id = self.task_mappings[task_id]
        self.network.eval()
        
        with torch.no_grad():
            outputs = self.network(x, column_id)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions

# Example progressive neural network
prog_learner = ProgressiveLearner(input_size=10, hidden_size=50, num_classes=5)

for task_id in range(3):
    task_data = torch.randn(100, 10)
    task_labels = torch.randint(0, 5, (100,))
    
    prog_learner.learn_task(task_id, task_data, task_labels)
    print(f"Learned task {task_id} with progressive neural network")
```

## Meta-Learning

### 1. Model-Agnostic Meta-Learning (MAML)
```python
class MAML(nn.Module):
    def __init__(self, model, alpha=0.01, beta=0.001):
        super(MAML, self).__init__()
        self.model = model
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta    # Outer loop learning rate
        self.meta_optimizer = optim.Adam(self.parameters(), lr=beta)
    
    def inner_loop(self, support_data, support_labels, num_steps=5):
        """Inner loop optimization for a specific task"""
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=self.alpha)
        
        for step in range(num_steps):
            task_optimizer.zero_grad()
            outputs = adapted_model(support_data)
            loss = nn.CrossEntropyLoss()(outputs, support_labels)
            loss.backward()
            task_optimizer.step()
        
        return adapted_model
    
    def outer_loop(self, task_batch):
        """Outer loop optimization across multiple tasks"""
        meta_loss = 0
        
        for support_data, support_labels, query_data, query_labels in task_batch:
            # Inner loop adaptation
            adapted_model = self.inner_loop(support_data, support_labels)
            
            # Evaluate on query set
            query_outputs = adapted_model(query_data)
            query_loss = nn.CrossEntropyLoss()(query_outputs, query_labels)
            meta_loss += query_loss
        
        return meta_loss / len(task_batch)
    
    def train_meta(self, task_distribution, num_meta_epochs=100):
        """Train the meta-learner"""
        for epoch in range(num_meta_epochs):
            # Sample batch of tasks
            task_batch = self.sample_task_batch(task_distribution)
            
            # Outer loop optimization
            meta_loss = self.outer_loop(task_batch)
            
            # Update meta-parameters
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Meta epoch {epoch}, Loss: {meta_loss.item():.4f}")
    
    def sample_task_batch(self, task_distribution, batch_size=4):
        """Sample a batch of tasks"""
        task_batch = []
        
        for _ in range(batch_size):
            # Sample a task from the distribution
            task = task_distribution.sample_task()
            support_data, support_labels = task.sample_support()
            query_data, query_labels = task.sample_query()
            
            task_batch.append((support_data, support_labels, query_data, query_labels))
        
        return task_batch
    
    def adapt_to_task(self, support_data, support_labels, num_steps=5):
        """Adapt to a new task using few-shot learning"""
        adapted_model = self.inner_loop(support_data, support_labels, num_steps)
        return adapted_model

# Example MAML implementation
class SimpleMAMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMAMLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

maml_model = SimpleMAMLModel(input_size=10, hidden_size=50, num_classes=5)
maml_learner = MAML(maml_model)

# Train meta-learner (would need task distribution)
# maml_learner.train_meta(task_distribution)
```

### 2. Prototypical Networks
```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
    
    def forward(self, x):
        """Encode input to embedding space"""
        return self.encoder(x)
    
    def compute_prototypes(self, support_data, support_labels, num_classes):
        """Compute class prototypes from support set"""
        prototypes = []
        
        for class_id in range(num_classes):
            # Get support examples for this class
            class_mask = (support_labels == class_id)
            class_embeddings = self.forward(support_data[class_mask])
            
            # Compute prototype as mean of embeddings
            prototype = torch.mean(class_embeddings, dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def compute_distances(self, query_embeddings, prototypes):
        """Compute Euclidean distances to prototypes"""
        # Expand dimensions for broadcasting
        query_embeddings = query_embeddings.unsqueeze(1)  # [N, 1, D]
        prototypes = prototypes.unsqueeze(0)               # [1, K, D]
        
        # Compute Euclidean distances
        distances = torch.sum((query_embeddings - prototypes) ** 2, dim=2)
        return distances
    
    def predict(self, query_data, support_data, support_labels, num_classes):
        """Predict classes for query examples"""
        # Compute prototypes
        prototypes = self.compute_prototypes(support_data, support_labels, num_classes)
        
        # Encode query examples
        query_embeddings = self.forward(query_data)
        
        # Compute distances to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Convert distances to probabilities (negative distances)
        logits = -distances
        
        return logits

class PrototypicalLearner:
    def __init__(self, input_size, hidden_size, embedding_size):
        self.network = PrototypicalNetwork(input_size, hidden_size, embedding_size)
        self.optimizer = optim.Adam(self.network.parameters())
    
    def train_episode(self, support_data, support_labels, query_data, query_labels, num_classes):
        """Train on a single episode"""
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.network.predict(support_data, support_labels, query_data, num_classes)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, query_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_episode(self, support_data, support_labels, query_data, query_labels, num_classes):
        """Evaluate on a single episode"""
        self.network.eval()
        
        with torch.no_grad():
            logits = self.network.predict(support_data, support_labels, query_data, num_classes)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        return accuracy

# Example prototypical network
proto_learner = PrototypicalLearner(input_size=10, hidden_size=50, embedding_size=20)

# Train on episodes
for episode in range(100):
    # Sample episode (few-shot learning scenario)
    support_data = torch.randn(5, 10)  # 5 support examples
    support_labels = torch.randint(0, 3, (5,))  # 3 classes
    query_data = torch.randn(15, 10)   # 15 query examples
    query_labels = torch.randint(0, 3, (15,))
    
    loss = proto_learner.train_episode(support_data, support_labels, query_data, query_labels, 3)
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Loss: {loss:.4f}")
```

### 3. Reptile Algorithm
```python
class Reptile(nn.Module):
    def __init__(self, model, epsilon=0.1):
        super(Reptile, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.parameters())
    
    def reptile_step(self, task_batch):
        """Perform one Reptile step"""
        # Store initial parameters
        initial_params = {}
        for name, param in self.model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # Inner loop for each task
        for support_data, support_labels in task_batch:
            # Adapt to task
            adapted_model = self.adapt_to_task(support_data, support_labels)
            
            # Reptile update: move towards adapted parameters
            for name, param in self.model.named_parameters():
                adapted_param = adapted_model[name]
                param.data += self.epsilon * (adapted_param - initial_params[name])
    
    def adapt_to_task(self, support_data, support_labels, num_steps=5):
        """Adapt model to a specific task"""
        # Clone model for adaptation
        adapted_model = {}
        for name, param in self.model.named_parameters():
            adapted_model[name] = param.data.clone()
        
        # Inner loop optimization
        for step in range(num_steps):
            # Forward pass
            outputs = self.model(support_data)
            loss = nn.CrossEntropyLoss()(outputs, support_labels)
            
            # Compute gradients
            gradients = torch.autograd.grad(loss, self.model.parameters())
            
            # Update adapted parameters
            for i, (name, param) in enumerate(self.model.named_parameters()):
                adapted_model[name] -= 0.01 * gradients[i]  # Inner learning rate
        
        return adapted_model
    
    def train_meta(self, task_distribution, num_meta_epochs=100):
        """Train the meta-learner using Reptile"""
        for epoch in range(num_meta_epochs):
            # Sample batch of tasks
            task_batch = self.sample_task_batch(task_distribution)
            
            # Perform Reptile step
            self.reptile_step(task_batch)
            
            if epoch % 10 == 0:
                print(f"Meta epoch {epoch}")
    
    def sample_task_batch(self, task_distribution, batch_size=4):
        """Sample a batch of tasks"""
        task_batch = []
        
        for _ in range(batch_size):
            # Sample a task
            task = task_distribution.sample_task()
            support_data, support_labels = task.sample_support()
            task_batch.append((support_data, support_labels))
        
        return task_batch

# Example Reptile implementation
reptile_model = SimpleMAMLModel(input_size=10, hidden_size=50, num_classes=5)
reptile_learner = Reptile(reptile_model)

# Train meta-learner (would need task distribution)
# reptile_learner.train_meta(task_distribution)
```

## Few-Shot Learning

### 1. Siamese Networks
```python
class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
    
    def forward(self, x1, x2):
        """Forward pass for pair of inputs"""
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        
        # Compute distance between embeddings
        distance = torch.sum((embedding1 - embedding2) ** 2, dim=1)
        
        return distance
    
    def predict_similarity(self, x1, x2):
        """Predict similarity between two inputs"""
        distance = self.forward(x1, x2)
        similarity = torch.exp(-distance)  # Convert distance to similarity
        return similarity

class SiameseLearner:
    def __init__(self, input_size, hidden_size, embedding_size):
        self.network = SiameseNetwork(input_size, hidden_size, embedding_size)
        self.optimizer = optim.Adam(self.network.parameters())
    
    def create_pairs(self, data, labels, num_pairs=1000):
        """Create positive and negative pairs for training"""
        pairs = []
        pair_labels = []
        
        for _ in range(num_pairs):
            # Randomly select two examples
            idx1, idx2 = np.random.choice(len(data), 2, replace=False)
            
            # Create pair
            pair = (data[idx1], data[idx2])
            pairs.append(pair)
            
            # Label: 1 if same class, 0 if different class
            label = 1 if labels[idx1] == labels[idx2] else 0
            pair_labels.append(label)
        
        return pairs, torch.tensor(pair_labels, dtype=torch.float32)
    
    def train_on_pairs(self, pairs, pair_labels, epochs=10):
        """Train on pairs of examples"""
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(pairs), 32):  # Batch size 32
                batch_pairs = pairs[i:i+32]
                batch_labels = pair_labels[i:i+32]
                
                # Prepare batch
                x1_batch = torch.stack([pair[0] for pair in batch_pairs])
                x2_batch = torch.stack([pair[1] for pair in batch_pairs])
                
                # Forward pass
                distances = self.network(x1_batch, x2_batch)
                
                # Convert distances to similarities
                similarities = torch.exp(-distances)
                
                # Compute loss (binary cross-entropy)
                loss = nn.BCELoss()(similarities, batch_labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def few_shot_classification(self, support_data, support_labels, query_data, num_classes):
        """Perform few-shot classification using similarity"""
        self.network.eval()
        
        predictions = []
        
        with torch.no_grad():
            for query_example in query_data:
                # Compare query example to all support examples
                similarities = []
                
                for support_example in support_data:
                    similarity = self.network.predict_similarity(
                        query_example.unsqueeze(0), 
                        support_example.unsqueeze(0)
                    )
                    similarities.append(similarity.item())
                
                # Find most similar support example
                most_similar_idx = np.argmax(similarities)
                predicted_class = support_labels[most_similar_idx]
                predictions.append(predicted_class)
        
        return torch.tensor(predictions)

# Example Siamese network
siamese_learner = SiameseLearner(input_size=10, hidden_size=50, embedding_size=20)

# Train on pairs
for task in range(5):
    # Generate task-specific data
    task_data = torch.randn(100, 10)
    task_labels = torch.randint(0, 5, (100,))
    
    # Create pairs and train
    pairs, pair_labels = siamese_learner.create_pairs(task_data, task_labels)
    siamese_learner.train_on_pairs(pairs, pair_labels)
    
    print(f"Trained on task {task + 1}")
```

### 2. Matching Networks
```python
class MatchingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(MatchingNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
        self.attention = nn.MultiheadAttention(embedding_size, num_heads=4)
    
    def forward(self, support_data, support_labels, query_data, num_classes):
        """Forward pass for few-shot classification"""
        # Encode support and query examples
        support_embeddings = self.encoder(support_data)
        query_embeddings = self.encoder(query_data)
        
        # Use attention mechanism
        query_embeddings = query_embeddings.unsqueeze(0)  # [1, N, D]
        support_embeddings = support_embeddings.unsqueeze(0)  # [1, K, D]
        
        attended_query, _ = self.attention(query_embeddings, support_embeddings, support_embeddings)
        attended_query = attended_query.squeeze(0)  # [N, D]
        
        # Compute similarities
        similarities = torch.mm(attended_query, support_embeddings.squeeze(0).t())
        
        # Convert to logits
        logits = similarities
        
        return logits

class MatchingLearner:
    def __init__(self, input_size, hidden_size, embedding_size):
        self.network = MatchingNetwork(input_size, hidden_size, embedding_size)
        self.optimizer = optim.Adam(self.network.parameters())
    
    def train_episode(self, support_data, support_labels, query_data, query_labels, num_classes):
        """Train on a single episode"""
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.network(support_data, support_labels, query_data, num_classes)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, query_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_episode(self, support_data, support_labels, query_data, query_labels, num_classes):
        """Evaluate on a single episode"""
        self.network.eval()
        
        with torch.no_grad():
            logits = self.network(support_data, support_labels, query_data, num_classes)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        return accuracy

# Example matching network
matching_learner = MatchingLearner(input_size=10, hidden_size=50, embedding_size=20)

# Train on episodes
for episode in range(100):
    # Sample episode
    support_data = torch.randn(5, 10)
    support_labels = torch.randint(0, 3, (5,))
    query_data = torch.randn(15, 10)
    query_labels = torch.randint(0, 3, (15,))
    
    loss = matching_learner.train_episode(support_data, support_labels, query_data, query_labels, 3)
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Loss: {loss:.4f}")
```

## Evaluation Metrics

### 1. Continual Learning Metrics
```python
class ContinualLearningEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_catastrophic_forgetting(self, model, task_datasets):
        """Evaluate catastrophic forgetting"""
        accuracies = []
        
        for task_id, (data, labels) in enumerate(task_datasets):
            model.eval()
            with torch.no_grad():
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                accuracies.append(accuracy)
        
        # Compute forgetting measure
        forgetting = []
        for i in range(1, len(accuracies)):
            forgetting.append(accuracies[i-1] - accuracies[i])
        
        return {
            'task_accuracies': accuracies,
            'average_accuracy': np.mean(accuracies),
            'forgetting': forgetting,
            'average_forgetting': np.mean(forgetting)
        }
    
    def evaluate_learning_efficiency(self, model, task_datasets):
        """Evaluate learning efficiency"""
        # Measure training time and data efficiency
        training_times = []
        data_efficiency = []
        
        for task_id, (data, labels) in enumerate(task_datasets):
            start_time = time.time()
            
            # Train on task
            # ... training code ...
            
            end_time = time.time()
            training_times.append(end_time - start_time)
            
            # Measure data efficiency (accuracy per training example)
            model.eval()
            with torch.no_grad():
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).float().mean().item()
            
            data_efficiency.append(accuracy / len(data))
        
        return {
            'training_times': training_times,
            'average_training_time': np.mean(training_times),
            'data_efficiency': data_efficiency,
            'average_data_efficiency': np.mean(data_efficiency)
        }
```

### 2. Meta-Learning Metrics
```python
class MetaLearningEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_few_shot_performance(self, model, test_tasks, num_shots=5):
        """Evaluate few-shot learning performance"""
        accuracies = []
        
        for task in test_tasks:
            support_data, support_labels = task.sample_support(num_shots)
            query_data, query_labels = task.sample_query()
            
            # Adapt model to task
            adapted_model = model.adapt_to_task(support_data, support_labels)
            
            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                outputs = adapted_model(query_data)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                accuracies.append(accuracy)
        
        return {
            'few_shot_accuracies': accuracies,
            'average_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
    
    def evaluate_adaptation_speed(self, model, test_tasks, adaptation_steps=[1, 5, 10]):
        """Evaluate adaptation speed"""
        results = {}
        
        for num_steps in adaptation_steps:
            accuracies = []
            
            for task in test_tasks:
                support_data, support_labels = task.sample_support()
                query_data, query_labels = task.sample_query()
                
                # Adapt with specified number of steps
                adapted_model = model.adapt_to_task(support_data, support_labels, num_steps)
                
                # Evaluate
                adapted_model.eval()
                with torch.no_grad():
                    outputs = adapted_model(query_data)
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == query_labels).float().mean().item()
                    accuracies.append(accuracy)
            
            results[f'{num_steps}_steps'] = {
                'accuracies': accuracies,
                'average_accuracy': np.mean(accuracies)
            }
        
        return results
```

## Tools and Libraries

- **PyTorch**: Deep learning framework
- **Learn2Learn**: Meta-learning library
- **Torchmeta**: Few-shot learning datasets
- **ContinualAI**: Continual learning resources
- **MAML-Pytorch**: MAML implementation

## Best Practices

1. **Memory Management**: Efficient storage and retrieval of past experiences
2. **Task Boundaries**: Clear definition of task transitions
3. **Regularization**: Prevent catastrophic forgetting
4. **Meta-Optimization**: Careful tuning of meta-learning hyperparameters
5. **Evaluation Protocols**: Proper evaluation of continual and meta-learning systems

## Next Steps

1. **Neural Architecture Search**: Automating architecture design
2. **Meta-Reinforcement Learning**: Meta-learning for RL
3. **Continual Meta-Learning**: Combining both approaches
4. **Lifelong Learning**: Long-term learning systems
5. **Few-Shot Learning**: Learning from minimal examples

---

*Continual and meta-learning represent the frontier of adaptive AI systems, enabling machines to learn continuously and adapt quickly to new tasks. These approaches are essential for building AI systems that can operate effectively in dynamic, real-world environments.* 