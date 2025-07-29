# Federated Learning: Decentralized AI & Privacy-Preserving ML

*"Training AI models across distributed data without centralization"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation](#implementation)
4. [Applications](#applications)
5. [Exercises and Projects](#exercises-and-projects)
6. [Further Reading](#further-reading)

---

## 🎯 Introduction

Federated Learning represents a paradigm shift in machine learning: training models across distributed data sources without centralizing sensitive information. In 2025, this approach is crucial for privacy, regulatory compliance, and scalable AI deployment.

### Historical Context

Traditional ML required centralized data collection, leading to privacy concerns and regulatory challenges. Federated Learning emerged to address these issues:

- **2016**: Google introduces federated learning for mobile keyboard prediction
- **2019**: FedAvg algorithm becomes the foundation for federated optimization
- **2020**: Differential privacy integration for enhanced security
- **2023**: Large-scale deployment in healthcare, finance, and IoT
- **2025**: Enterprise adoption with advanced privacy guarantees

### 2025 Federated Learning Landscape

**Global Challenges:**
- Data privacy regulations (GDPR, CCPA, EU AI Act)
- Cross-border data sharing restrictions
- Need for collaborative AI without data centralization
- Edge computing and IoT device proliferation
- Healthcare and financial data sensitivity

**FL Solutions:**
- Privacy-preserving model training
- Cross-silo and cross-device federated learning
- Secure aggregation protocols
- Differential privacy integration
- Federated analytics and inference

---

## 🧮 Mathematical Foundations

### 1. Federated Averaging (FedAvg)

**Core Algorithm:**

```
w^(t+1) = Σᵢ (nᵢ/n) × wᵢ^(t+1)
```

Where:
- w^(t+1) = Global model parameters at round t+1
- nᵢ = Number of samples at client i
- n = Total number of samples across all clients
- wᵢ^(t+1) = Local model parameters at client i

**Local Training:**

```
wᵢ^(t+1) = wᵢ^t - η × ∇Lᵢ(wᵢ^t)
```

Where:
- η = Learning rate
- Lᵢ = Loss function for client i
- ∇Lᵢ = Gradient of loss function

### 2. Secure Aggregation

**Homomorphic Encryption for Aggregation:**

```
E(w₁) ⊕ E(w₂) = E(w₁ + w₂)
```

**Secret Sharing Protocol:**

```
w = Σᵢ wᵢ = Σᵢ Σⱼ sᵢⱼ
```

Where:
- sᵢⱼ = Secret share of client i's contribution to server j
- w = Final aggregated model

### 3. Differential Privacy in FL

**Local Differential Privacy:**

```
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ
```

Where:
- M = Mechanism (algorithm)
- D, D' = Adjacent datasets
- ε = Privacy budget
- δ = Privacy failure probability

**Gaussian Mechanism:**

```
M(x) = x + N(0, σ²)
```

Where σ² = (Δf)² × log(1/δ) / ε²

---

## 💻 Implementation

### 1. Basic Federated Learning System

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
import copy
import random

class FederatedLearningSystem:
    def __init__(self, model_fn, num_clients=10):
        self.model_fn = model_fn
        self.num_clients = num_clients
        self.global_model = None
        self.client_models = []
        self.client_data = []
        
    def initialize_global_model(self):
        """Initialize global model"""
        self.global_model = self.model_fn()
        return self.global_model
    
    def create_client_data(self, dataset, split_ratio=0.8):
        """Create distributed data for clients"""
        # Simulate distributed data
        x, y = dataset
        
        # Split data among clients
        data_per_client = len(x) // self.num_clients
        self.client_data = []
        
        for i in range(self.num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client
            
            client_x = x[start_idx:end_idx]
            client_y = y[start_idx:end_idx]
            
            # Add some heterogeneity (different distributions)
            if i % 3 == 0:
                # Client with more positive samples
                pos_indices = np.where(client_y == 1)[0]
                neg_indices = np.where(client_y == 0)[0]
                if len(pos_indices) > len(neg_indices):
                    # Add more positive samples
                    extra_pos = np.random.choice(pos_indices, size=len(pos_indices)//2)
                    client_x = np.concatenate([client_x, client_x[extra_pos]])
                    client_y = np.concatenate([client_y, client_y[extra_pos]])
            
            self.client_data.append((client_x, client_y))
    
    def train_client_model(self, client_id, epochs=5, batch_size=32):
        """Train model on a specific client's data"""
        if client_id >= len(self.client_data):
            raise ValueError(f"Client {client_id} does not exist")
        
        # Create local model copy
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.set_weights(self.global_model.get_weights())
        
        # Get client data
        x_train, y_train = self.client_data[client_id]
        
        # Compile model
        local_model.compile(
            optimizer=SGD(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train locally
        history = local_model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        return local_model, history
    
    def federated_averaging(self, client_models):
        """Perform federated averaging of client models"""
        if not client_models:
            return self.global_model
        
        # Get model weights
        weights = [model.get_weights() for model in client_models]
        
        # Calculate average weights
        averaged_weights = []
        for layer_idx in range(len(weights[0])):
            layer_weights = np.array([client_weights[layer_idx] for client_weights in weights])
            averaged_layer = np.mean(layer_weights, axis=0)
            averaged_weights.append(averaged_layer)
        
        # Update global model
        self.global_model.set_weights(averaged_weights)
        
        return self.global_model
    
    def run_federated_training(self, rounds=10, clients_per_round=5, local_epochs=5):
        """Run federated learning for multiple rounds"""
        training_history = {
            'round': [],
            'global_accuracy': [],
            'client_accuracies': []
        }
        
        for round_num in range(rounds):
            print(f"Federated Round {round_num + 1}/{rounds}")
            
            # Select clients for this round
            selected_clients = random.sample(range(self.num_clients), clients_per_round)
            
            # Train on selected clients
            client_models = []
            client_accuracies = []
            
            for client_id in selected_clients:
                print(f"  Training client {client_id}")
                local_model, history = self.train_client_model(client_id, epochs=local_epochs)
                client_models.append(local_model)
                
                # Evaluate local model
                x_test, y_test = self.client_data[client_id]
                accuracy = local_model.evaluate(x_test, y_test, verbose=0)[1]
                client_accuracies.append(accuracy)
            
            # Aggregate models
            self.federated_averaging(client_models)
            
            # Evaluate global model
            global_accuracy = self.evaluate_global_model()
            
            # Record metrics
            training_history['round'].append(round_num + 1)
            training_history['global_accuracy'].append(global_accuracy)
            training_history['client_accuracies'].append(client_accuracies)
            
            print(f"  Global accuracy: {global_accuracy:.4f}")
            print(f"  Average client accuracy: {np.mean(client_accuracies):.4f}")
        
        return training_history
    
    def evaluate_global_model(self):
        """Evaluate global model on all client data"""
        if not self.global_model:
            return 0.0
        
        # Compile global model
        self.global_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Evaluate on all client data
        total_correct = 0
        total_samples = 0
        
        for client_data in self.client_data:
            x_test, y_test = client_data
            predictions = self.global_model.predict(x_test, verbose=0)
            predictions = (predictions > 0.5).astype(int)
            
            correct = np.sum(predictions.flatten() == y_test)
            total_correct += correct
            total_samples += len(y_test)
        
        return total_correct / total_samples if total_samples > 0 else 0.0

# Usage example
def create_simple_model():
    """Create a simple neural network model"""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Initialize federated learning system
fl_system = FederatedLearningSystem(create_simple_model, num_clients=10)

# Generate synthetic data (simulating MNIST binary classification)
def generate_synthetic_data(num_samples=10000):
    """Generate synthetic data for demonstration"""
    x = np.random.rand(num_samples, 28, 28)
    y = np.random.randint(0, 2, num_samples)
    return x, y

# Create distributed data
synthetic_data = generate_synthetic_data()
fl_system.create_client_data(synthetic_data)

# Initialize global model
fl_system.initialize_global_model()

# Run federated training
training_history = fl_system.run_federated_training(rounds=5, clients_per_round=3)

print("Federated Learning Training Complete!")
print(f"Final global accuracy: {training_history['global_accuracy'][-1]:.4f}")
```

### 2. Secure Aggregation with Homomorphic Encryption

```python
import numpy as np
from cryptography.fernet import Fernet
import hashlib
import secrets

class SecureFederatedLearning:
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.clients = {}
        self.server = None
        self.setup_encryption()
    
    def setup_encryption(self):
        """Setup encryption keys for secure aggregation"""
        self.master_key = Fernet.generate_key()
        self.cipher = Fernet(self.master_key)
        
        # Generate keys for each client
        for i in range(self.num_clients):
            client_key = Fernet.generate_key()
            self.clients[f'client_{i}'] = {
                'key': client_key,
                'cipher': Fernet(client_key),
                'data': None,
                'model': None
            }
    
    def encrypt_model_weights(self, weights, client_id):
        """Encrypt model weights for secure transmission"""
        client = self.clients[client_id]
        
        # Convert weights to bytes
        weights_bytes = self.weights_to_bytes(weights)
        
        # Encrypt weights
        encrypted_weights = client['cipher'].encrypt(weights_bytes)
        
        return encrypted_weights
    
    def decrypt_model_weights(self, encrypted_weights, client_id):
        """Decrypt model weights"""
        client = self.clients[client_id]
        
        # Decrypt weights
        weights_bytes = client['cipher'].decrypt(encrypted_weights)
        
        # Convert back to weights
        weights = self.bytes_to_weights(weights_bytes)
        
        return weights
    
    def weights_to_bytes(self, weights):
        """Convert model weights to bytes for encryption"""
        import pickle
        return pickle.dumps(weights)
    
    def bytes_to_weights(self, weights_bytes):
        """Convert bytes back to model weights"""
        import pickle
        return pickle.loads(weights_bytes)
    
    def secure_aggregation(self, encrypted_weights_list):
        """Perform secure aggregation of encrypted model weights"""
        if not encrypted_weights_list:
            return None
        
        # For demonstration, we'll decrypt and then aggregate
        # In practice, this would use homomorphic encryption
        decrypted_weights_list = []
        
        for i, encrypted_weights in enumerate(encrypted_weights_list):
            client_id = f'client_{i}'
            decrypted_weights = self.decrypt_model_weights(encrypted_weights, client_id)
            decrypted_weights_list.append(decrypted_weights)
        
        # Perform federated averaging
        aggregated_weights = self.federated_averaging(decrypted_weights_list)
        
        return aggregated_weights
    
    def federated_averaging(self, weights_list):
        """Perform federated averaging of model weights"""
        if not weights_list:
            return None
        
        # Average weights across clients
        averaged_weights = []
        for layer_idx in range(len(weights_list[0])):
            layer_weights = np.array([weights[layer_idx] for weights in weights_list])
            averaged_layer = np.mean(layer_weights, axis=0)
            averaged_weights.append(averaged_layer)
        
        return averaged_weights
    
    def simulate_secure_training(self, global_model, client_data_list):
        """Simulate secure federated training"""
        training_rounds = 5
        training_history = []
        
        for round_num in range(training_rounds):
            print(f"Secure Federated Round {round_num + 1}")
            
            # Simulate local training on each client
            client_models = []
            encrypted_weights_list = []
            
            for client_id in range(self.num_clients):
                # Simulate local training (simplified)
                local_model = self.simulate_local_training(global_model, client_data_list[client_id])
                client_models.append(local_model)
                
                # Encrypt model weights
                weights = local_model.get_weights()
                encrypted_weights = self.encrypt_model_weights(weights, f'client_{client_id}')
                encrypted_weights_list.append(encrypted_weights)
            
            # Secure aggregation
            aggregated_weights = self.secure_aggregation(encrypted_weights_list)
            
            # Update global model
            if aggregated_weights:
                global_model.set_weights(aggregated_weights)
            
            # Evaluate global model
            global_accuracy = self.evaluate_model(global_model, client_data_list)
            training_history.append(global_accuracy)
            
            print(f"  Global accuracy: {global_accuracy:.4f}")
        
        return training_history
    
    def simulate_local_training(self, global_model, client_data):
        """Simulate local training on client data"""
        # Create local model copy
        local_model = tf.keras.models.clone_model(global_model)
        local_model.set_weights(global_model.get_weights())
        
        # Simulate training (simplified)
        x_train, y_train = client_data
        
        # Add some noise to simulate training
        weights = local_model.get_weights()
        for i in range(len(weights)):
            noise = np.random.normal(0, 0.01, weights[i].shape)
            weights[i] += noise
        
        local_model.set_weights(weights)
        
        return local_model
    
    def evaluate_model(self, model, client_data_list):
        """Evaluate model on all client data"""
        total_correct = 0
        total_samples = 0
        
        for client_data in client_data_list:
            x_test, y_test = client_data
            predictions = model.predict(x_test, verbose=0)
            predictions = (predictions > 0.5).astype(int)
            
            correct = np.sum(predictions.flatten() == y_test)
            total_correct += correct
            total_samples += len(y_test)
        
        return total_correct / total_samples if total_samples > 0 else 0.0

# Usage example
secure_fl = SecureFederatedLearning(num_clients=5)

# Generate synthetic client data
client_data_list = []
for i in range(5):
    x = np.random.rand(1000, 28, 28)
    y = np.random.randint(0, 2, 1000)
    client_data_list.append((x, y))

# Create global model
global_model = create_simple_model()

# Run secure federated training
secure_training_history = secure_fl.simulate_secure_training(global_model, client_data_list)

print("Secure Federated Learning Complete!")
print(f"Final accuracy: {secure_training_history[-1]:.4f}")
```

### 3. Differential Privacy in Federated Learning

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import random

class DifferentialPrivateFL:
    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_scale = self.calculate_noise_scale()
    
    def calculate_noise_scale(self):
        """Calculate noise scale for differential privacy"""
        # Using Gaussian mechanism
        noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return noise_scale
    
    def add_dp_noise(self, gradients):
        """Add differential privacy noise to gradients"""
        noise = np.random.normal(0, self.noise_scale, gradients.shape)
        return gradients + noise
    
    def clip_gradients(self, gradients, clip_norm=1.0):
        """Clip gradients to bound sensitivity"""
        norm = np.linalg.norm(gradients)
        if norm > clip_norm:
            gradients = gradients * clip_norm / norm
        return gradients
    
    def train_with_dp(self, model, x_train, y_train, epochs=5):
        """Train model with differential privacy"""
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        
        for epoch in range(epochs):
            # Forward pass
            with tf.GradientTape() as tape:
                predictions = model(x_train)
                loss = tf.keras.losses.binary_crossentropy(y_train, predictions)
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Clip gradients
            clipped_gradients = []
            for grad in gradients:
                clipped_gradients.append(self.clip_gradients(grad.numpy()))
            
            # Add differential privacy noise
            noisy_gradients = []
            for grad in clipped_gradients:
                noisy_gradients.append(self.add_dp_noise(grad))
            
            # Apply gradients
            for grad, var in zip(noisy_gradients, model.trainable_variables):
                var.assign_sub(0.01 * grad)
    
    def federated_learning_with_dp(self, global_model, client_data_list, rounds=10):
        """Run federated learning with differential privacy"""
        training_history = []
        
        for round_num in range(rounds):
            print(f"DP Federated Round {round_num + 1}")
            
            # Train on each client with DP
            client_models = []
            
            for client_id, client_data in enumerate(client_data_list):
                # Create local model copy
                local_model = tf.keras.models.clone_model(global_model)
                local_model.set_weights(global_model.get_weights())
                
                # Train with differential privacy
                x_train, y_train = client_data
                self.train_with_dp(local_model, x_train, y_train, epochs=3)
                
                client_models.append(local_model)
            
            # Aggregate models
            aggregated_weights = self.aggregate_models(client_models)
            global_model.set_weights(aggregated_weights)
            
            # Evaluate
            accuracy = self.evaluate_model(global_model, client_data_list)
            training_history.append(accuracy)
            
            print(f"  Global accuracy: {accuracy:.4f}")
            print(f"  Privacy budget used: {(round_num + 1) * self.epsilon:.2f}")
        
        return training_history
    
    def aggregate_models(self, client_models):
        """Aggregate client models"""
        if not client_models:
            return None
        
        weights = [model.get_weights() for model in client_models]
        
        # Average weights
        averaged_weights = []
        for layer_idx in range(len(weights[0])):
            layer_weights = np.array([client_weights[layer_idx] for client_weights in weights])
            averaged_layer = np.mean(layer_weights, axis=0)
            averaged_weights.append(averaged_layer)
        
        return averaged_weights
    
    def evaluate_model(self, model, client_data_list):
        """Evaluate model on all client data"""
        total_correct = 0
        total_samples = 0
        
        for client_data in client_data_list:
            x_test, y_test = client_data
            predictions = model.predict(x_test, verbose=0)
            predictions = (predictions > 0.5).astype(int)
            
            correct = np.sum(predictions.flatten() == y_test)
            total_correct += correct
            total_samples += len(y_test)
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def calculate_privacy_budget(self, rounds, epsilon_per_round):
        """Calculate total privacy budget used"""
        total_epsilon = rounds * epsilon_per_round
        return total_epsilon

# Usage example
dp_fl = DifferentialPrivateFL(epsilon=0.5, delta=1e-5)

# Generate client data
client_data_list = []
for i in range(5):
    x = np.random.rand(1000, 28, 28)
    y = np.random.randint(0, 2, 1000)
    client_data_list.append((x, y))

# Create global model
global_model = create_simple_model()

# Run federated learning with differential privacy
dp_training_history = dp_fl.federated_learning_with_dp(global_model, client_data_list, rounds=5)

print("Differential Private Federated Learning Complete!")
print(f"Final accuracy: {dp_training_history[-1]:.4f}")
print(f"Total privacy budget used: {dp_fl.calculate_privacy_budget(5, 0.5):.2f}")
```

---

## 🎯 Applications

### 1. Healthcare Federated Learning

**Google Health's FL Implementation:**
- Training medical AI models across hospitals
- Preserving patient privacy
- Improving diagnostic accuracy
- 20+ healthcare partners

**Owkin's Federated Learning Platform:**
- Cancer research collaboration
- Drug discovery acceleration
- Multi-institutional studies
- Regulatory compliance

### 2. Financial Services

**JPMorgan Chase's FL System:**
- Fraud detection across branches
- Risk assessment models
- Compliance with data regulations
- Real-time threat detection

**Federated Learning for Credit Scoring:**
- Cross-bank collaboration
- Privacy-preserving credit assessment
- Regulatory compliance
- Improved risk models

### 3. Mobile and Edge Computing

**Google's Federated Learning for Mobile:**
- Keyboard prediction improvement
- On-device personalization
- Privacy-preserving recommendations
- 100+ million devices

**Apple's Federated Learning:**
- Siri voice recognition
- Health data analysis
- Privacy-preserving analytics
- On-device intelligence

### 4. IoT and Smart Cities

**Smart City FL Applications:**
- Traffic prediction across cities
- Energy consumption optimization
- Environmental monitoring
- Privacy-preserving urban analytics

**Industrial IoT FL:**
- Predictive maintenance
- Quality control optimization
- Supply chain optimization
- Cross-factory collaboration

---

## 🧪 Exercises and Projects

### Exercise 1: Basic Federated Learning

**Task**: Implement federated averaging on MNIST dataset.

**Requirements**:
- Split data among 5 clients
- Implement FedAvg algorithm
- Compare with centralized training
- Analyze convergence behavior

### Exercise 2: Secure Aggregation

**Task**: Implement secure aggregation using homomorphic encryption.

**Components**:
- Additive homomorphic encryption
- Secret sharing protocols
- Secure multi-party computation
- Privacy guarantees analysis

### Exercise 3: Differential Privacy in FL

**Task**: Add differential privacy to federated learning.

**Requirements**:
- Implement Gaussian mechanism
- Privacy budget management
- Accuracy-privacy trade-off analysis
- Adaptive noise scaling

### Project: Cross-Silo Federated Learning System

**Objective**: Build a complete federated learning system for healthcare.

**Components**:
1. **Data Distribution**: Heterogeneous data across hospitals
2. **Model Training**: Federated averaging with privacy
3. **Secure Aggregation**: Homomorphic encryption
4. **Privacy Analysis**: Differential privacy guarantees
5. **Performance Evaluation**: Accuracy and privacy metrics

**Implementation Steps**:
```python
# 1. Data management
class HealthcareFLDataManager:
    def distribute_data(self, dataset, hospitals):
        # Distribute medical data across hospitals
        pass
    
    def ensure_privacy(self, data):
        # Apply privacy-preserving techniques
        pass

# 2. Federated training
class HealthcareFLTrainer:
    def train_local_models(self, hospitals):
        # Train models on local hospital data
        pass
    
    def aggregate_models(self, local_models):
        # Securely aggregate hospital models
        pass

# 3. Privacy analysis
class PrivacyAnalyzer:
    def calculate_privacy_budget(self, training_rounds):
        # Calculate total privacy budget used
        pass
    
    def analyze_privacy_guarantees(self):
        # Analyze privacy guarantees
        pass
```

### Quiz Questions

1. **What is the main advantage of federated learning?**
   - A) Faster training
   - B) Privacy-preserving collaborative training
   - C) Lower computational costs
   - D) Better model accuracy

2. **Which algorithm is the foundation of federated learning?**
   - A) Stochastic gradient descent
   - B) Federated averaging (FedAvg)
   - C) Adam optimizer
   - D) Backpropagation

3. **What is the primary challenge in federated learning?**
   - A) High computational costs
   - B) Communication overhead and privacy
   - C) Model complexity
   - D) Data availability

**Answers**: 1-B, 2-B, 3-B

---

## 📖 Further Reading

### Essential Papers
1. **"Communication-Efficient Learning of Deep Networks from Decentralized Data"** - McMahan et al. (2017)
2. **"Practical Secure Aggregation for Privacy-Preserving Machine Learning"** - Bonawitz et al. (2017)
3. **"Differential Privacy for Federated Learning"** - Wei et al. (2020)

### Books
1. **"Federated Learning: Theory and Practice"** - Li et al. (2022)
2. **"Privacy-Preserving Machine Learning"** - Dwork & Roth (2014)
3. **"Secure Multi-Party Computation"** - Goldreich (2002)

### Online Resources
1. **TensorFlow Federated**: https://www.tensorflow.org/federated
2. **PySyft**: https://github.com/OpenMined/PySyft
3. **FedML**: https://fedml.ai/

### Next Steps
1. **Advanced Topics**: Explore federated analytics
2. **Related Modules**: 
   - [Privacy-Preserving ML](advanced_topics/51_ai_ethics_safety.md)
   - [Edge AI](infrastructure/49_edge_ai.md)
   - [AI Security](ai_security/32_ai_security_fundamentals.md)

---

## 🎯 Key Takeaways

1. **Privacy Preservation**: Federated learning enables collaborative AI without data centralization
2. **Secure Aggregation**: Cryptographic techniques protect model updates during aggregation
3. **Differential Privacy**: Mathematical guarantees for privacy in distributed learning
4. **Cross-Silo Applications**: Healthcare, finance, and enterprise collaboration
5. **Edge Computing**: On-device training for mobile and IoT applications
6. **Regulatory Compliance**: Meets GDPR, HIPAA, and other privacy regulations

---

*"Federated learning enables AI collaboration while preserving the privacy of distributed data."*

**Next: [AI Ethics & Safety](advanced_topics/51_ai_ethics_safety.md) → Alignment, robustness, and responsible AI development**