# Federated Learning

## Overview
Federated Learning enables training machine learning models across decentralized data sources while preserving data privacy and reducing communication overhead.

## Core Concepts

### 1. Federated Learning Framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import copy

@dataclass
class ClientConfig:
    """Configuration for federated learning client"""
    client_id: str
    data_size: int
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32

@dataclass
class ServerConfig:
    """Configuration for federated learning server"""
    num_rounds: int = 100
    num_clients_per_round: int = 10
    aggregation_method: str = "fedavg"
    min_clients: int = 5

class FederatedServer:
    """Central server for federated learning"""
    
    def __init__(self, global_model: nn.Module, config: ServerConfig):
        self.global_model = global_model
        self.config = config
        self.clients = {}
        self.round_history = []
    
    def register_client(self, client_id: str, client: 'FederatedClient'):
        """Register a client"""
        self.clients[client_id] = client
    
    def select_clients(self, round_num: int) -> List[str]:
        """Select clients for current round"""
        available_clients = list(self.clients.keys())
        
        if len(available_clients) < self.config.min_clients:
            raise ValueError(f"Not enough clients available. Need at least {self.config.min_clients}")
        
        # Random selection
        num_to_select = min(self.config.num_clients_per_round, len(available_clients))
        selected_clients = np.random.choice(available_clients, num_to_select, replace=False)
        
        return selected_clients.tolist()
    
    def aggregate_models(self, client_models: List[nn.Module], 
                        client_weights: List[float]) -> nn.Module:
        """Aggregate client models"""
        if self.config.aggregation_method == "fedavg":
            return self._federated_averaging(client_models, client_weights)
        elif self.config.aggregation_method == "fedprox":
            return self._federated_proximal(client_models, client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
    
    def _federated_averaging(self, client_models: List[nn.Module], 
                            client_weights: List[float]) -> nn.Module:
        """Federated averaging aggregation"""
        # Create new global model
        global_model = copy.deepcopy(self.global_model)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        for name, param in global_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param.data)
        
        # Aggregate parameters
        total_weight = sum(client_weights)
        for client_model, weight in zip(client_models, client_weights):
            for name, param in client_model.named_parameters():
                aggregated_params[name] += (weight / total_weight) * param.data
        
        # Update global model
        for name, param in global_model.named_parameters():
            param.data = aggregated_params[name]
        
        return global_model
    
    def _federated_proximal(self, client_models: List[nn.Module], 
                           client_weights: List[float]) -> nn.Module:
        """Federated proximal aggregation with regularization"""
        # Similar to FedAvg but with proximal term
        return self._federated_averaging(client_models, client_weights)
    
    def train_round(self, round_num: int) -> Dict:
        """Execute one training round"""
        # Select clients
        selected_clients = self.select_clients(round_num)
        
        # Distribute global model to clients
        client_models = []
        client_weights = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Send global model to client
            client_model = copy.deepcopy(self.global_model)
            client.set_model(client_model)
            
            # Train on client
            client.train_local()
            
            # Get updated model and weight
            updated_model = client.get_model()
            client_models.append(updated_model)
            client_weights.append(client.config.data_size)
        
        # Aggregate models
        self.global_model = self.aggregate_models(client_models, client_weights)
        
        # Record round information
        round_info = {
            'round': round_num,
            'selected_clients': selected_clients,
            'num_clients': len(selected_clients)
        }
        self.round_history.append(round_info)
        
        return round_info
    
    def train(self) -> List[Dict]:
        """Train federated model"""
        for round_num in range(self.config.num_rounds):
            round_info = self.train_round(round_num)
            print(f"Round {round_num}: {round_info}")
        
        return self.round_history

class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def set_model(self, model: nn.Module):
        """Set model parameters"""
        self.model.load_state_dict(model.state_dict())
    
    def get_model(self) -> nn.Module:
        """Get current model"""
        return copy.deepcopy(self.model)
    
    def train_local(self):
        """Train model on local data"""
        self.model.train()
        
        for epoch in range(self.config.local_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Client {self.config.client_id}, Epoch {epoch}: Loss = {avg_loss:.4f}")
```

### 2. Privacy-Preserving Federated Learning

```python
class DifferentialPrivacyClient(FederatedClient):
    """Federated client with differential privacy"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader, 
                 noise_scale: float = 1.0, clip_norm: float = 1.0):
        super().__init__(config, model, train_data)
        
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
    
    def train_local(self):
        """Train with differential privacy"""
        self.model.train()
        
        for epoch in range(self.config.local_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients for differential privacy
                self._clip_gradients()
                
                # Add noise to gradients
                self._add_noise_to_gradients()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"DP Client {self.config.client_id}, Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def _clip_gradients(self):
        """Clip gradients to L2 norm"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    
    def _add_noise_to_gradients(self):
        """Add Gaussian noise to gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_scale
                param.grad += noise

class SecureAggregationServer(FederatedServer):
    """Server with secure aggregation"""
    
    def __init__(self, global_model: nn.Module, config: ServerConfig):
        super().__init__(global_model, config)
        self.encryption_scheme = None  # Would implement actual encryption
    
    def aggregate_models_securely(self, client_models: List[nn.Module], 
                                client_weights: List[float]) -> nn.Module:
        """Securely aggregate client models"""
        # This is a simplified implementation
        # In practice, would use homomorphic encryption or secure multiparty computation
        
        # Encrypt client models
        encrypted_models = []
        for model in client_models:
            encrypted_model = self._encrypt_model(model)
            encrypted_models.append(encrypted_model)
        
        # Aggregate encrypted models
        aggregated_encrypted = self._aggregate_encrypted_models(encrypted_models, client_weights)
        
        # Decrypt aggregated model
        global_model = self._decrypt_model(aggregated_encrypted)
        
        return global_model
    
    def _encrypt_model(self, model: nn.Module) -> Dict:
        """Encrypt model parameters"""
        # Simplified encryption
        encrypted_params = {}
        for name, param in model.named_parameters():
            # In practice, would use actual encryption
            encrypted_params[name] = param.data + torch.randn_like(param.data) * 0.1
        return encrypted_params
    
    def _aggregate_encrypted_models(self, encrypted_models: List[Dict], 
                                  client_weights: List[float]) -> Dict:
        """Aggregate encrypted models"""
        # Simplified aggregation
        aggregated = {}
        total_weight = sum(client_weights)
        
        for name in encrypted_models[0].keys():
            aggregated[name] = torch.zeros_like(encrypted_models[0][name])
            for encrypted_model, weight in zip(encrypted_models, client_weights):
                aggregated[name] += (weight / total_weight) * encrypted_model[name]
        
        return aggregated
    
    def _decrypt_model(self, encrypted_model: Dict) -> nn.Module:
        """Decrypt model parameters"""
        # Simplified decryption
        model = copy.deepcopy(self.global_model)
        for name, param in model.named_parameters():
            param.data = encrypted_model[name]
        return model
```

### 3. Advanced Federated Optimization

```python
class FedProxClient(FederatedClient):
    """Federated client with proximal term"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader, mu: float = 0.01):
        super().__init__(config, model, train_data)
        self.mu = mu
        self.global_model_state = None
    
    def set_global_model_state(self, global_state_dict: Dict):
        """Set global model state for proximal term"""
        self.global_model_state = global_state_dict
    
    def train_local(self):
        """Train with proximal term"""
        self.model.train()
        
        for epoch in range(self.config.local_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Add proximal term
                if self.global_model_state is not None:
                    proximal_loss = self._compute_proximal_loss()
                    loss += self.mu * proximal_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"FedProx Client {self.config.client_id}, Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def _compute_proximal_loss(self) -> torch.Tensor:
        """Compute proximal term loss"""
        proximal_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.global_model_state:
                global_param = self.global_model_state[name]
                proximal_loss += torch.norm(param - global_param, p=2) ** 2
        
        return proximal_loss

class FedNovaClient(FederatedClient):
    """Federated client with normalized averaging"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader):
        super().__init__(config, model, train_data)
        self.local_steps = 0
    
    def train_local(self):
        """Train with step counting"""
        self.model.train()
        self.local_steps = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_data):
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.local_steps += 1
        
        print(f"FedNova Client {self.config.client_id}: {self.local_steps} steps")

class FedNovaServer(FederatedServer):
    """Server with normalized averaging"""
    
    def _federated_averaging(self, client_models: List[nn.Module], 
                            client_weights: List[float]) -> nn.Module:
        """Normalized federated averaging"""
        global_model = copy.deepcopy(self.global_model)
        
        # Get client step counts
        client_steps = []
        for client_id in self.clients:
            if hasattr(self.clients[client_id], 'local_steps'):
                client_steps.append(self.clients[client_id].local_steps)
            else:
                client_steps.append(1)  # Default
        
        # Normalize by local steps
        normalized_weights = []
        for weight, steps in zip(client_weights, client_steps):
            normalized_weight = weight / max(steps, 1)
            normalized_weights.append(normalized_weight)
        
        # Aggregate with normalized weights
        aggregated_params = {}
        for name, param in global_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param.data)
        
        total_weight = sum(normalized_weights)
        for client_model, weight in zip(client_models, normalized_weights):
            for name, param in client_model.named_parameters():
                aggregated_params[name] += (weight / total_weight) * param.data
        
        # Update global model
        for name, param in global_model.named_parameters():
            param.data = aggregated_params[name]
        
        return global_model
```

## Communication-Efficient Federated Learning

### 1. Gradient Compression

```python
class GradientCompressionClient(FederatedClient):
    """Client with gradient compression"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader, 
                 compression_ratio: float = 0.1):
        super().__init__(config, model, train_data)
        self.compression_ratio = compression_ratio
    
    def compress_gradients(self) -> Dict:
        """Compress gradients before sending"""
        compressed_gradients = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Top-k sparsification
                grad_flat = param.grad.flatten()
                k = int(len(grad_flat) * self.compression_ratio)
                
                # Keep top-k values
                top_k_values, top_k_indices = torch.topk(torch.abs(grad_flat), k)
                compressed_grad = torch.zeros_like(grad_flat)
                compressed_grad[top_k_indices] = grad_flat[top_k_indices]
                
                compressed_gradients[name] = compressed_grad.reshape(param.grad.shape)
        
        return compressed_gradients
    
    def decompress_gradients(self, compressed_gradients: Dict):
        """Decompress gradients"""
        for name, param in self.model.named_parameters():
            if name in compressed_gradients:
                param.grad = compressed_gradients[name]

class QuantizedClient(FederatedClient):
    """Client with gradient quantization"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader, 
                 num_bits: int = 8):
        super().__init__(config, model, train_data)
        self.num_bits = num_bits
        self.max_value = 2 ** (num_bits - 1) - 1
    
    def quantize_gradients(self) -> Dict:
        """Quantize gradients"""
        quantized_gradients = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Normalize gradients
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0:
                    normalized_grad = param.grad / grad_norm
                else:
                    normalized_grad = param.grad
                
                # Quantize to specified number of bits
                quantized_grad = torch.round(normalized_grad * self.max_value) / self.max_value
                quantized_gradients[name] = quantized_grad * grad_norm
        
        return quantized_gradients
    
    def dequantize_gradients(self, quantized_gradients: Dict):
        """Dequantize gradients"""
        for name, param in self.model.named_parameters():
            if name in quantized_gradients:
                param.grad = quantized_gradients[name]
```

### 2. Asynchronous Federated Learning

```python
class AsynchronousFederatedServer(FederatedServer):
    """Asynchronous federated learning server"""
    
    def __init__(self, global_model: nn.Module, config: ServerConfig):
        super().__init__(global_model, config)
        self.client_updates = {}
        self.update_counter = 0
    
    def receive_update(self, client_id: str, model_update: nn.Module, 
                      client_weight: float):
        """Receive asynchronous update from client"""
        self.client_updates[client_id] = {
            'model': model_update,
            'weight': client_weight,
            'timestamp': self.update_counter
        }
        self.update_counter += 1
    
    def process_updates(self, min_updates: int = 5):
        """Process accumulated updates"""
        if len(self.client_updates) >= min_updates:
            # Aggregate updates
            client_models = []
            client_weights = []
            
            for client_id, update in self.client_updates.items():
                client_models.append(update['model'])
                client_weights.append(update['weight'])
            
            # Update global model
            self.global_model = self.aggregate_models(client_models, client_weights)
            
            # Clear processed updates
            self.client_updates.clear()
            
            return True
        return False

class AsynchronousFederatedClient(FederatedClient):
    """Asynchronous federated learning client"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader, server):
        super().__init__(config, model, train_data)
        self.server = server
        self.async_mode = True
    
    def train_and_send_update(self):
        """Train locally and send update asynchronously"""
        # Train locally
        self.train_local()
        
        # Send update to server
        updated_model = self.get_model()
        self.server.receive_update(
            self.config.client_id, 
            updated_model, 
            self.config.data_size
        )
```

## Federated Learning Applications

### 1. Cross-Device Federated Learning

```python
class CrossDeviceFederatedLearning:
    """Cross-device federated learning system"""
    
    def __init__(self, global_model: nn.Module, server_config: ServerConfig):
        self.server = FederatedServer(global_model, server_config)
        self.devices = {}
    
    def add_device(self, device_id: str, device_data: torch.utils.data.DataLoader, 
                   device_config: ClientConfig):
        """Add a device to the federated learning system"""
        device_client = FederatedClient(device_config, copy.deepcopy(self.server.global_model), device_data)
        self.server.register_client(device_id, device_client)
        self.devices[device_id] = device_client
    
    def train_cross_device(self, num_rounds: int = 100):
        """Train across multiple devices"""
        for round_num in range(num_rounds):
            # Select devices for this round
            selected_devices = self.server.select_clients(round_num)
            
            # Train on selected devices
            for device_id in selected_devices:
                device = self.devices[device_id]
                device.train_local()
            
            # Aggregate updates
            self.server.train_round(round_num)
            
            print(f"Cross-device round {round_num} completed")

class MobileFederatedClient(FederatedClient):
    """Mobile device federated learning client"""
    
    def __init__(self, config: ClientConfig, model: nn.Module, 
                 train_data: torch.utils.data.DataLoader):
        super().__init__(config, model, train_data)
        self.battery_level = 100.0
        self.network_quality = 1.0
    
    def should_participate(self) -> bool:
        """Determine if device should participate based on conditions"""
        # Check battery level
        if self.battery_level < 20.0:
            return False
        
        # Check network quality
        if self.network_quality < 0.5:
            return False
        
        return True
    
    def adaptive_training(self):
        """Adaptive training based on device conditions"""
        if not self.should_participate():
            return False
        
        # Adjust training based on conditions
        if self.battery_level < 50.0:
            # Reduce local epochs
            original_epochs = self.config.local_epochs
            self.config.local_epochs = max(1, original_epochs // 2)
        
        # Train
        self.train_local()
        
        # Restore original config
        self.config.local_epochs = original_epochs
        
        return True
```

### 2. Federated Learning for Healthcare

```python
class HealthcareFederatedLearning:
    """Federated learning for healthcare applications"""
    
    def __init__(self, global_model: nn.Module, server_config: ServerConfig):
        self.server = FederatedServer(global_model, server_config)
        self.hospitals = {}
        self.privacy_level = "high"
    
    def add_hospital(self, hospital_id: str, hospital_data: torch.utils.data.DataLoader, 
                    hospital_config: ClientConfig):
        """Add a hospital to the federated learning system"""
        # Create privacy-preserving client
        hospital_client = DifferentialPrivacyClient(
            hospital_config, 
            copy.deepcopy(self.server.global_model), 
            hospital_data,
            noise_scale=0.1  # Higher noise for healthcare
        )
        
        self.server.register_client(hospital_id, hospital_client)
        self.hospitals[hospital_id] = hospital_client
    
    def train_healthcare_model(self, num_rounds: int = 50):
        """Train healthcare model with privacy guarantees"""
        for round_num in range(num_rounds):
            # Train with differential privacy
            round_info = self.server.train_round(round_num)
            
            # Log privacy metrics
            self._log_privacy_metrics(round_num, round_info)
            
            print(f"Healthcare round {round_num} completed with privacy guarantees")
    
    def _log_privacy_metrics(self, round_num: int, round_info: Dict):
        """Log privacy and performance metrics"""
        # Calculate privacy budget
        epsilon = self._calculate_privacy_budget(round_num)
        
        # Log metrics
        print(f"Round {round_num}: Privacy budget ε = {epsilon:.4f}")
    
    def _calculate_privacy_budget(self, round_num: int) -> float:
        """Calculate privacy budget for differential privacy"""
        # Simplified privacy budget calculation
        base_epsilon = 0.1
        return base_epsilon * (round_num + 1)
```

## Implementation Checklist

### Phase 1: Basic Federated Learning
- [ ] Implement federated server and client
- [ ] Build federated averaging
- [ ] Create client selection
- [ ] Add basic aggregation

### Phase 2: Privacy and Security
- [ ] Add differential privacy
- [ ] Implement secure aggregation
- [ ] Create privacy-preserving protocols
- [ ] Build audit trails

### Phase 3: Advanced Optimization
- [ ] Implement FedProx
- [ ] Add FedNova
- [ ] Create communication-efficient methods
- [ ] Build asynchronous FL

### Phase 4: Applications
- [ ] Add cross-device FL
- [ ] Implement healthcare FL
- [ ] Create mobile FL
- [ ] Build industry-specific FL

## Resources

### Key Papers
- "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- "Federated Learning with Non-IID Data" (FedProx)
- "Federated Learning with Differential Privacy"
- "Federated Learning: Challenges, Methods, and Future Directions"

### Tools and Libraries
- **PySyft**: Privacy-preserving deep learning
- **FedML**: Federated learning framework
- **TensorFlow Federated**: Google's FL framework
- **Flower**: Federated learning framework

### Advanced Topics
- Federated optimization
- Privacy-preserving ML
- Communication efficiency
- Heterogeneous FL
- Federated analytics

This comprehensive guide covers federated learning techniques essential for distributed, privacy-preserving machine learning in 2025. 