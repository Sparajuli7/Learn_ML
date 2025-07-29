# Ray Deep Dive

## Overview
Ray is a unified framework for distributed computing and ML that scales from laptops to clusters. This guide covers Ray Core, Ray Train, Ray Serve, and production deployments for 2025.

## Table of Contents
1. [Ray Fundamentals](#ray-fundamentals)
2. [Ray Core](#ray-core)
3. [Ray Train](#ray-train)
4. [Ray Serve](#ray-serve)
5. [Ray Tune](#ray-tune)
6. [Production Deployments](#production-deployments)

## Ray Fundamentals

### Basic Ray Setup
```python
import ray
import numpy as np
from typing import Dict, List, Any
import time

# Initialize Ray
ray.init()

# Basic remote function
@ray.remote
def square(x):
    return x * x

# Remote function with resources
@ray.remote(num_cpus=2, num_gpus=1)
def gpu_compute(data):
    # GPU computation
    return np.array(data) ** 2

# Remote class
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value

# Usage
futures = [square.remote(i) for i in range(10)]
results = ray.get(futures)
print(f"Results: {results}")

# Remote object
counter = Counter.remote()
futures = [counter.increment.remote() for _ in range(5)]
values = ray.get(futures)
print(f"Counter values: {values}")
```

### Ray Object Store
```python
import ray
import numpy as np

# Put objects in object store
array_ref = ray.put(np.random.rand(1000, 1000))

# Get objects from object store
array = ray.get(array_ref)

# Pass object references to remote functions
@ray.remote
def process_array(array_ref):
    array = ray.get(array_ref)
    return np.mean(array)

# Use object references
result = process_array.remote(array_ref)
mean_value = ray.get(result)
print(f"Mean value: {mean_value}")
```

## Ray Core

### Distributed Computing Patterns
```python
import ray
import numpy as np
from typing import List, Dict, Any

# Map-Reduce Pattern
@ray.remote
def map_function(data_chunk):
    """Process a chunk of data"""
    return np.sum(data_chunk)

@ray.remote
def reduce_function(results):
    """Combine results from map operations"""
    return np.sum(results)

def map_reduce_example(data: List[np.ndarray]):
    """Map-reduce example using Ray"""
    # Map phase
    map_futures = [map_function.remote(chunk) for chunk in data]
    map_results = ray.get(map_futures)
    
    # Reduce phase
    final_result = ray.get(reduce_function.remote(map_results))
    return final_result

# Pipeline Pattern
@ray.remote
class PipelineStage:
    def __init__(self, stage_id):
        self.stage_id = stage_id
    
    def process(self, data):
        # Simulate processing
        time.sleep(0.1)
        return f"Stage {self.stage_id}: {data}"

def pipeline_example():
    """Pipeline example using Ray"""
    stages = [PipelineStage.remote(i) for i in range(3)]
    
    # Create pipeline
    data = "input"
    for stage in stages:
        data = stage.process.remote(data)
    
    result = ray.get(data)
    return result

# Actor Pattern
@ray.remote
class WorkerPool:
    def __init__(self, num_workers):
        self.workers = [Worker.remote() for _ in range(num_workers)]
        self.current_worker = 0
    
    def submit_task(self, task):
        """Submit task to next available worker"""
        worker = self.workers[self.current_worker]
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return worker.process.remote(task)
    
    def get_results(self, futures):
        """Get results from workers"""
        return ray.get(futures)

@ray.remote
class Worker:
    def __init__(self):
        self.processed_count = 0
    
    def process(self, task):
        """Process a task"""
        self.processed_count += 1
        return f"Processed task {task} (total: {self.processed_count})"
```

### Resource Management
```python
import ray
from typing import Dict, Any

# Resource-aware remote functions
@ray.remote(num_cpus=2, num_gpus=1, memory=1000*1024*1024)  # 1GB memory
def resource_intensive_task(data):
    """Task that requires specific resources"""
    # Simulate GPU computation
    result = np.array(data) ** 2
    return result

# Custom resources
@ray.remote(resources={"custom_resource": 1})
def custom_resource_task():
    """Task that requires custom resource"""
    return "Custom resource task completed"

# Resource placement
@ray.remote(num_cpus=1)
class ResourceAwareActor:
    def __init__(self, node_id):
        self.node_id = node_id
    
    def get_node_info(self):
        return f"Running on node: {self.node_id}"

# Resource monitoring
def monitor_resources():
    """Monitor cluster resources"""
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    
    print(f"Cluster resources: {cluster_resources}")
    print(f"Available resources: {available_resources}")
    
    return cluster_resources, available_resources
```

## Ray Train

### Distributed Training Setup
```python
import ray
from ray import train
import torch
import torch.nn as nn
from typing import Dict, Any

# Define model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Training function
def train_func(config: Dict[str, Any]):
    """Training function for Ray Train"""
    # Get dataset
    dataset = train.get_dataset_shard("train")
    
    # Create model
    model = SimpleModel()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        total_loss = 0
        num_batches = 0
        
        for batch in dataset.iter_batches(batch_size=config["batch_size"]):
            x, y = batch["x"], batch["y"]
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Report metrics
        avg_loss = total_loss / num_batches
        train.report({"loss": avg_loss, "epoch": epoch})
    
    # Save model
    train.save_checkpoint(model.state_dict())

# Launch training
def launch_distributed_training():
    """Launch distributed training with Ray Train"""
    from ray.train.torch import TorchTrainer
    
    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "lr": 0.001,
            "epochs": 10,
            "batch_size": 32
        },
        scaling_config={"num_workers": 4, "use_gpu": True}
    )
    
    result = trainer.fit()
    return result
```

### Hyperparameter Tuning with Ray Train
```python
import ray
from ray import train, tune
from ray.train.torch import TorchTrainer
import torch.nn as nn

def hyperparameter_tuning():
    """Hyperparameter tuning with Ray Train and Tune"""
    
    def train_with_tune(config):
        """Training function for hyperparameter tuning"""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        
        for epoch in range(config["epochs"]):
            # Training logic
            loss = train_epoch(model, optimizer, config)
            
            # Report to Tune
            tune.report(loss=loss, epoch=epoch)
    
    # Define search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": tune.choice([5, 10, 15])
    }
    
    # Launch tuning
    analysis = tune.run(
        train_with_tune,
        config=search_space,
        num_samples=10,
        scheduler=tune.schedulers.ASHAScheduler(),
        resources_per_trial={"cpu": 2, "gpu": 0.5}
    )
    
    return analysis
```

## Ray Serve

### Model Serving Setup
```python
import ray
from ray import serve
import torch
import torch.nn as nn
from typing import Dict, Any
import json

# Define model
class ServingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Ray Serve deployment
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
class ModelDeployment:
    def __init__(self):
        self.model = ServingModel()
        # Load model weights
        # self.model.load_state_dict(torch.load("model.pth"))
    
    def __call__(self, request):
        """Handle inference request"""
        # Parse request
        data = request.json()
        input_data = torch.tensor(data["input"], dtype=torch.float32)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_data)
        
        # Return response
        return {"prediction": prediction.item()}

# Advanced deployment with batching
@serve.deployment(
    num_replicas=3,
    max_concurrent_queries=10,
    batch_wait_timeout_s=1.0
)
class BatchedModelDeployment:
    def __init__(self):
        self.model = ServingModel()
    
    @serve.batch(batch_size=32, max_wait_time_s=1.0)
    async def __call__(self, request):
        """Handle batched inference requests"""
        data = request.json()
        inputs = torch.tensor(data["inputs"], dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.model(inputs)
        
        return {"predictions": predictions.tolist()}

# Deployment with custom metrics
@serve.deployment
class MonitoredModelDeployment:
    def __init__(self):
        self.model = ServingModel()
        self.request_count = 0
    
    def __call__(self, request):
        """Handle request with monitoring"""
        self.request_count += 1
        
        # Record custom metrics
        serve.context.get_internal_replica_context().record_metric(
            "request_count", self.request_count
        )
        
        # Process request
        data = request.json()
        input_data = torch.tensor(data["input"], dtype=torch.float32)
        
        with torch.no_grad():
            prediction = self.model(input_data)
        
        return {"prediction": prediction.item()}

# Deploy and serve
def deploy_model():
    """Deploy model with Ray Serve"""
    # Deploy the model
    deployment = ModelDeployment.bind()
    serve.run(deployment)
    
    # Test the deployment
    import requests
    
    response = requests.post(
        "http://localhost:8000/ModelDeployment",
        json={"input": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    
    return response.json()
```

## Ray Tune

### Hyperparameter Optimization
```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search import OptunaSearch, HyperOptSearch
import torch
import torch.nn as nn

def hyperparameter_optimization():
    """Comprehensive hyperparameter optimization with Ray Tune"""
    
    def trainable(config):
        """Trainable function for hyperparameter optimization"""
        model = SimpleModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"]
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        # Training loop
        for epoch in range(config["epochs"]):
            # Training logic here
            loss = train_epoch(model, optimizer, config)
            
            # Report metrics
            tune.report(
                loss=loss,
                accuracy=compute_accuracy(model),
                epoch=epoch
            )
    
    # Define search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "epochs": tune.choice([10, 20, 30])
    }
    
    # Define scheduler
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="loss",
        mode="min",
        max_t=30,
        grace_period=5,
        reduction_factor=2
    )
    
    # Define search algorithm
    search_alg = OptunaSearch(
        metric="loss",
        mode="min"
    )
    
    # Run optimization
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=50,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        local_dir="./ray_results"
    )
    
    return analysis

# Multi-objective optimization
def multi_objective_optimization():
    """Multi-objective hyperparameter optimization"""
    
    def multi_objective_trainable(config):
        """Trainable with multiple objectives"""
        # Training logic
        model = train_model(config)
        
        # Evaluate multiple metrics
        accuracy = evaluate_accuracy(model)
        latency = measure_latency(model)
        model_size = get_model_size(model)
        
        # Report multiple objectives
        tune.report(
            accuracy=accuracy,
            latency=latency,
            model_size=model_size
        )
    
    # Pareto frontier optimization
    analysis = tune.run(
        multi_objective_trainable,
        config=search_space,
        num_samples=100,
        scheduler=tune.schedulers.PopulationBasedTraining(
            time_attr="epoch",
            metric="accuracy",
            mode="max"
        )
    )
    
    return analysis
```

## Production Deployments

### Ray Cluster Management
```python
import ray
import yaml
from typing import Dict, Any

class RayClusterManager:
    def __init__(self, cluster_config_path: str):
        self.config = self._load_config(cluster_config_path)
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load cluster configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def start_cluster(self):
        """Start Ray cluster"""
        # Initialize Ray with cluster configuration
        ray.init(
            address=self.config["head_node_address"],
            _redis_password=self.config["redis_password"]
        )
    
    def scale_cluster(self, num_workers: int):
        """Scale cluster up or down"""
        # Implementation for scaling
        pass
    
    def monitor_cluster(self):
        """Monitor cluster health"""
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        return {
            "cluster_resources": cluster_resources,
            "available_resources": available_resources,
            "node_count": len(ray.nodes())
        }

# Production deployment configuration
def production_deployment():
    """Production deployment with Ray"""
    
    # 1. Model training with Ray Train
    trainer = TorchTrainer(
        train_func,
        scaling_config={"num_workers": 8, "use_gpu": True},
        run_config={"storage_path": "s3://my-bucket/ray-results"}
    )
    
    result = trainer.fit()
    
    # 2. Model serving with Ray Serve
    deployment = ModelDeployment.bind()
    serve.run(deployment, host="0.0.0.0", port=8000)
    
    # 3. Monitoring and logging
    from ray import dashboard
    
    # Start dashboard
    dashboard.start(host="0.0.0.0", port=8265)
    
    return result

# Kubernetes integration
def kubernetes_deployment():
    """Deploy Ray on Kubernetes"""
    
    # Ray cluster configuration for K8s
    cluster_config = {
        "cluster_name": "ray-cluster",
        "provider": {
            "type": "kubernetes",
            "namespace": "ray",
            "image": "rayproject/ray:latest"
        },
        "head_node": {
            "cpu": 4,
            "memory": 8,
            "gpu": 1
        },
        "worker_nodes": {
            "cpu": 2,
            "memory": 4,
            "gpu": 0.5,
            "min_workers": 2,
            "max_workers": 10
        }
    }
    
    return cluster_config
```

## Conclusion

Ray provides a comprehensive framework for distributed computing and ML. Key areas include:

1. **Ray Core**: Distributed computing primitives and patterns
2. **Ray Train**: Distributed training and hyperparameter tuning
3. **Ray Serve**: Model serving and deployment
4. **Ray Tune**: Hyperparameter optimization
5. **Production Deployments**: Cluster management and monitoring

The framework continues to evolve with new features for more efficient distributed computing and ML workflows.

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Train Guide](https://docs.ray.io/en/latest/train/train.html)
- [Ray Serve Guide](https://docs.ray.io/en/latest/serve/index.html)
- [Ray Tune Guide](https://docs.ray.io/en/latest/tune/index.html) 