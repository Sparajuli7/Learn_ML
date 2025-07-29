# Kubeflow Deep Dive

## Overview
Kubeflow is a machine learning toolkit for Kubernetes that makes ML workflows on Kubernetes simple, portable, and scalable. This guide covers pipelines, components, and production deployments for 2025.

## Table of Contents
1. [Kubeflow Fundamentals](#kubeflow-fundamentals)
2. [Kubeflow Pipelines](#kubeflow-pipelines)
3. [Components and Operations](#components-and-operations)
4. [Training and Serving](#training-and-serving)
5. [Production Deployments](#production-deployments)

## Kubeflow Fundamentals

### Basic Setup
```python
import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from typing import Dict, List, Any
import os

# Initialize Kubeflow client
client = kfp.Client()

# Basic component function
def preprocess_data(input_path: str, output_path: str):
    """Preprocess data component"""
    import pandas as pd
    import numpy as np
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Preprocessing steps
    df = df.dropna()
    df = df.select_dtypes(include=[np.number])
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Create component
preprocess_op = create_component_from_func(
    preprocess_data,
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy']
)
```

### Pipeline Definition
```python
@dsl.pipeline(
    name='ML Pipeline',
    description='A simple ML pipeline'
)
def ml_pipeline(
    input_path: str = '/data/input.csv',
    output_path: str = '/data/output.csv'
):
    """Define ML pipeline"""
    
    # Preprocessing step
    preprocess_task = preprocess_op(
        input_path=input_path,
        output_path=output_path
    )
    
    # Training step
    train_task = train_op(
        input_path=preprocess_task.output,
        model_path='/models/model.pkl'
    )
    
    # Evaluation step
    evaluate_task = evaluate_op(
        model_path=train_task.output,
        test_path='/data/test.csv'
    )
    
    return train_task.output
```

## Kubeflow Pipelines

### Pipeline with Multiple Steps
```python
from kfp import dsl
from kfp.components import create_component_from_func
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Data preprocessing component
def preprocess_data(input_path: str, train_path: str, test_path: str):
    """Preprocess and split data"""
    df = pd.read_csv(input_path)
    
    # Clean data
    df = df.dropna()
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save splits
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Data split saved: {train_path}, {test_path}")

# Training component
def train_model(train_path: str, model_path: str):
    """Train machine learning model"""
    df = pd.read_csv(train_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")

# Evaluation component
def evaluate_model(model_path: str, test_path: str, metrics_path: str):
    """Evaluate model performance"""
    import json
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load model and test data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(test_path)
    X_test = df.drop('target', axis=1)
    y_test = df['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Accuracy: {accuracy}")
    print(f"Metrics saved to {metrics_path}")

# Create components
preprocess_op = create_component_from_func(
    preprocess_data,
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)

train_op = create_component_from_func(
    train_model,
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)

evaluate_op = create_component_from_func(
    evaluate_model,
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)

# Define pipeline
@dsl.pipeline(
    name='ML Training Pipeline',
    description='Complete ML training pipeline'
)
def ml_training_pipeline(
    input_path: str = '/data/dataset.csv',
    model_path: str = '/models/model.pkl',
    metrics_path: str = '/metrics/metrics.json'
):
    """Complete ML training pipeline"""
    
    # Preprocessing
    preprocess_task = preprocess_op(
        input_path=input_path,
        train_path='/data/train.csv',
        test_path='/data/test.csv'
    )
    
    # Training
    train_task = train_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path=model_path
    )
    
    # Evaluation
    evaluate_task = evaluate_op(
        model_path=train_task.output,
        test_path=preprocess_task.outputs['test_path'],
        metrics_path=metrics_path
    )
    
    return train_task.output
```

## Components and Operations

### Custom Component with Artifacts
```python
from kfp import dsl
from kfp.components import create_component_from_func
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Component with typed inputs/outputs
def train_model_with_artifacts(
    train_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics]
):
    """Train model with typed artifacts"""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    import json
    
    # Load training data
    df = pd.read_csv(train_dataset.path)
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train model
    model_obj = RandomForestClassifier(n_estimators=100, random_state=42)
    model_obj.fit(X, y)
    
    # Save model
    with open(model.path, 'wb') as f:
        pickle.dump(model_obj, f)
    
    # Calculate and save metrics
    train_accuracy = model_obj.score(X, y)
    metrics_dict = {
        'train_accuracy': train_accuracy,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    with open(metrics.path, 'w') as f:
        json.dump(metrics_dict, f)
    
    print(f"Model saved to {model.path}")
    print(f"Metrics saved to {metrics.path}")

# Create component
train_with_artifacts_op = create_component_from_func(
    train_model_with_artifacts,
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)
```

### Conditional Pipeline
```python
@dsl.pipeline(
    name='Conditional ML Pipeline',
    description='Pipeline with conditional execution'
)
def conditional_pipeline(
    input_path: str = '/data/dataset.csv',
    threshold: float = 0.8
):
    """Pipeline with conditional steps"""
    
    # Preprocessing
    preprocess_task = preprocess_op(
        input_path=input_path,
        train_path='/data/train.csv',
        test_path='/data/test.csv'
    )
    
    # Training
    train_task = train_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path='/models/model.pkl'
    )
    
    # Evaluation
    evaluate_task = evaluate_op(
        model_path=train_task.output,
        test_path=preprocess_task.outputs['test_path'],
        metrics_path='/metrics/metrics.json'
    )
    
    # Conditional deployment
    with dsl.Condition(evaluate_task.output > threshold):
        deploy_task = deploy_op(
            model_path=train_task.output,
            deployment_name='ml-model'
        )
    
    return train_task.output
```

### Parallel Pipeline
```python
@dsl.pipeline(
    name='Parallel ML Pipeline',
    description='Pipeline with parallel execution'
)
def parallel_pipeline(
    input_path: str = '/data/dataset.csv'
):
    """Pipeline with parallel steps"""
    
    # Preprocessing
    preprocess_task = preprocess_op(
        input_path=input_path,
        train_path='/data/train.csv',
        test_path='/data/test.csv'
    )
    
    # Parallel training with different algorithms
    rf_task = train_rf_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path='/models/rf_model.pkl'
    )
    
    xgb_task = train_xgb_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path='/models/xgb_model.pkl'
    )
    
    # Parallel evaluation
    rf_eval_task = evaluate_op(
        model_path=rf_task.output,
        test_path=preprocess_task.outputs['test_path'],
        metrics_path='/metrics/rf_metrics.json'
    )
    
    xgb_eval_task = evaluate_op(
        model_path=xgb_task.output,
        test_path=preprocess_task.outputs['test_path'],
        metrics_path='/metrics/xgb_metrics.json'
    )
    
    # Compare models
    compare_task = compare_models_op(
        metrics1_path=rf_eval_task.output,
        metrics2_path=xgb_eval_task.output,
        comparison_path='/metrics/comparison.json'
    )
    
    return compare_task.output
```

## Training and Serving

### Distributed Training
```python
from kfp import dsl
from kfp.components import create_component_from_func

def distributed_training(
    train_path: str,
    model_path: str,
    num_workers: int = 4
):
    """Distributed training with multiple workers"""
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import pandas as pd
    import pickle
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    
    # Load data
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(X.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, len(set(y)))
    )
    
    # Wrap model with DDP
    model = DDP(model)
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Save model
    if dist.get_rank() == 0:
        with open(model_path, 'wb') as f:
            pickle.dump(model.module, f)
    
    dist.destroy_process_group()

# Create distributed training component
distributed_train_op = create_component_from_func(
    distributed_training,
    base_image='pytorch/pytorch:latest',
    packages_to_install=['pandas', 'numpy']
)
```

### Model Serving
```python
from kfp import dsl
from kfp.components import create_component_from_func

def deploy_model(
    model_path: str,
    service_name: str,
    replicas: int = 3
):
    """Deploy model as Kubernetes service"""
    from kubernetes import client, config
    import yaml
    
    # Load kubeconfig
    config.load_kube_config()
    
    # Create deployment
    deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=service_name),
        spec=client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": service_name}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": service_name}
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name=service_name,
                            image="ml-serving:latest",
                            ports=[client.V1ContainerPort(container_port=8080)],
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="model-volume",
                                    mount_path="/models"
                                )
                            ]
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="model-volume",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name="model-pvc"
                            )
                        )
                    ]
                )
            )
        )
    )
    
    # Create service
    service = client.V1Service(
        metadata=client.V1ObjectMeta(name=service_name),
        spec=client.V1ServiceSpec(
            selector={"app": service_name},
            ports=[client.V1ServicePort(port=8080)]
        )
    )
    
    # Apply to cluster
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    
    apps_v1.create_namespaced_deployment(
        namespace="default", body=deployment
    )
    core_v1.create_namespaced_service(
        namespace="default", body=service
    )
    
    print(f"Model deployed as service: {service_name}")

# Create deployment component
deploy_op = create_component_from_func(
    deploy_model,
    base_image='python:3.9',
    packages_to_install=['kubernetes']
)
```

## Production Deployments

### Pipeline with Monitoring
```python
@dsl.pipeline(
    name='Production ML Pipeline',
    description='Production pipeline with monitoring'
)
def production_pipeline(
    input_path: str = '/data/dataset.csv',
    model_path: str = '/models/model.pkl'
):
    """Production pipeline with monitoring and validation"""
    
    # Data validation
    validate_task = validate_data_op(
        input_path=input_path,
        validation_report='/reports/validation.json'
    )
    
    # Preprocessing
    preprocess_task = preprocess_op(
        input_path=input_path,
        train_path='/data/train.csv',
        test_path='/data/test.csv'
    ).after(validate_task)
    
    # Training with monitoring
    train_task = train_with_monitoring_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path=model_path,
        metrics_path='/metrics/training_metrics.json'
    )
    
    # Model validation
    validate_model_task = validate_model_op(
        model_path=train_task.output,
        test_path=preprocess_task.outputs['test_path'],
        validation_report='/reports/model_validation.json'
    )
    
    # Deployment with rollback capability
    with dsl.Condition(validate_model_task.output['is_valid']):
        deploy_task = deploy_with_rollback_op(
            model_path=train_task.output,
            service_name='ml-model',
            rollback_version='/models/previous_model.pkl'
        )
    
    return train_task.output
```

### Pipeline with Resource Management
```python
@dsl.pipeline(
    name='Resource-Managed ML Pipeline',
    description='Pipeline with resource management'
)
def resource_managed_pipeline(
    input_path: str = '/data/dataset.csv'
):
    """Pipeline with resource management and scheduling"""
    
    # Data preprocessing with resource limits
    preprocess_task = preprocess_op(
        input_path=input_path,
        train_path='/data/train.csv',
        test_path='/data/test.csv'
    ).set_cpu_limit('2').set_memory_limit('4Gi')
    
    # Training with GPU
    train_task = train_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path='/models/model.pkl'
    ).set_gpu_limit(1).set_cpu_limit('4').set_memory_limit('8Gi')
    
    # Evaluation
    evaluate_task = evaluate_op(
        model_path=train_task.output,
        test_path=preprocess_task.outputs['test_path'],
        metrics_path='/metrics/metrics.json'
    ).set_cpu_limit('1').set_memory_limit('2Gi')
    
    return train_task.output
```

### Pipeline with Caching
```python
@dsl.pipeline(
    name='Cached ML Pipeline',
    description='Pipeline with caching for efficiency'
)
def cached_pipeline(
    input_path: str = '/data/dataset.csv'
):
    """Pipeline with caching for repeated steps"""
    
    # Preprocessing with caching
    preprocess_task = preprocess_op(
        input_path=input_path,
        train_path='/data/train.csv',
        test_path='/data/test.csv'
    ).set_caching_options(enable_cache=True)
    
    # Training with caching
    train_task = train_op(
        train_path=preprocess_task.outputs['train_path'],
        model_path='/models/model.pkl'
    ).set_caching_options(enable_cache=True)
    
    # Evaluation
    evaluate_task = evaluate_op(
        model_path=train_task.output,
        test_path=preprocess_task.outputs['test_path'],
        metrics_path='/metrics/metrics.json'
    ).set_caching_options(enable_cache=True)
    
    return train_task.output
```

## Conclusion

Kubeflow provides a comprehensive platform for ML orchestration on Kubernetes. Key areas include:

1. **Pipelines**: Complex ML workflows with conditional and parallel execution
2. **Components**: Reusable ML operations with typed inputs/outputs
3. **Training**: Distributed training and hyperparameter tuning
4. **Serving**: Model deployment and serving
5. **Production**: Monitoring, resource management, and caching

The platform continues to evolve with new features for more efficient ML workflows and production deployments.

## Resources

- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [Kubeflow Training](https://www.kubeflow.org/docs/components/training/)
- [Kubeflow Serving](https://www.kubeflow.org/docs/components/serving/) 