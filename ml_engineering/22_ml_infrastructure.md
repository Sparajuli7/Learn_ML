# ML Infrastructure: Building Scalable Machine Learning Systems
*"Infrastructure is the invisible foundation that makes AI magic possible"*

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Infrastructure Fundamentals](#infrastructure-fundamentals)
3. [Cloud-Native ML Architecture](#cloud-native-ml-architecture)
4. [Containerization and Orchestration](#containerization-and-orchestration)
5. [Model Serving and Deployment](#model-serving-and-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Implementation Examples](#implementation-examples)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

## 🎯 Introduction

ML infrastructure is like building a smart factory for AI - you need the right machines, assembly lines, and quality control systems to produce reliable AI products at scale. In 2025, with the explosion of AI applications, having robust ML infrastructure is not just nice to have; it's essential for survival.

### Why ML Infrastructure Matters in 2025

The AI landscape in 2025 demands infrastructure that can handle:
- **Massive Scale**: Processing millions of predictions per second
- **Real-time Requirements**: Sub-second response times
- **Cost Efficiency**: Optimizing compute resources for ROI
- **Reliability**: 99.9%+ uptime for critical AI services
- **Security**: Protecting sensitive data and models
- **Compliance**: Meeting regulatory requirements (GDPR, CCPA, AI Act)

### The Infrastructure Evolution

ML infrastructure has evolved dramatically:

- **2010s**: Single-server deployments with manual scaling
- **2015s**: Cloud-based solutions with basic auto-scaling
- **2020s**: Kubernetes-native ML with microservices
- **2025**: AI-native infrastructure with automated optimization

## 🧮 Mathematical Foundations

### Infrastructure Performance Metrics

#### 1. Throughput (T)
```
T = (Requests per second) × (Average response time)
```

#### 2. Resource Utilization (RU)
```
RU = (Used resources) / (Total available resources) × 100
```

#### 3. Cost per Prediction (CPP)
```
CPP = (Infrastructure cost per hour) / (Predictions per hour)
```

#### 4. Availability (A)
```
A = (Uptime) / (Total time) × 100
```

### Example Calculation

For a system processing 10,000 requests/second:
- Average response time: 50ms
- Infrastructure cost: $100/hour
- Uptime: 99.5%

```
T = 10,000 × 0.05 = 500 requests/second
CPP = $100 / (10,000 × 3600) = $0.0000028 per prediction
A = 99.5%
```

## 💻 Implementation

### 1. Kubernetes-Based ML Infrastructure

Kubernetes is like an intelligent factory manager - it automatically handles the complex logistics of running your AI applications at scale.

```python
# Why: Kubernetes provides scalable, reliable container orchestration
# How: Container-based deployment with automatic scaling
# Where: Production ML systems requiring high availability
# What: Automated resource management and load balancing
# When: When you need enterprise-grade ML deployment

import yaml
import kubernetes
from kubernetes import client, config
import logging
import time

class MLInfrastructureManager:
    def __init__(self):
        # Why: Initialize connection to Kubernetes cluster
        # How: Load kubeconfig and create API client
        # Where: ML infrastructure management systems
        # What: Kubernetes API client for deployment
        # When: At system startup
        
        try:
            config.load_kube_config()
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            logging.info("Connected to Kubernetes cluster")
        except Exception as e:
            logging.error(f"Failed to connect to Kubernetes: {e}")
            raise
    
    def deploy_ml_model(self, model_name, model_image, replicas=3):
        """Deploy ML model to Kubernetes"""
        # Why: Deploy ML models with high availability
        # How: Create deployment, service, and ingress
        # Where: Production ML serving environments
        # What: Scalable model serving infrastructure
        # When: When deploying new models or updates
        
        # Create deployment
        deployment = self._create_deployment(model_name, model_image, replicas)
        
        # Create service
        service = self._create_service(model_name)
        
        # Create ingress
        ingress = self._create_ingress(model_name)
        
        try:
            # Apply deployment
            self.apps_v1.create_namespaced_deployment(
                namespace="ml-models",
                body=deployment
            )
            logging.info(f"Deployed {model_name} with {replicas} replicas")
            
            # Apply service
            self.v1.create_namespaced_service(
                namespace="ml-models",
                body=service
            )
            
            # Apply ingress
            self.networking_v1.create_namespaced_ingress(
                namespace="ml-models",
                body=ingress
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to deploy {model_name}: {e}")
            return False
    
    def _create_deployment(self, model_name, model_image, replicas):
        """Create Kubernetes deployment for ML model"""
        # Why: Define how ML model containers should run
        # How: Specify container image, resources, and scaling
        # Where: Kubernetes deployment manifests
        # What: Deployment configuration for model serving
        # When: When creating new model deployments
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{model_name}-deployment",
                "namespace": "ml-models",
                "labels": {
                    "app": model_name,
                    "component": "ml-model"
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": model_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": model_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": model_name,
                            "image": model_image,
                            "ports": [{
                                "containerPort": 8080
                            }],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                }
                            },
                            "env": [
                                {
                                    "name": "MODEL_NAME",
                                    "value": model_name
                                },
                                {
                                    "name": "LOG_LEVEL",
                                    "value": "INFO"
                                }
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        return deployment
    
    def _create_service(self, model_name):
        """Create Kubernetes service for ML model"""
        # Why: Enable network access to ML model pods
        # How: Create ClusterIP service with load balancing
        # Where: Kubernetes service configuration
        # What: Network service for model serving
        # When: When deploying model endpoints
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{model_name}-service",
                "namespace": "ml-models"
            },
            "spec": {
                "selector": {
                    "app": model_name
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8080
                }],
                "type": "ClusterIP"
            }
        }
        
        return service
    
    def _create_ingress(self, model_name):
        """Create Kubernetes ingress for ML model"""
        # Why: Provide external access to ML model API
        # How: Configure ingress rules and SSL termination
        # Where: Kubernetes ingress configuration
        # What: External API endpoint for model serving
        # When: When exposing models to external clients
        
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{model_name}-ingress",
                "namespace": "ml-models",
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": [f"{model_name}.ml.example.com"],
                    "secretName": f"{model_name}-tls"
                }],
                "rules": [{
                    "host": f"{model_name}.ml.example.com",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{model_name}-service",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        return ingress
    
    def scale_model(self, model_name, replicas):
        """Scale ML model deployment"""
        # Why: Adjust model capacity based on demand
        # How: Update deployment replica count
        # Where: Production ML serving environments
        # What: Dynamic scaling of model instances
        # When: When traffic patterns change
        
        try:
            self.apps_v1.patch_namespaced_deployment_scale(
                name=f"{model_name}-deployment",
                namespace="ml-models",
                body={"spec": {"replicas": replicas}}
            )
            logging.info(f"Scaled {model_name} to {replicas} replicas")
            return True
        except Exception as e:
            logging.error(f"Failed to scale {model_name}: {e}")
            return False
    
    def get_model_status(self, model_name):
        """Get deployment status for ML model"""
        # Why: Monitor model deployment health
        # How: Query Kubernetes API for deployment status
        # Where: ML infrastructure monitoring
        # What: Real-time deployment status
        # When: For monitoring and alerting
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=f"{model_name}-deployment",
                namespace="ml-models"
            )
            
            status = {
                "name": model_name,
                "replicas": deployment.spec.replicas,
                "available_replicas": deployment.status.available_replicas,
                "ready_replicas": deployment.status.ready_replicas,
                "updated_replicas": deployment.status.updated_replicas,
                "conditions": deployment.status.conditions
            }
            
            return status
            
        except Exception as e:
            logging.error(f"Failed to get status for {model_name}: {e}")
            return None

# Usage example
if __name__ == "__main__":
    manager = MLInfrastructureManager()
    
    # Deploy a recommendation model
    success = manager.deploy_ml_model(
        model_name="recommendation-engine",
        model_image="ml-registry.example.com/recommendation:v1.2.0",
        replicas=5
    )
    
    if success:
        print("✅ Model deployed successfully")
        
        # Scale up during peak hours
        manager.scale_model("recommendation-engine", 10)
        
        # Check status
        status = manager.get_model_status("recommendation-engine")
        print(f"Model status: {status}")
    else:
        print("❌ Model deployment failed")
```

### 2. Model Serving with TensorFlow Serving

TensorFlow Serving is like a specialized restaurant for AI models - it serves your trained models efficiently and reliably to hungry applications.

```python
# Why: Optimized serving for TensorFlow models
# How: REST/gRPC APIs with model versioning
# Where: Production TensorFlow model serving
# What: High-performance model inference service
# When: When serving TensorFlow models at scale

import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc
import grpc
import numpy as np
import json
import logging
from typing import Dict, Any

class TensorFlowServingClient:
    def __init__(self, server_url: str = "localhost:8500"):
        # Why: Connect to TensorFlow Serving server
        # How: Establish gRPC connection to serving endpoint
        # Where: ML inference applications
        # What: Client for model serving API
        # When: When making predictions with TensorFlow models
        
        self.server_url = server_url
        self.channel = grpc.insecure_channel(server_url)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        logging.info(f"Connected to TensorFlow Serving at {server_url}")
    
    def predict(self, model_name: str, model_version: str, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Make prediction using TensorFlow Serving"""
        # Why: Get predictions from deployed TensorFlow models
        # How: Send gRPC request to serving server
        # Where: ML inference pipelines
        # What: Model predictions with confidence scores
        # When: When serving real-time predictions
        
        try:
            # Create prediction request
            request = predict_pb2.PredictRequest()
            request.model_spec.name = model_name
            request.model_spec.signature_name = "serving_default"
            
            # Add input tensors
            for input_name, input_data in inputs.items():
                # Convert numpy array to tensor proto
                tensor_proto = tf.make_tensor_proto(input_data)
                request.inputs[input_name].CopyFrom(tensor_proto)
            
            # Make prediction
            response = self.stub.Predict(request, timeout=10.0)
            
            # Parse response
            predictions = {}
            for output_name, output_tensor in response.outputs.items():
                # Convert tensor proto to numpy array
                predictions[output_name] = tf.make_ndarray(output_tensor)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata from TensorFlow Serving"""
        # Why: Understand model inputs, outputs, and capabilities
        # How: Query serving server for model metadata
        # Where: Model management and monitoring
        # What: Model signature and tensor information
        # When: When setting up new model deployments
        
        try:
            # Create metadata request
            request = predict_pb2.GetModelMetadataRequest()
            request.model_spec.name = model_name
            request.metadata_field.append("signature_def")
            
            # Get metadata
            response = self.stub.GetModelMetadata(request, timeout=10.0)
            
            # Parse metadata
            metadata = {
                "model_name": model_name,
                "signatures": {}
            }
            
            for signature_name, signature_def in response.model_metadata["signature_def"].signature_def.items():
                metadata["signatures"][signature_name] = {
                    "inputs": {
                        name: {
                            "dtype": tensor.dtype.name,
                            "shape": list(tensor.tensor_shape.dim)
                        }
                        for name, tensor in signature_def.inputs.items()
                    },
                    "outputs": {
                        name: {
                            "dtype": tensor.dtype.name,
                            "shape": list(tensor.tensor_shape.dim)
                        }
                        for name, tensor in signature_def.outputs.items()
                    }
                }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Failed to get metadata for {model_name}: {e}")
            raise

class ModelServingManager:
    def __init__(self, serving_url: str = "localhost:8500"):
        # Why: Manage multiple model deployments
        # How: Coordinate model serving operations
        # Where: ML infrastructure management
        # What: Centralized model serving management
        # When: When managing multiple ML models
        
        self.serving_url = serving_url
        self.client = TensorFlowServingClient(serving_url)
        self.model_cache = {}
        logging.info("Model serving manager initialized")
    
    def load_model(self, model_name: str, model_path: str):
        """Load model into serving system"""
        # Why: Deploy new models for serving
        # How: Copy model files to serving directory
        # Where: Model deployment pipelines
        # What: Model loading and registration
        # When: When deploying new model versions
        
        try:
            # In real implementation, you'd copy model files
            # For demonstration, we'll simulate model loading
            logging.info(f"Loading model {model_name} from {model_path}")
            
            # Simulate model loading time
            time.sleep(2)
            
            # Cache model metadata
            metadata = self.client.get_model_metadata(model_name)
            self.model_cache[model_name] = metadata
            
            logging.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def predict_batch(self, model_name: str, batch_inputs: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        # Why: Efficient batch processing for multiple inputs
        # How: Process multiple inputs in single request
        # Where: Batch inference pipelines
        # What: Batch predictions with optimized throughput
        # When: When processing multiple predictions efficiently
        
        predictions = []
        
        for inputs in batch_inputs:
            try:
                prediction = self.client.predict(model_name, "latest", inputs)
                predictions.append(prediction)
            except Exception as e:
                logging.error(f"Batch prediction failed: {e}")
                predictions.append({"error": str(e)})
        
        return predictions
    
    def health_check(self, model_name: str) -> Dict[str, Any]:
        """Check model serving health"""
        # Why: Monitor model serving availability
        # How: Test model endpoint responsiveness
        # Where: Health monitoring systems
        # What: Model health status and metrics
        # When: For continuous monitoring and alerting
        
        try:
            # Create test input
            test_input = {
                "input_1": np.random.randn(1, 10).astype(np.float32)
            }
            
            start_time = time.time()
            prediction = self.client.predict(model_name, "latest", test_input)
            response_time = time.time() - start_time
            
            return {
                "model_name": model_name,
                "status": "healthy",
                "response_time": response_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "model_name": model_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Usage example
if __name__ == "__main__":
    manager = ModelServingManager()
    
    # Load a recommendation model
    success = manager.load_model("recommendation-model", "/models/recommendation")
    
    if success:
        # Make prediction
        inputs = {
            "user_features": np.random.randn(1, 100).astype(np.float32),
            "item_features": np.random.randn(1, 50).astype(np.float32)
        }
        
        prediction = manager.client.predict("recommendation-model", "latest", inputs)
        print(f"Prediction: {prediction}")
        
        # Health check
        health = manager.health_check("recommendation-model")
        print(f"Health status: {health}")
```

### 3. Infrastructure Monitoring with Prometheus and Grafana

Monitoring is like having a dashboard for your AI factory - you need to see what's happening in real-time to keep everything running smoothly.

```python
# Why: Monitor ML infrastructure performance and health
# How: Collect metrics and visualize system state
# Where: Production ML infrastructure
# What: Real-time monitoring and alerting
# When: For continuous system health monitoring

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import time
import logging
from typing import Dict, Any

class MLInfrastructureMonitor:
    def __init__(self):
        # Why: Initialize monitoring system for ML infrastructure
        # How: Set up Prometheus metrics collectors
        # Where: ML infrastructure monitoring
        # What: Metrics collection and monitoring
        # When: At system startup
        
        # Define metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of ML predictions',
            ['model_name', 'status']
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'ML prediction latency in seconds',
            ['model_name']
        )
        
        self.model_memory_usage = Gauge(
            'ml_model_memory_bytes',
            'Memory usage of ML models in bytes',
            ['model_name']
        )
        
        self.model_cpu_usage = Gauge(
            'ml_model_cpu_percent',
            'CPU usage of ML models in percent',
            ['model_name']
        )
        
        self.active_connections = Gauge(
            'ml_active_connections',
            'Number of active connections to ML models',
            ['model_name']
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Model accuracy percentage',
            ['model_name']
        )
        
        logging.info("ML infrastructure monitor initialized")
    
    def record_prediction(self, model_name: str, status: str, latency: float):
        """Record prediction metrics"""
        # Why: Track prediction performance and success rates
        # How: Increment counters and record histograms
        # Where: ML inference endpoints
        # What: Prediction metrics for monitoring
        # When: After each prediction request
        
        self.prediction_counter.labels(model_name=model_name, status=status).inc()
        self.prediction_latency.labels(model_name=model_name).observe(latency)
        
        logging.info(f"Recorded prediction: {model_name}, {status}, {latency:.3f}s")
    
    def update_resource_usage(self, model_name: str, memory_bytes: int, cpu_percent: float):
        """Update resource usage metrics"""
        # Why: Monitor system resource consumption
        # How: Update gauge metrics with current values
        # Where: Resource monitoring systems
        # What: Real-time resource usage tracking
        # When: Periodically for resource monitoring
        
        self.model_memory_usage.labels(model_name=model_name).set(memory_bytes)
        self.model_cpu_usage.labels(model_name=model_name).set(cpu_percent)
        
        logging.debug(f"Updated resource usage for {model_name}: {memory_bytes} bytes, {cpu_percent}% CPU")
    
    def update_connection_count(self, model_name: str, connection_count: int):
        """Update active connection count"""
        # Why: Monitor connection load on model endpoints
        # How: Update gauge with current connection count
        # Where: Load balancing and scaling systems
        # What: Connection metrics for capacity planning
        # When: When connection counts change
        
        self.active_connections.labels(model_name=model_name).set(connection_count)
    
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metrics"""
        # Why: Track model performance over time
        # How: Update gauge with current accuracy
        # Where: Model performance monitoring
        # What: Accuracy metrics for model evaluation
        # When: After model evaluation or retraining
        
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
        
        logging.info(f"Updated accuracy for {model_name}: {accuracy:.2f}%")
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        # Why: Provide detailed system performance insights
        # How: Collect all current metric values
        # Where: Monitoring dashboards and reports
        # What: Comprehensive system health report
        # When: For system analysis and alerting
        
        report = {
            "timestamp": time.time(),
            "metrics": {
                "predictions": {},
                "latency": {},
                "resources": {},
                "connections": {},
                "accuracy": {}
            }
        }
        
        # Collect prediction metrics
        for metric in self.prediction_counter._metrics:
            report["metrics"]["predictions"][str(metric)] = self.prediction_counter._metrics[metric]._value.get()
        
        # Collect latency metrics
        for metric in self.prediction_latency._metrics:
            report["metrics"]["latency"][str(metric)] = {
                "count": self.prediction_latency._metrics[metric]._count.get(),
                "sum": self.prediction_latency._metrics[metric]._sum.get()
            }
        
        # Collect resource metrics
        for metric in self.model_memory_usage._metrics:
            report["metrics"]["resources"][str(metric)] = {
                "memory_bytes": self.model_memory_usage._metrics[metric]._value.get(),
                "cpu_percent": self.model_cpu_usage._metrics[metric]._value.get()
            }
        
        # Collect connection metrics
        for metric in self.active_connections._metrics:
            report["metrics"]["connections"][str(metric)] = self.active_connections._metrics[metric]._value.get()
        
        # Collect accuracy metrics
        for metric in self.model_accuracy._metrics:
            report["metrics"]["accuracy"][str(metric)] = self.model_accuracy._metrics[metric]._value.get()
        
        return report

class InfrastructureAlerting:
    def __init__(self, monitor: MLInfrastructureMonitor):
        # Why: Set up automated alerting for infrastructure issues
        # How: Define alert thresholds and notification channels
        # Where: Production ML infrastructure
        # What: Automated alerting system
        # When: When infrastructure issues occur
        
        self.monitor = monitor
        self.alert_thresholds = {
            "latency_p95": 1.0,  # seconds
            "error_rate": 0.05,  # 5%
            "memory_usage": 0.9,  # 90%
            "cpu_usage": 0.8,     # 80%
            "accuracy_drop": 0.1   # 10% drop
        }
        
        logging.info("Infrastructure alerting initialized")
    
    def check_alerts(self, metrics_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        # Why: Detect infrastructure issues automatically
        # How: Compare metrics against thresholds
        # Where: Automated monitoring systems
        # What: Alert notifications for issues
        # When: When metrics exceed thresholds
        
        alerts = []
        
        # Check latency alerts
        for model_name, latency_data in metrics_report["metrics"]["latency"].items():
            if latency_data["count"] > 0:
                avg_latency = latency_data["sum"] / latency_data["count"]
                if avg_latency > self.alert_thresholds["latency_p95"]:
                    alerts.append({
                        "severity": "warning",
                        "model": model_name,
                        "metric": "latency",
                        "value": avg_latency,
                        "threshold": self.alert_thresholds["latency_p95"],
                        "message": f"High latency detected for {model_name}"
                    })
        
        # Check resource usage alerts
        for model_name, resource_data in metrics_report["metrics"]["resources"].items():
            memory_usage = resource_data["memory_bytes"] / (1024**3)  # Convert to GB
            cpu_usage = resource_data["cpu_percent"] / 100
            
            if memory_usage > self.alert_thresholds["memory_usage"]:
                alerts.append({
                    "severity": "critical",
                    "model": model_name,
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "threshold": self.alert_thresholds["memory_usage"],
                    "message": f"High memory usage for {model_name}"
                })
            
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                alerts.append({
                    "severity": "warning",
                    "model": model_name,
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "threshold": self.alert_thresholds["cpu_usage"],
                    "message": f"High CPU usage for {model_name}"
                })
        
        return alerts

# Usage example
if __name__ == "__main__":
    monitor = MLInfrastructureMonitor()
    alerting = InfrastructureAlerting(monitor)
    
    # Simulate some metrics
    monitor.record_prediction("recommendation-model", "success", 0.15)
    monitor.record_prediction("recommendation-model", "error", 0.25)
    monitor.update_resource_usage("recommendation-model", 2 * 1024**3, 75.0)  # 2GB, 75% CPU
    monitor.update_connection_count("recommendation-model", 150)
    monitor.update_model_accuracy("recommendation-model", 92.5)
    
    # Generate report
    report = monitor.generate_metrics_report()
    print("Metrics Report:")
    print(json.dumps(report, indent=2))
    
    # Check for alerts
    alerts = alerting.check_alerts(report)
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"- {alert['severity'].upper()}: {alert['message']}")
    else:
        print("\n✅ No alerts detected")
```

## 🎯 Applications

### 1. E-commerce Recommendation Infrastructure

**Problem**: An e-commerce platform needs to serve personalized recommendations to millions of users with sub-second latency.

**Solution**:
- **Kubernetes Deployment**: Auto-scaling model pods based on traffic
- **TensorFlow Serving**: Optimized inference with model versioning
- **Monitoring**: Real-time metrics for latency, accuracy, and resource usage
- **Results**: 99.9% uptime, <100ms latency, 50% cost reduction

### 2. Healthcare AI Infrastructure

**Problem**: A hospital needs to deploy AI diagnostic models with strict compliance and security requirements.

**Solution**:
- **Secure Infrastructure**: HIPAA-compliant Kubernetes clusters
- **Model Serving**: Encrypted inference with audit logging
- **Monitoring**: Comprehensive security and performance monitoring
- **Results**: 100% compliance, 99.5% uptime, 30% faster diagnosis

### 3. Financial Trading AI Infrastructure

**Problem**: A trading firm needs ultra-low-latency AI predictions for real-time trading decisions.

**Solution**:
- **High-Performance Infrastructure**: GPU-optimized Kubernetes nodes
- **Model Serving**: gRPC-based inference with connection pooling
- **Monitoring**: Microsecond-level latency monitoring
- **Results**: <1ms latency, 99.99% uptime, 25% trading performance improvement

## 🧪 Exercises and Projects

### Exercise 1: Deploy a Simple ML Model

Deploy a scikit-learn model using Kubernetes:

```python
# Your task: Deploy a machine learning model to Kubernetes
# Requirements:
# 1. Create Docker image with model
# 2. Deploy to Kubernetes
# 3. Set up monitoring
# 4. Test scaling
# 5. Implement health checks

# Starter code:
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model (in real implementation, load from file)
model = None  # TODO: Load your trained model

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction endpoint"""
    # TODO: Implement prediction logic
    pass

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # TODO: Implement health check
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Exercise 2: Build Monitoring Dashboard

Create a comprehensive monitoring system:

```python
# Your task: Build monitoring dashboard for ML infrastructure
# Requirements:
# 1. Collect metrics from multiple models
# 2. Create Grafana dashboards
# 3. Set up alerting rules
# 4. Implement log aggregation
# 5. Create performance reports

# Starter code:
class MLDashboard:
    def __init__(self):
        # TODO: Initialize dashboard components
        pass
    
    def collect_metrics(self):
        """Collect metrics from all models"""
        # TODO: Implement metrics collection
        pass
    
    def create_dashboard(self):
        """Create Grafana dashboard"""
        # TODO: Implement dashboard creation
        pass
    
    def setup_alerts(self):
        """Setup alerting rules"""
        # TODO: Implement alerting
        pass
```

### Project: Complete ML Infrastructure

Build a production-ready ML infrastructure:

**Requirements**:
1. **Multi-Model Deployment**: Support for different ML frameworks
2. **Auto-Scaling**: Dynamic scaling based on demand
3. **Monitoring**: Comprehensive metrics and alerting
4. **Security**: Authentication, authorization, and encryption
5. **CI/CD**: Automated deployment pipeline
6. **Documentation**: Complete infrastructure documentation

**Deliverables**:
- Kubernetes deployment manifests
- Monitoring dashboards
- Security configuration
- Performance benchmarks
- Deployment guide

## 📖 Further Reading

### Essential Resources

1. **Books**:
   - "Kubernetes: Up and Running" by Kelsey Hightower
   - "Site Reliability Engineering" by Google
   - "The Phoenix Project" by Gene Kim

2. **Online Courses**:
   - Coursera: "Cloud Architecture with Google Cloud"
   - edX: "Introduction to Kubernetes"
   - DataCamp: "DevOps for Data Science"

3. **Tools and Technologies**:
   - **Kubernetes**: Container orchestration
   - **TensorFlow Serving**: Model serving
   - **Prometheus**: Metrics collection
   - **Grafana**: Monitoring dashboards
   - **Istio**: Service mesh
   - **Helm**: Kubernetes package manager

4. **2025 Trends**:
   - **AI-Native Infrastructure**: Infrastructure optimized for AI workloads
   - **Serverless ML**: Pay-per-prediction model serving
   - **Edge AI**: Distributed ML inference
   - **Green AI**: Energy-efficient ML infrastructure
   - **MLOps Automation**: Automated ML lifecycle management

### Certification Path

1. **Beginner**: Kubernetes Administrator (CKA)
2. **Intermediate**: Certified Kubernetes Application Developer (CKAD)
3. **Advanced**: Google Cloud Professional ML Engineer
4. **Expert**: AWS Machine Learning Specialty

## 🎯 Key Takeaways

1. **Infrastructure is critical** for successful ML deployments
2. **Kubernetes provides** scalable, reliable container orchestration
3. **Monitoring is essential** for production ML systems
4. **Security must be built-in** from the start
5. **Auto-scaling** optimizes costs and performance
6. **CI/CD pipelines** enable rapid model deployment

*"Good infrastructure is invisible - users only notice when it's broken"*

**Next: [Model Deployment](ml_engineering/23_model_deployment.md) → Deploying ML models to production with best practices**