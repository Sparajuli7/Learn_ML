# Model Deployment: Production Serving & API Development

*"Deployment is where ML meets reality—where models serve real users at scale."*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Deployment Strategies](#deployment-strategies)
3. [API Development](#api-development)
4. [Containerization](#containerization)
5. [Model Serving](#model-serving)
6. [Scalability & Performance](#scalability--performance)
7. [Security & Monitoring](#security--monitoring)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

---

## 🎯 Introduction

Model deployment transforms trained models into production-ready services that can handle real-world traffic, maintain performance under load, and provide reliable predictions. This chapter covers the complete deployment pipeline from model packaging to production serving, including modern practices like containerization, microservices, and serverless deployment.

### Deployment Challenges

1. **Model Versioning**: Managing multiple model versions in production
2. **Performance Optimization**: Balancing accuracy with inference speed
3. **Scalability**: Handling varying load patterns
4. **Reliability**: Ensuring 99.9%+ uptime
5. **Security**: Protecting models and data
6. **Monitoring**: Real-time performance tracking

### 2025 Deployment Trends

- **Edge AI**: On-device deployment for low-latency applications
- **Serverless ML**: Pay-per-prediction scaling
- **Model Compression**: Quantization and pruning for efficiency
- **Multi-cloud Deployment**: Vendor-agnostic serving
- **AI-specific Hardware**: GPU/TPU optimization
- **Federated Deployment**: Distributed model serving

---

## 🚀 Deployment Strategies

### Blue-Green Deployment

```python
import docker
import kubernetes
from kubernetes import client, config
import time

class BlueGreenDeployment:
    def __init__(self, model_registry_url, namespace="ml-production"):
        self.model_registry_url = model_registry_url
        self.namespace = namespace
        self.current_version = None
        self.new_version = None
        
        # Initialize Kubernetes client
        config.load_kube_config()
        self.k8s_client = client.CoreV1Api()
        self.apps_client = client.AppsV1Api()
    
    def deploy_new_version(self, model_version, replicas=3):
        """
        Deploy new model version alongside existing one
        """
        # Create new deployment (green)
        deployment_name = f"ml-model-v{model_version}"
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": deployment_name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": deployment_name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="ml-model",
                                image=f"{self.model_registry_url}:v{model_version}",
                                ports=[client.V1ContainerPort(container_port=8000)],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "500m", "memory": "1Gi"},
                                    limits={"cpu": "1000m", "memory": "2Gi"}
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        # Create deployment
        self.apps_client.create_namespaced_deployment(
            namespace=self.namespace,
            body=deployment
        )
        
        self.new_version = model_version
        return deployment_name
    
    def create_service(self, deployment_name, service_name):
        """
        Create Kubernetes service for deployment
        """
        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=service_name),
            spec=client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[client.V1ServicePort(port=8000, target_port=8000)]
            )
        )
        
        self.k8s_client.create_namespaced_service(
            namespace=self.namespace,
            body=service
        )
    
    def switch_traffic(self, new_service_name, percentage=100):
        """
        Gradually switch traffic to new version
        """
        # Update ingress to route traffic
        ingress = client.V1Ingress(
            metadata=client.V1ObjectMeta(name="ml-model-ingress"),
            spec=client.V1IngressSpec(
                rules=[
                    client.V1IngressRule(
                        host="ml-api.example.com",
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path="/",
                                    path_type="Prefix",
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=new_service_name,
                                            port=client.V1ServiceBackendPort(number=8000)
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        )
        
        # Apply ingress configuration
        networking_client = client.NetworkingV1Api()
        networking_client.create_namespaced_ingress(
            namespace=self.namespace,
            body=ingress
        )
    
    def rollback(self):
        """
        Rollback to previous version if issues detected
        """
        if self.current_version:
            # Switch traffic back to current version
            current_service = f"ml-model-v{self.current_version}"
            self.switch_traffic(current_service)
            
            # Delete new deployment
            if self.new_version:
                deployment_name = f"ml-model-v{self.new_version}"
                self.apps_client.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace
                )
    
    def health_check(self, service_name):
        """
        Perform health check on deployment
        """
        import requests
        
        try:
            response = requests.get(f"http://{service_name}:8000/health")
            return response.status_code == 200
        except:
            return False

# Example usage
deployer = BlueGreenDeployment("registry.example.com/ml-model")

# Deploy new version
new_deployment = deployer.deploy_new_version("2.1.0", replicas=3)
deployer.create_service(new_deployment, "ml-model-v2")

# Wait for deployment to be ready
time.sleep(30)

# Switch traffic
deployer.switch_traffic("ml-model-v2", percentage=100)

# Health check
if deployer.health_check("ml-model-v2"):
    print("Deployment successful!")
else:
    deployer.rollback()
    print("Rollback initiated due to health check failure")
```

### Canary Deployment

```python
class CanaryDeployment:
    def __init__(self, base_traffic_percentage=5):
        self.base_traffic_percentage = base_traffic_percentage
        self.canary_percentage = 0
        self.metrics = []
    
    def deploy_canary(self, model_version, initial_percentage=5):
        """
        Deploy canary with initial traffic percentage
        """
        self.canary_percentage = initial_percentage
        
        # Create canary deployment
        canary_deployment = f"ml-model-canary-v{model_version}"
        
        # Update traffic split
        self._update_traffic_split(canary_percentage=initial_percentage)
        
        return canary_deployment
    
    def _update_traffic_split(self, canary_percentage):
        """
        Update traffic split between stable and canary
        """
        stable_percentage = 100 - canary_percentage
        
        # Update Kubernetes service mesh or load balancer
        # This is a simplified example
        traffic_config = {
            "stable": stable_percentage,
            "canary": canary_percentage
        }
        
        print(f"Traffic split updated: {traffic_config}")
    
    def monitor_canary(self, metrics_window=300):
        """
        Monitor canary performance
        """
        # Collect metrics for canary
        canary_metrics = self._collect_metrics("canary")
        stable_metrics = self._collect_metrics("stable")
        
        # Compare performance
        comparison = self._compare_metrics(canary_metrics, stable_metrics)
        
        return comparison
    
    def _collect_metrics(self, deployment_type):
        """
        Collect performance metrics
        """
        # In practice, this would collect from monitoring system
        return {
            "latency_p95": np.random.normal(100, 10),
            "error_rate": np.random.normal(0.01, 0.005),
            "throughput": np.random.normal(1000, 100),
            "cpu_usage": np.random.normal(0.6, 0.1),
            "memory_usage": np.random.normal(0.7, 0.1)
        }
    
    def _compare_metrics(self, canary_metrics, stable_metrics):
        """
        Compare canary vs stable metrics
        """
        comparison = {}
        
        for metric in canary_metrics:
            canary_val = canary_metrics[metric]
            stable_val = stable_metrics[metric]
            
            # Calculate relative difference
            if stable_val != 0:
                relative_diff = (canary_val - stable_val) / stable_val
                comparison[metric] = {
                    "canary": canary_val,
                    "stable": stable_val,
                    "relative_diff": relative_diff,
                    "acceptable": abs(relative_diff) < 0.1  # 10% threshold
                }
        
        return comparison
    
    def promote_canary(self, target_percentage=100):
        """
        Gradually promote canary to full deployment
        """
        steps = [25, 50, 75, 100]
        
        for step in steps:
            if step <= target_percentage:
                self.canary_percentage = step
                self._update_traffic_split(self.canary_percentage)
                
                # Monitor for issues
                comparison = self.monitor_canary()
                
                # Check if any metrics are unacceptable
                unacceptable_metrics = [
                    metric for metric, data in comparison.items()
                    if not data["acceptable"]
                ]
                
                if unacceptable_metrics:
                    print(f"Unacceptable metrics detected: {unacceptable_metrics}")
                    self.rollback_canary()
                    return False
                
                print(f"Canary promoted to {step}% traffic")
                time.sleep(60)  # Wait before next promotion
        
        return True
    
    def rollback_canary(self):
        """
        Rollback canary deployment
        """
        self.canary_percentage = 0
        self._update_traffic_split(canary_percentage=0)
        print("Canary rolled back to 0% traffic")

# Example usage
canary = CanaryDeployment()

# Deploy canary
canary_deployment = canary.deploy_canary("2.1.0", initial_percentage=5)

# Monitor and promote
if canary.monitor_canary():
    success = canary.promote_canary(target_percentage=100)
    if success:
        print("Canary deployment successful!")
    else:
        print("Canary deployment failed, rolled back")
```

---

## 🔌 API Development

### FastAPI Model Serving

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import joblib
import time
import logging
from typing import List, Dict, Any
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "latest"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    inference_time: float
    timestamp: str

class ModelService:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_version = "1.0.0"
        self.request_count = 0
        self.total_inference_time = 0.0
        
        # Load model metadata
        self.feature_names = getattr(self.model, 'feature_names_', None)
        self.class_names = getattr(self.model, 'classes_', None)
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction with timing
        """
        start_time = time.time()
        
        # Validate input
        if len(features) != len(self.feature_names) if self.feature_names else True:
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(features_array)[0]
            prediction = self.model.predict(features_array)[0]
            confidence = np.max(prediction_proba)
        else:
            prediction = self.model.predict(features_array)[0]
            confidence = 1.0
        
        inference_time = time.time() - start_time
        
        # Update metrics
        self.request_count += 1
        self.total_inference_time += inference_time
        
        return {
            "prediction": float(prediction),
            "confidence": float(confidence),
            "inference_time": inference_time,
            "model_version": self.model_version
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics
        """
        avg_inference_time = (self.total_inference_time / self.request_count 
                             if self.request_count > 0 else 0)
        
        return {
            "total_requests": self.request_count,
            "average_inference_time": avg_inference_time,
            "model_version": self.model_version,
            "uptime": time.time() - self.start_time
        }

# Initialize model service
model_service = ModelService("model.joblib")
model_service.start_time = time.time()

# Create FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production ML model serving API",
    version="1.0.0"
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Make prediction endpoint
    """
    try:
        # Add request to background monitoring
        background_tasks.add_task(log_request, request)
        
        # Make prediction
        result = model_service.predict(request.features)
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            inference_time=result["inference_time"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_version": model_service.model_version,
        "uptime": time.time() - model_service.start_time
    }

@app.get("/metrics")
async def get_metrics():
    """
    Get service metrics
    """
    return model_service.get_metrics()

@app.get("/model-info")
async def get_model_info():
    """
    Get model information
    """
    return {
        "model_version": model_service.model_version,
        "feature_names": model_service.feature_names,
        "class_names": model_service.class_names,
        "model_type": type(model_service.model).__name__
    }

async def log_request(request: PredictionRequest):
    """
    Background task to log request
    """
    logger.info(f"Request processed: {request.model_version}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### gRPC Model Serving

```python
import grpc
from concurrent import futures
import numpy as np
import joblib
import time
import ml_pb2
import ml_pb2_grpc

class MLServiceServicer(ml_pb2_grpc.MLServiceServicer):
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_version = "1.0.0"
    
    def Predict(self, request, context):
        """
        gRPC prediction method
        """
        try:
            # Convert request to numpy array
            features = np.array(request.features).reshape(1, -1)
            
            # Make prediction
            start_time = time.time()
            
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features)[0]
                prediction = self.model.predict(features)[0]
                confidence = float(np.max(prediction_proba))
            else:
                prediction = self.model.predict(features)[0]
                confidence = 1.0
            
            inference_time = time.time() - start_time
            
            # Return response
            return ml_pb2.PredictionResponse(
                prediction=float(prediction),
                confidence=confidence,
                model_version=self.model_version,
                inference_time=inference_time
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_pb2.PredictionResponse()
    
    def HealthCheck(self, request, context):
        """
        Health check method
        """
        return ml_pb2.HealthResponse(
            status="healthy",
            model_version=self.model_version
        )

def serve():
    """
    Start gRPC server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_pb2_grpc.add_MLServiceServicer_to_server(
        MLServiceServicer("model.joblib"), server
    )
    
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
```

---

## 🐳 Containerization

### Dockerfile for ML Models

```dockerfile
# Multi-stage build for ML model serving
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for ML Stack

```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.joblib
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    depends_on:
      - redis
      - postgres
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ml-network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ml_models
      POSTGRES_USER: ml_user
      POSTGRES_PASSWORD: ml_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - ml-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - ml-network

volumes:
  redis_data:
  postgres_data:
  grafana_data:

networks:
  ml-network:
    driver: bridge
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: registry.example.com/ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MODEL_PATH
          value: "/app/models/model.joblib"
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

---

## ⚡ Model Serving

### TensorFlow Serving

```python
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

class TensorFlowServingClient:
    def __init__(self, server_url: str, model_name: str):
        self.server_url = server_url
        self.model_name = model_name
        self.channel = grpc.insecure_channel(server_url)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
    
    def predict(self, input_data: np.ndarray, signature_name: str = "serving_default"):
        """
        Make prediction using TensorFlow Serving
        """
        # Create prediction request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = signature_name
        
        # Add input data
        request.inputs["input"].CopyFrom(
            tf.make_tensor_proto(input_data, dtype=tf.float32)
        )
        
        # Make prediction
        response = self.stub.Predict(request, timeout=10.0)
        
        # Extract predictions
        predictions = tf.make_ndarray(response.outputs["output"])
        
        return predictions
    
    def get_model_metadata(self):
        """
        Get model metadata
        """
        # Implementation for getting model metadata
        pass

# Example usage
client = TensorFlowServingClient("localhost:8500", "my_model")

# Make prediction
input_data = np.random.random((1, 28, 28, 1))
predictions = client.predict(input_data)
print(f"Predictions: {predictions}")
```

### ONNX Runtime Serving

```python
import onnxruntime as ort
import numpy as np
import time

class ONNXModelServer:
    def __init__(self, model_path: str, providers: List[str] = None):
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        
        # Create inference session
        self.session = ort.InferenceSession(
            model_path, 
            providers=self.providers
        )
        
        # Get model metadata
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Performance optimization
        self.session.set_providers(self.providers)
    
    def predict(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Make prediction using ONNX Runtime
        """
        start_time = time.time()
        
        # Validate input
        for input_name in self.input_names:
            if input_name not in input_data:
                raise ValueError(f"Missing input: {input_name}")
        
        # Run inference
        outputs = self.session.run(self.output_names, input_data)
        
        inference_time = time.time() - start_time
        
        # Create output dictionary
        output_dict = dict(zip(self.output_names, outputs))
        output_dict['inference_time'] = inference_time
        
        return output_dict
    
    def benchmark(self, input_data: Dict[str, np.ndarray], num_runs: int = 100):
        """
        Benchmark model performance
        """
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(input_data)
            times.append(time.time() - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': 1.0 / np.mean(times)
        }

# Example usage
model_server = ONNXModelServer("model.onnx")

# Prepare input data
input_data = {
    "input": np.random.random((1, 3, 224, 224)).astype(np.float32)
}

# Make prediction
outputs = model_server.predict(input_data)
print(f"Predictions: {outputs}")

# Benchmark
benchmark_results = model_server.benchmark(input_data)
print(f"Benchmark results: {benchmark_results}")
```

---

## 📈 Scalability & Performance

### Load Balancing

```python
import asyncio
import aiohttp
import time
from typing import List, Dict
import random

class LoadBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.health_status = {endpoint: True for endpoint in endpoints}
        self.response_times = {endpoint: [] for endpoint in endpoints}
        self.request_counts = {endpoint: 0 for endpoint in endpoints}
    
    async def health_check(self, endpoint: str) -> bool:
        """
        Check endpoint health
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health", timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def update_health_status(self):
        """
        Update health status of all endpoints
        """
        tasks = [self.health_check(endpoint) for endpoint in self.endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for endpoint, result in zip(self.endpoints, results):
            self.health_status[endpoint] = result
    
    def get_healthy_endpoints(self) -> List[str]:
        """
        Get list of healthy endpoints
        """
        return [endpoint for endpoint in self.endpoints if self.health_status[endpoint]]
    
    def round_robin_select(self) -> str:
        """
        Round-robin load balancing
        """
        healthy_endpoints = self.get_healthy_endpoints()
        if not healthy_endpoints:
            raise Exception("No healthy endpoints available")
        
        # Simple round-robin
        endpoint = healthy_endpoints[self.request_counts[healthy_endpoints[0]] % len(healthy_endpoints)]
        self.request_counts[endpoint] += 1
        
        return endpoint
    
    def least_connections_select(self) -> str:
        """
        Least connections load balancing
        """
        healthy_endpoints = self.get_healthy_endpoints()
        if not healthy_endpoints:
            raise Exception("No healthy endpoints available")
        
        return min(healthy_endpoints, key=lambda x: self.request_counts[x])
    
    def weighted_select(self, weights: Dict[str, float]) -> str:
        """
        Weighted load balancing
        """
        healthy_endpoints = self.get_healthy_endpoints()
        if not healthy_endpoints:
            raise Exception("No healthy endpoints available")
        
        # Filter weights for healthy endpoints
        healthy_weights = {k: v for k, v in weights.items() if k in healthy_endpoints}
        
        # Normalize weights
        total_weight = sum(healthy_weights.values())
        normalized_weights = {k: v/total_weight for k, v in healthy_weights.items()}
        
        # Select based on weights
        rand = random.random()
        cumulative = 0
        
        for endpoint, weight in normalized_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return endpoint
        
        return healthy_endpoints[0]  # Fallback

# Example usage
endpoints = [
    "http://ml-api-1:8000",
    "http://ml-api-2:8000",
    "http://ml-api-3:8000"
]

lb = LoadBalancer(endpoints)

# Health check
asyncio.run(lb.update_health_status())

# Load balancing
try:
    selected_endpoint = lb.round_robin_select()
    print(f"Selected endpoint: {selected_endpoint}")
except Exception as e:
    print(f"Error: {e}")
```

### Caching Strategy

```python
import redis
import pickle
import hashlib
import json
from typing import Any, Optional

class ModelCache:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # 1 hour default
    
    def _generate_cache_key(self, features: List[float], model_version: str) -> str:
        """
        Generate cache key for prediction
        """
        # Create hash of features and model version
        data = json.dumps(features, sort_keys=True) + model_version
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_cached_prediction(self, features: List[float], model_version: str) -> Optional[Dict]:
        """
        Get cached prediction if available
        """
        cache_key = self._generate_cache_key(features, model_version)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_prediction(self, features: List[float], model_version: str, prediction: Dict):
        """
        Cache prediction result
        """
        cache_key = self._generate_cache_key(features, model_version)
        
        try:
            serialized_data = pickle.dumps(prediction)
            self.redis_client.setex(cache_key, self.cache_ttl, serialized_data)
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    def invalidate_cache(self, model_version: str):
        """
        Invalidate all cache entries for a model version
        """
        # In practice, you'd use Redis patterns to delete keys
        # This is a simplified version
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory", 0),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            return {"error": str(e)}

# Example usage
cache = ModelCache()

# Cache prediction
features = [1.0, 2.0, 3.0, 4.0]
model_version = "1.0.0"
prediction = {"prediction": 0.85, "confidence": 0.92}

cache.cache_prediction(features, model_version, prediction)

# Retrieve cached prediction
cached_result = cache.get_cached_prediction(features, model_version)
if cached_result:
    print(f"Cached prediction: {cached_result}")
else:
    print("No cached prediction found")

# Get cache stats
stats = cache.get_cache_stats()
print(f"Cache stats: {stats}")
```

---

## 🔒 Security & Monitoring

### API Security

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time
from typing import Optional

# Security configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class SecurityManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: dict, expires_delta: Optional[int] = None):
        """
        Create JWT access token
        """
        to_encode = data.copy()
        if expires_delta:
            expire = time.time() + expires_delta
        else:
            expire = time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """
        Verify JWT token
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

security_manager = SecurityManager(SECRET_KEY)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current user from token
    """
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    return payload

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_with_auth(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Protected prediction endpoint
    """
    # Add user context to request
    request.user_id = current_user.get("user_id")
    
    # Make prediction
    result = model_service.predict(request.features)
    
    return PredictionResponse(**result)
```

### Monitoring & Observability

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time
import logging

# Prometheus metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_DURATION = Histogram('ml_prediction_duration_seconds', 'Prediction duration')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy')
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Active connections')

class MonitoringMiddleware:
    def __init__(self):
        self.start_time = time.time()
    
    async def __call__(self, request, call_next):
        # Track active connections
        ACTIVE_CONNECTIONS.inc()
        
        # Track request start time
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Track prediction metrics
            if request.url.path == "/predict":
                PREDICTION_COUNTER.inc()
                duration = time.time() - start_time
                PREDICTION_DURATION.observe(duration)
            
            return response
        
        finally:
            ACTIVE_CONNECTIONS.dec()

# Add middleware to FastAPI app
app.add_middleware(MonitoringMiddleware)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    return prometheus_client.generate_latest()

# Structured logging
import structlog

logger = structlog.get_logger()

def log_prediction(features: List[float], prediction: float, user_id: str):
    """
    Log prediction with structured logging
    """
    logger.info(
        "prediction_made",
        user_id=user_id,
        prediction=prediction,
        feature_count=len(features),
        model_version=model_service.model_version
    )

# Update prediction endpoint to include logging
@app.post("/predict")
@limiter.limit("10/minute")
async def predict_with_monitoring(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Prediction endpoint with monitoring
    """
    try:
        # Make prediction
        result = model_service.predict(request.features)
        
        # Log prediction
        log_prediction(
            request.features,
            result["prediction"],
            current_user.get("user_id", "anonymous")
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(
            "prediction_error",
            error=str(e),
            user_id=current_user.get("user_id", "anonymous")
        )
        raise
```

---

## 🧪 Exercises and Projects

### Exercise 1: Multi-Model Deployment System

Create a system that can deploy and manage multiple model versions:

1. **Model Registry**: Store and version models
2. **Deployment Orchestrator**: Manage blue-green deployments
3. **Traffic Management**: Route requests to different model versions
4. **Rollback Mechanism**: Automatic rollback on failures

### Exercise 2: High-Performance API Gateway

Build an API gateway for ML services:

1. **Load Balancing**: Distribute traffic across model instances
2. **Rate Limiting**: Prevent abuse and ensure fair usage
3. **Authentication**: JWT-based authentication
4. **Caching**: Redis-based response caching
5. **Monitoring**: Real-time metrics and alerting

### Exercise 3: Serverless ML Deployment

Create a serverless deployment using AWS Lambda or similar:

1. **Model Packaging**: Optimize models for serverless execution
2. **Cold Start Optimization**: Minimize startup time
3. **Auto-scaling**: Handle varying load automatically
4. **Cost Optimization**: Monitor and optimize costs

### Project: Complete ML Platform

Build a complete ML serving platform:

1. **Model Registry**: Version and store models
2. **Deployment Pipeline**: Automated deployment with testing
3. **Monitoring Dashboard**: Real-time performance monitoring
4. **A/B Testing**: Traffic splitting between model versions
5. **Auto-scaling**: Automatic scaling based on load
6. **Security**: Authentication, authorization, and encryption

### Advanced Project: Edge ML Deployment

Create an edge ML deployment system:

1. **Model Optimization**: Quantization and pruning
2. **Edge Runtime**: Lightweight inference engine
3. **Federated Updates**: Update models across edge devices
4. **Offline Capability**: Work without internet connection
5. **Battery Optimization**: Minimize power consumption

---

## 📖 Further Reading

### Essential Papers

1. **"TensorFlow Serving: Flexible, High-Performance ML Serving"** - Olston et al.
2. **"Clipper: A Low-Latency Online Prediction Serving System"** - Crankshaw et al.
3. **"ModelDB: A System for ML Model Management"** - Vartak et al.

### Books

1. **"Kubernetes: Up and Running"** - Kelsey Hightower
2. **"Designing Data-Intensive Applications"** - Martin Kleppmann
3. **"Site Reliability Engineering"** - Google

### Online Resources

1. **TensorFlow Serving**: Production ML serving
2. **Kubeflow**: ML toolkit for Kubernetes
3. **Seldon Core**: ML model serving platform
4. **BentoML**: Model serving library

### Tools and Frameworks

1. **TensorFlow Serving**: High-performance serving
2. **ONNX Runtime**: Cross-platform inference
3. **Kubernetes**: Container orchestration
4. **Docker**: Containerization
5. **FastAPI**: Modern Python web framework
6. **Prometheus**: Monitoring and alerting

---

## 🎯 Key Takeaways

1. **Containerization**: Package models with dependencies for consistent deployment
2. **Orchestration**: Use Kubernetes for scalable deployment
3. **API Design**: Create clean, documented APIs with proper error handling
4. **Monitoring**: Implement comprehensive monitoring and alerting
5. **Security**: Protect models and data with authentication and encryption
6. **Performance**: Optimize for latency, throughput, and resource usage

---

*"Deployment is not the end—it's the beginning of the real work."*

**Next: [MLOps Basics](ml_engineering/29_mlops_basics.md) → CI/CD, versioning, and monitoring**