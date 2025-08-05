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

### Mathematical Foundations of Deployment (2025)

**Queueing Theory for Model Serving**

Model deployment systems can be modeled using queueing theory to optimize performance:

```
λ = arrival rate (requests/second)
μ = service rate (predictions/second) 
ρ = λ/μ = utilization factor
```

**Little's Law**: `L = λW` where L is average queue length, W is average waiting time.

**Response Time Optimization**:
```
T_response = T_queue + T_processing + T_network
```

Where:
- `T_queue = ρ/(μ(1-ρ))` for M/M/1 queue
- `T_processing = 1/μ`
- `T_network = latency + data_size/bandwidth`

**Load Balancing Mathematics**

For n servers with capacities μ₁, μ₂, ..., μₙ:

**Optimal Load Distribution**:
```
p_i = μ_i / Σ(μ_j) for j=1 to n
```

**Performance Bounds**:
- Lower bound: `T_min = 1/max(μ_i)`
- Upper bound: `T_max = n/Σ(μ_i)`

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

## 🚀 Serverless ML Deployment (2025)

### AWS Lambda ML Serving

```python
import json
import boto3
import pickle
import numpy as np
from typing import Dict, Any

class ServerlessMLHandler:
    """Serverless ML model handler for AWS Lambda"""
    
    def __init__(self):
        self.model = None
        self.s3_client = boto3.client('s3')
        self.model_bucket = 'ml-models-prod'
        self.model_key = 'latest/model.pkl'
        
    def load_model(self):
        """Load model from S3 with caching"""
        if self.model is None:
            try:
                # Download model from S3
                response = self.s3_client.get_object(
                    Bucket=self.model_bucket,
                    Key=self.model_key
                )
                
                # Load model
                model_data = response['Body'].read()
                self.model = pickle.loads(model_data)
                
                print(f"Model loaded from s3://{self.model_bucket}/{self.model_key}")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
    
    def preprocess(self, data: Dict) -> np.ndarray:
        """Preprocess input data"""
        try:
            # Extract features
            features = [
                data.get('feature1', 0.0),
                data.get('feature2', 0.0),
                data.get('feature3', 0.0)
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            raise ValueError(f"Preprocessing error: {str(e)}")
    
    def predict(self, features: np.ndarray) -> Dict:
        """Make prediction"""
        try:
            # Load model if not cached
            self.load_model()
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            return {
                'prediction': float(prediction),
                'probabilities': probability.tolist(),
                'model_version': '1.0.0'
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")

# Lambda handler instance
ml_handler = ServerlessMLHandler()

def lambda_handler(event, context):
    """AWS Lambda entry point"""
    try:
        # Parse input
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        # Preprocess
        features = ml_handler.preprocess(body['data'])
        
        # Predict
        result = ml_handler.predict(features)
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'result': result,
                'timestamp': context.aws_request_id
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'timestamp': context.aws_request_id
            })
        }

### Serverless Deployment Configuration

```yaml
# serverless.yml
service: ml-model-api

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  timeout: 30
  memorySize: 1024
  environment:
    MODEL_BUCKET: ml-models-prod
    MODEL_KEY: latest/model.pkl
  iamRoleStatements:
    - Effect: Allow
      Action:
        - s3:GetObject
      Resource: "arn:aws:s3:::ml-models-prod/*"

functions:
  predict:
    handler: handler.lambda_handler
    events:
      - http:
          path: predict
          method: post
          cors: true
    reservedConcurrency: 100
    
  batch-predict:
    handler: batch_handler.lambda_handler
    events:
      - s3:
          bucket: ml-input-data
          event: s3:ObjectCreated:*
          suffix: .json
    timeout: 900
    memorySize: 3008

plugins:
  - serverless-python-requirements
  - serverless-plugin-warmup

custom:
  pythonRequirements:
    dockerizePip: true
    noDeploy:
      - boto3
      - botocore
  warmup:
    enabled: true
    prewarm: true
```

### Azure Functions ML Deployment

```python
import azure.functions as func
import logging
import json
import os
from azure.storage.blob import BlobServiceClient
import joblib
import numpy as np

class AzureMLFunction:
    """Azure Functions ML model handler"""
    
    def __init__(self):
        self.model = None
        self.blob_service = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )
        self.container_name = 'ml-models'
        self.model_blob = 'production/model.joblib'
    
    def load_model(self):
        """Load model from Azure Blob Storage"""
        if self.model is None:
            try:
                # Download model
                blob_client = self.blob_service.get_blob_client(
                    container=self.container_name,
                    blob=self.model_blob
                )
                
                # Load model
                model_data = blob_client.download_blob().readall()
                self.model = joblib.loads(model_data)
                
                logging.info(f"Model loaded from {self.model_blob}")
                
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise
    
    def predict(self, data):
        """Make prediction"""
        try:
            self.load_model()
            
            # Preprocess data
            features = np.array(data['features']).reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(features)[0]
            
            return {
                'prediction': float(prediction),
                'model_version': '1.0.0',
                'confidence': 0.95
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

# Global instance
ml_function = AzureMLFunction()

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function entry point"""
    logging.info('ML prediction function triggered')
    
    try:
        # Parse request
        req_body = req.get_json()
        
        # Make prediction
        result = ml_function.predict(req_body)
        
        return func.HttpResponse(
            json.dumps({
                'success': True,
                'result': result
            }),
            status_code=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        logging.error(f"Function error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                'success': False,
                'error': str(e)
            }),
            status_code=500,
            mimetype='application/json'
        )

### Google Cloud Functions ML Deployment

```python
import functions_framework
from google.cloud import storage
import pickle
import json
import numpy as np
from flask import jsonify

class GCPMLFunction:
    """Google Cloud Functions ML handler"""
    
    def __init__(self):
        self.model = None
        self.client = storage.Client()
        self.bucket_name = 'ml-models-gcp'
        self.model_path = 'production/model.pkl'
    
    def load_model(self):
        """Load model from Google Cloud Storage"""
        if self.model is None:
            try:
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(self.model_path)
                
                # Download and load model
                model_data = blob.download_as_bytes()
                self.model = pickle.loads(model_data)
                
                print(f"Model loaded from gs://{self.bucket_name}/{self.model_path}")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
    
    def predict(self, data):
        """Make prediction"""
        self.load_model()
        
        # Preprocess
        features = np.array(data['features']).reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0].max()
        
        return {
            'prediction': float(prediction),
            'confidence': float(probability),
            'model_version': '1.0.0'
        }

# Global instance
ml_function = GCPMLFunction()

@functions_framework.http
def predict(request):
    """HTTP Cloud Function entry point"""
    try:
        # Handle CORS
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600'
            }
            return ('', 204, headers)
        
        # Parse request
        request_json = request.get_json()
        
        # Make prediction
        result = ml_function.predict(request_json)
        
        # Return response
        headers = {'Access-Control-Allow-Origin': '*'}
        return (jsonify({
            'success': True,
            'result': result
        }), 200, headers)
        
    except Exception as e:
        headers = {'Access-Control-Allow-Origin': '*'}
        return (jsonify({
            'success': False,
            'error': str(e)
        }), 500, headers)

## 🌐 Edge AI Deployment (2025)

### Edge Deployment Architecture

```python
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List
import json
import time
import psutil
import threading
from queue import Queue

class EdgeMLDeployment:
    """Edge ML deployment with optimization and monitoring"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cpu',
                 max_batch_size: int = 8,
                 cache_size: int = 1000):
        """Initialize edge deployment"""
        self.model_path = model_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.cache_size = cache_size
        
        # Load and optimize model
        self.model = self._load_optimized_model()
        
        # Initialize caching and monitoring
        self.cache = {}
        self.metrics = {
            'requests': 0,
            'cache_hits': 0,
            'inference_times': [],
            'cpu_usage': [],
            'memory_usage': []
        }
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources
        )
        self.monitor_thread.start()
    
    def _load_optimized_model(self):
        """Load and optimize model for edge deployment"""
        try:
            # Load model
            if self.model_path.endswith('.tflite'):
                # TensorFlow Lite model
                interpreter = tf.lite.Interpreter(
                    model_path=self.model_path
                )
                interpreter.allocate_tensors()
                return interpreter
            
            elif self.model_path.endswith('.onnx'):
                # ONNX model
                import onnxruntime as ort
                
                # Create optimized session
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                
                if self.device == 'cuda':
                    providers = ['CUDAExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                
                session = ort.InferenceSession(
                    self.model_path,
                    session_options,
                    providers=providers
                )
                return session
                
            else:
                # TensorFlow SavedModel
                model = tf.saved_model.load(self.model_path)
                
                # Apply optimizations
                if self.device == 'cpu':
                    # Quantize for CPU
                    converter = tf.lite.TFLiteConverter.from_saved_model(
                        self.model_path
                    )
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_model = converter.convert()
                    
                    # Save optimized model
                    with open('/tmp/optimized_model.tflite', 'wb') as f:
                        f.write(tflite_model)
                    
                    # Load optimized model
                    interpreter = tf.lite.Interpreter(
                        model_path='/tmp/optimized_model.tflite'
                    )
                    interpreter.allocate_tensors()
                    return interpreter
                
                return model
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _generate_cache_key(self, data: np.ndarray) -> str:
        """Generate cache key for input data"""
        # Use hash of rounded data for caching
        rounded_data = np.round(data, decimals=3)
        return str(hash(rounded_data.tobytes()))
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append(memory.percent)
                
                # Keep only recent metrics
                if len(self.metrics['cpu_usage']) > 100:
                    self.metrics['cpu_usage'] = self.metrics['cpu_usage'][-100:]
                    self.metrics['memory_usage'] = self.metrics['memory_usage'][-100:]
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
    
    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """Make prediction with caching and optimization"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics['requests'] += 1
            
            # Check cache
            cache_key = self._generate_cache_key(data)
            if cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                result = self.cache[cache_key]
                inference_time = time.time() - start_time
                
                return {
                    'prediction': result,
                    'inference_time': inference_time,
                    'cached': True,
                    'device': self.device
                }
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                # TensorFlow model
                prediction = self.model.predict(data)
            
            elif hasattr(self.model, 'run'):
                # ONNX model
                input_name = self.model.get_inputs()[0].name
                prediction = self.model.run(None, {input_name: data})[0]
            
            else:
                # TensorFlow Lite model
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                
                self.model.set_tensor(input_details[0]['index'], data)
                self.model.invoke()
                prediction = self.model.get_tensor(
                    output_details[0]['index']
                )
            
            # Cache result
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = prediction.tolist()
            
            inference_time = time.time() - start_time
            self.metrics['inference_times'].append(inference_time)
            
            # Keep only recent inference times
            if len(self.metrics['inference_times']) > 1000:
                self.metrics['inference_times'] = (
                    self.metrics['inference_times'][-1000:]
                )
            
            return {
                'prediction': prediction.tolist(),
                'inference_time': inference_time,
                'cached': False,
                'device': self.device
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            return {
                'error': str(e),
                'inference_time': inference_time,
                'cached': False,
                'device': self.device
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        inference_times = self.metrics['inference_times']
        
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            p95_inference_time = np.percentile(inference_times, 95)
            p99_inference_time = np.percentile(inference_times, 99)
        else:
            avg_inference_time = p95_inference_time = p99_inference_time = 0
        
        cache_hit_rate = (
            self.metrics['cache_hits'] / max(self.metrics['requests'], 1)
        )
        
        return {
            'total_requests': self.metrics['requests'],
            'cache_hit_rate': cache_hit_rate,
            'avg_inference_time': avg_inference_time,
            'p95_inference_time': p95_inference_time,
            'p99_inference_time': p99_inference_time,
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'avg_memory_usage': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'cache_size': len(self.cache),
            'device': self.device
        }
    
    def shutdown(self):
        """Shutdown deployment"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

### Mobile Deployment with TensorFlow Lite

```python
import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any

class MobileMLDeployment:
    """Mobile-optimized ML deployment"""
    
    def __init__(self, model_path: str):
        """Initialize mobile deployment"""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow Lite model"""
        try:
            # Load interpreter
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model loaded: {self.model_path}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, 
                        image_data: bytes, 
                        target_size: tuple = (224, 224)) -> np.ndarray:
        """Preprocess image for mobile inference"""
        try:
            # Decode image
            import PIL.Image
            import io
            
            image = PIL.Image.open(io.BytesIO(image_data))
            
            # Resize
            image = image.resize(target_size)
            
            # Convert to array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction"""
        try:
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                input_data
            )
            
            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Process output
            predictions = output_data[0]
            top_prediction = np.argmax(predictions)
            confidence = float(predictions[top_prediction])
            
            return {
                'prediction': int(top_prediction),
                'confidence': confidence,
                'inference_time': inference_time,
                'all_predictions': predictions.tolist()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'inference_time': 0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'input_shape': self.input_details[0]['shape'].tolist(),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype']),
            'model_size': os.path.getsize(self.model_path)
        }

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

## 🎯 Career Paths and Industry Case Studies (2025)

### ML Deployment Engineer Career Path

**Entry Level (0-2 years): Junior ML Deployment Engineer**
- **Salary Range**: $70,000 - $95,000
- **Key Skills**: Docker, Kubernetes, Python, basic ML concepts
- **Responsibilities**:
  - Deploy pre-trained models using standard frameworks
  - Monitor basic deployment metrics
  - Write deployment documentation
  - Support model versioning and rollbacks

**Mid Level (2-5 years): ML Deployment Engineer**
- **Salary Range**: $95,000 - $130,000
- **Key Skills**: Advanced containerization, cloud platforms, CI/CD, performance optimization
- **Responsibilities**:
  - Design scalable deployment architectures
  - Implement blue-green and canary deployments
  - Optimize model inference performance
  - Lead deployment automation initiatives

**Senior Level (5-8 years): Senior ML Deployment Engineer**
- **Salary Range**: $130,000 - $180,000
- **Key Skills**: Multi-cloud architecture, edge deployment, advanced monitoring, team leadership
- **Responsibilities**:
  - Architect enterprise-scale deployment platforms
  - Design disaster recovery and high-availability systems
  - Mentor junior engineers
  - Drive deployment standards and best practices

**Expert Level (8+ years): Principal ML Infrastructure Engineer**
- **Salary Range**: $180,000 - $250,000+
- **Key Skills**: Platform engineering, organizational strategy, emerging technologies
- **Responsibilities**:
  - Define company-wide ML deployment strategy
  - Research and evaluate cutting-edge deployment technologies
  - Cross-functional leadership and technical vision
  - Industry thought leadership and speaking

### Industry Case Studies

**Case Study 1: Netflix - Global Video Recommendation Deployment**

```python
class NetflixDeploymentStrategy:
    """Netflix-style global ML deployment architecture"""
    
    def __init__(self):
        self.regions = [
            'us-east-1', 'us-west-2', 'eu-west-1', 
            'ap-southeast-1', 'ap-northeast-1'
        ]
        self.models = {
            'recommendation': 'personalization-v2.3',
            'ranking': 'content-ranking-v1.8',
            'thumbnail': 'thumbnail-selection-v3.1'
        }
        self.deployment_config = {
            'canary_percentage': 5,
            'full_rollout_hours': 24,
            'fallback_enabled': True
        }
    
    def deploy_globally(self, model_name: str, version: str):
        """Deploy model globally with staged rollout"""
        
        # Stage 1: Deploy to staging environment
        self._deploy_staging(model_name, version)
        
        # Stage 2: Canary deployment in one region
        self._canary_deploy('us-west-2', model_name, version)
        
        # Stage 3: Monitor canary metrics
        canary_success = self._monitor_canary_metrics()
        
        if canary_success:
            # Stage 4: Gradual rollout to all regions
            for region in self.regions:
                self._deploy_region(region, model_name, version)
                time.sleep(300)  # 5-minute delay between regions
        else:
            # Rollback canary
            self._rollback_canary('us-west-2', model_name)
    
    def _monitor_canary_metrics(self) -> bool:
        """Monitor canary deployment metrics"""
        metrics = {
            'error_rate': 0.001,  # 0.1%
            'latency_p99': 45,    # 45ms
            'engagement_rate': 0.85,  # 85%
            'revenue_impact': 1.02    # +2%
        }
        
        # Netflix's criteria for successful canary
        return (
            metrics['error_rate'] < 0.005 and
            metrics['latency_p99'] < 50 and
            metrics['engagement_rate'] > 0.80 and
            metrics['revenue_impact'] > 0.98
        )

# Key Netflix Deployment Principles:
# 1. Global scale: 200M+ users across 190+ countries
# 2. A/B testing: 1000+ experiments running simultaneously
# 3. Personalization: 2000+ model variants per user
# 4. Availability: 99.99% uptime requirement
# 5. Performance: <50ms latency for recommendations
```

**Case Study 2: Tesla - Edge AI Deployment for Autonomous Driving**

```python
class TeslaEdgeDeployment:
    """Tesla-style edge AI deployment for autonomous vehicles"""
    
    def __init__(self):
        self.vehicle_fleet_size = 3000000  # 3M+ vehicles
        self.models = {
            'perception': 'vision-transformer-v4.2',
            'planning': 'trajectory-planner-v3.8',
            'control': 'vehicle-controller-v2.1'
        }
        self.hardware = {
            'fsd_chip': 'Tesla FSD Chip (144 TOPS)',
            'cameras': '8x high-resolution cameras',
            'compute': '2x redundant systems'
        }
    
    def deploy_ota_update(self, model_name: str, version: str):
        """Deploy over-the-air model update to vehicle fleet"""
        
        # Stage 1: Fleet selection strategy
        deployment_cohorts = self._select_deployment_cohorts()
        
        # Stage 2: Staged rollout
        for cohort in deployment_cohorts:
            self._deploy_to_cohort(cohort, model_name, version)
            
            # Monitor safety metrics
            safety_metrics = self._monitor_safety_metrics(cohort)
            
            if not self._safety_criteria_met(safety_metrics):
                self._emergency_rollback(cohort, model_name)
                break
    
    def _select_deployment_cohorts(self) -> List[Dict]:
        """Select vehicle cohorts for staged deployment"""
        return [
            {
                'name': 'beta_testers',
                'size': 1000,
                'criteria': 'opt-in beta users',
                'regions': ['california', 'texas']
            },
            {
                'name': 'employee_fleet',
                'size': 5000,
                'criteria': 'tesla employees',
                'regions': ['global']
            },
            {
                'name': 'low_complexity',
                'size': 50000,
                'criteria': 'highway-dominant driving',
                'regions': ['us_highways']
            },
            {
                'name': 'general_fleet',
                'size': 2944000,
                'criteria': 'all remaining vehicles',
                'regions': ['global']
            }
        ]
    
    def _safety_criteria_met(self, metrics: Dict) -> bool:
        """Evaluate safety criteria for deployment"""
        return (
            metrics['disengagement_rate'] < 0.001 and
            metrics['collision_rate'] < 0.0001 and
            metrics['false_positive_rate'] < 0.05 and
            metrics['intervention_rate'] < 0.01
        )

# Key Tesla Deployment Principles:
# 1. Safety-first: Zero tolerance for safety regressions
# 2. Data-driven: 5M+ miles driven daily for validation
# 3. Real-time: <10ms inference time requirement
# 4. Redundancy: Multiple failsafe systems
# 5. Continuous learning: Fleet data improves models
```

**Case Study 3: Spotify - Real-Time Music Recommendation**

```python
class SpotifyDeploymentPlatform:
    """Spotify-style real-time ML deployment platform"""
    
    def __init__(self):
        self.user_base = 400000000  # 400M+ users
        self.daily_streams = 4000000000  # 4B+ daily streams
        self.models = {
            'discover_weekly': 'collaborative-filtering-v5.1',
            'radio': 'content-based-v3.2',
            'search': 'neural-search-v2.4',
            'ads': 'ad-targeting-v4.0'
        }
    
    def real_time_deployment(self):
        """Spotify's real-time ML deployment architecture"""
        
        deployment_config = {
            'streaming_platform': 'Apache Kafka',
            'feature_store': 'Redis Cluster',
            'model_serving': 'TensorFlow Serving + ONNX',
            'a_b_testing': 'Custom experimentation platform',
            'monitoring': 'Prometheus + Grafana',
            'deployment': 'Kubernetes + Istio'
        }
        
        # Real-time feature pipeline
        features = self._compute_real_time_features()
        
        # Model inference
        recommendations = self._get_recommendations(features)
        
        # A/B test assignment
        experiment_variant = self._assign_experiment_variant()
        
        # Personalized ranking
        ranked_content = self._rank_content(
            recommendations, 
            experiment_variant
        )
        
        return ranked_content
    
    def _compute_real_time_features(self) -> Dict:
        """Compute real-time user features"""
        return {
            'listening_history': 'last_50_tracks',
            'current_context': 'time_of_day, device, location',
            'engagement_signals': 'skip_rate, repeat_rate, save_rate',
            'social_signals': 'friend_activity, trending_content'
        }
    
    def deployment_metrics(self) -> Dict:
        """Spotify's key deployment metrics"""
        return {
            'latency': {
                'p50': '12ms',
                'p95': '45ms',
                'p99': '120ms'
            },
            'throughput': '500K requests/second',
            'availability': '99.95%',
            'user_engagement': {
                'stream_completion_rate': '0.85',
                'discovery_click_rate': '0.32',
                'session_length': '28 minutes'
            },
            'business_impact': {
                'user_retention': '+5.2%',
                'premium_conversion': '+3.8%',
                'ad_revenue': '+12.5%'
            }
        }

# Key Spotify Deployment Principles:
# 1. Real-time: <50ms recommendation latency
# 2. Personalization: Unique experience for each user
# 3. Experimentation: 1000+ A/B tests running
# 4. Scale: 500K+ recommendations per second
# 5. Discovery: Balance familiarity with novelty
```

### Required Skills by Experience Level

```python
class MLDeploymentSkillsFramework:
    """Skills framework for ML deployment engineers"""
    
    def __init__(self):
        self.skill_matrix = {
            'junior': {
                'core_skills': [
                    'Python programming',
                    'Docker containers',
                    'Basic Kubernetes',
                    'Git version control',
                    'Linux command line'
                ],
                'ml_skills': [
                    'Scikit-learn',
                    'Model serialization (pickle, joblib)',
                    'Basic ML concepts',
                    'API development (Flask/FastAPI)'
                ],
                'cloud_skills': [
                    'AWS/GCP/Azure basics',
                    'Cloud storage (S3, GCS)',
                    'Basic monitoring'
                ]
            },
            'mid_level': {
                'core_skills': [
                    'Advanced containerization',
                    'Kubernetes orchestration',
                    'CI/CD pipelines',
                    'Infrastructure as Code',
                    'Performance optimization'
                ],
                'ml_skills': [
                    'TensorFlow/PyTorch serving',
                    'Model optimization',
                    'A/B testing frameworks',
                    'Feature stores',
                    'ML monitoring'
                ],
                'cloud_skills': [
                    'Multi-cloud deployment',
                    'Serverless architectures',
                    'Load balancing',
                    'Auto-scaling',
                    'Security best practices'
                ]
            },
            'senior': {
                'core_skills': [
                    'System architecture',
                    'Platform engineering',
                    'Disaster recovery',
                    'Team leadership',
                    'Technical strategy'
                ],
                'ml_skills': [
                    'Edge deployment',
                    'Real-time inference',
                    'Model governance',
                    'Advanced monitoring',
                    'Cost optimization'
                ],
                'cloud_skills': [
                    'Enterprise architecture',
                    'Compliance frameworks',
                    'Global deployment',
                    'Vendor management',
                    'Technology evaluation'
                ]
            }
        }
    
    def get_learning_path(self, current_level: str, target_level: str) -> List[str]:
        """Generate learning path between skill levels"""
        # Implementation would generate specific learning roadmap
        pass

### Salary Benchmarks (2025)

| Role Level | Experience | Base Salary | Total Comp | Top Companies |
|------------|------------|-------------|------------|---------------|
| Junior | 0-2 years | $70K-$95K | $80K-$110K | $90K-$130K |
| Mid-Level | 2-5 years | $95K-$130K | $110K-$160K | $130K-$200K |
| Senior | 5-8 years | $130K-$180K | $160K-$230K | $200K-$300K |
| Principal | 8+ years | $180K-$250K | $230K-$350K | $300K-$500K |

*Top Companies: FAANG, Tesla, Netflix, Spotify, Uber, Airbnb*
```

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