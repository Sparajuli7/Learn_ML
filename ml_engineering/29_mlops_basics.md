# MLOps Basics: CI/CD, Versioning & Monitoring

*"MLOps is the bridge between ML development and production—where automation meets reliability."*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [CI/CD for ML](#cicd-for-ml)
3. [Model Versioning](#model-versioning)
4. [Monitoring & Observability](#monitoring--observability)
5. [Infrastructure as Code](#infrastructure-as-code)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction

MLOps (Machine Learning Operations) combines DevOps practices with ML-specific workflows to create reliable, scalable, and maintainable ML systems. This chapter covers the foundational MLOps practices including continuous integration/deployment, model versioning, and comprehensive monitoring.

### MLOps Challenges

1. **Model Drift**: Models degrade over time as data distributions change
2. **Reproducibility**: Ensuring consistent results across environments
3. **Scalability**: Managing multiple models and versions
4. **Monitoring**: Tracking model performance in production
5. **Compliance**: Meeting regulatory and audit requirements

### 2025 MLOps Trends

- **GitOps for ML**: Infrastructure and model management through Git
- **Automated Retraining**: Trigger retraining based on performance degradation
- **Model Governance**: Centralized model lifecycle management
- **Observability**: Comprehensive monitoring and alerting
- **Security**: Secure model deployment and data handling

---

## 🔄 CI/CD for ML

### GitHub Actions ML Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Train model
      run: python src/train.py
    
    - name: Evaluate model
      run: python src/evaluate.py
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-artifacts
        path: models/
    
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        kubectl apply -f k8s/staging/
    
    - name: Run integration tests
      run: |
        # Run integration tests against staging
        python tests/integration_test.py
    
    - name: Deploy to production
      if: success()
      run: |
        # Deploy to production
        kubectl apply -f k8s/production/
```

### MLflow Integration

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

class MLflowPipeline:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def train_and_log(self, X_train, y_train, X_test, y_test, params=None):
        """
        Train model and log to MLflow
        """
        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Train model
            model = RandomForestClassifier(**params) if params else RandomForestClassifier()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            })
            mlflow.log_artifact(feature_importance.to_csv(), "feature_importance.csv")
            
            return model, accuracy
    
    def load_model(self, run_id: str):
        """
        Load model from MLflow
        """
        logged_model = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(logged_model)
        return loaded_model
    
    def compare_models(self, run_ids: list):
        """
        Compare multiple model runs
        """
        comparison = {}
        
        for run_id in run_ids:
            with mlflow.start_run(run_id=run_id):
                run = mlflow.get_run(run_id)
                comparison[run_id] = {
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
        
        return comparison

# Example usage
pipeline = MLflowPipeline("fraud_detection")

# Train with different parameters
params1 = {'n_estimators': 100, 'max_depth': 10}
params2 = {'n_estimators': 200, 'max_depth': 15}

model1, acc1 = pipeline.train_and_log(X_train, y_train, X_test, y_test, params1)
model2, acc2 = pipeline.train_and_log(X_train, y_train, X_test, y_test, params2)

# Compare models
comparison = pipeline.compare_models([run1.info.run_id, run2.info.run_id])
```

---

## 📦 Model Versioning

### DVC Model Versioning

```python
import dvc.api
import joblib
import os
from datetime import datetime

class ModelVersioning:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.models_dir = "models"
        
    def save_model_version(self, model, model_name: str, version: str = None):
        """
        Save model with versioning
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_path = os.path.join(self.models_dir, model_name, version)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, "model.joblib")
        joblib.dump(model, model_file)
        
        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'model_path': model_file
        }
        
        metadata_file = os.path.join(model_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to DVC
        os.system(f"dvc add {model_file}")
        os.system(f"dvc push")
        
        return version
    
    def load_model_version(self, model_name: str, version: str):
        """
        Load specific model version
        """
        model_path = os.path.join(self.models_dir, model_name, version, "model.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model version {version} not found")
        
        model = joblib.load(model_path)
        return model
    
    def list_versions(self, model_name: str):
        """
        List all versions of a model
        """
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            return []
        
        versions = []
        for version_dir in os.listdir(model_dir):
            metadata_file = os.path.join(model_dir, version_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    versions.append(metadata)
        
        return sorted(versions, key=lambda x: x['created_at'], reverse=True)
    
    def promote_version(self, model_name: str, version: str):
        """
        Promote version to production
        """
        # Create production symlink
        prod_path = os.path.join(self.models_dir, model_name, "production")
        version_path = os.path.join(self.models_dir, model_name, version, "model.joblib")
        
        if os.path.exists(prod_path):
            os.remove(prod_path)
        
        os.symlink(version_path, prod_path)
        
        print(f"Promoted {model_name} version {version} to production")

# Example usage
versioning = ModelVersioning(".")

# Save model versions
version1 = versioning.save_model_version(model1, "fraud_detector", "v1.0.0")
version2 = versioning.save_model_version(model2, "fraud_detector", "v1.0.1")

# List versions
versions = versioning.list_versions("fraud_detector")
for version in versions:
    print(f"Version: {version['version']}, Created: {version['created_at']}")

# Promote to production
versioning.promote_version("fraud_detector", "v1.0.1")
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
PREDICTION_REQUESTS = Counter('ml_prediction_requests_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy')
ACTIVE_MODELS = Gauge('ml_active_models', 'Number of active models')

class ModelMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def record_prediction(self, model_name: str, latency: float, accuracy: float = None):
        """
        Record prediction metrics
        """
        PREDICTION_REQUESTS.inc()
        PREDICTION_LATENCY.observe(latency)
        
        if accuracy is not None:
            MODEL_ACCURACY.set(accuracy)
    
    def update_active_models(self, count: int):
        """
        Update active models count
        """
        ACTIVE_MODELS.set(count)
    
    def get_uptime(self):
        """
        Get service uptime
        """
        return time.time() - self.start_time

# Example usage
monitor = ModelMonitor()

# Record prediction
start_time = time.time()
prediction = model.predict(X_test[0])
latency = time.time() - start_time

monitor.record_prediction("fraud_detector", latency, 0.95)
```

### Data Drift Detection

```python
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.drift_threshold = 0.05
    
    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Detect data drift using statistical tests
        """
        drift_results = {}
        
        for column in self.reference_data.columns:
            if column in current_data.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(
                    self.reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                
                drift_detected = p_value < self.drift_threshold
                
                drift_results[column] = {
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'drift_severity': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
                }
        
        return drift_results
    
    def alert_drift(self, drift_results: dict):
        """
        Generate alerts for significant drift
        """
        alerts = []
        
        for column, result in drift_results.items():
            if result['drift_detected']:
                alert = {
                    'column': column,
                    'severity': result['drift_severity'],
                    'p_value': result['p_value'],
                    'message': f"Data drift detected in {column}"
                }
                alerts.append(alert)
        
        return alerts

# Example usage
detector = DriftDetector(X_train)

# Detect drift
drift_results = detector.detect_drift(X_test)
alerts = detector.alert_drift(drift_results)

for alert in alerts:
    print(f"ALERT: {alert['message']} (p-value: {alert['p_value']:.4f})")
```

---

## 🏗️ Infrastructure as Code

### Terraform ML Infrastructure

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

# S3 bucket for model artifacts
resource "aws_s3_bucket" "ml_models" {
  bucket = "ml-models-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "ML Models"
    Environment = "production"
  }
}

# ECR repository for model containers
resource "aws_ecr_repository" "ml_model" {
  name                 = "ml-model"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
}

# EKS cluster for model serving
resource "aws_eks_cluster" "ml_cluster" {
  name     = "ml-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  
  vpc_config {
    subnet_ids = var.subnet_ids
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
}

# SageMaker endpoint for model serving
resource "aws_sagemaker_endpoint" "ml_endpoint" {
  name = "ml-model-endpoint"
  
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ml_config.name
}

resource "aws_sagemaker_endpoint_configuration" "ml_config" {
  name = "ml-model-config"
  
  production_variants {
    variant_name           = "primary"
    model_name            = aws_sagemaker_model.ml_model.name
    initial_instance_count = 1
    instance_type         = "ml.m5.large"
  }
}

# CloudWatch alarms for monitoring
resource "aws_cloudwatch_metric_alarm" "model_latency" {
  alarm_name          = "model-latency-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = "300"
  statistic           = "Average"
  threshold           = "1000"
  alarm_description   = "Model latency is too high"
}
```

### Kubernetes ML Infrastructure

```yaml
# ml-infrastructure.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-production

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: ml-production
data:
  MODEL_PATH: "/app/models/model.joblib"
  LOG_LEVEL: "INFO"
  MONITORING_ENABLED: "true"

---
apiVersion: v1
kind: Secret
metadata:
  name: ml-secrets
  namespace: ml-production
type: Opaque
data:
  API_KEY: <base64-encoded-api-key>
  DATABASE_URL: <base64-encoded-db-url>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  namespace: ml-production
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
        image: ml-model:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ml-config
        - secretRef:
            name: ml-secrets
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

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
  namespace: ml-production
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 🧪 Exercises and Projects

### Exercise 1: Complete CI/CD Pipeline

Build a complete CI/CD pipeline for ML models:

1. **Automated Testing**: Unit tests, integration tests, model validation
2. **Model Training**: Automated training with hyperparameter optimization
3. **Model Evaluation**: Performance metrics and drift detection
4. **Deployment**: Staging and production deployment
5. **Monitoring**: Real-time monitoring and alerting

### Exercise 2: Model Registry System

Create a comprehensive model registry:

1. **Version Management**: Track model versions and metadata
2. **Model Lineage**: Track data, code, and model relationships
3. **Approval Workflow**: Model approval and promotion process
4. **Rollback Capability**: Easy rollback to previous versions
5. **Audit Trail**: Complete history of model changes

### Exercise 3: Monitoring Dashboard

Build a real-time monitoring dashboard:

1. **Performance Metrics**: Accuracy, latency, throughput
2. **System Metrics**: CPU, memory, network usage
3. **Business Metrics**: Revenue impact, user satisfaction
4. **Alerting**: Automated alerts for issues
5. **Visualization**: Interactive charts and graphs

### Project: End-to-End MLOps Platform

Create a complete MLOps platform:

1. **Data Pipeline**: Automated data ingestion and preprocessing
2. **Model Development**: Experiment tracking and model training
3. **Model Registry**: Centralized model management
4. **Deployment Pipeline**: Automated deployment with testing
5. **Monitoring**: Comprehensive monitoring and alerting
6. **Governance**: Model governance and compliance

---

## 📖 Further Reading

### Essential Papers

1. **"Hidden Technical Debt in Machine Learning Systems"** - Sculley et al.
2. **"MLOps: Continuous Delivery and Automation Pipelines in Machine Learning"** - Kreuzberger et al.
3. **"A Survey of MLOps Tools and Platforms"** - Testi et al.

### Books

1. **"Building Machine Learning Pipelines"** - Hannes Hapke
2. **"Kubeflow: A Complete Guide"** - Holden Karau
3. **"MLOps Engineering at Scale"** - Carl Osipov

### Tools and Frameworks

1. **MLflow**: Model lifecycle management
2. **Kubeflow**: ML toolkit for Kubernetes
3. **DVC**: Data version control
4. **Weights & Biases**: Experiment tracking
5. **Prometheus**: Monitoring and alerting
6. **Grafana**: Visualization and dashboards

---

## 🎯 Key Takeaways

1. **Automation**: Automate every step of the ML lifecycle
2. **Versioning**: Version code, data, and models
3. **Monitoring**: Monitor everything in production
4. **Infrastructure**: Use IaC for reproducible infrastructure
5. **Governance**: Implement model governance and compliance
6. **Collaboration**: Enable team collaboration and knowledge sharing

---

*"MLOps is not just about tools—it's about culture, process, and automation."*

**Next: [MLOps Advanced](ml_engineering/30_mlops_advanced.md) → Data lineage, concept drift, and automated retraining**