# Model Deployment: From Training to Production
*"A model is only as good as its deployment - the bridge between research and real-world impact"*

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Deployment Fundamentals](#deployment-fundamentals)
3. [Model Packaging and Versioning](#model-packaging-and-versioning)
4. [Deployment Strategies](#deployment-strategies)
5. [A/B Testing and Canary Deployments](#ab-testing-and-canary-deployments)
6. [Rollback and Recovery](#rollback-and-recovery)
7. [Implementation Examples](#implementation-examples)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

## 🎯 Introduction

Model deployment is like launching a rocket - you need perfect coordination between all systems, careful monitoring, and the ability to abort if something goes wrong. In 2025, with AI systems becoming increasingly critical to business operations, deploying models safely and efficiently is more important than ever.

### Why Model Deployment Matters in 2025

The AI landscape in 2025 demands deployment systems that can handle:
- **Rapid Iteration**: Deploying model updates multiple times per day
- **Zero Downtime**: Continuous service availability during deployments
- **Safety First**: Preventing bad models from affecting users
- **Compliance**: Meeting regulatory requirements (GDPR, AI Act)
- **Cost Efficiency**: Optimizing deployment costs and resource usage
- **Observability**: Complete visibility into model behavior

### The Deployment Evolution

Model deployment has evolved significantly:

- **2010s**: Manual deployment with long release cycles
- **2015s**: Automated deployment with basic monitoring
- **2020s**: CI/CD pipelines with A/B testing
- **2025**: AI-native deployment with automated safety checks

## 🧮 Mathematical Foundations

### Deployment Performance Metrics

#### 1. Deployment Success Rate (DSR)
```
DSR = (Successful deployments) / (Total deployments) × 100
```

#### 2. Mean Time to Recovery (MTTR)
```
MTTR = (Total downtime) / (Number of incidents)
```

#### 3. Deployment Frequency (DF)
```
DF = (Number of deployments) / (Time period)
```

#### 4. Change Failure Rate (CFR)
```
CFR = (Failed deployments) / (Total deployments) × 100
```

### Example Calculation

For a system with 100 deployments per month:
- Successful deployments: 95
- Failed deployments: 5
- Total downtime: 2 hours
- Number of incidents: 3

```
DSR = (95 / 100) × 100 = 95%
MTTR = 2 / 3 = 0.67 hours
DF = 100 / 1 = 100 deployments/month
CFR = (5 / 100) × 100 = 5%
```

## 💻 Implementation

### 1. Model Packaging with MLflow

MLflow is like a smart packaging system for AI models - it ensures your models are properly packaged, versioned, and ready for deployment.

```python
# Why: Standardized model packaging and versioning
# How: Model registry with metadata tracking
# Where: ML development and deployment pipelines
# What: Reproducible model packaging and deployment
# When: When transitioning from development to production

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ModelPackager:
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        # Why: Initialize MLflow for model tracking and packaging
        # How: Set up MLflow tracking server and registry
        # Where: ML development environments
        # What: Model packaging and versioning system
        # When: At the start of model development
        
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        logging.info(f"Initialized MLflow with tracking URI: {tracking_uri}")
    
    def train_and_log_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                           model_params: Dict[str, Any], model_type: str = "sklearn"):
        """Train and log model to MLflow"""
        # Why: Track model training and package for deployment
        # How: Train model and log to MLflow registry
        # Where: Model development pipelines
        # What: Versioned model with metadata
        # When: After model training and validation
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            
            # Train model based on type
            if model_type == "sklearn":
                model = RandomForestClassifier(**model_params)
                model.fit(X_train, y_train)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
            elif model_type == "tensorflow":
                # TensorFlow model training
                import tensorflow as tf
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                
                # Log model
                mlflow.tensorflow.log_model(model, "model")
            
            # Log metrics
            train_score = model.score(X_train, y_train) if hasattr(model, 'score') else model.evaluate(X_train, y_train)[1]
            mlflow.log_metric("train_score", train_score)
            
            # Log model metadata
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("deployment_ready", "true")
            
            run_id = mlflow.active_run().info.run_id
            logging.info(f"Logged model {model_name} with run_id: {run_id}")
            
            return run_id
    
    def register_model(self, model_name: str, run_id: str, stage: str = "Staging"):
        """Register model in MLflow Model Registry"""
        # Why: Create versioned model for deployment
        # How: Register model from MLflow run
        # Where: Model deployment pipelines
        # What: Versioned model ready for deployment
        # When: After successful model training
        
        try:
            # Create model version
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition to specified stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            
            logging.info(f"Registered model {model_name} version {model_version.version} in {stage} stage")
            return model_version.version
            
        except Exception as e:
            logging.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def load_model_for_deployment(self, model_name: str, stage: str = "Production"):
        """Load model for deployment"""
        # Why: Load production-ready model for serving
        # How: Load model from MLflow registry
        # Where: Model serving applications
        # What: Deployed model with metadata
        # When: When starting model serving
        
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model metadata
            model_info = self.client.get_latest_versions(model_name, [stage])[0]
            
            logging.info(f"Loaded model {model_name} version {model_info.version} from {stage} stage")
            return model, model_info
            
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        # Why: Track model version history and performance
        # How: Query MLflow registry for model versions
        # Where: Model management and monitoring
        # What: Complete model version history
        # When: For model comparison and rollback decisions
        
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_info = []
            for version in versions:
                version_info.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "created_at": version.creation_timestamp,
                    "last_updated": version.last_updated_timestamp,
                    "status": version.status
                })
            
            return version_info
            
        except Exception as e:
            logging.error(f"Failed to get versions for {model_name}: {e}")
            return []

# Usage example
if __name__ == "__main__":
    packager = ModelPackager()
    
    # Generate sample data
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train and log model
    model_params = {"n_estimators": 100, "max_depth": 10}
    run_id = packager.train_and_log_model(
        model_name="recommendation-model",
        X_train=X_train,
        y_train=y_train,
        model_params=model_params
    )
    
    # Register model
    version = packager.register_model("recommendation-model", run_id, "Staging")
    print(f"Registered model version: {version}")
    
    # Load model for deployment
    model, model_info = packager.load_model_for_deployment("recommendation-model", "Staging")
    print(f"Loaded model version {model_info.version}")
```

### 2. Deployment Strategies with Kubernetes

Kubernetes deployment strategies are like having multiple safety nets - they ensure your model deployments are safe, reliable, and can be quickly rolled back if needed.

```python
# Why: Implement safe deployment strategies for ML models
# How: Use Kubernetes deployment patterns for zero-downtime updates
# Where: Production ML model deployments
# What: Reliable model deployment with rollback capabilities
# When: When deploying new model versions to production

import yaml
import kubernetes
from kubernetes import client, config
import time
import logging
from typing import Dict, Any, List

class ModelDeploymentManager:
    def __init__(self):
        # Why: Initialize Kubernetes deployment manager
        # How: Connect to Kubernetes cluster and set up clients
        # Where: Production ML deployment environments
        # What: Automated model deployment system
        # When: At system startup
        
        try:
            config.load_kube_config()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            logging.info("Connected to Kubernetes cluster")
        except Exception as e:
            logging.error(f"Failed to connect to Kubernetes: {e}")
            raise
    
    def deploy_rolling_update(self, model_name: str, new_image: str, replicas: int = 3):
        """Deploy model using rolling update strategy"""
        # Why: Update model with zero downtime
        # How: Gradually replace old pods with new ones
        # Where: Production model deployments
        # What: Seamless model updates without service interruption
        # When: When deploying new model versions
        
        deployment = self._create_rolling_update_deployment(model_name, new_image, replicas)
        
        try:
            # Apply deployment
            self.apps_v1.patch_namespaced_deployment(
                name=f"{model_name}-deployment",
                namespace="ml-models",
                body=deployment
            )
            
            logging.info(f"Started rolling update for {model_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to deploy {model_name}: {e}")
            return False
    
    def deploy_blue_green(self, model_name: str, new_image: str, replicas: int = 3):
        """Deploy model using blue-green strategy"""
        # Why: Deploy new model version alongside existing one
        # How: Create new deployment and switch traffic
        # Where: Critical model deployments requiring safety
        # What: Safe deployment with instant rollback capability
        # When: When deploying high-risk model changes
        
        # Create new deployment (green)
        green_deployment = self._create_deployment(
            f"{model_name}-green",
            new_image,
            replicas,
            labels={"app": model_name, "version": "green"}
        )
        
        try:
            # Deploy green version
            self.apps_v1.create_namespaced_deployment(
                namespace="ml-models",
                body=green_deployment
            )
            
            # Wait for green deployment to be ready
            self._wait_for_deployment_ready(f"{model_name}-green")
            
            # Switch traffic to green
            self._switch_traffic_to_green(model_name)
            
            # Scale down blue deployment
            self._scale_down_blue_deployment(model_name)
            
            logging.info(f"Completed blue-green deployment for {model_name}")
            return True
            
        except Exception as e:
            logging.error(f"Blue-green deployment failed for {model_name}: {e}")
            # Rollback to blue
            self._switch_traffic_to_blue(model_name)
            return False
    
    def deploy_canary(self, model_name: str, new_image: str, canary_percentage: int = 10):
        """Deploy model using canary strategy"""
        # Why: Gradually roll out new model to small percentage of traffic
        # How: Deploy new version alongside existing with traffic splitting
        # Where: Model deployments requiring gradual validation
        # What: Gradual rollout with monitoring and rollback
        # When: When testing new model versions safely
        
        # Create canary deployment
        canary_deployment = self._create_deployment(
            f"{model_name}-canary",
            new_image,
            max(1, int(canary_percentage / 10)),  # At least 1 replica
            labels={"app": model_name, "version": "canary"}
        )
        
        try:
            # Deploy canary
            self.apps_v1.create_namespaced_deployment(
                namespace="ml-models",
                body=canary_deployment
            )
            
            # Update service to split traffic
            self._update_service_traffic_split(model_name, canary_percentage)
            
            logging.info(f"Started canary deployment for {model_name} with {canary_percentage}% traffic")
            return True
            
        except Exception as e:
            logging.error(f"Canary deployment failed for {model_name}: {e}")
            return False
    
    def _create_rolling_update_deployment(self, model_name: str, image: str, replicas: int):
        """Create deployment with rolling update strategy"""
        # Why: Define rolling update deployment configuration
        # How: Set up deployment with rolling update strategy
        # Where: Kubernetes deployment manifests
        # What: Deployment configuration for rolling updates
        # When: When creating rolling update deployments
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{model_name}-deployment",
                "namespace": "ml-models"
            },
            "spec": {
                "replicas": replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    }
                },
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
                            "image": image,
                            "ports": [{"containerPort": 8080}],
                            "resources": {
                                "requests": {"memory": "512Mi", "cpu": "250m"},
                                "limits": {"memory": "1Gi", "cpu": "500m"}
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }
    
    def _wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300):
        """Wait for deployment to be ready"""
        # Why: Ensure deployment is fully ready before proceeding
        # How: Poll deployment status until ready
        # Where: Deployment orchestration
        # What: Deployment readiness confirmation
        # When: Before switching traffic or proceeding with deployment
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace="ml-models"
                )
                
                if (deployment.status.ready_replicas == deployment.spec.replicas and
                    deployment.status.updated_replicas == deployment.spec.replicas):
                    logging.info(f"Deployment {deployment_name} is ready")
                    return True
                
                time.sleep(5)
                
            except Exception as e:
                logging.warning(f"Error checking deployment status: {e}")
                time.sleep(5)
        
        raise TimeoutError(f"Deployment {deployment_name} did not become ready within {timeout} seconds")
    
    def _switch_traffic_to_green(self, model_name: str):
        """Switch traffic from blue to green deployment"""
        # Why: Complete blue-green deployment by switching traffic
        # How: Update service selector to point to green deployment
        # Where: Blue-green deployment orchestration
        # What: Traffic switching for zero-downtime deployment
        # When: After green deployment is ready
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{model_name}-service",
                "namespace": "ml-models"
            },
            "spec": {
                "selector": {
                    "app": model_name,
                    "version": "green"
                },
                "ports": [{"port": 80, "targetPort": 8080}],
                "type": "ClusterIP"
            }
        }
        
        self.core_v1.patch_namespaced_service(
            name=f"{model_name}-service",
            namespace="ml-models",
            body=service
        )
        
        logging.info(f"Switched traffic to green deployment for {model_name}")
    
    def _update_service_traffic_split(self, model_name: str, canary_percentage: int):
        """Update service to split traffic between stable and canary"""
        # Why: Implement traffic splitting for canary deployments
        # How: Use Kubernetes service with multiple selectors
        # Where: Canary deployment orchestration
        # What: Traffic splitting between model versions
        # When: During canary deployment rollout
        
        # In real implementation, you'd use Istio or similar for traffic splitting
        # For demonstration, we'll simulate traffic splitting
        logging.info(f"Updated traffic split: {100-canary_percentage}% stable, {canary_percentage}% canary")
    
    def rollback_deployment(self, model_name: str, target_version: str):
        """Rollback deployment to previous version"""
        # Why: Quickly revert to previous working version
        # How: Update deployment to use previous image
        # Where: Emergency rollback scenarios
        # What: Fast rollback to stable version
        # When: When new deployment causes issues
        
        try:
            # Get previous deployment image
            previous_image = self._get_previous_deployment_image(model_name, target_version)
            
            # Update deployment
            deployment = self._create_rolling_update_deployment(model_name, previous_image, 3)
            
            self.apps_v1.patch_namespaced_deployment(
                name=f"{model_name}-deployment",
                namespace="ml-models",
                body=deployment
            )
            
            logging.info(f"Rolled back {model_name} to version {target_version}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to rollback {model_name}: {e}")
            return False
    
    def _get_previous_deployment_image(self, model_name: str, target_version: str) -> str:
        """Get image for target version"""
        # Why: Retrieve previous deployment image for rollback
        # How: Query deployment history or registry
        # Where: Rollback operations
        # What: Previous deployment image
        # When: When performing rollback
        
        # In real implementation, you'd query your image registry
        # For demonstration, we'll return a placeholder
        return f"ml-registry.example.com/{model_name}:{target_version}"

# Usage example
if __name__ == "__main__":
    manager = ModelDeploymentManager()
    
    # Deploy using rolling update
    success = manager.deploy_rolling_update(
        model_name="recommendation-model",
        new_image="ml-registry.example.com/recommendation:v2.0.0",
        replicas=5
    )
    
    if success:
        print("✅ Rolling update deployment successful")
        
        # Deploy using blue-green strategy
        success = manager.deploy_blue_green(
            model_name="fraud-detection-model",
            new_image="ml-registry.example.com/fraud-detection:v1.5.0"
        )
        
        if success:
            print("✅ Blue-green deployment successful")
        else:
            print("❌ Blue-green deployment failed")
    else:
        print("❌ Rolling update deployment failed")
```

### 3. A/B Testing Framework for ML Models

A/B testing is like having a scientific experiment for your models - you can test different versions and measure which one performs better in real-world conditions.

```python
# Why: Test model versions with real user traffic
# How: Split traffic between model versions and measure performance
# Where: Production ML model evaluation
# What: Statistical comparison of model performance
# When: When deploying new model versions

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json

class ABTestingFramework:
    def __init__(self):
        # Why: Initialize A/B testing framework for ML models
        # How: Set up testing infrastructure and metrics collection
        # Where: Production ML model evaluation
        # What: A/B testing system for model comparison
        # When: At system startup
        
        self.experiments = {}
        self.metrics_collector = {}
        logging.info("A/B testing framework initialized")
    
    def create_experiment(self, experiment_name: str, model_a: str, model_b: str, 
                         traffic_split: float = 0.5, duration_days: int = 7):
        """Create A/B testing experiment"""
        # Why: Set up controlled experiment between model versions
        # How: Define experiment parameters and traffic allocation
        # Where: Model deployment and evaluation
        # What: Structured experiment for model comparison
        # When: When testing new model versions
        
        experiment = {
            "name": experiment_name,
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=duration_days),
            "status": "running",
            "metrics": {
                "model_a": {},
                "model_b": {}
            }
        }
        
        self.experiments[experiment_name] = experiment
        logging.info(f"Created A/B experiment: {experiment_name}")
        
        return experiment
    
    def assign_traffic(self, experiment_name: str, user_id: str) -> str:
        """Assign user to model version based on traffic split"""
        # Why: Route users to different model versions
        # How: Use consistent hashing for traffic assignment
        # Where: Model serving endpoints
        # What: Traffic routing for A/B testing
        # When: For each user request
        
        if experiment_name not in self.experiments:
            return "default"
        
        experiment = self.experiments[experiment_name]
        
        # Use consistent hashing for traffic assignment
        hash_value = hash(user_id) % 100
        traffic_split = experiment["traffic_split"] * 100
        
        if hash_value < traffic_split:
            return experiment["model_a"]
        else:
            return experiment["model_b"]
    
    def record_prediction(self, experiment_name: str, model_version: str, 
                         user_id: str, prediction: Any, actual: Any = None):
        """Record prediction for A/B testing metrics"""
        # Why: Collect performance data for model comparison
        # How: Store prediction results with metadata
        # Where: Model serving endpoints
        # What: Performance metrics for statistical analysis
        # When: After each model prediction
        
        if experiment_name not in self.experiments:
            return
        
        if experiment_name not in self.metrics_collector:
            self.metrics_collector[experiment_name] = {
                "model_a": [],
                "model_b": []
            }
        
        record = {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "model_version": model_version,
            "prediction": prediction,
            "actual": actual
        }
        
        self.metrics_collector[experiment_name][model_version].append(record)
        
        logging.debug(f"Recorded prediction for {experiment_name}: {model_version}")
    
    def calculate_metrics(self, experiment_name: str) -> Dict[str, Any]:
        """Calculate performance metrics for experiment"""
        # Why: Analyze experiment results for statistical significance
        # How: Calculate performance metrics and statistical tests
        # Where: A/B testing analysis
        # What: Performance comparison between model versions
        # When: During and after experiment completion
        
        if experiment_name not in self.metrics_collector:
            return {}
        
        metrics = {}
        
        for model_version in ["model_a", "model_b"]:
            predictions = self.metrics_collector[experiment_name][model_version]
            
            if not predictions:
                continue
            
            # Calculate basic metrics
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p["actual"] is not None and p["prediction"] == p["actual"])
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Calculate latency (simplified)
            latencies = [0.1 + np.random.normal(0, 0.02) for _ in predictions]  # Simulated
            avg_latency = np.mean(latencies)
            
            metrics[model_version] = {
                "total_predictions": total_predictions,
                "accuracy": accuracy,
                "avg_latency": avg_latency,
                "throughput": total_predictions / max(1, (datetime.now() - self.experiments[experiment_name]["start_date"]).total_seconds() / 3600)
            }
        
        # Perform statistical significance test
        if "model_a" in metrics and "model_b" in metrics:
            significance = self._calculate_statistical_significance(experiment_name)
            metrics["statistical_significance"] = significance
        
        return metrics
    
    def _calculate_statistical_significance(self, experiment_name: str) -> Dict[str, Any]:
        """Calculate statistical significance between model versions"""
        # Why: Determine if performance differences are statistically significant
        # How: Perform statistical tests (t-test, chi-square, etc.)
        # Where: A/B testing analysis
        # What: Statistical significance results
        # When: When comparing model performance
        
        model_a_data = self.metrics_collector[experiment_name]["model_a"]
        model_b_data = self.metrics_collector[experiment_name]["model_b"]
        
        if not model_a_data or not model_b_data:
            return {"significant": False, "p_value": 1.0}
        
        # Extract accuracy scores (simplified)
        model_a_scores = [1 if p["prediction"] == p["actual"] else 0 for p in model_a_data if p["actual"] is not None]
        model_b_scores = [1 if p["prediction"] == p["actual"] else 0 for p in model_b_data if p["actual"] is not None]
        
        if not model_a_scores or not model_b_scores:
            return {"significant": False, "p_value": 1.0}
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
        
        return {
            "significant": p_value < 0.05,
            "p_value": p_value,
            "t_statistic": t_stat,
            "confidence_level": 0.95
        }
    
    def get_experiment_recommendation(self, experiment_name: str) -> Dict[str, Any]:
        """Get recommendation based on experiment results"""
        # Why: Provide clear recommendation for model deployment
        # How: Analyze metrics and statistical significance
        # Where: Model deployment decision making
        # What: Clear recommendation with confidence
        # When: When experiment is complete
        
        metrics = self.calculate_metrics(experiment_name)
        
        if not metrics or "model_a" not in metrics or "model_b" not in metrics:
            return {"recommendation": "insufficient_data", "confidence": 0.0}
        
        model_a_metrics = metrics["model_a"]
        model_b_metrics = metrics["model_b"]
        
        # Compare performance
        accuracy_diff = model_b_metrics["accuracy"] - model_a_metrics["accuracy"]
        latency_diff = model_a_metrics["avg_latency"] - model_b_metrics["avg_latency"]
        
        significance = metrics.get("statistical_significance", {})
        
        if significance.get("significant", False):
            if accuracy_diff > 0.01:  # 1% improvement threshold
                return {
                    "recommendation": "deploy_model_b",
                    "confidence": 1 - significance["p_value"],
                    "reason": f"Model B shows {accuracy_diff:.3f} better accuracy with statistical significance"
                }
            elif accuracy_diff < -0.01:
                return {
                    "recommendation": "keep_model_a",
                    "confidence": 1 - significance["p_value"],
                    "reason": f"Model A shows {abs(accuracy_diff):.3f} better accuracy with statistical significance"
                }
            else:
                return {
                    "recommendation": "no_significant_difference",
                    "confidence": 1 - significance["p_value"],
                    "reason": "No significant difference in performance"
                }
        else:
            return {
                "recommendation": "continue_experiment",
                "confidence": 0.0,
                "reason": "Insufficient data for statistical significance"
            }
    
    def end_experiment(self, experiment_name: str):
        """End A/B testing experiment"""
        # Why: Conclude experiment and provide final results
        # How: Stop traffic assignment and analyze final results
        # Where: Experiment management
        # What: Final experiment results and recommendation
        # When: When experiment duration is complete or manually stopped
        
        if experiment_name in self.experiments:
            self.experiments[experiment_name]["status"] = "completed"
            self.experiments[experiment_name]["end_date"] = datetime.now()
            
            # Get final recommendation
            recommendation = self.get_experiment_recommendation(experiment_name)
            
            logging.info(f"Experiment {experiment_name} completed. Recommendation: {recommendation['recommendation']}")
            
            return recommendation

# Usage example
if __name__ == "__main__":
    framework = ABTestingFramework()
    
    # Create experiment
    experiment = framework.create_experiment(
        experiment_name="recommendation_v2_test",
        model_a="recommendation-v1",
        model_b="recommendation-v2",
        traffic_split=0.5,
        duration_days=7
    )
    
    # Simulate traffic assignment
    for i in range(1000):
        user_id = f"user_{i}"
        assigned_model = framework.assign_traffic("recommendation_v2_test", user_id)
        
        # Simulate prediction
        prediction = np.random.choice([0, 1], p=[0.3, 0.7])
        actual = np.random.choice([0, 1], p=[0.3, 0.7])
        
        framework.record_prediction("recommendation_v2_test", assigned_model, user_id, prediction, actual)
    
    # Calculate metrics
    metrics = framework.calculate_metrics("recommendation_v2_test")
    print("Experiment Metrics:")
    print(json.dumps(metrics, indent=2, default=str))
    
    # Get recommendation
    recommendation = framework.get_experiment_recommendation("recommendation_v2_test")
    print(f"\nRecommendation: {recommendation['recommendation']}")
    print(f"Reason: {recommendation['reason']}")
```

## 🎯 Applications

### 1. E-commerce Recommendation A/B Testing

**Problem**: An e-commerce platform wants to test a new recommendation algorithm against the current one.

**Solution**:
- **A/B Testing**: 50/50 traffic split between old and new models
- **Metrics**: Click-through rate, conversion rate, revenue per user
- **Duration**: 2 weeks with statistical significance testing
- **Results**: New model shows 15% improvement in conversion rate

### 2. Fraud Detection Model Deployment

**Problem**: A bank needs to deploy a new fraud detection model with zero downtime.

**Solution**:
- **Blue-Green Deployment**: Deploy new model alongside existing one
- **Traffic Switching**: Gradually shift traffic to new model
- **Monitoring**: Real-time fraud detection accuracy and false positive rates
- **Results**: Zero downtime deployment with 20% improvement in fraud detection

### 3. Healthcare AI Model Rollout

**Problem**: A hospital needs to safely deploy AI diagnostic models with regulatory compliance.

**Solution**:
- **Canary Deployment**: Start with 5% of cases using new model
- **Monitoring**: Track diagnostic accuracy and patient outcomes
- **Rollback Plan**: Immediate rollback if performance degrades
- **Results**: Gradual rollout with 99.9% safety record

## 🧪 Exercises and Projects

### Exercise 1: Deploy a Model with Rolling Updates

Deploy a scikit-learn model using rolling update strategy:

```python
# Your task: Implement rolling update deployment
# Requirements:
# 1. Create model packaging with MLflow
# 2. Deploy using Kubernetes rolling update
# 3. Implement health checks
# 4. Add monitoring and metrics
# 5. Test rollback functionality

# Starter code:
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def package_model(model, model_name):
    """Package model with MLflow"""
    # TODO: Implement model packaging
    pass

def deploy_model(model_name, image):
    """Deploy model to Kubernetes"""
    # TODO: Implement deployment
    pass

def monitor_deployment(model_name):
    """Monitor deployment health"""
    # TODO: Implement monitoring
    pass
```

### Exercise 2: Implement A/B Testing

Create an A/B testing framework for model comparison:

```python
# Your task: Build A/B testing framework
# Requirements:
# 1. Traffic splitting between model versions
# 2. Metrics collection and analysis
# 3. Statistical significance testing
# 4. Automated recommendations
# 5. Experiment management

# Starter code:
class ABTestManager:
    def __init__(self):
        # TODO: Initialize A/B testing manager
        pass
    
    def create_experiment(self, model_a, model_b):
        """Create A/B testing experiment"""
        # TODO: Implement experiment creation
        pass
    
    def assign_traffic(self, user_id):
        """Assign user to model version"""
        # TODO: Implement traffic assignment
        pass
    
    def analyze_results(self):
        """Analyze experiment results"""
        # TODO: Implement results analysis
        pass
```

### Project: Complete Model Deployment Pipeline

Build a production-ready model deployment system:

**Requirements**:
1. **Model Packaging**: MLflow integration with versioning
2. **Deployment Strategies**: Rolling update, blue-green, canary
3. **A/B Testing**: Traffic splitting and statistical analysis
4. **Monitoring**: Real-time metrics and alerting
5. **Rollback**: Automated rollback capabilities
6. **Documentation**: Complete deployment guide

**Deliverables**:
- Model packaging pipeline
- Deployment automation scripts
- A/B testing framework
- Monitoring dashboards
- Rollback procedures

## 📖 Further Reading

### Essential Resources

1. **Books**:
   - "Continuous Delivery" by Jez Humble
   - "Site Reliability Engineering" by Google
   - "A/B Testing: The Most Powerful Way to Test" by Dan Siroker

2. **Online Courses**:
   - Coursera: "Machine Learning Engineering for Production"
   - edX: "DevOps for Data Science"
   - DataCamp: "MLOps Fundamentals"

3. **Tools and Technologies**:
   - **MLflow**: Model lifecycle management
   - **Kubernetes**: Container orchestration
   - **Istio**: Service mesh for traffic management
   - **Prometheus**: Metrics collection
   - **Grafana**: Monitoring dashboards
   - **ArgoCD**: GitOps deployment

4. **2025 Trends**:
   - **GitOps for ML**: Git-based model deployment
   - **Automated Rollbacks**: AI-powered deployment safety
   - **Multi-Model Serving**: Unified serving for different frameworks
   - **Edge Deployment**: Distributed model serving
   - **Compliance Automation**: Automated regulatory compliance

### Certification Path

1. **Beginner**: Google Cloud Professional ML Engineer
2. **Intermediate**: AWS Machine Learning Specialty
3. **Advanced**: Kubernetes Administrator (CKA)
4. **Expert**: Site Reliability Engineering (SRE)

## 🎯 Key Takeaways

1. **Model deployment is critical** for ML success
2. **Multiple deployment strategies** provide safety and flexibility
3. **A/B testing** ensures evidence-based deployment decisions
4. **Monitoring and rollback** are essential for production systems
5. **Automation** reduces human error and speeds deployment
6. **Compliance** must be built into deployment processes

*"Deploy with confidence, monitor with vigilance, rollback with speed"*

**Next: [Model Monitoring](ml_engineering/24_model_monitoring.md) → Monitoring ML models in production for performance and drift**