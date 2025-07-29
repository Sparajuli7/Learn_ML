# MLflow Advanced

## Overview
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. This guide covers advanced experiment tracking, model management, and production deployments for 2025.

## Table of Contents
1. [MLflow Fundamentals](#mlflow-fundamentals)
2. [Experiment Tracking](#experiment-tracking)
3. [Model Management](#model-management)
4. [Model Serving](#model-serving)
5. [Production Deployments](#production-deployments)

## MLflow Fundamentals

### Basic Setup
```python
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("mlflow-demo")

# Initialize client
client = MlflowClient()
```

### Basic Experiment Tracking
```python
def train_and_log_model(X_train, X_test, y_train, y_test):
    """Train model and log to MLflow"""
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Log artifacts
        mlflow.log_artifact("data.csv")
        
        return model, accuracy
```

## Experiment Tracking

### Advanced Parameter Logging
```python
def advanced_experiment_tracking():
    """Advanced experiment tracking with multiple runs"""
    
    # Define parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    }
    
    # Generate all combinations
    from itertools import product
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    best_accuracy = 0
    best_run_id = None
    
    for params in combinations:
        with mlflow.start_run():
            # Log all parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Train model with current parameters
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = mlflow.active_run().info.run_id
    
    return best_run_id, best_accuracy
```

### Custom Metrics and Artifacts
```python
def custom_metrics_and_artifacts():
    """Log custom metrics and artifacts"""
    
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Custom metrics
        from sklearn.metrics import roc_auc_score, confusion_matrix
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log custom metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("true_positives", conf_matrix[1, 1])
        mlflow.log_metric("false_positives", conf_matrix[0, 1])
        mlflow.log_metric("true_negatives", conf_matrix[0, 0])
        mlflow.log_metric("false_negatives", conf_matrix[1, 0])
        
        # Log confusion matrix as artifact
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        mlflow.log_artifact("confusion_matrix.png")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model
```

### Nested Runs and Parent-Child Relationships
```python
def nested_experiments():
    """Demonstrate nested runs and parent-child relationships"""
    
    with mlflow.start_run(run_name="parent_experiment") as parent_run:
        # Log parent parameters
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("algorithm", "random_forest")
        
        # Child run 1: Data preprocessing
        with mlflow.start_run(run_name="data_preprocessing", nested=True) as child_run1:
            mlflow.log_param("preprocessing_method", "standard_scaler")
            mlflow.log_metric("preprocessing_time", 2.5)
            
            # Simulate preprocessing
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            mlflow.sklearn.log_model(scaler, "preprocessor")
        
        # Child run 2: Model training
        with mlflow.start_run(run_name="model_training", nested=True) as child_run2:
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
        
        # Child run 3: Model evaluation
        with mlflow.start_run(run_name="model_evaluation", nested=True) as child_run3:
            from sklearn.metrics import classification_report
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
            
            # Log detailed evaluation report
            with open("evaluation_report.txt", "w") as f:
                f.write(classification_report(y_test, y_pred))
            
            mlflow.log_artifact("evaluation_report.txt")
```

## Model Management

### Model Registry Operations
```python
def model_registry_operations():
    """Demonstrate model registry operations"""
    
    # Register model
    model_name = "iris_classifier"
    
    # Get the best run
    experiment = client.get_experiment_by_name("mlflow-demo")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    best_run = runs[0]
    
    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"Registered model version: {model_version.version}")
    
    # Transition model to staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    
    # Add model description
    client.update_registered_model(
        name=model_name,
        description="Random Forest classifier for iris dataset"
    )
    
    # Add version description
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description="Best performing model with 95% accuracy"
    )
    
    return model_name, model_version.version
```

### Model Versioning and Lifecycle
```python
def model_lifecycle_management():
    """Manage model lifecycle with versioning"""
    
    model_name = "iris_classifier"
    
    # Get all versions of a model
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    for version in model_versions:
        print(f"Version {version.version}: {version.status}")
    
    # Promote best model to production
    best_version = None
    best_accuracy = 0
    
    for version in model_versions:
        if version.status == "Staging":
            # Load model and evaluate
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_version = version.version
    
    if best_version:
        # Transition to production
        client.transition_model_version_stage(
            name=model_name,
            version=best_version,
            stage="Production"
        )
        
        # Archive old production models
        for version in model_versions:
            if version.status == "Production" and version.version != best_version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
    
    return best_version
```

### Model Comparison and Selection
```python
def model_comparison():
    """Compare different model versions"""
    
    model_name = "iris_classifier"
    
    # Get all versions
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    comparison_results = []
    
    for version in model_versions:
        # Load model
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get run details
        run = client.get_run(version.run_id)
        
        comparison_results.append({
            "version": version.version,
            "accuracy": accuracy,
            "stage": version.status,
            "created_time": version.creation_timestamp,
            "parameters": run.data.params
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values("accuracy", ascending=False)
    
    # Log comparison as artifact
    comparison_df.to_csv("model_comparison.csv", index=False)
    
    with mlflow.start_run():
        mlflow.log_artifact("model_comparison.csv")
        mlflow.log_metric("best_accuracy", comparison_df.iloc[0]["accuracy"])
        mlflow.log_param("best_version", comparison_df.iloc[0]["version"])
    
    return comparison_df
```

## Model Serving

### Local Model Serving
```python
def local_model_serving():
    """Serve model locally with MLflow"""
    
    model_name = "iris_classifier"
    model_version = 1
    
    # Load model
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    
    # Serve model locally
    mlflow.sklearn.serve(
        model_uri=f"models:/{model_name}/{model_version}",
        port=5001
    )
    
    print("Model served at http://localhost:5001")
    
    # Test prediction
    import requests
    import json
    
    # Sample data
    sample_data = {
        "data": [[5.1, 3.5, 1.4, 0.2]]
    }
    
    response = requests.post(
        "http://localhost:5001/invocations",
        json=sample_data,
        headers={"Content-Type": "application/json"}
    )
    
    prediction = response.json()
    print(f"Prediction: {prediction}")
    
    return model
```

### Production Model Serving
```python
def production_model_serving():
    """Deploy model to production"""
    
    model_name = "iris_classifier"
    
    # Get production model
    production_models = client.search_model_versions(
        f"name='{model_name}' AND status='Production'"
    )
    
    if production_models:
        production_model = production_models[0]
        
        # Deploy to production environment
        mlflow.sklearn.serve(
            model_uri=f"models:/{model_name}/{production_model.version}",
            port=5002,
            host="0.0.0.0"
        )
        
        print(f"Production model deployed at http://0.0.0.0:5002")
        
        return production_model
    else:
        print("No production model found")
        return None
```

## Production Deployments

### Docker Deployment
```python
def docker_deployment():
    """Deploy model using Docker"""
    
    model_name = "iris_classifier"
    model_version = 1
    
    # Build Docker image
    mlflow.sklearn.build_image(
        model_uri=f"models:/{model_name}/{model_version}",
        name="iris-classifier",
        tag="latest"
    )
    
    # Run Docker container
    import subprocess
    
    subprocess.run([
        "docker", "run", "-p", "5003:8080",
        "iris-classifier:latest"
    ])
    
    print("Model deployed in Docker container at http://localhost:5003")
```

### Kubernetes Deployment
```python
def kubernetes_deployment():
    """Deploy model to Kubernetes"""
    
    model_name = "iris_classifier"
    model_version = 1
    
    # Create Kubernetes deployment
    deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-classifier
  template:
    metadata:
      labels:
        app: iris-classifier
    spec:
      containers:
      - name: iris-classifier
        image: iris-classifier:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: iris-classifier-service
spec:
  selector:
    app: iris-classifier
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
"""
    
    # Apply to Kubernetes
    with open("deployment.yaml", "w") as f:
        f.write(deployment_yaml)
    
    import subprocess
    subprocess.run(["kubectl", "apply", "-f", "deployment.yaml"])
    
    print("Model deployed to Kubernetes")
```

### Model Monitoring
```python
def model_monitoring():
    """Monitor model performance in production"""
    
    model_name = "iris_classifier"
    
    # Get production model
    production_models = client.search_model_versions(
        f"name='{model_name}' AND status='Production'"
    )
    
    if production_models:
        production_model = production_models[0]
        
        # Monitor model performance
        with mlflow.start_run(run_name="production_monitoring"):
            # Load model
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{production_model.version}")
            
            # Simulate production predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log production metrics
            mlflow.log_metric("production_accuracy", accuracy)
            mlflow.log_metric("prediction_count", len(X_test))
            mlflow.log_param("model_version", production_model.version)
            
            # Alert if accuracy drops
            if accuracy < 0.9:
                mlflow.log_param("alert", "Accuracy below threshold")
                print("WARNING: Model accuracy below threshold!")
            
            # Log drift metrics
            from sklearn.metrics import drift_score
            
            # Simulate data drift
            drift_score_value = drift_score(X_train, X_test)
            mlflow.log_metric("data_drift_score", drift_score_value)
            
            if drift_score_value > 0.1:
                mlflow.log_param("alert", "Data drift detected")
                print("WARNING: Data drift detected!")
```

## Conclusion

MLflow provides a comprehensive platform for ML lifecycle management. Key areas include:

1. **Experiment Tracking**: Advanced parameter logging and custom metrics
2. **Model Management**: Registry operations and lifecycle management
3. **Model Serving**: Local and production deployment
4. **Production Deployments**: Docker, Kubernetes, and monitoring

The platform continues to evolve with new features for more efficient ML lifecycle management.

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples) 