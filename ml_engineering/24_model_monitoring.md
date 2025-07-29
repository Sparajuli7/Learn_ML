# Model Monitoring: Keeping AI Systems Healthy in Production
*"Monitoring is the heartbeat of AI systems - without it, you're flying blind"*

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Monitoring Fundamentals](#monitoring-fundamentals)
3. [Data Drift Detection](#data-drift-detection)
4. [Model Performance Monitoring](#model-performance-monitoring)
5. [Infrastructure Monitoring](#infrastructure-monitoring)
6. [Alerting and Incident Response](#alerting-and-incident-response)
7. [Implementation Examples](#implementation-examples)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

## 🎯 Introduction

Model monitoring is like having a health checkup system for your AI - you need to continuously monitor vital signs, detect problems early, and take action before things go wrong. In 2025, with AI systems becoming increasingly critical to business operations, comprehensive monitoring is not just important; it's essential for survival.

### Why Model Monitoring Matters in 2025

The AI landscape in 2025 demands monitoring systems that can handle:
- **Real-time Detection**: Immediate identification of model degradation
- **Data Drift**: Monitoring changes in input data distributions
- **Performance Degradation**: Tracking model accuracy and latency
- **Bias Detection**: Monitoring for fairness and bias issues
- **Compliance**: Ensuring regulatory requirements are met
- **Cost Optimization**: Monitoring resource usage and efficiency

### The Monitoring Evolution

Model monitoring has evolved dramatically:

- **2010s**: Basic accuracy tracking with manual checks
- **2015s**: Automated monitoring with simple alerts
- **2020s**: Comprehensive monitoring with drift detection
- **2025**: AI-native monitoring with predictive maintenance

## 🧮 Mathematical Foundations

### Monitoring Performance Metrics

#### 1. Model Accuracy (MA)
```
MA = (Correct predictions) / (Total predictions) × 100
```

#### 2. Data Drift Score (DDS)
```
DDS = KL_divergence(reference_distribution, current_distribution)
```

#### 3. Model Latency (ML)
```
ML = (Total processing time) / (Number of requests)
```

#### 4. Drift Threshold (DT)
```
DT = μ ± (2 × σ)  # 95% confidence interval
```

### Example Calculation

For a model processing 10,000 predictions:
- Correct predictions: 9,500
- Average latency: 50ms
- Drift score: 0.15 (threshold: 0.2)

```
MA = (9,500 / 10,000) × 100 = 95%
ML = 50ms
DDS = 0.15 (within threshold)
Status: Healthy
```

## 💻 Implementation

### 1. Data Drift Detection with Evidently AI

Evidently AI is like a smart doctor for your data - it continuously monitors your data health and alerts you when something looks suspicious.

```python
# Why: Detect data drift and model performance degradation
# How: Statistical analysis of data distributions and model outputs
# Where: Production ML monitoring systems
# What: Automated drift detection and alerting
# When: For continuous model health monitoring

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

class DataDriftMonitor:
    def __init__(self):
        # Why: Initialize data drift monitoring system
        # How: Set up Evidently AI for drift detection
        # Where: Production ML monitoring environments
        # What: Automated drift detection and reporting
        # When: At system startup
        
        self.reference_data = None
        self.drift_thresholds = {
            "data_drift": 0.2,
            "target_drift": 0.15,
            "column_drift": 0.25
        }
        logging.info("Data drift monitor initialized")
    
    def set_reference_data(self, reference_df: pd.DataFrame):
        """Set reference dataset for drift detection"""
        # Why: Establish baseline data distribution for comparison
        # How: Store reference dataset for drift calculations
        # Where: Model monitoring setup
        # What: Baseline data for drift detection
        # When: When setting up monitoring for a new model
        
        self.reference_data = reference_df.copy()
        logging.info(f"Set reference data with {len(reference_df)} samples")
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         target_column: Optional[str] = None) -> Dict[str, Any]:
        """Detect data drift between reference and current data"""
        # Why: Identify changes in data distribution that may affect model performance
        # How: Compare current data distribution with reference using statistical tests
        # Where: Production data pipelines
        # What: Drift detection results with confidence scores
        # When: When new data arrives or periodically
        
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        # Create column mapping for Evidently
        column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=[col for col in current_data.columns if current_data[col].dtype in ['int64', 'float64']],
            categorical_features=[col for col in current_data.columns if current_data[col].dtype == 'object']
        )
        
        # Generate drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric()
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Extract drift metrics
        drift_metrics = drift_report.metrics[0].result
        dataset_drift = drift_report.metrics[1].result
        
        # Analyze drift results
        drift_analysis = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": {
                "detected": dataset_drift.dataset_drift,
                "drift_score": dataset_drift.drift_score,
                "threshold": self.drift_thresholds["data_drift"]
            },
            "column_drift": {},
            "overall_status": "healthy"
        }
        
        # Check individual column drift
        drifted_columns = []
        for column_name, drift_info in drift_metrics.drift_by_columns.items():
            drift_score = drift_info.drift_score
            drift_analysis["column_drift"][column_name] = {
                "drift_score": drift_score,
                "detected": drift_score > self.drift_thresholds["column_drift"],
                "threshold": self.drift_thresholds["column_drift"]
            }
            
            if drift_score > self.drift_thresholds["column_drift"]:
                drifted_columns.append(column_name)
        
        # Determine overall status
        if dataset_drift.dataset_drift or len(drifted_columns) > 0:
            drift_analysis["overall_status"] = "drift_detected"
            drift_analysis["drifted_columns"] = drifted_columns
        
        logging.info(f"Data drift analysis completed. Status: {drift_analysis['overall_status']}")
        return drift_analysis
    
    def detect_target_drift(self, current_data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect target drift for supervised learning models"""
        # Why: Monitor changes in target variable distribution
        # How: Compare target distribution between reference and current data
        # Where: Supervised learning model monitoring
        # What: Target drift detection results
        # When: When monitoring supervised learning models
        
        if self.reference_data is None or target_column not in self.reference_data.columns:
            raise ValueError("Reference data not set or target column not found")
        
        # Create column mapping
        column_mapping = ColumnMapping(target=target_column)
        
        # Generate target drift report
        target_drift_report = Report(metrics=[
            TargetDriftPreset()
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Extract target drift metrics
        target_drift_metrics = target_drift_report.metrics[0].result
        
        target_analysis = {
            "timestamp": datetime.now().isoformat(),
            "target_drift": {
                "detected": target_drift_metrics.target_drift,
                "drift_score": target_drift_metrics.drift_score,
                "threshold": self.drift_thresholds["target_drift"]
            },
            "status": "healthy" if not target_drift_metrics.target_drift else "drift_detected"
        }
        
        logging.info(f"Target drift analysis completed. Status: {target_analysis['status']}")
        return target_analysis
    
    def generate_drift_report(self, current_data: pd.DataFrame, 
                            target_column: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive drift report"""
        # Why: Provide complete drift analysis for decision making
        # How: Combine data drift and target drift analysis
        # Where: Model monitoring dashboards
        # What: Comprehensive drift report with recommendations
        # When: For regular monitoring or when drift is detected
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_drift": self.detect_data_drift(current_data, target_column),
            "recommendations": []
        }
        
        # Add target drift if target column is provided
        if target_column:
            report["target_drift"] = self.detect_target_drift(current_data, target_column)
        
        # Generate recommendations
        if report["data_drift"]["overall_status"] == "drift_detected":
            report["recommendations"].append({
                "type": "warning",
                "message": "Data drift detected. Consider retraining model.",
                "severity": "medium"
            })
        
        if target_column and report.get("target_drift", {}).get("status") == "drift_detected":
            report["recommendations"].append({
                "type": "critical",
                "message": "Target drift detected. Model retraining required.",
                "severity": "high"
            })
        
        if not report["recommendations"]:
            report["recommendations"].append({
                "type": "info",
                "message": "No drift detected. Model performing normally.",
                "severity": "low"
            })
        
        return report

# Usage example
if __name__ == "__main__":
    monitor = DataDriftMonitor()
    
    # Generate reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(0, 1, 1000),
        'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    monitor.set_reference_data(reference_data)
    
    # Generate current data with some drift
    current_data = pd.DataFrame({
        'feature_1': np.random.normal(0.2, 1, 500),  # Slight drift
        'feature_2': np.random.normal(0, 1, 500),
        'feature_3': np.random.choice(['A', 'B', 'C'], 500),
        'target': np.random.choice([0, 1], 500, p=[0.6, 0.4])  # Target drift
    })
    
    # Detect drift
    drift_report = monitor.generate_drift_report(current_data, target_column='target')
    print("Drift Report:")
    print(json.dumps(drift_report, indent=2))
```

### 2. Model Performance Monitoring with Prometheus

Prometheus is like a sophisticated health monitoring system - it tracks every vital sign of your AI system and alerts you when something goes wrong.

```python
# Why: Monitor model performance metrics in real-time
# How: Collect and store time-series metrics
# Where: Production ML monitoring systems
# What: Real-time performance monitoring and alerting
# When: For continuous model performance tracking

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import time
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

class ModelPerformanceMonitor:
    def __init__(self):
        # Why: Initialize comprehensive model performance monitoring
        # How: Set up Prometheus metrics collectors
        # Where: Production ML monitoring environments
        # What: Real-time performance monitoring system
        # When: At system startup
        
        # Define Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'model_version', 'status']
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Model prediction latency in seconds',
            ['model_name', 'model_version']
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy_percentage',
            'Model accuracy percentage',
            ['model_name', 'model_version']
        )
        
        self.model_throughput = Gauge(
            'model_throughput_predictions_per_second',
            'Model throughput in predictions per second',
            ['model_name', 'model_version']
        )
        
        self.data_drift_score = Gauge(
            'model_data_drift_score',
            'Data drift score for model',
            ['model_name', 'model_version']
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage of model in bytes',
            ['model_name', 'model_version']
        )
        
        self.model_cpu_usage = Gauge(
            'model_cpu_usage_percent',
            'CPU usage of model in percent',
            ['model_name', 'model_version']
        )
        
        logging.info("Model performance monitor initialized")
    
    def record_prediction(self, model_name: str, model_version: str, 
                         status: str, latency: float, prediction: Any = None, 
                         actual: Any = None):
        """Record prediction metrics"""
        # Why: Track individual prediction performance and success rates
        # How: Increment counters and record histograms for each prediction
        # Where: Model serving endpoints
        # What: Detailed prediction metrics for analysis
        # When: After each model prediction
        
        # Record prediction count
        self.prediction_counter.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        
        # Record latency
        self.prediction_latency.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(latency)
        
        logging.debug(f"Recorded prediction: {model_name} v{model_version}, {status}, {latency:.3f}s")
    
    def update_accuracy(self, model_name: str, model_version: str, accuracy: float):
        """Update model accuracy metric"""
        # Why: Track model accuracy over time
        # How: Update gauge metric with current accuracy
        # Where: Model evaluation and monitoring
        # What: Real-time accuracy tracking
        # When: After accuracy calculations or model evaluation
        
        self.model_accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
        
        logging.info(f"Updated accuracy for {model_name} v{model_version}: {accuracy:.2f}%")
    
    def update_throughput(self, model_name: str, model_version: str, 
                         predictions_count: int, time_window: float):
        """Update model throughput metric"""
        # Why: Monitor model processing capacity
        # How: Calculate and update throughput metric
        # Where: Performance monitoring systems
        # What: Real-time throughput tracking
        # When: Periodically or after batch processing
        
        throughput = predictions_count / time_window if time_window > 0 else 0
        self.model_throughput.labels(
            model_name=model_name,
            model_version=model_version
        ).set(throughput)
        
        logging.debug(f"Updated throughput for {model_name} v{model_version}: {throughput:.2f} pred/s")
    
    def update_drift_score(self, model_name: str, model_version: str, drift_score: float):
        """Update data drift score metric"""
        # Why: Track data drift over time
        # How: Update gauge metric with current drift score
        # Where: Drift monitoring systems
        # What: Real-time drift tracking
        # When: After drift detection calculations
        
        self.data_drift_score.labels(
            model_name=model_name,
            model_version=model_version
        ).set(drift_score)
        
        logging.info(f"Updated drift score for {model_name} v{model_version}: {drift_score:.3f}")
    
    def update_resource_usage(self, model_name: str, model_version: str, 
                            memory_bytes: int, cpu_percent: float):
        """Update resource usage metrics"""
        # Why: Monitor system resource consumption
        # How: Update gauge metrics with current resource usage
        # Where: Infrastructure monitoring
        # What: Real-time resource usage tracking
        # When: Periodically for resource monitoring
        
        self.model_memory_usage.labels(
            model_name=model_name,
            model_version=model_version
        ).set(memory_bytes)
        
        self.model_cpu_usage.labels(
            model_name=model_name,
            model_version=model_version
        ).set(cpu_percent)
        
        logging.debug(f"Updated resource usage for {model_name} v{model_version}: {memory_bytes} bytes, {cpu_percent}% CPU")
    
    def generate_performance_report(self, model_name: str, model_version: str, 
                                  time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Why: Provide detailed performance insights for decision making
        # How: Aggregate metrics over specified time window
        # Where: Performance monitoring dashboards
        # What: Comprehensive performance analysis
        # When: For regular reporting or when issues are detected
        
        # In real implementation, you'd query Prometheus for metrics
        # For demonstration, we'll create a simulated report
        
        report = {
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "time_window": str(time_window),
            "metrics": {
                "total_predictions": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "current_accuracy": 0.0,
                "current_throughput": 0.0,
                "current_drift_score": 0.0,
                "memory_usage_bytes": 0,
                "cpu_usage_percent": 0.0
            },
            "status": "healthy",
            "alerts": []
        }
        
        # Simulate metrics (in real implementation, query Prometheus)
        report["metrics"]["total_predictions"] = 10000
        report["metrics"]["successful_predictions"] = 9500
        report["metrics"]["failed_predictions"] = 500
        report["metrics"]["avg_latency"] = 0.15
        report["metrics"]["p95_latency"] = 0.25
        report["metrics"]["p99_latency"] = 0.35
        report["metrics"]["current_accuracy"] = 95.0
        report["metrics"]["current_throughput"] = 100.0
        report["metrics"]["current_drift_score"] = 0.12
        report["metrics"]["memory_usage_bytes"] = 2 * 1024**3  # 2GB
        report["metrics"]["cpu_usage_percent"] = 75.0
        
        # Determine status and generate alerts
        if report["metrics"]["current_accuracy"] < 90.0:
            report["status"] = "degraded"
            report["alerts"].append({
                "type": "accuracy_degradation",
                "severity": "high",
                "message": f"Model accuracy below threshold: {report['metrics']['current_accuracy']:.1f}%"
            })
        
        if report["metrics"]["current_drift_score"] > 0.2:
            report["status"] = "drift_detected"
            report["alerts"].append({
                "type": "data_drift",
                "severity": "medium",
                "message": f"Data drift detected: {report['metrics']['current_drift_score']:.3f}"
            })
        
        if report["metrics"]["avg_latency"] > 0.5:
            report["status"] = "performance_issue"
            report["alerts"].append({
                "type": "high_latency",
                "severity": "medium",
                "message": f"High latency detected: {report['metrics']['avg_latency']:.3f}s"
            })
        
        return report

class ModelAlerting:
    def __init__(self, monitor: ModelPerformanceMonitor):
        # Why: Set up automated alerting for model issues
        # How: Define alert thresholds and notification channels
        # Where: Production ML monitoring systems
        # What: Automated alerting system for model issues
        # When: When model performance issues are detected
        
        self.monitor = monitor
        self.alert_thresholds = {
            "accuracy_min": 90.0,
            "latency_max": 0.5,
            "drift_max": 0.2,
            "error_rate_max": 0.05,
            "memory_usage_max": 0.9,
            "cpu_usage_max": 0.8
        }
        
        logging.info("Model alerting system initialized")
    
    def check_alerts(self, performance_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions based on performance report"""
        # Why: Detect model performance issues automatically
        # How: Compare metrics against thresholds
        # Where: Automated monitoring systems
        # What: Alert notifications for issues
        # When: When performance metrics exceed thresholds
        
        alerts = []
        metrics = performance_report["metrics"]
        
        # Check accuracy alert
        if metrics["current_accuracy"] < self.alert_thresholds["accuracy_min"]:
            alerts.append({
                "type": "accuracy_degradation",
                "severity": "high",
                "model": performance_report["model_name"],
                "version": performance_report["model_version"],
                "metric": "accuracy",
                "value": metrics["current_accuracy"],
                "threshold": self.alert_thresholds["accuracy_min"],
                "message": f"Model accuracy below threshold: {metrics['current_accuracy']:.1f}%"
            })
        
        # Check latency alert
        if metrics["avg_latency"] > self.alert_thresholds["latency_max"]:
            alerts.append({
                "type": "high_latency",
                "severity": "medium",
                "model": performance_report["model_name"],
                "version": performance_report["model_version"],
                "metric": "latency",
                "value": metrics["avg_latency"],
                "threshold": self.alert_thresholds["latency_max"],
                "message": f"High latency detected: {metrics['avg_latency']:.3f}s"
            })
        
        # Check drift alert
        if metrics["current_drift_score"] > self.alert_thresholds["drift_max"]:
            alerts.append({
                "type": "data_drift",
                "severity": "medium",
                "model": performance_report["model_name"],
                "version": performance_report["model_version"],
                "metric": "drift_score",
                "value": metrics["current_drift_score"],
                "threshold": self.alert_thresholds["drift_max"],
                "message": f"Data drift detected: {metrics['current_drift_score']:.3f}"
            })
        
        # Check error rate alert
        error_rate = metrics["failed_predictions"] / metrics["total_predictions"] if metrics["total_predictions"] > 0 else 0
        if error_rate > self.alert_thresholds["error_rate_max"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "high",
                "model": performance_report["model_name"],
                "version": performance_report["model_version"],
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self.alert_thresholds["error_rate_max"],
                "message": f"High error rate detected: {error_rate:.3f}"
            })
        
        return alerts

# Usage example
if __name__ == "__main__":
    monitor = ModelPerformanceMonitor()
    alerting = ModelAlerting(monitor)
    
    # Simulate some metrics
    monitor.record_prediction("recommendation-model", "v1.2.0", "success", 0.15)
    monitor.record_prediction("recommendation-model", "v1.2.0", "error", 0.25)
    monitor.update_accuracy("recommendation-model", "v1.2.0", 92.5)
    monitor.update_drift_score("recommendation-model", "v1.2.0", 0.18)
    monitor.update_resource_usage("recommendation-model", "v1.2.0", 2 * 1024**3, 75.0)
    
    # Generate performance report
    report = monitor.generate_performance_report("recommendation-model", "v1.2.0")
    print("Performance Report:")
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

### 3. Infrastructure Monitoring with Grafana

Grafana is like a comprehensive dashboard for your AI factory - it visualizes all the important metrics and helps you spot problems before they become critical.

```python
# Why: Visualize and monitor ML infrastructure metrics
# How: Create dashboards and alerts for infrastructure monitoring
# Where: Production ML infrastructure monitoring
# What: Real-time infrastructure monitoring and visualization
# When: For continuous infrastructure health monitoring

import requests
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

class GrafanaMLDashboard:
    def __init__(self, grafana_url: str, api_key: str):
        # Why: Initialize Grafana dashboard for ML monitoring
        # How: Connect to Grafana API with authentication
        # Where: ML infrastructure monitoring
        # What: Dashboard management for ML metrics
        # When: At system startup
        
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        logging.info(f"Initialized Grafana dashboard at {grafana_url}")
    
    def create_ml_dashboard(self, dashboard_name: str) -> Dict[str, Any]:
        """Create comprehensive ML monitoring dashboard"""
        # Why: Set up complete monitoring dashboard for ML systems
        # How: Create Grafana dashboard with ML-specific panels
        # Where: ML infrastructure monitoring
        # What: Comprehensive monitoring dashboard
        # When: When setting up monitoring for new ML systems
        
        dashboard = {
            "dashboard": {
                "title": dashboard_name,
                "tags": ["ml", "monitoring"],
                "timezone": "browser",
                "panels": [
                    # Model Performance Panel
                    {
                        "title": "Model Performance",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "model_accuracy_percentage",
                                "legendFormat": "{{model_name}} v{{model_version}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": 90},
                                        {"color": "green", "value": 95}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    
                    # Prediction Latency Panel
                    {
                        "title": "Prediction Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(model_prediction_latency_seconds_sum[5m]) / rate(model_prediction_latency_seconds_count[5m])",
                                "legendFormat": "{{model_name}} v{{model_version}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    
                    # Throughput Panel
                    {
                        "title": "Model Throughput",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_throughput_predictions_per_second",
                                "legendFormat": "{{model_name}} v{{model_version}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    
                    # Data Drift Panel
                    {
                        "title": "Data Drift Score",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_data_drift_score",
                                "legendFormat": "{{model_name}} v{{model_version}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 0.1},
                                        {"color": "red", "value": 0.2}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    
                    # Resource Usage Panel
                    {
                        "title": "Resource Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_memory_usage_bytes / 1024 / 1024 / 1024",
                                "legendFormat": "Memory (GB) - {{model_name}} v{{model_version}}"
                            },
                            {
                                "expr": "model_cpu_usage_percent",
                                "legendFormat": "CPU (%) - {{model_name}} v{{model_version}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                    },
                    
                    # Prediction Count Panel
                    {
                        "title": "Total Predictions",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "increase(model_predictions_total[1h])",
                                "legendFormat": "{{model_name}} v{{model_version}} - {{status}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24}
                    },
                    
                    # Error Rate Panel
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(model_predictions_total{status=\"error\"}[5m]) / rate(model_predictions_total[5m])",
                                "legendFormat": "{{model_name}} v{{model_version}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 0.01},
                                        {"color": "red", "value": 0.05}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "10s"
            },
            "folderId": 0,
            "overwrite": True
        }
        
        try:
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                headers=self.headers,
                json=dashboard
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Created ML dashboard: {result['url']}")
                return result
            else:
                logging.error(f"Failed to create dashboard: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating dashboard: {e}")
            return None
    
    def create_alert_rule(self, alert_name: str, condition: str, 
                         duration: str = "5m") -> Dict[str, Any]:
        """Create alert rule for ML monitoring"""
        # Why: Set up automated alerts for critical issues
        # How: Create Grafana alert rules with conditions
        # Where: ML monitoring systems
        # What: Automated alerting for model issues
        # When: When setting up monitoring alerts
        
        alert_rule = {
            "alert": {
                "name": alert_name,
                "condition": condition,
                "for": duration,
                "annotations": {
                    "description": f"Alert for {alert_name}",
                    "summary": f"{alert_name} threshold exceeded"
                },
                "labels": {
                    "severity": "warning",
                    "team": "ml-ops"
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.grafana_url}/api/v1/provisioning/alerting/rules",
                headers=self.headers,
                json=alert_rule
            )
            
            if response.status_code == 200:
                logging.info(f"Created alert rule: {alert_name}")
                return response.json()
            else:
                logging.error(f"Failed to create alert rule: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating alert rule: {e}")
            return None
    
    def get_dashboard_metrics(self, dashboard_uid: str, 
                             time_range: str = "1h") -> Dict[str, Any]:
        """Get metrics from dashboard"""
        # Why: Retrieve current metrics for analysis
        # How: Query dashboard metrics from Grafana API
        # Where: ML monitoring analysis
        # What: Current metric values for decision making
        # When: For regular monitoring or when issues are detected
        
        try:
            response = requests.get(
                f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                dashboard_data = response.json()
                
                # Extract metrics (simplified - in real implementation, query actual metrics)
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "dashboard_uid": dashboard_uid,
                    "time_range": time_range,
                    "metrics": {
                        "model_accuracy": 95.2,
                        "avg_latency": 0.15,
                        "throughput": 100.5,
                        "drift_score": 0.12,
                        "memory_usage_gb": 2.1,
                        "cpu_usage_percent": 75.0,
                        "total_predictions": 10000,
                        "error_rate": 0.02
                    }
                }
                
                return metrics
            else:
                logging.error(f"Failed to get dashboard metrics: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting dashboard metrics: {e}")
            return None

# Usage example
if __name__ == "__main__":
    # Initialize dashboard (in real implementation, provide actual credentials)
    dashboard = GrafanaMLDashboard(
        grafana_url="http://localhost:3000",
        api_key="your-api-key"
    )
    
    # Create ML monitoring dashboard
    result = dashboard.create_ml_dashboard("ML Model Monitoring")
    
    if result:
        print(f"✅ Dashboard created: {result['url']}")
        
        # Create alert rules
        alerts = [
            ("Model Accuracy Alert", "model_accuracy_percentage < 90"),
            ("High Latency Alert", "avg(model_prediction_latency_seconds) > 0.5"),
            ("Data Drift Alert", "model_data_drift_score > 0.2"),
            ("High Error Rate Alert", "rate(model_predictions_total{status=\"error\"}[5m]) / rate(model_predictions_total[5m]) > 0.05")
        ]
        
        for alert_name, condition in alerts:
            alert_result = dashboard.create_alert_rule(alert_name, condition)
            if alert_result:
                print(f"✅ Alert rule created: {alert_name}")
            else:
                print(f"❌ Failed to create alert rule: {alert_name}")
    else:
        print("❌ Failed to create dashboard")
```

## 🎯 Applications

### 1. E-commerce Recommendation Monitoring

**Problem**: An e-commerce platform needs to monitor recommendation model performance in real-time.

**Solution**:
- **Data Drift Detection**: Monitor user behavior changes
- **Performance Monitoring**: Track click-through rates and conversion
- **Alerting**: Immediate alerts for performance degradation
- **Results**: 99.9% uptime, 15% improvement in conversion rates

### 2. Healthcare AI Monitoring

**Problem**: A hospital needs to monitor AI diagnostic models with strict compliance requirements.

**Solution**:
- **Accuracy Monitoring**: Real-time diagnostic accuracy tracking
- **Bias Detection**: Monitor for fairness issues
- **Compliance Monitoring**: Ensure regulatory requirements are met
- **Results**: 100% compliance, 30% faster diagnosis, 99.5% accuracy

### 3. Financial Fraud Detection Monitoring

**Problem**: A bank needs to monitor fraud detection models with zero tolerance for false positives.

**Solution**:
- **Real-time Monitoring**: Sub-second detection of model issues
- **Drift Detection**: Monitor transaction pattern changes
- **Performance Tracking**: Track fraud detection accuracy
- **Results**: 95% fraud detection rate, 0.1% false positives

## 🧪 Exercises and Projects

### Exercise 1: Build a Drift Detection System

Create a comprehensive drift detection system:

```python
# Your task: Implement drift detection for ML models
# Requirements:
# 1. Statistical drift detection
# 2. Feature-level drift analysis
# 3. Automated alerting
# 4. Drift visualization
# 5. Historical drift tracking

# Starter code:
import pandas as pd
import numpy as np
from scipy import stats

class DriftDetector:
    def __init__(self):
        # TODO: Initialize drift detection system
        pass
    
    def detect_drift(self, reference_data, current_data):
        """Detect data drift between reference and current data"""
        # TODO: Implement drift detection
        pass
    
    def calculate_drift_score(self, feature_name, reference_dist, current_dist):
        """Calculate drift score for a feature"""
        # TODO: Implement drift score calculation
        pass
    
    def generate_drift_report(self):
        """Generate comprehensive drift report"""
        # TODO: Implement drift reporting
        pass
```

### Exercise 2: Create Monitoring Dashboard

Build a comprehensive monitoring dashboard:

```python
# Your task: Create ML monitoring dashboard
# Requirements:
# 1. Real-time metrics visualization
# 2. Performance tracking
# 3. Alert management
# 4. Historical analysis
# 5. Custom metrics

# Starter code:
class MLDashboard:
    def __init__(self):
        # TODO: Initialize dashboard components
        pass
    
    def add_metric_panel(self, metric_name, query):
        """Add metric panel to dashboard"""
        # TODO: Implement panel addition
        pass
    
    def create_alert(self, condition, threshold):
        """Create alert rule"""
        # TODO: Implement alert creation
        pass
    
    def generate_report(self):
        """Generate monitoring report"""
        # TODO: Implement report generation
        pass
```

### Project: Complete ML Monitoring System

Build a production-ready ML monitoring system:

**Requirements**:
1. **Data Drift Detection**: Statistical and ML-based drift detection
2. **Performance Monitoring**: Real-time accuracy and latency tracking
3. **Infrastructure Monitoring**: Resource usage and health monitoring
4. **Alerting System**: Automated alerts with escalation
5. **Visualization**: Comprehensive dashboards and reports
6. **Compliance**: Regulatory compliance monitoring

**Deliverables**:
- Drift detection system
- Performance monitoring framework
- Alerting and notification system
- Monitoring dashboards
- Compliance reporting

## 📖 Further Reading

### Essential Resources

1. **Books**:
   - "Monitoring and Observability" by Charity Majors
   - "Site Reliability Engineering" by Google
   - "Data Quality" by Thomas C. Redman

2. **Online Courses**:
   - Coursera: "Machine Learning Engineering for Production"
   - edX: "Monitoring and Observability"
   - DataCamp: "MLOps Monitoring"

3. **Tools and Technologies**:
   - **Evidently AI**: Data drift detection
   - **Prometheus**: Metrics collection
   - **Grafana**: Monitoring dashboards
   - **MLflow**: Model monitoring
   - **Weights & Biases**: Experiment tracking
   - **Arize AI**: ML observability

4. **2025 Trends**:
   - **AI-Native Monitoring**: Automated monitoring with AI
   - **Predictive Monitoring**: Proactive issue detection
   - **Multi-Model Monitoring**: Unified monitoring for different frameworks
   - **Edge Monitoring**: Distributed monitoring for edge AI
   - **Compliance Automation**: Automated regulatory compliance

### Certification Path

1. **Beginner**: Google Cloud Professional ML Engineer
2. **Intermediate**: AWS Machine Learning Specialty
3. **Advanced**: Site Reliability Engineering (SRE)
4. **Expert**: ML Observability Specialist

## 🎯 Key Takeaways

1. **Monitoring is essential** for production ML systems
2. **Data drift detection** prevents model degradation
3. **Real-time alerting** enables quick response to issues
4. **Comprehensive dashboards** provide visibility into system health
5. **Automated monitoring** reduces manual effort and errors
6. **Compliance monitoring** ensures regulatory requirements are met

*"Good monitoring is like having a crystal ball - you can see problems before they happen"*

**Next: [MLOps Automation](ml_engineering/25_mlops_automation.md) → Automating the complete ML lifecycle with CI/CD pipelines**