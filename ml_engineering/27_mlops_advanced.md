# MLOps Advanced: Production ML Systems

*"From experimentation to production: Building robust, scalable ML systems"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Data Lineage and Governance](#data-lineage-and-governance)
3. [Concept Drift Detection](#concept-drift-detection)
4. [Automated Retraining](#automated-retraining)
5. [Model Serving and Deployment](#model-serving-and-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Practical Implementation](#practical-implementation)
8. [Real-World Applications](#real-world-applications)
9. [Exercises and Projects](#exercises-and-projects)
10. [Further Reading](#further-reading)

---

## 🎯 Introduction

Advanced MLOps represents the intersection of machine learning, software engineering, and DevOps, focusing on building production-ready ML systems that are reliable, scalable, and maintainable. In 2025, the field has evolved beyond basic CI/CD to encompass sophisticated data management, automated retraining, and intelligent monitoring systems.

### Historical Context

MLOps emerged in the 2010s as organizations struggled to deploy ML models at scale. The early 2020s saw the rise of dedicated MLOps platforms and tools, while the mid-2020s brought focus on data lineage, automated retraining, and production-grade monitoring. Today, MLOps encompasses the entire ML lifecycle with emphasis on reliability and automation.

### Current State (2025)

- **Data Lineage**: End-to-end traceability of data transformations
- **Concept Drift**: Real-time detection and automated response
- **Automated Retraining**: Self-healing ML systems
- **Model Serving**: High-performance inference at scale
- **Observability**: Comprehensive monitoring and debugging
- **Governance**: Compliance and audit trails

---

## 📊 Data Lineage and Governance

### Data Lineage Fundamentals

**Data Lineage** tracks the complete journey of data through the ML pipeline:

```
Raw Data → Preprocessing → Feature Engineering → Training → Model → Inference
```

**Lineage Components**:
- **Data Sources**: Origin and metadata
- **Transformations**: Processing steps and parameters
- **Dependencies**: Data relationships and flows
- **Provenance**: Version history and changes

### Lineage Implementation

```python
import mlflow
from dataclasses import dataclass
from typing import Dict, List, Any
import hashlib
import json

@dataclass
class DataLineage:
    """Track data lineage through ML pipeline"""
    data_id: str
    source: str
    transformations: List[Dict]
    metadata: Dict[str, Any]
    timestamp: str
    hash: str
    
    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _compute_hash(self):
        """Compute hash of data lineage"""
        content = f"{self.data_id}{self.source}{json.dumps(self.transformations)}"
        return hashlib.sha256(content.encode()).hexdigest()

class LineageTracker:
    def __init__(self):
        self.lineages = []
    
    def track_transformation(self, data_id: str, source: str, 
                           transformation: Dict, metadata: Dict = None):
        """Track a data transformation"""
        lineage = DataLineage(
            data_id=data_id,
            source=source,
            transformations=[transformation],
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            hash=""
        )
        self.lineages.append(lineage)
        return lineage
    
    def get_lineage(self, data_id: str) -> List[DataLineage]:
        """Get complete lineage for data ID"""
        return [l for l in self.lineages if l.data_id == data_id]
    
    def export_lineage(self, format: str = "json"):
        """Export lineage data"""
        if format == "json":
            return json.dumps([l.__dict__ for l in self.lineages], indent=2)
        elif format == "graph":
            return self._to_graph_format()
    
    def _to_graph_format(self):
        """Convert to graph format for visualization"""
        nodes = []
        edges = []
        
        for lineage in self.lineages:
            nodes.append({
                "id": lineage.data_id,
                "label": lineage.source,
                "metadata": lineage.metadata
            })
            
            for transformation in lineage.transformations:
                edges.append({
                    "source": lineage.data_id,
                    "target": transformation.get("output_id", "unknown"),
                    "label": transformation.get("type", "transform")
                })
        
        return {"nodes": nodes, "edges": edges}
```

### Data Governance

**Data Quality Monitoring**:
```python
class DataQualityMonitor:
    def __init__(self):
        self.quality_metrics = {}
    
    def check_completeness(self, data):
        """Check for missing values"""
        missing_ratio = data.isnull().sum() / len(data)
        return {
            "completeness": 1 - missing_ratio.mean(),
            "missing_columns": missing_ratio[missing_ratio > 0.1].index.tolist()
        }
    
    def check_consistency(self, data, schema):
        """Check data against schema"""
        violations = []
        for column, expected_type in schema.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    violations.append({
                        "column": column,
                        "expected": expected_type,
                        "actual": actual_type
                    })
        return violations
    
    def check_freshness(self, data_timestamp, expected_frequency):
        """Check if data is fresh"""
        current_time = datetime.now()
        age = current_time - data_timestamp
        
        if age > expected_frequency:
            return {
                "freshness": "stale",
                "age_hours": age.total_seconds() / 3600
            }
        return {"freshness": "fresh"}
    
    def comprehensive_check(self, data, schema, timestamp):
        """Run comprehensive data quality check"""
        results = {
            "completeness": self.check_completeness(data),
            "consistency": self.check_consistency(data, schema),
            "freshness": self.check_freshness(timestamp, timedelta(hours=24))
        }
        
        # Overall quality score
        quality_score = 1.0
        if results["completeness"]["completeness"] < 0.9:
            quality_score *= 0.8
        if results["consistency"]:
            quality_score *= 0.7
        if results["freshness"]["freshness"] == "stale":
            quality_score *= 0.6
        
        results["quality_score"] = quality_score
        return results
```

---

## 🔄 Concept Drift Detection

### Drift Detection Methods

**Statistical Methods**:
```python
import numpy as np
from scipy import stats
from sklearn.metrics import kl_divergence

class ConceptDriftDetector:
    def __init__(self, window_size=1000, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_distribution = None
        self.drift_history = []
    
    def fit(self, reference_data):
        """Fit detector with reference data"""
        self.reference_distribution = self._compute_distribution(reference_data)
    
    def detect_drift(self, current_data):
        """Detect concept drift in current data"""
        current_distribution = self._compute_distribution(current_data)
        
        # Multiple drift detection methods
        drift_scores = {
            "ks_test": self._kolmogorov_smirnov_test(current_distribution),
            "kl_divergence": self._kl_divergence_test(current_distribution),
            "wasserstein": self._wasserstein_distance(current_distribution),
            "chi_square": self._chi_square_test(current_distribution)
        }
        
        # Combined drift score
        combined_score = np.mean(list(drift_scores.values()))
        drift_detected = combined_score > self.threshold
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": combined_score,
            "individual_scores": drift_scores,
            "timestamp": datetime.now()
        }
        
        self.drift_history.append(result)
        return result
    
    def _compute_distribution(self, data):
        """Compute distribution statistics"""
        if isinstance(data, pd.DataFrame):
            # For multivariate data, compute per-feature
            distributions = {}
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    distributions[column] = {
                        'mean': data[column].mean(),
                        'std': data[column].std(),
                        'histogram': np.histogram(data[column], bins=20)[0]
                    }
            return distributions
        else:
            # For univariate data
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'histogram': np.histogram(data, bins=20)[0]
            }
    
    def _kolmogorov_smirnov_test(self, current_dist):
        """KS test for distribution comparison"""
        if self.reference_distribution is None:
            return 0.0
        
        # Compare histograms
        ref_hist = self.reference_distribution.get('histogram', [])
        curr_hist = current_dist.get('histogram', [])
        
        if len(ref_hist) > 0 and len(curr_hist) > 0:
            # Normalize histograms
            ref_norm = ref_hist / np.sum(ref_hist)
            curr_norm = curr_hist / np.sum(curr_hist)
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_norm, curr_norm)
            return 1 - p_value  # Higher value = more drift
        
        return 0.0
    
    def _kl_divergence_test(self, current_dist):
        """KL divergence for distribution comparison"""
        if self.reference_distribution is None:
            return 0.0
        
        ref_hist = self.reference_distribution.get('histogram', [])
        curr_hist = current_dist.get('histogram', [])
        
        if len(ref_hist) > 0 and len(curr_hist) > 0:
            # Normalize and add small epsilon to avoid log(0)
            ref_norm = ref_hist / np.sum(ref_hist) + 1e-10
            curr_norm = curr_hist / np.sum(curr_hist) + 1e-10
            
            # KL divergence
            kl_div = kl_divergence(ref_norm, curr_norm)
            return min(kl_div, 1.0)  # Cap at 1.0
        
        return 0.0
    
    def _wasserstein_distance(self, current_dist):
        """Wasserstein distance for distribution comparison"""
        if self.reference_distribution is None:
            return 0.0
        
        ref_mean = self.reference_distribution.get('mean', 0)
        curr_mean = current_dist.get('mean', 0)
        
        # Simple Wasserstein distance approximation
        return abs(ref_mean - curr_mean) / (self.reference_distribution.get('std', 1) + 1e-10)
    
    def _chi_square_test(self, current_dist):
        """Chi-square test for distribution comparison"""
        if self.reference_distribution is None:
            return 0.0
        
        ref_hist = self.reference_distribution.get('histogram', [])
        curr_hist = current_dist.get('histogram', [])
        
        if len(ref_hist) > 0 and len(curr_hist) > 0:
            # Chi-square test
            chi2_stat, p_value = stats.chisquare(curr_hist, ref_hist)
            return 1 - p_value  # Higher value = more drift
        
        return 0.0
```

### Real-time Drift Detection

```python
class RealTimeDriftDetector:
    def __init__(self, window_size=100, update_frequency=1000):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.data_buffer = []
        self.drift_detector = ConceptDriftDetector()
        self.last_update = 0
    
    def add_data_point(self, data_point):
        """Add new data point and check for drift"""
        self.data_buffer.append(data_point)
        
        # Check if we have enough data and it's time to update
        if (len(self.data_buffer) >= self.window_size and 
            len(self.data_buffer) - self.last_update >= self.update_frequency):
            
            # Get current window
            current_window = self.data_buffer[-self.window_size:]
            
            # Detect drift
            drift_result = self.drift_detector.detect_drift(current_window)
            
            # Update reference if no drift detected
            if not drift_result["drift_detected"]:
                self.drift_detector.fit(current_window)
            
            self.last_update = len(self.data_buffer)
            
            return drift_result
        
        return None
    
    def get_drift_summary(self):
        """Get summary of drift detection history"""
        if not self.drift_detector.drift_history:
            return {"total_checks": 0, "drift_count": 0}
        
        total_checks = len(self.drift_detector.drift_history)
        drift_count = sum(1 for result in self.drift_detector.drift_history 
                         if result["drift_detected"])
        
        return {
            "total_checks": total_checks,
            "drift_count": drift_count,
            "drift_rate": drift_count / total_checks if total_checks > 0 else 0,
            "last_drift": self.drift_detector.drift_history[-1] if self.drift_detector.drift_history else None
        }
```

---

## 🤖 Automated Retraining

### Retraining Triggers

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class RetrainingTrigger(Enum):
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_ACCUMULATION = "data_accumulation"
    MANUAL = "manual"

@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    trigger_type: RetrainingTrigger
    schedule: str = None  # Cron expression for scheduled retraining
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    data_threshold: int = 10000
    max_retraining_frequency: int = 24  # hours

class AutomatedRetrainer:
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.last_retraining = None
        self.retraining_history = []
        self.drift_detector = ConceptDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()
    
    def should_retrain(self, current_metrics: Dict[str, Any]) -> bool:
        """Determine if retraining is needed"""
        current_time = datetime.now()
        
        # Check if enough time has passed since last retraining
        if (self.last_retraining and 
            (current_time - self.last_retraining).total_seconds() < 
            self.config.max_retraining_frequency * 3600):
            return False
        
        # Check different triggers
        if self.config.trigger_type == RetrainingTrigger.SCHEDULED:
            return self._check_schedule(current_time)
        
        elif self.config.trigger_type == RetrainingTrigger.DRIFT_DETECTED:
            return self._check_drift(current_metrics)
        
        elif self.config.trigger_type == RetrainingTrigger.PERFORMANCE_DEGRADATION:
            return self._check_performance(current_metrics)
        
        elif self.config.trigger_type == RetrainingTrigger.DATA_ACCUMULATION:
            return self._check_data_accumulation(current_metrics)
        
        return False
    
    def _check_schedule(self, current_time):
        """Check if it's time for scheduled retraining"""
        if not self.config.schedule:
            return False
        
        # Simple cron-like schedule checking
        # In practice, use a proper cron parser
        return True  # Placeholder
    
    def _check_drift(self, current_metrics):
        """Check for concept drift"""
        if "drift_score" in current_metrics:
            return current_metrics["drift_score"] > self.config.drift_threshold
        return False
    
    def _check_performance(self, current_metrics):
        """Check for performance degradation"""
        if "performance_degradation" in current_metrics:
            return current_metrics["performance_degradation"] > self.config.performance_threshold
        return False
    
    def _check_data_accumulation(self, current_metrics):
        """Check if enough new data has accumulated"""
        if "new_data_count" in current_metrics:
            return current_metrics["new_data_count"] > self.config.data_threshold
        return False
    
    def trigger_retraining(self, trigger_reason: str):
        """Trigger the retraining process"""
        retraining_event = {
            "timestamp": datetime.now(),
            "trigger": trigger_reason,
            "config": self.config.__dict__
        }
        
        # Execute retraining
        success = self._execute_retraining()
        retraining_event["success"] = success
        
        self.retraining_history.append(retraining_event)
        self.last_retraining = datetime.now()
        
        return success
    
    def _execute_retraining(self) -> bool:
        """Execute the actual retraining process"""
        try:
            # This would integrate with your training pipeline
            # For example, using Airflow, Kubeflow, or custom orchestration
            
            # 1. Prepare new training data
            # 2. Train new model
            # 3. Evaluate new model
            # 4. Compare with current model
            # 5. Deploy if better
            
            print("Executing retraining...")
            return True
            
        except Exception as e:
            print(f"Retraining failed: {e}")
            return False
```

### Model Performance Monitoring

```python
class ModelPerformanceMonitor:
    def __init__(self, baseline_metrics: Dict[str, float]):
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        self.degradation_threshold = 0.05
    
    def update_performance(self, current_metrics: Dict[str, float]):
        """Update performance metrics and detect degradation"""
        current_time = datetime.now()
        
        # Calculate degradation
        degradation = {}
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                degradation[metric] = (baseline_value - current_value) / baseline_value
        
        # Overall degradation score
        overall_degradation = np.mean(list(degradation.values())) if degradation else 0
        
        performance_record = {
            "timestamp": current_time,
            "metrics": current_metrics,
            "degradation": degradation,
            "overall_degradation": overall_degradation,
            "degradation_detected": overall_degradation > self.degradation_threshold
        }
        
        self.performance_history.append(performance_record)
        
        return performance_record
    
    def get_performance_trend(self, metric: str, window_size: int = 100):
        """Get performance trend for a specific metric"""
        if len(self.performance_history) < window_size:
            return None
        
        recent_history = self.performance_history[-window_size:]
        values = [record["metrics"].get(metric, 0) for record in recent_history]
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        return {
            "trend": slope,
            "trend_direction": "improving" if slope > 0 else "degrading",
            "current_value": values[-1],
            "baseline_value": self.baseline_metrics.get(metric, 0)
        }
```

---

## 🚀 Model Serving and Deployment

### Model Serving Architecture

```python
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import Dict, Any, List

class ModelServer:
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model = self._load_model(model_path)
        self.config = config
        self.app = FastAPI()
        self._setup_routes()
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        # Load model based on framework
        if model_path.endswith('.pth'):
            model = torch.load(model_path, map_location='cpu')
        elif model_path.endswith('.onnx'):
            import onnxruntime as ort
            model = ort.InferenceSession(model_path)
        else:
            # Load other formats
            model = self._load_custom_model(model_path)
        
        return model
    
    def _setup_routes(self):
        """Setup API routes"""
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                prediction = await self._predict(request.data)
                return PredictionResponse(
                    prediction=prediction,
                    model_version=self.config.get("version", "1.0"),
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "model_loaded": self.model is not None}
        
        @self.app.get("/metrics")
        async def metrics():
            return self._get_metrics()
    
    async def _predict(self, data: Dict[str, Any]):
        """Make prediction"""
        # Preprocess input
        processed_data = self._preprocess(data)
        
        # Make prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(processed_data)
        elif hasattr(self.model, '__call__'):
            prediction = self.model(processed_data)
        else:
            # Handle different model types
            prediction = self._custom_predict(processed_data)
        
        # Postprocess output
        return self._postprocess(prediction)
    
    def _preprocess(self, data: Dict[str, Any]):
        """Preprocess input data"""
        # Implement preprocessing logic
        return data
    
    def _postprocess(self, prediction):
        """Postprocess prediction output"""
        # Implement postprocessing logic
        return prediction
    
    def _get_metrics(self):
        """Get model serving metrics"""
        return {
            "requests_processed": len(self.performance_history),
            "average_latency": self._calculate_average_latency(),
            "error_rate": self._calculate_error_rate(),
            "model_version": self.config.get("version", "1.0")
        }
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the model server"""
        uvicorn.run(self.app, host=host, port=port)

class PredictionRequest(BaseModel):
    data: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: Any
    model_version: str
    timestamp: str
```

### Load Balancing and Scaling

```python
import asyncio
from typing import List
import aiohttp
import random

class LoadBalancer:
    def __init__(self, model_servers: List[str]):
        self.servers = model_servers
        self.health_checks = {}
        self.request_counts = {server: 0 for server in model_servers}
    
    async def health_check(self, server: str) -> bool:
        """Check if server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server}/health") as response:
                    return response.status == 200
        except:
            return False
    
    async def update_health_status(self):
        """Update health status of all servers"""
        for server in self.servers:
            self.health_checks[server] = await self.health_check(server)
    
    def get_healthy_servers(self) -> List[str]:
        """Get list of healthy servers"""
        return [server for server, healthy in self.health_checks.items() if healthy]
    
    def select_server(self, strategy: str = "round_robin") -> str:
        """Select server based on strategy"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        if strategy == "round_robin":
            # Simple round-robin
            server = healthy_servers[self.request_counts[healthy_servers[0]] % len(healthy_servers)]
            self.request_counts[server] += 1
            return server
        
        elif strategy == "random":
            return random.choice(healthy_servers)
        
        elif strategy == "least_connections":
            # Select server with least requests
            return min(healthy_servers, key=lambda s: self.request_counts[s])
        
        else:
            return healthy_servers[0]
    
    async def make_request(self, data: Dict[str, Any], strategy: str = "round_robin"):
        """Make request through load balancer"""
        server = self.select_server(strategy)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server}/predict",
                    json={"data": data}
                ) as response:
                    return await response.json()
        except Exception as e:
            # Mark server as unhealthy and retry
            self.health_checks[server] = False
            if len(self.get_healthy_servers()) > 0:
                return await self.make_request(data, strategy)
            else:
                raise e
```

---

## 📊 Monitoring and Observability

### Comprehensive Monitoring

```python
import time
import psutil
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class MonitoringMetrics:
    """Comprehensive monitoring metrics"""
    # System metrics
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    
    # Model metrics
    prediction_latency: float
    throughput: float
    error_rate: float
    accuracy: float
    
    # Business metrics
    revenue_impact: float
    user_satisfaction: float
    cost_per_prediction: float

class ModelMonitor:
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.alert_thresholds = alert_thresholds
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> MonitoringMetrics:
        """Collect current metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_metrics = {
            "bytes_sent": network_io.bytes_sent,
            "bytes_recv": network_io.bytes_recv
        }
        
        # Model metrics (placeholder - would come from actual model serving)
        prediction_latency = self._get_prediction_latency()
        throughput = self._get_throughput()
        error_rate = self._get_error_rate()
        accuracy = self._get_accuracy()
        
        # Business metrics (placeholder)
        revenue_impact = self._get_revenue_impact()
        user_satisfaction = self._get_user_satisfaction()
        cost_per_prediction = self._get_cost_per_prediction()
        
        return MonitoringMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_metrics,
            prediction_latency=prediction_latency,
            throughput=throughput,
            error_rate=error_rate,
            accuracy=accuracy,
            revenue_impact=revenue_impact,
            user_satisfaction=user_satisfaction,
            cost_per_prediction=cost_per_prediction
        )
    
    def _check_alerts(self, metrics: MonitoringMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # System alerts
        if metrics.cpu_usage > self.alert_thresholds.get("cpu_usage", 80):
            alerts.append({
                "type": "system",
                "metric": "cpu_usage",
                "value": metrics.cpu_usage,
                "threshold": self.alert_thresholds["cpu_usage"],
                "timestamp": datetime.now()
            })
        
        if metrics.memory_usage > self.alert_thresholds.get("memory_usage", 80):
            alerts.append({
                "type": "system",
                "metric": "memory_usage",
                "value": metrics.memory_usage,
                "threshold": self.alert_thresholds["memory_usage"],
                "timestamp": datetime.now()
            })
        
        # Model alerts
        if metrics.error_rate > self.alert_thresholds.get("error_rate", 0.05):
            alerts.append({
                "type": "model",
                "metric": "error_rate",
                "value": metrics.error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "timestamp": datetime.now()
            })
        
        if metrics.prediction_latency > self.alert_thresholds.get("latency", 1000):
            alerts.append({
                "type": "model",
                "metric": "prediction_latency",
                "value": metrics.prediction_latency,
                "threshold": self.alert_thresholds["latency"],
                "timestamp": datetime.now()
            })
        
        # Business alerts
        if metrics.user_satisfaction < self.alert_thresholds.get("satisfaction", 0.7):
            alerts.append({
                "type": "business",
                "metric": "user_satisfaction",
                "value": metrics.user_satisfaction,
                "threshold": self.alert_thresholds["satisfaction"],
                "timestamp": datetime.now()
            })
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Send alerts (in practice, integrate with alerting system)
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        # In practice, integrate with:
        # - Email/SMS services
        # - Slack/Discord webhooks
        # - PagerDuty
        # - Custom alerting systems
        
        print(f"ALERT: {alert['type']} - {alert['metric']} = {alert['value']}")
    
    # Placeholder methods for metrics collection
    def _get_prediction_latency(self) -> float:
        return random.uniform(50, 200)  # ms
    
    def _get_throughput(self) -> float:
        return random.uniform(100, 500)  # requests/sec
    
    def _get_error_rate(self) -> float:
        return random.uniform(0.01, 0.05)  # percentage
    
    def _get_accuracy(self) -> float:
        return random.uniform(0.85, 0.95)  # percentage
    
    def _get_revenue_impact(self) -> float:
        return random.uniform(1000, 5000)  # dollars
    
    def _get_user_satisfaction(self) -> float:
        return random.uniform(0.7, 0.9)  # score
    
    def _get_cost_per_prediction(self) -> float:
        return random.uniform(0.001, 0.01)  # dollars
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 minutes
        
        return {
            "current_metrics": recent_metrics[-1].__dict__,
            "average_metrics": self._calculate_averages(recent_metrics),
            "trends": self._calculate_trends(recent_metrics),
            "alerts_last_hour": len([a for a in self.alerts 
                                   if (datetime.now() - a["timestamp"]).seconds < 3600])
        }
    
    def _calculate_averages(self, metrics: List[MonitoringMetrics]) -> Dict[str, float]:
        """Calculate average metrics"""
        if not metrics:
            return {}
        
        return {
            "avg_cpu_usage": np.mean([m.cpu_usage for m in metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in metrics]),
            "avg_latency": np.mean([m.prediction_latency for m in metrics]),
            "avg_throughput": np.mean([m.throughput for m in metrics]),
            "avg_error_rate": np.mean([m.error_rate for m in metrics])
        }
    
    def _calculate_trends(self, metrics: List[MonitoringMetrics]) -> Dict[str, str]:
        """Calculate trends for metrics"""
        if len(metrics) < 2:
            return {}
        
        trends = {}
        
        # Calculate trends for key metrics
        latency_trend = metrics[-1].prediction_latency - metrics[0].prediction_latency
        trends["latency_trend"] = "increasing" if latency_trend > 0 else "decreasing"
        
        error_trend = metrics[-1].error_rate - metrics[0].error_rate
        trends["error_trend"] = "increasing" if error_trend > 0 else "decreasing"
        
        return trends
```

---

## 💻 Practical Implementation

### Complete MLOps Pipeline

```python
class MLOpsPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lineage_tracker = LineageTracker()
        self.drift_detector = ConceptDriftDetector()
        self.retrainer = AutomatedRetrainer(config.get("retraining_config"))
        self.monitor = ModelMonitor(config.get("alert_thresholds"))
        self.model_server = None
        
    def setup_pipeline(self):
        """Setup the complete MLOps pipeline"""
        # Initialize components
        self._setup_data_pipeline()
        self._setup_training_pipeline()
        self._setup_deployment_pipeline()
        self._setup_monitoring_pipeline()
        
        print("MLOps pipeline setup complete")
    
    def _setup_data_pipeline(self):
        """Setup data processing pipeline"""
        # Data ingestion
        # Data validation
        # Data lineage tracking
        # Data quality monitoring
        pass
    
    def _setup_training_pipeline(self):
        """Setup model training pipeline"""
        # Automated retraining triggers
        # Model versioning
        # Experiment tracking
        # Model evaluation
        pass
    
    def _setup_deployment_pipeline(self):
        """Setup model deployment pipeline"""
        # Model serving
        # Load balancing
        # A/B testing
        # Rollback mechanisms
        pass
    
    def _setup_monitoring_pipeline(self):
        """Setup monitoring pipeline"""
        # Performance monitoring
        # Drift detection
        # Alerting
        # Logging
        pass
    
    def run_pipeline(self):
        """Run the complete MLOps pipeline"""
        try:
            # 1. Data processing
            processed_data = self._process_data()
            
            # 2. Drift detection
            drift_result = self.drift_detector.detect_drift(processed_data)
            
            # 3. Retraining decision
            if self.retrainer.should_retrain({"drift_score": drift_result.get("drift_score", 0)}):
                self.retrainer.trigger_retraining("drift_detected")
            
            # 4. Model serving
            if self.model_server is None:
                self._deploy_model()
            
            # 5. Monitoring
            metrics = self.monitor._collect_metrics()
            self.monitor._check_alerts(metrics)
            
            return {
                "status": "success",
                "drift_detected": drift_result.get("drift_detected", False),
                "retraining_triggered": False,  # Would be set based on retraining decision
                "metrics": metrics.__dict__
            }
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _process_data(self):
        """Process incoming data"""
        # Placeholder for data processing
        return pd.DataFrame(np.random.randn(100, 10))
    
    def _deploy_model(self):
        """Deploy model to serving infrastructure"""
        # Placeholder for model deployment
        print("Deploying model...")
        self.model_server = "deployed"

# Example usage
if __name__ == "__main__":
    config = {
        "retraining_config": RetrainingConfig(
            trigger_type=RetrainingTrigger.DRIFT_DETECTED,
            drift_threshold=0.1
        ),
        "alert_thresholds": {
            "cpu_usage": 80,
            "memory_usage": 80,
            "error_rate": 0.05,
            "latency": 1000,
            "satisfaction": 0.7
        }
    }
    
    pipeline = MLOpsPipeline(config)
    pipeline.setup_pipeline()
    
    # Run pipeline
    result = pipeline.run_pipeline()
    print(f"Pipeline result: {result}")
```

---

## 🎯 Real-World Applications

### 1. E-commerce Recommendation Systems

**Challenges**:
- High traffic and low latency requirements
- Concept drift in user preferences
- A/B testing for optimization

**MLOps Solutions**:
- Real-time drift detection
- Automated retraining on new data
- Load balancing for high availability

### 2. Financial Trading Systems

**Challenges**:
- Regulatory compliance
- High-frequency predictions
- Risk management

**MLOps Solutions**:
- Comprehensive audit trails
- Real-time monitoring
- Automated rollback mechanisms

### 3. Healthcare Diagnostics

**Challenges**:
- Data privacy and security
- Model interpretability
- Regulatory approval

**MLOps Solutions**:
- Secure data lineage
- Explainable AI integration
- Compliance monitoring

### 4. Autonomous Vehicles

**Challenges**:
- Safety-critical systems
- Real-time processing
- Edge deployment

**MLOps Solutions**:
- Robust monitoring
- Edge model serving
- Safety validation

### 5. Content Recommendation

**Challenges**:
- Personalization at scale
- Content freshness
- User engagement

**MLOps Solutions**:
- Automated retraining
- Real-time personalization
- Engagement monitoring

---

## 🧪 Exercises and Projects

### Beginner Exercises

1. **Basic Monitoring Setup**
   ```python
   # Set up basic model monitoring
   # Implement drift detection
   # Create alerting system
   ```

2. **Data Lineage Tracking**
   ```python
   # Implement data lineage tracking
   # Visualize data flows
   # Track data quality metrics
   ```

3. **Simple Retraining Pipeline**
   ```python
   # Build automated retraining triggers
   # Implement model versioning
   # Create deployment pipeline
   ```

### Intermediate Projects

1. **End-to-End MLOps Pipeline**
   - Build complete pipeline from data to deployment
   - Implement monitoring and alerting
   - Add automated retraining

2. **Multi-Model Serving System**
   - Implement load balancing
   - Add A/B testing capabilities
   - Build rollback mechanisms

3. **Advanced Drift Detection**
   - Implement multiple drift detection methods
   - Build adaptive thresholds
   - Create drift visualization dashboard

### Advanced Projects

1. **Production ML Platform**
   - Build scalable ML infrastructure
   - Implement multi-tenant architecture
   - Add comprehensive governance

2. **Real-Time ML Systems**
   - Build streaming ML pipelines
   - Implement real-time drift detection
   - Create low-latency serving

3. **ML Governance Platform**
   - Implement comprehensive audit trails
   - Add compliance monitoring
   - Build explainability features

### Quiz Questions

1. **Conceptual Questions**
   - What is the difference between data lineage and data provenance?
   - How does concept drift differ from data drift?
   - What are the trade-offs in automated retraining?

2. **Technical Questions**
   - How would you implement real-time drift detection?
   - What are the challenges in model serving at scale?
   - How do you ensure model reproducibility?

3. **Architecture Questions**
   - How would you design a multi-model serving system?
   - What are the considerations for edge ML deployment?
   - How do you handle model versioning and rollbacks?

---

## 📖 Further Reading

### Essential Papers

1. **"Hidden Technical Debt in Machine Learning Systems"** - Sculley et al. (2015)
2. **"MLOps: Continuous Delivery and Automation Pipelines in Machine Learning"** - Kreuzberger et al. (2023)
3. **"Concept Drift Detection and Adaptation for Machine Learning"** - Gama et al. (2014)

### Books

1. **"Building Machine Learning Pipelines"** - Hapke & Nelson
2. **"Machine Learning Engineering"** - Zinkevich
3. **"Designing Machine Learning Systems"** - Huyen

### Online Resources

1. **Platforms**: Kubeflow, MLflow, Airflow
2. **Tools**: Prometheus, Grafana, Jaeger
3. **Frameworks**: TensorFlow Serving, TorchServe, Seldon

### Next Steps

1. **Advanced Topics**: Edge ML, federated learning, privacy-preserving ML
2. **Production Systems**: Large-scale deployment, multi-region serving
3. **Domain Specialization**: Industry-specific MLOps patterns

---

## 🎯 Key Takeaways

1. **Automation**: Reduce manual intervention in ML lifecycle
2. **Monitoring**: Comprehensive observability is crucial
3. **Governance**: Ensure compliance and auditability
4. **Scalability**: Design for growth from the start
5. **Reliability**: Build robust, fault-tolerant systems

---

*"MLOps is not just about deploying models—it's about building reliable, scalable ML systems that deliver value continuously."*

**Next: [Full Stack ML](ml_engineering/28_full_stack_ml.md) → End-to-end applications and microservices**