# ML System Design

## Overview
ML system design encompasses architecture patterns, scalability principles, and production-ready system development for machine learning applications.

## Core Design Principles

### 1. Scalability Patterns
- **Horizontal Scaling**: Add more machines to handle increased load
- **Vertical Scaling**: Increase resources on existing machines
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Load Balancing**: Distribute requests across multiple instances

### 2. Reliability Patterns
- **Fault Tolerance**: System continues operating despite failures
- **Redundancy**: Multiple components for critical functions
- **Circuit Breakers**: Prevent cascade failures
- **Retry Mechanisms**: Handle transient failures gracefully

### 3. Performance Patterns
- **Caching**: Store frequently accessed data
- **Asynchronous Processing**: Non-blocking operations
- **Batch Processing**: Efficient bulk operations
- **Streaming**: Real-time data processing

## System Architecture Patterns

### 1. Microservices Architecture

```python
# Service Definition
class MLPredictionService:
    def __init__(self, model_registry, cache_client):
        self.model_registry = model_registry
        self.cache_client = cache_client
        self.load_balancer = LoadBalancer()
    
    async def predict(self, request):
        # Load balancing
        service_instance = self.load_balancer.get_instance()
        
        # Cache check
        cache_key = self._generate_cache_key(request)
        cached_result = await self.cache_client.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Model prediction
        model = await self.model_registry.get_model(request.model_id)
        prediction = await model.predict(request.data)
        
        # Cache result
        await self.cache_client.set(cache_key, prediction, ttl=3600)
        
        return prediction

# Service Discovery
class ServiceRegistry:
    def __init__(self):
        self.services = {}
    
    def register_service(self, service_name, service_instance):
        self.services[service_name] = service_instance
    
    def get_service(self, service_name):
        return self.services.get(service_name)
    
    def health_check(self):
        return {name: service.is_healthy() for name, service in self.services.items()}
```

### 2. Event-Driven Architecture

```python
import asyncio
from typing import Dict, List, Callable

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, event_data: dict):
        if event_type in self.subscribers:
            tasks = [handler(event_data) for handler in self.subscribers[event_type]]
            await asyncio.gather(*tasks, return_exceptions=True)

# Event Handlers
class ModelTrainingEventHandler:
    def __init__(self, model_registry, notification_service):
        self.model_registry = model_registry
        self.notification_service = notification_service
    
    async def handle_training_completed(self, event_data):
        model_id = event_data['model_id']
        metrics = event_data['metrics']
        
        # Update model registry
        await self.model_registry.update_model_metrics(model_id, metrics)
        
        # Send notification
        await self.notification_service.send_training_complete_notification(model_id, metrics)

class DataPipelineEventHandler:
    def __init__(self, feature_store, model_registry):
        self.feature_store = feature_store
        self.model_registry = model_registry
    
    async def handle_data_updated(self, event_data):
        dataset_id = event_data['dataset_id']
        
        # Trigger feature engineering
        await self.feature_store.update_features(dataset_id)
        
        # Notify models that depend on this data
        dependent_models = await self.model_registry.get_dependent_models(dataset_id)
        for model_id in dependent_models:
            await self.model_registry.trigger_retraining(model_id)
```

### 3. CQRS (Command Query Responsibility Segregation)

```python
from abc import ABC, abstractmethod
from typing import List, Optional
import asyncio

# Commands (Write Operations)
class Command(ABC):
    pass

class TrainModelCommand(Command):
    def __init__(self, model_config: dict, training_data_id: str):
        self.model_config = model_config
        self.training_data_id = training_data_id

class UpdateModelCommand(Command):
    def __init__(self, model_id: str, new_weights: bytes):
        self.model_id = model_id
        self.new_weights = new_weights

# Queries (Read Operations)
class Query(ABC):
    pass

class GetModelQuery(Query):
    def __init__(self, model_id: str):
        self.model_id = model_id

class GetModelPredictionsQuery(Query):
    def __init__(self, model_id: str, input_data: dict):
        self.model_id = model_id
        self.input_data = input_data

# Command Handler
class ModelCommandHandler:
    def __init__(self, model_repository, event_bus):
        self.model_repository = model_repository
        self.event_bus = event_bus
    
    async def handle_train_model(self, command: TrainModelCommand):
        # Execute training
        model_id = await self.model_repository.create_model(command.model_config)
        
        # Publish event
        await self.event_bus.publish('model_training_started', {
            'model_id': model_id,
            'config': command.model_config
        })
        
        return model_id
    
    async def handle_update_model(self, command: UpdateModelCommand):
        # Update model weights
        await self.model_repository.update_model_weights(
            command.model_id, 
            command.new_weights
        )
        
        # Publish event
        await self.event_bus.publish('model_updated', {
            'model_id': command.model_id
        })

# Query Handler
class ModelQueryHandler:
    def __init__(self, model_repository, cache_service):
        self.model_repository = model_repository
        self.cache_service = cache_service
    
    async def handle_get_model(self, query: GetModelQuery):
        # Check cache first
        cached_model = await self.cache_service.get(f"model:{query.model_id}")
        if cached_model:
            return cached_model
        
        # Fetch from repository
        model = await self.model_repository.get_model(query.model_id)
        
        # Cache result
        await self.cache_service.set(f"model:{query.model_id}", model, ttl=3600)
        
        return model
    
    async def handle_get_predictions(self, query: GetModelPredictionsQuery):
        model = await self.handle_get_model(GetModelQuery(query.model_id))
        return await model.predict(query.input_data)
```

## Scalable Data Processing

### 1. Batch Processing Pipeline

```python
from apache_beam import Pipeline, PTransform, ParDo, GroupByKey
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam

class DataProcessingPipeline:
    def __init__(self, pipeline_options: PipelineOptions):
        self.pipeline_options = pipeline_options
    
    def create_batch_pipeline(self, input_path: str, output_path: str):
        with Pipeline(options=self.pipeline_options) as pipeline:
            # Read data
            data = (pipeline 
                   | 'ReadData' >> ReadFromText(input_path)
                   | 'ParseJSON' >> ParDo(ParseJSON())
                   | 'FilterValid' >> ParDo(FilterValidRecords())
                   | 'ExtractFeatures' >> ParDo(ExtractFeatures())
                   | 'GroupByKey' >> GroupByKey()
                   | 'AggregateFeatures' >> ParDo(AggregateFeatures())
                   | 'WriteOutput' >> WriteToText(output_path))
        
        return pipeline

class ParseJSON(beam.DoFn):
    def process(self, element):
        import json
        try:
            yield json.loads(element)
        except json.JSONDecodeError:
            pass

class FilterValidRecords(beam.DoFn):
    def process(self, element):
        if self._is_valid(element):
            yield element
    
    def _is_valid(self, record):
        required_fields = ['id', 'features', 'label']
        return all(field in record for field in required_fields)

class ExtractFeatures(beam.DoFn):
    def process(self, element):
        features = self._extract_features(element)
        yield (element['id'], features)
    
    def _extract_features(self, record):
        # Feature extraction logic
        return record.get('features', {})

class AggregateFeatures(beam.DoFn):
    def process(self, element):
        key, features_list = element
        aggregated_features = self._aggregate(features_list)
        yield {'id': key, 'features': aggregated_features}
    
    def _aggregate(self, features_list):
        # Aggregation logic
        return {'aggregated': True}
```

### 2. Stream Processing Pipeline

```python
import apache_beam as beam
from apache_beam.transforms.window import FixedWindows
from apache_beam.transforms.trigger import AfterWatermark, AfterCount

class StreamProcessingPipeline:
    def __init__(self, pipeline_options):
        self.pipeline_options = pipeline_options
    
    def create_stream_pipeline(self, input_topic: str, output_topic: str):
        with Pipeline(options=self.pipeline_options) as pipeline:
            # Read from streaming source
            messages = (pipeline 
                       | 'ReadFromPubSub' >> beam.io.ReadFromPubSub(topic=input_topic)
                       | 'ParseMessage' >> ParDo(ParseStreamMessage())
                       | 'Window' >> beam.WindowInto(FixedWindows(60))  # 1-minute windows
                       | 'ProcessStream' >> ParDo(ProcessStreamData())
                       | 'WriteToPubSub' >> beam.io.WriteToPubSub(topic=output_topic))
        
        return pipeline

class ParseStreamMessage(beam.DoFn):
    def process(self, element):
        import json
        try:
            data = json.loads(element.decode('utf-8'))
            yield data
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

class ProcessStreamData(beam.DoFn):
    def process(self, element, window=beam.DoFn.WindowParam):
        # Process streaming data
        processed_data = self._process_element(element)
        yield processed_data
    
    def _process_element(self, element):
        # Real-time processing logic
        return {'processed': True, 'data': element}
```

## Model Serving Architecture

### 1. Model Server with Load Balancing

```python
import asyncio
from typing import Dict, List
import aiohttp
from aiohttp import web
import numpy as np

class ModelServer:
    def __init__(self, model_registry, load_balancer):
        self.model_registry = model_registry
        self.load_balancer = load_balancer
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        self.app.router.add_post('/predict', self.handle_predict)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_post('/models/{model_id}/load', self.handle_load_model)
    
    async def handle_predict(self, request):
        try:
            data = await request.json()
            model_id = data.get('model_id')
            input_data = data.get('input')
            
            # Get model instance
            model = await self.model_registry.get_model(model_id)
            if not model:
                return web.json_response({'error': 'Model not found'}, status=404)
            
            # Make prediction
            prediction = await model.predict(input_data)
            
            return web.json_response({
                'prediction': prediction,
                'model_id': model_id,
                'timestamp': asyncio.get_event_loop().time()
            })
        
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_health(self, request):
        return web.json_response({'status': 'healthy'})
    
    async def handle_load_model(self, request):
        model_id = request.match_info['model_id']
        await self.model_registry.load_model(model_id)
        return web.json_response({'status': 'Model loaded'})

class LoadBalancer:
    def __init__(self):
        self.servers = []
        self.current_index = 0
    
    def add_server(self, server_url: str):
        self.servers.append(server_url)
    
    def get_next_server(self) -> str:
        if not self.servers:
            raise ValueError("No servers available")
        
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    async def health_check(self) -> Dict[str, bool]:
        async with aiohttp.ClientSession() as session:
            health_results = {}
            for server in self.servers:
                try:
                    async with session.get(f"{server}/health") as response:
                        health_results[server] = response.status == 200
                except:
                    health_results[server] = False
            return health_results
```

### 2. Model Versioning and A/B Testing

```python
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib

class ModelVersion:
    def __init__(self, model_id: str, version: str, model_data: bytes, metadata: Dict):
        self.model_id = model_id
        self.version = version
        self.model_data = model_data
        self.metadata = metadata
        self.created_at = datetime.now()
        self.checksum = hashlib.md5(model_data).hexdigest()

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Dict[str, ModelVersion]] = {}
        self.active_versions: Dict[str, str] = {}
        self.experiments: Dict[str, Dict] = {}
    
    async def register_model(self, model_id: str, version: str, model_data: bytes, metadata: Dict):
        if model_id not in self.models:
            self.models[model_id] = {}
        
        model_version = ModelVersion(model_id, version, model_data, metadata)
        self.models[model_id][version] = model_version
        
        # Set as active if first version
        if model_id not in self.active_versions:
            self.active_versions[model_id] = version
    
    async def set_active_version(self, model_id: str, version: str):
        if model_id in self.models and version in self.models[model_id]:
            self.active_versions[model_id] = version
        else:
            raise ValueError(f"Model {model_id} version {version} not found")
    
    async def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        if model_id not in self.models:
            return None
        
        if version is None:
            version = self.active_versions.get(model_id)
        
        return self.models[model_id].get(version)
    
    async def create_experiment(self, experiment_id: str, model_id: str, 
                               traffic_split: Dict[str, float]):
        """Create A/B testing experiment"""
        self.experiments[experiment_id] = {
            'model_id': model_id,
            'traffic_split': traffic_split,
            'created_at': datetime.now(),
            'metrics': {}
        }
    
    async def get_experiment_model(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Get model version for A/B testing"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        model_id = experiment['model_id']
        traffic_split = experiment['traffic_split']
        
        # Deterministic assignment based on user_id
        user_hash = hash(user_id) % 100
        cumulative_prob = 0
        
        for version, probability in traffic_split.items():
            cumulative_prob += probability
            if user_hash < cumulative_prob * 100:
                return version
        
        return list(traffic_split.keys())[0]  # Fallback
```

## Monitoring and Observability

### 1. Metrics Collection

```python
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List
import prometheus_client as prom

@dataclass
class PredictionMetrics:
    model_id: str
    prediction_time: float
    input_size: int
    output_size: int
    error: bool = False
    error_message: str = ""

class MetricsCollector:
    def __init__(self):
        # Prometheus metrics
        self.prediction_duration = prom.Histogram(
            'prediction_duration_seconds',
            'Time spent on prediction',
            ['model_id']
        )
        self.prediction_requests = prom.Counter(
            'prediction_requests_total',
            'Total prediction requests',
            ['model_id', 'status']
        )
        self.model_memory_usage = prom.Gauge(
            'model_memory_bytes',
            'Memory usage by model',
            ['model_id']
        )
        
        # Custom metrics
        self.metrics_buffer: List[PredictionMetrics] = []
        self.buffer_size = 1000
    
    async def record_prediction(self, metrics: PredictionMetrics):
        # Record Prometheus metrics
        self.prediction_duration.labels(metrics.model_id).observe(metrics.prediction_time)
        
        status = 'error' if metrics.error else 'success'
        self.prediction_requests.labels(metrics.model_id, status).inc()
        
        # Buffer for batch processing
        self.metrics_buffer.append(metrics)
        
        if len(self.metrics_buffer) >= self.buffer_size:
            await self.flush_metrics()
    
    async def flush_metrics(self):
        """Flush buffered metrics to storage"""
        if not self.metrics_buffer:
            return
        
        # Process metrics in batch
        metrics_to_flush = self.metrics_buffer.copy()
        self.metrics_buffer.clear()
        
        # Store metrics (implement storage logic)
        await self.store_metrics(metrics_to_flush)
    
    async def store_metrics(self, metrics: List[PredictionMetrics]):
        """Store metrics to time-series database"""
        # Implementation for storing to InfluxDB, TimescaleDB, etc.
        pass
    
    def get_model_performance(self, model_id: str, time_window: str = "1h"):
        """Get performance metrics for a model"""
        # Implementation for querying metrics
        pass
```

### 2. Distributed Tracing

```python
import opentracing
from opentracing import tags
import asyncio
from contextlib import asynccontextmanager

class TracingService:
    def __init__(self, tracer):
        self.tracer = tracer
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **kwargs):
        span = self.tracer.start_span(operation_name)
        
        # Add tags
        for key, value in kwargs.items():
            span.set_tag(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_tag(tags.ERROR, True)
            span.log_kv({'event': 'error', 'error.object': e})
            raise
        finally:
            span.finish()
    
    async def trace_prediction(self, model_id: str, input_data: dict):
        async with self.trace_operation('model_prediction', model_id=model_id) as span:
            # Add input data size
            span.set_tag('input_size', len(str(input_data)))
            
            # Simulate prediction
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Add result
            span.set_tag('prediction_completed', True)
            
            return {'prediction': 'result'}

# Usage example
async def predict_with_tracing(tracing_service, model_id, input_data):
    return await tracing_service.trace_prediction(model_id, input_data)
```

## Deployment Strategies

### 1. Blue-Green Deployment

```python
import asyncio
from typing import Dict, List
import aiohttp

class BlueGreenDeployment:
    def __init__(self, load_balancer, health_checker):
        self.load_balancer = load_balancer
        self.health_checker = health_checker
        self.environments = {
            'blue': {'active': True, 'servers': []},
            'green': {'active': False, 'servers': []}
        }
    
    async def deploy_new_version(self, new_servers: List[str]):
        # Determine inactive environment
        inactive_env = 'green' if self.environments['blue']['active'] else 'blue'
        active_env = 'blue' if self.environments['blue']['active'] else 'green'
        
        # Deploy to inactive environment
        self.environments[inactive_env]['servers'] = new_servers
        
        # Health check new environment
        health_status = await self.health_checker.check_servers(new_servers)
        
        if all(health_status.values()):
            # Switch traffic
            await self.switch_traffic(active_env, inactive_env)
            return True
        else:
            # Rollback
            self.environments[inactive_env]['servers'] = []
            return False
    
    async def switch_traffic(self, from_env: str, to_env: str):
        # Update load balancer
        self.load_balancer.update_servers(self.environments[to_env]['servers'])
        
        # Update environment status
        self.environments[from_env]['active'] = False
        self.environments[to_env]['active'] = True
        
        # Graceful shutdown of old servers
        await self.graceful_shutdown(self.environments[from_env]['servers'])
    
    async def graceful_shutdown(self, servers: List[str]):
        """Gracefully shutdown servers"""
        for server in servers:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(f"{server}/shutdown")
            except:
                pass  # Server might already be down

class HealthChecker:
    async def check_servers(self, servers: List[str]) -> Dict[str, bool]:
        async with aiohttp.ClientSession() as session:
            health_checks = []
            for server in servers:
                health_checks.append(self.check_server(session, server))
            
            results = await asyncio.gather(*health_checks, return_exceptions=True)
            return {server: isinstance(result, bool) and result 
                   for server, result in zip(servers, results)}
    
    async def check_server(self, session: aiohttp.ClientSession, server: str) -> bool:
        try:
            async with session.get(f"{server}/health", timeout=5) as response:
                return response.status == 200
        except:
            return False
```

### 2. Canary Deployment

```python
class CanaryDeployment:
    def __init__(self, load_balancer, metrics_collector):
        self.load_balancer = load_balancer
        self.metrics_collector = metrics_collector
        self.canary_configs = {}
    
    async def start_canary(self, model_id: str, canary_config: Dict):
        """Start canary deployment"""
        self.canary_configs[model_id] = {
            'traffic_percentage': canary_config.get('traffic_percentage', 5),
            'duration_minutes': canary_config.get('duration_minutes', 30),
            'success_threshold': canary_config.get('success_threshold', 0.95),
            'error_threshold': canary_config.get('error_threshold', 0.05),
            'start_time': asyncio.get_event_loop().time(),
            'metrics': {'success': 0, 'error': 0, 'total': 0}
        }
        
        # Update load balancer with canary routing
        await self.load_balancer.set_canary_routing(model_id, canary_config['traffic_percentage'])
    
    async def monitor_canary(self, model_id: str):
        """Monitor canary deployment metrics"""
        if model_id not in self.canary_configs:
            return
        
        config = self.canary_configs[model_id]
        current_time = asyncio.get_event_loop().time()
        
        # Check if canary period is complete
        if (current_time - config['start_time']) > (config['duration_minutes'] * 60):
            await self.evaluate_canary(model_id)
    
    async def evaluate_canary(self, model_id: str):
        """Evaluate canary deployment success"""
        config = self.canary_configs[model_id]
        metrics = config['metrics']
        
        if metrics['total'] == 0:
            await self.rollback_canary(model_id)
            return
        
        success_rate = metrics['success'] / metrics['total']
        error_rate = metrics['error'] / metrics['total']
        
        if success_rate >= config['success_threshold'] and error_rate <= config['error_threshold']:
            await self.promote_canary(model_id)
        else:
            await self.rollback_canary(model_id)
    
    async def promote_canary(self, model_id: str):
        """Promote canary to full deployment"""
        await self.load_balancer.set_full_traffic(model_id)
        del self.canary_configs[model_id]
    
    async def rollback_canary(self, model_id: str):
        """Rollback canary deployment"""
        await self.load_balancer.remove_canary_routing(model_id)
        del self.canary_configs[model_id]
```

## Implementation Checklist

### Phase 1: Basic Architecture
- [ ] Design microservices architecture
- [ ] Implement service discovery
- [ ] Set up load balancing
- [ ] Create health check endpoints

### Phase 2: Data Processing
- [ ] Implement batch processing pipeline
- [ ] Set up stream processing
- [ ] Create data validation
- [ ] Add error handling

### Phase 3: Model Serving
- [ ] Build model server
- [ ] Implement model versioning
- [ ] Add A/B testing framework
- [ ] Create model registry

### Phase 4: Monitoring
- [ ] Set up metrics collection
- [ ] Implement distributed tracing
- [ ] Create alerting system
- [ ] Add performance monitoring

### Phase 5: Deployment
- [ ] Implement blue-green deployment
- [ ] Add canary deployment
- [ ] Create rollback mechanisms
- [ ] Set up CI/CD pipeline

## Resources

### Key Papers
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Building Microservices" by Sam Newman
- "Site Reliability Engineering" by Google

### Tools and Frameworks
- **Orchestration**: Kubernetes, Docker Swarm
- **Service Mesh**: Istio, Linkerd
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Message Queues**: Apache Kafka, RabbitMQ
- **Databases**: PostgreSQL, Redis, Cassandra

### Advanced Topics
- Multi-region deployment
- Disaster recovery
- Security patterns
- Cost optimization
- Performance tuning

This comprehensive guide covers ML system design patterns essential for building scalable, reliable ML systems in 2025.