# ML Frameworks Advanced: Hugging Face, ONNX, JAX

*"Master the cutting-edge frameworks that define the future of ML"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Hugging Face Ecosystem](#hugging-face-ecosystem)
3. [ONNX Interoperability](#onnx-interoperability)
4. [JAX Acceleration](#jax-acceleration)
5. [Advanced Integration](#advanced-integration)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction

Advanced ML frameworks provide specialized capabilities for specific use cases. Hugging Face dominates the NLP landscape, ONNX enables cross-framework model deployment, and JAX offers unprecedented performance for research and production. These frameworks represent the cutting edge of ML development in 2025.

### Advanced Framework Landscape

| Framework | Specialization | Key Features | Use Cases |
|-----------|---------------|--------------|-----------|
| **Hugging Face** | NLP/Transformers | Model Hub, AutoML | Language models, research |
| **ONNX** | Model Interop | Cross-platform, optimization | Production deployment |
| **JAX** | Performance | GPU/TPU acceleration | Research, high-performance |

### 2025 Advanced Trends

- **Model Compression**: Efficient deployment of large models
- **Federated Learning**: Distributed training across devices
- **AutoML Integration**: Automated model development
- **Edge Optimization**: Mobile and IoT deployment

---

## 🤗 Hugging Face Ecosystem

### 1. Advanced Transformers Usage

```python
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import torch
import numpy as np

class AdvancedHuggingFace:
    """Advanced Hugging Face implementation"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
    
    def setup_model(self, num_labels=2, task="classification"):
        """Setup model for specific task"""
        if task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
        elif task == "regression":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=1, problem_type="regression"
            )
        
        return self.model
    
    def advanced_tokenization(self, texts, max_length=512, truncation=True):
        """Advanced tokenization with custom settings"""
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt"
        )
        return tokenized
    
    def custom_training_loop(self, train_dataset, val_dataset, epochs=3):
        """Custom training with advanced features"""
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        self.trainer.train()
        return self.trainer
    
    def compute_metrics(self, eval_pred):
        """Custom metrics computation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def model_interpretation(self, text, class_names=None):
        """Model interpretation using attention weights"""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions[-1]  # Last layer attention
        
        # Average attention across heads
        attention_weights = torch.mean(attentions, dim=1)
        
        # Get token-level attention
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attention_scores = attention_weights[0].mean(dim=0)
        
        return {
            'tokens': tokens,
            'attention_scores': attention_scores.tolist(),
            'predictions': torch.softmax(outputs.logits, dim=-1).tolist()
        }

# Usage
hf_model = AdvancedHuggingFace("bert-base-uncased")
```

### 2. Custom Dataset and Pipeline

```python
class CustomHuggingFaceDataset:
    """Custom dataset for Hugging Face"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class HuggingFacePipeline:
    """Advanced Hugging Face pipeline"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
    
    def create_pipeline(self, task="text-classification"):
        """Create Hugging Face pipeline"""
        from transformers import pipeline
        
        self.pipeline = pipeline(
            task=task,
            model=self.model_name,
            tokenizer=self.tokenizer
        )
        return self.pipeline
    
    def batch_inference(self, texts, batch_size=32):
        """Batch inference for efficiency"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.pipeline(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def model_optimization(self, model, optimization_level=1):
        """Model optimization for deployment"""
        from transformers import AutoModelForSequenceClassification
        
        if optimization_level == 1:
            # Basic optimization
            model.eval()
            return model
        elif optimization_level == 2:
            # Quantization
            from torch.quantization import quantize_dynamic
            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            return model
        elif optimization_level == 3:
            # ONNX export
            import onnx
            dummy_input = torch.randn(1, 512)
            torch.onnx.export(model, dummy_input, "model.onnx")
            return model

# Usage
hf_pipeline = HuggingFacePipeline()
```

---

## 🔄 ONNX Interoperability

### 1. Cross-Framework Model Conversion

```python
import onnx
import onnxruntime as ort
import torch
import tensorflow as tf
import numpy as np

class ONNXInteroperability:
    """ONNX interoperability utilities"""
    
    def __init__(self):
        self.onnx_model = None
        self.session = None
    
    def pytorch_to_onnx(self, pytorch_model, dummy_input, output_path="model.onnx"):
        """Convert PyTorch model to ONNX"""
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Validate ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        self.onnx_model = onnx_model
        return output_path
    
    def tensorflow_to_onnx(self, tf_model, output_path="model.onnx"):
        """Convert TensorFlow model to ONNX"""
        import tf2onnx
        
        # Convert TensorFlow model
        onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        self.onnx_model = onnx_model
        return output_path
    
    def optimize_onnx_model(self, model_path, optimization_level="all"):
        """Optimize ONNX model"""
        from onnxruntime.transformers import optimizer
        
        # Optimize model
        opt_model = optimizer.optimize_model(
            model_path,
            model_type="bert",
            opt_level=optimization_level,
            use_gpu=True
        )
        
        return opt_model
    
    def onnx_inference(self, model_path, input_data):
        """Perform inference with ONNX model"""
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_data})
        
        return outputs
    
    def benchmark_models(self, pytorch_model, onnx_path, test_input):
        """Benchmark PyTorch vs ONNX performance"""
        import time
        
        # PyTorch inference
        pytorch_model.eval()
        start_time = time.time()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        pytorch_time = time.time() - start_time
        
        # ONNX inference
        start_time = time.time()
        onnx_output = self.onnx_inference(onnx_path, test_input.numpy())
        onnx_time = time.time() - start_time
        
        return {
            'pytorch_time': pytorch_time,
            'onnx_time': onnx_time,
            'speedup': pytorch_time / onnx_time
        }

# Usage
onnx_tools = ONNXInteroperability()
```

### 2. Advanced ONNX Operations

```python
class AdvancedONNX:
    """Advanced ONNX operations"""
    
    def __init__(self):
        self.models = {}
    
    def model_fusion(self, model_paths, output_path="fused_model.onnx"):
        """Fuse multiple ONNX models"""
        import onnx.compose
        
        models = []
        for path in model_paths:
            model = onnx.load(path)
            models.append(model)
        
        # Fuse models (simplified example)
        fused_model = onnx.compose.merge_models(models)
        
        # Save fused model
        onnx.save(fused_model, output_path)
        return output_path
    
    def model_quantization(self, model_path, output_path="quantized_model.onnx"):
        """Quantize ONNX model for efficiency"""
        from onnxruntime.quantization import quantize_dynamic
        
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=onnx.TensorProto.INT8
        )
        
        return output_path
    
    def model_pruning(self, model_path, output_path="pruned_model.onnx"):
        """Prune ONNX model"""
        # Load model
        model = onnx.load(model_path)
        
        # Simple pruning: remove nodes with low weights
        # This is a simplified example
        pruned_model = model
        
        # Save pruned model
        onnx.save(pruned_model, output_path)
        return output_path
    
    def cross_platform_deployment(self, model_path, target_platform):
        """Deploy ONNX model to different platforms"""
        
        if target_platform == "mobile":
            # Mobile optimization
            return self.optimize_for_mobile(model_path)
        elif target_platform == "web":
            # Web deployment
            return self.optimize_for_web(model_path)
        elif target_platform == "edge":
            # Edge device optimization
            return self.optimize_for_edge(model_path)
        else:
            return model_path
    
    def optimize_for_mobile(self, model_path):
        """Optimize model for mobile deployment"""
        # Mobile-specific optimizations
        return model_path
    
    def optimize_for_web(self, model_path):
        """Optimize model for web deployment"""
        # Web-specific optimizations
        return model_path
    
    def optimize_for_edge(self, model_path):
        """Optimize model for edge devices"""
        # Edge-specific optimizations
        return model_path

# Usage
advanced_onnx = AdvancedONNX()
```

---

## ⚡ JAX Acceleration

### 1. High-Performance JAX Implementation

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.nn import relu, softmax
import optax
from flax import linen as nn
import numpy as np

class AdvancedJAX:
    """Advanced JAX implementation"""
    
    def __init__(self):
        # Enable JIT compilation
        jax.config.update('jax_enable_x64', True)
    
    def create_high_performance_model(self, input_size, hidden_size, num_classes):
        """Create high-performance JAX model"""
        
        class JAXModel(nn.Module):
            hidden_size: int
            num_classes: int
            
            @nn.compact
            def __call__(self, x, training=True):
                x = nn.Dense(self.hidden_size)(x)
                x = relu(x)
                x = nn.Dropout(rate=0.3, deterministic=not training)(x)
                x = nn.Dense(self.hidden_size // 2)(x)
                x = relu(x)
                x = nn.Dropout(rate=0.2, deterministic=not training)(x)
                x = nn.Dense(self.num_classes)(x)
                return softmax(x, axis=-1)
        
        return JAXModel(hidden_size=hidden_size, num_classes=num_classes)
    
    @jit
    def jax_training_step(self, params, opt_state, batch, optimizer):
        """JIT-compiled training step"""
        
        def loss_fn(params, batch):
            x, y = batch
            logits = self.model.apply(params, x, training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean()
        
        loss, grads = grad(loss_fn, has_aux=False)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    @jit
    def jax_evaluation(self, params, batch):
        """JIT-compiled evaluation"""
        x, y = batch
        logits = self.model.apply(params, x, training=False)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return accuracy
    
    def vectorized_operations(self, data):
        """Vectorized operations with JAX"""
        
        # Vectorized normalization
        @jit
        def normalize(x):
            return (x - jnp.mean(x)) / jnp.std(x)
        
        # Vectorized matrix operations
        @jit
        def matrix_ops(x, y):
            return jnp.dot(x, y) + jnp.sin(x) * jnp.cos(y)
        
        # Apply operations
        normalized_data = normalize(data)
        result = matrix_ops(normalized_data, normalized_data.T)
        
        return result
    
    def parallel_training(self, params, data, num_epochs=10):
        """Parallel training with JAX"""
        
        # Setup optimizer
        optimizer = optax.adam(learning_rate=0.001)
        opt_state = optimizer.init(params)
        
        # Training loop
        for epoch in range(num_epochs):
            for batch in data:
                params, opt_state, loss = self.jax_training_step(
                    params, opt_state, batch, optimizer
                )
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return params
    
    def custom_gradients(self, x, y):
        """Custom gradient computation"""
        
        @jit
        def custom_loss(x, y):
            return jnp.sum(jnp.square(x - y)) + jnp.sum(jnp.abs(x))
        
        # Compute gradients
        grad_fn = grad(custom_loss)
        gradients = grad_fn(x, y)
        
        return gradients

# Usage
jax_model = AdvancedJAX()
```

### 2. JAX for Research and Production

```python
class JAXResearchTools:
    """JAX tools for research and production"""
    
    def __init__(self):
        self.devices = jax.devices()
    
    def distributed_training(self, model, data, num_devices=4):
        """Distributed training across multiple devices"""
        
        # Replicate model across devices
        replicated_model = jax.pmap(model)
        
        # Distribute data across devices
        distributed_data = jax.device_put_sharded(data, self.devices)
        
        # Training on multiple devices
        def train_step(model_params, batch):
            loss, grads = jax.value_and_grad(self.loss_fn)(model_params, batch)
            return loss, grads
        
        # Parallel training step
        parallel_train_step = jax.pmap(train_step)
        
        return parallel_train_step
    
    def custom_optimizer(self, learning_rate=0.001):
        """Custom optimizer with JAX"""
        
        def custom_adam(learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
            def init_fn(params):
                return {
                    'count': jnp.zeros([]),
                    'mu': jax.tree_map(jnp.zeros_like, params),
                    'nu': jax.tree_map(jnp.zeros_like, params)
                }
            
            def update_fn(updates, state, params=None):
                mu = jax.tree_map(
                    lambda u, s: beta1 * s + (1 - beta1) * u, updates, state['mu']
                )
                nu = jax.tree_map(
                    lambda u, s: beta2 * s + (1 - beta2) * jnp.square(u), 
                    updates, state['nu']
                )
                mu_hat = jax.tree_map(lambda t: t / (1 - beta1 ** (state['count'] + 1)), mu)
                nu_hat = jax.tree_map(lambda t: t / (1 - beta2 ** (state['count'] + 1)), nu)
                updates = jax.tree_map(
                    lambda m, v: learning_rate * m / (jnp.sqrt(v) + eps), mu_hat, nu_hat
                )
                return updates, {
                    'count': state['count'] + 1,
                    'mu': mu,
                    'nu': nu
                }
            
            return optax.GradientTransformation(init_fn, update_fn)
        
        return custom_adam(learning_rate)
    
    def model_interpretation(self, model, data):
        """Model interpretation with JAX"""
        
        @jit
        def compute_gradients(model_params, inputs, targets):
            def loss_fn(params, x, y):
                predictions = model.apply(params, x)
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, y))
            
            return grad(loss_fn)(model_params, inputs, targets)
        
        gradients = compute_gradients(model, data[0], data[1])
        return gradients
    
    def uncertainty_quantification(self, model, data, num_samples=100):
        """Uncertainty quantification with JAX"""
        
        @jit
        def sample_predictions(model_params, inputs, key):
            # Monte Carlo dropout
            predictions = []
            for _ in range(num_samples):
                pred = model.apply(model_params, inputs, training=True, rngs={'dropout': key})
                predictions.append(pred)
            return jnp.stack(predictions)
        
        # Generate predictions with uncertainty
        key = jax.random.PRNGKey(0)
        predictions = sample_predictions(model, data, key)
        
        # Compute uncertainty metrics
        mean_prediction = jnp.mean(predictions, axis=0)
        uncertainty = jnp.std(predictions, axis=0)
        
        return mean_prediction, uncertainty

# Usage
jax_research = JAXResearchTools()
```

---

## 🔗 Advanced Integration

### 1. Multi-Framework Pipeline

```python
class MultiFrameworkPipeline:
    """Integration of multiple frameworks"""
    
    def __init__(self):
        self.models = {}
        self.optimizers = {}
    
    def create_hybrid_pipeline(self, task_type="nlp"):
        """Create hybrid pipeline using multiple frameworks"""
        
        if task_type == "nlp":
            # Hugging Face for tokenization and base model
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            base_model = AutoModel.from_pretrained("bert-base-uncased")
            
            # Convert to ONNX for deployment
            import torch
            dummy_input = torch.randn(1, 512)
            torch.onnx.export(base_model, dummy_input, "bert_model.onnx")
            
            # JAX for fine-tuning
            jax_model = self.create_jax_classifier(base_model.config.hidden_size)
            
            return {
                'tokenizer': tokenizer,
                'base_model': base_model,
                'jax_classifier': jax_model,
                'onnx_model': "bert_model.onnx"
            }
    
    def create_jax_classifier(self, input_size, num_classes=2):
        """Create JAX classifier"""
        class JAXClassifier(nn.Module):
            num_classes: int
            
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(256)(x)
                x = relu(x)
                x = nn.Dropout(rate=0.3)(x)
                x = nn.Dense(self.num_classes)(x)
                return softmax(x, axis=-1)
        
        return JAXClassifier(num_classes=num_classes)
    
    def unified_inference(self, pipeline, text):
        """Unified inference across frameworks"""
        
        # Tokenization with Hugging Face
        inputs = pipeline['tokenizer'](text, return_tensors="pt")
        
        # Base model inference (PyTorch)
        with torch.no_grad():
            outputs = pipeline['base_model'](**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        
        # Classification with JAX
        jax_features = jnp.array(features.numpy())
        predictions = pipeline['jax_classifier'].apply(
            pipeline['jax_params'], jax_features
        )
        
        return predictions
    
    def performance_benchmark(self, pipeline, test_data):
        """Benchmark performance across frameworks"""
        
        results = {}
        
        # Hugging Face benchmark
        start_time = time.time()
        for text in test_data:
            pipeline['pipeline'](text)
        results['hugging_face'] = time.time() - start_time
        
        # ONNX benchmark
        start_time = time.time()
        for text in test_data:
            inputs = pipeline['tokenizer'](text, return_tensors="np")
            pipeline['onnx_session'].run(None, inputs)
        results['onnx'] = time.time() - start_time
        
        # JAX benchmark
        start_time = time.time()
        for text in test_data:
            inputs = pipeline['tokenizer'](text, return_tensors="np")
            features = jnp.array(inputs['input_ids'])
            pipeline['jax_model'].apply(pipeline['jax_params'], features)
        results['jax'] = time.time() - start_time
        
        return results

# Usage
multi_pipeline = MultiFrameworkPipeline()
```

---

## 🧪 Exercises and Projects

### Exercise 1: Hugging Face Advanced
1. Build custom transformer model
2. Implement advanced tokenization
3. Create custom training loop
4. Add model interpretation

### Exercise 2: ONNX Optimization
1. Convert PyTorch model to ONNX
2. Optimize for different platforms
3. Benchmark performance
4. Deploy to edge device

### Exercise 3: JAX Research
1. Implement custom neural network
2. Add distributed training
3. Create uncertainty quantification
4. Optimize for TPU

### Project: Multi-Framework ML System

**Objective**: Build comprehensive ML system using advanced frameworks

**Requirements**:
- Hugging Face for NLP preprocessing
- ONNX for model deployment
- JAX for high-performance training
- Unified evaluation and deployment

**Deliverables**:
- Complete multi-framework pipeline
- Performance benchmarks
- Deployment strategy
- Research tools

---

## 📖 Further Reading

### Essential Resources

1. **Hugging Face**
   - [Transformers Documentation](https://huggingface.co/docs/transformers/)
   - [Datasets Library](https://huggingface.co/docs/datasets/)
   - [Model Hub](https://huggingface.co/models)

2. **ONNX**
   - [ONNX Documentation](https://onnx.ai/)
   - [ONNX Runtime](https://onnxruntime.ai/)
   - [Model Optimization](https://github.com/microsoft/onnxruntime)

3. **JAX**
   - [JAX Documentation](https://jax.readthedocs.io/)
   - [Flax Library](https://flax.readthedocs.io/)
   - [Optax Optimizers](https://optax.readthedocs.io/)

### Advanced Topics

- **Model Compression**: Quantization, pruning, distillation
- **Distributed Training**: Multi-device, multi-node training
- **Edge Deployment**: Mobile, IoT, embedded systems
- **Research Tools**: Custom gradients, uncertainty quantification

### 2025 Trends

- **Unified APIs**: Cross-framework compatibility
- **AutoML Integration**: Automated model development
- **Edge Computing**: Mobile and IoT optimization
- **Federated Learning**: Privacy-preserving distributed training

---

## 🎯 Key Takeaways

1. **Framework Specialization**: Each framework excels in specific areas
2. **Performance Optimization**: Advanced frameworks provide significant speedups
3. **Deployment Flexibility**: ONNX enables cross-platform deployment
4. **Research Capabilities**: JAX enables cutting-edge research
5. **Integration Benefits**: Combining frameworks provides optimal solutions

---

*"The best framework is the one that solves your specific problem."*

**Next: [Custom Silicon](tools_and_ides/37_custom_silicon.md) → AI-specific hardware and optimization**