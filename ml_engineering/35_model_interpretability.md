# Advanced Model Interpretability

## Overview
Advanced model interpretability techniques enable understanding of complex ML model decisions, building trust, and ensuring regulatory compliance.

## Core Concepts

### Interpretability vs Explainability
- **Interpretability**: Ability to understand model behavior
- **Explainability**: Ability to provide human-readable explanations
- **Transparency**: Model architecture and training process visibility

### Interpretability Levels
1. **Global Interpretability**: Overall model behavior
2. **Local Interpretability**: Individual prediction explanations
3. **Feature Importance**: Relative contribution of inputs

## Advanced Interpretability Techniques

### 1. SHAP (SHapley Additive exPlanations)

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
```

**Key Components:**
- **Shapley Values**: Fair attribution of prediction to features
- **Additive Property**: Sum of feature contributions equals prediction
- **Consistency**: Monotonic feature importance

### 2. LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

# Create explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['0', '1'],
    mode='classification'
)

# Explain individual prediction
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)
exp.show_in_notebook()
```

**Advantages:**
- Model-agnostic approach
- Local approximation
- Human-readable explanations

### 3. Attention Mechanisms

```python
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output, attn_weights

# Visualize attention weights
def visualize_attention(attention_weights, tokens):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights.detach().numpy())
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title('Attention Weights')
    plt.show()
```

### 4. Integrated Gradients

```python
import tensorflow as tf

def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = tf.zeros_like(input_tensor)
    
    # Generate interpolated inputs
    alphas = tf.linspace(0.0, 1.0, steps)
    interpolated_inputs = baseline + alphas[:, tf.newaxis] * (input_tensor - baseline)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)
    
    gradients = tape.gradient(predictions, interpolated_inputs)
    
    # Average gradients
    avg_gradients = tf.reduce_mean(gradients, axis=0)
    
    # Compute integrated gradients
    integrated_grads = (input_tensor - baseline) * avg_gradients
    
    return integrated_grads
```

## Interpretable Model Architectures

### 1. Decision Trees and Rule-Based Models

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train interpretable decision tree
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
tree.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns, class_names=['0', '1'], filled=True)
plt.show()

# Extract rules
from sklearn.tree import export_text
rules = export_text(tree, feature_names=X.columns)
print(rules)
```

### 2. Generalized Additive Models (GAMs)

```python
from pygam import LogisticGAM
import numpy as np

# Train GAM
gam = LogisticGAM(terms='auto', max_iter=1000)
gam.fit(X_train, y_train)

# Visualize feature effects
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    if i < X_train.shape[1]:
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, i], pdep)
        ax.plot(XX[:, i], confi, c='r', ls='--')
        ax.set_title(f'Feature {i}')
plt.tight_layout()
plt.show()
```

### 3. Neural Additive Models

```python
import torch
import torch.nn as nn

class NeuralAdditiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])
        self.final_layer = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Apply each feature network separately
        feature_outputs = []
        for i, net in enumerate(self.feature_nets):
            feature_outputs.append(net(x[:, i:i+1]))
        
        # Combine features
        combined = torch.cat(feature_outputs, dim=1)
        return self.final_layer(combined)
```

## Model-Agnostic Methods

### 1. Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance
result = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

# Visualize
importances = result.importances_mean
std = result.importances_std

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances, yerr=std)
plt.xticks(range(len(importances)), X.columns, rotation=45)
plt.title('Permutation Importance')
plt.tight_layout()
plt.show()
```

### 2. Partial Dependence Plots

```python
from sklearn.inspection import partial_dependence

# Compute partial dependence
feature_idx = 0
pdp = partial_dependence(
    model, X_train, [feature_idx], 
    percentiles=(0.05, 0.95), 
    grid_resolution=50
)

# Visualize
plt.figure(figsize=(8, 6))
plt.plot(pdp[1][0], pdp[0][0])
plt.xlabel(X.columns[feature_idx])
plt.ylabel('Partial Dependence')
plt.title(f'Partial Dependence Plot for {X.columns[feature_idx]}')
plt.show()
```

### 3. Individual Conditional Expectation (ICE)

```python
from sklearn.inspection import partial_dependence

def ice_plot(model, X, feature_idx, n_samples=100):
    # Sample instances
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X.iloc[sample_indices]
    
    # Compute ICE for each sample
    ice_curves = []
    for i in range(n_samples):
        pdp = partial_dependence(
            model, X_sample.iloc[i:i+1], [feature_idx],
            percentiles=(0.05, 0.95), grid_resolution=50
        )
        ice_curves.append(pdp[0][0])
    
    # Plot
    plt.figure(figsize=(8, 6))
    for curve in ice_curves:
        plt.plot(pdp[1][0], curve, alpha=0.3, color='blue')
    
    # Add average curve
    mean_curve = np.mean(ice_curves, axis=0)
    plt.plot(pdp[1][0], mean_curve, color='red', linewidth=2, label='Average')
    
    plt.xlabel(X.columns[feature_idx])
    plt.ylabel('Prediction')
    plt.title(f'ICE Plot for {X.columns[feature_idx]}')
    plt.legend()
    plt.show()
```

## Advanced Visualization Techniques

### 1. Feature Interaction Analysis

```python
import seaborn as sns

def interaction_heatmap(model, X, feature1, feature2, grid_size=20):
    # Create grid
    x1_range = np.linspace(X[feature1].min(), X[feature1].max(), grid_size)
    x2_range = np.linspace(X[feature2].min(), X[feature2].max(), grid_size)
    
    # Create meshgrid
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Compute predictions
    grid_data = X.iloc[:1].copy()
    predictions = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            grid_data[feature1] = X1[i, j]
            grid_data[feature2] = X2[i, j]
            predictions[i, j] = model.predict_proba(grid_data)[0, 1]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(predictions, xticklabels=x1_range[::2], 
                yticklabels=x2_range[::2], cmap='viridis')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Interaction between {feature1} and {feature2}')
    plt.show()
```

### 2. Model Performance by Subgroups

```python
def subgroup_analysis(model, X, y, sensitive_features):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    results = {}
    
    for feature in sensitive_features:
        unique_values = X[feature].unique()
        feature_results = {}
        
        for value in unique_values:
            mask = X[feature] == value
            X_subgroup = X[mask]
            y_subgroup = y[mask]
            
            if len(y_subgroup) > 0:
                y_pred = model.predict(X_subgroup)
                
                feature_results[value] = {
                    'accuracy': accuracy_score(y_subgroup, y_pred),
                    'precision': precision_score(y_subgroup, y_pred, average='weighted'),
                    'recall': recall_score(y_subgroup, y_pred, average='weighted'),
                    'count': len(y_subgroup)
                }
        
        results[feature] = feature_results
    
    return results
```

## Regulatory Compliance

### 1. GDPR Compliance

```python
class GDPRCompliantModel:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
        self.explanations = {}
    
    def predict_with_explanation(self, X):
        predictions = self.model.predict(X)
        explanations = []
        
        for i in range(len(X)):
            exp = self.explainer.explain_instance(
                X.iloc[i].values,
                self.model.predict_proba,
                num_features=10
            )
            explanations.append(exp)
            self.explanations[i] = exp
        
        return predictions, explanations
    
    def get_explanation(self, instance_id):
        return self.explanations.get(instance_id, None)
    
    def delete_explanation(self, instance_id):
        if instance_id in self.explanations:
            del self.explanations[instance_id]
```

### 2. Right to Explanation Implementation

```python
class ExplanationService:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
        self.explanation_database = {}
    
    def generate_explanation(self, instance, user_id):
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance.values,
            self.model.predict_proba,
            num_features=10
        )
        
        # Store with metadata
        explanation_id = f"{user_id}_{len(self.explanation_database)}"
        self.explanation_database[explanation_id] = {
            'explanation': explanation,
            'timestamp': datetime.now(),
            'user_id': user_id,
            'instance': instance.to_dict()
        }
        
        return explanation_id, explanation
    
    def get_explanation(self, explanation_id):
        return self.explanation_database.get(explanation_id)
    
    def delete_user_explanations(self, user_id):
        to_delete = [k for k, v in self.explanation_database.items() 
                    if v['user_id'] == user_id]
        for key in to_delete:
            del self.explanation_database[key]
```

## Best Practices

### 1. Explanation Quality Metrics

```python
def evaluate_explanation_quality(explainer, model, X_test, y_test):
    from sklearn.metrics import accuracy_score
    
    # Fidelity: How well explanations match model behavior
    fidelity_scores = []
    
    for i in range(len(X_test)):
        # Generate explanation
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=10
        )
        
        # Create simplified model from explanation
        simplified_pred = exp.predict_fn(X_test.iloc[i:i+1].values)
        original_pred = model.predict(X_test.iloc[i:i+1])
        
        # Compare predictions
        fidelity = accuracy_score(original_pred, simplified_pred)
        fidelity_scores.append(fidelity)
    
    return np.mean(fidelity_scores)
```

### 2. Human Evaluation Framework

```python
class HumanEvaluationFramework:
    def __init__(self):
        self.evaluation_metrics = {
            'comprehensibility': [],
            'completeness': [],
            'correctness': [],
            'usefulness': []
        }
    
    def evaluate_explanation(self, explanation, ground_truth, evaluator_rating):
        """
        Evaluate explanation quality through human assessment
        
        Args:
            explanation: Generated explanation
            ground_truth: True explanation
            evaluator_rating: Human rating (1-5 scale)
        """
        self.evaluation_metrics['comprehensibility'].append(evaluator_rating['comprehensibility'])
        self.evaluation_metrics['completeness'].append(evaluator_rating['completeness'])
        self.evaluation_metrics['correctness'].append(evaluator_rating['correctness'])
        self.evaluation_metrics['usefulness'].append(evaluator_rating['usefulness'])
    
    def get_average_scores(self):
        return {metric: np.mean(scores) for metric, scores in self.evaluation_metrics.items()}
```

## Implementation Checklist

### Phase 1: Basic Interpretability
- [ ] Implement SHAP for feature importance
- [ ] Add LIME for local explanations
- [ ] Create partial dependence plots
- [ ] Set up permutation importance analysis

### Phase 2: Advanced Techniques
- [ ] Implement attention mechanisms
- [ ] Add integrated gradients
- [ ] Create ICE plots
- [ ] Build interaction analysis

### Phase 3: Regulatory Compliance
- [ ] Implement GDPR-compliant explanations
- [ ] Add right to explanation service
- [ ] Create explanation quality metrics
- [ ] Set up human evaluation framework

### Phase 4: Production Deployment
- [ ] Optimize explanation generation
- [ ] Add caching for explanations
- [ ] Implement explanation versioning
- [ ] Create monitoring dashboard

## Resources

### Key Papers
- "A Unified Approach to Interpreting Model Predictions" (SHAP)
- "Why Should I Trust You?" (LIME)
- "Deep Inside Convolutional Networks" (CAM)
- "Learning Important Features Through Propagating Activation Differences" (DeepLIFT)

### Tools and Libraries
- SHAP: `pip install shap`
- LIME: `pip install lime`
- Interpret: `pip install interpret`
- ELI5: `pip install eli5`
- Alibi: `pip install alibi`

### Advanced Topics
- Counterfactual explanations
- Adversarial robustness in explanations
- Multi-modal interpretability
- Temporal interpretability for time series
- Causal interpretability

This comprehensive guide covers advanced model interpretability techniques essential for building trustworthy AI systems in 2025.