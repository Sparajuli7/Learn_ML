# Model Fairness and Explainability: Building Trustworthy AI

## Table of Contents
1. [Introduction](#introduction)
2. [AI Fairness Fundamentals](#ai-fairness-fundamentals)
3. [Bias Detection and Measurement](#bias-detection-and-measurement)
4. [Fairness Metrics and Algorithms](#fairness-metrics-and-algorithms)
5. [Explainable AI Techniques](#explainable-ai-techniques)
6. [Model Interpretability](#model-interpretability)
7. [Fairness in Practice](#fairness-in-practice)
8. [Regulatory Compliance](#regulatory-compliance)
9. [Practical Implementation](#practical-implementation)
10. [Exercises and Projects](#exercises-and-projects)

## Introduction

Model fairness and explainability are critical for building trustworthy AI systems. This chapter covers techniques for detecting bias, ensuring fairness, and making ML models interpretable and transparent.

### Key Learning Objectives
- Understand AI fairness concepts and bias types
- Implement bias detection and measurement techniques
- Apply fairness metrics and algorithms
- Use explainable AI techniques for model interpretation
- Ensure regulatory compliance for AI systems

## AI Fairness Fundamentals

### Types of Bias in ML

```python
# AI Bias Classification
class AIBiasTypes:
    def __init__(self):
        self.bias_categories = {
            'data_bias': {
                'sampling_bias': 'Under/over-representation of groups',
                'measurement_bias': 'Inaccurate or biased measurements',
                'label_bias': 'Biased ground truth labels',
                'historical_bias': 'Bias from historical data'
            },
            'algorithm_bias': {
                'model_bias': 'Bias in model architecture',
                'optimization_bias': 'Bias in training objectives',
                'evaluation_bias': 'Bias in evaluation metrics'
            },
            'deployment_bias': {
                'feedback_bias': 'Bias from user feedback loops',
                'temporal_bias': 'Bias from changing distributions',
                'interaction_bias': 'Bias from user interactions'
            }
        }
    
    def assess_bias_impact(self, model, data, sensitive_attributes):
        """Assess the impact of different bias types"""
        
        bias_assessment = {
            'data_bias': self._assess_data_bias(data, sensitive_attributes),
            'algorithm_bias': self._assess_algorithm_bias(model, data),
            'deployment_bias': self._assess_deployment_bias(model, data)
        }
        
        return bias_assessment
```

### Fairness Definitions

```python
# Fairness Definitions and Metrics
class FairnessDefinitions:
    def __init__(self):
        self.fairness_metrics = {
            'demographic_parity': 'Equal positive prediction rates across groups',
            'equalized_odds': 'Equal TPR and FPR across groups',
            'equal_opportunity': 'Equal TPR across groups',
            'individual_fairness': 'Similar individuals get similar predictions',
            'counterfactual_fairness': 'Fairness under counterfactual scenarios'
        }
    
    def demographic_parity(self, predictions, sensitive_attributes):
        """Calculate demographic parity"""
        
        groups = np.unique(sensitive_attributes)
        parity_scores = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            positive_rate = np.mean(group_predictions == 1)
            parity_scores[group] = positive_rate
        
        # Calculate disparity
        rates = list(parity_scores.values())
        disparity = max(rates) - min(rates)
        
        return {
            'parity_scores': parity_scores,
            'disparity': disparity,
            'is_fair': disparity < 0.1  # 10% threshold
        }
    
    def equalized_odds(self, predictions, true_labels, sensitive_attributes):
        """Calculate equalized odds"""
        
        groups = np.unique(sensitive_attributes)
        odds_scores = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            group_labels = true_labels[group_mask]
            
            # Calculate TPR and FPR
            tpr = np.mean(group_predictions[group_labels == 1] == 1)
            fpr = np.mean(group_predictions[group_labels == 0] == 1)
            
            odds_scores[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate disparities
        tprs = [scores['tpr'] for scores in odds_scores.values()]
        fprs = [scores['fpr'] for scores in odds_scores.values()]
        
        tpr_disparity = max(tprs) - min(tprs)
        fpr_disparity = max(fprs) - min(fprs)
        
        return {
            'odds_scores': odds_scores,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'is_fair': tpr_disparity < 0.1 and fpr_disparity < 0.1
        }
```

## Bias Detection and Measurement

### Statistical Parity Testing

```python
# Statistical Parity Testing
class StatisticalParityTest:
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
    
    def test_statistical_parity(self, predictions, sensitive_attributes):
        """Test for statistical parity across groups"""
        
        groups = np.unique(sensitive_attributes)
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Calculate positive rates for each group
        positive_rates = {}
        group_sizes = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            positive_rates[group] = np.mean(group_predictions == 1)
            group_sizes[group] = np.sum(group_mask)
        
        # Perform chi-square test
        from scipy.stats import chi2_contingency
        
        # Create contingency table
        contingency_table = []
        for group in groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            positive_count = np.sum(group_predictions == 1)
            negative_count = np.sum(group_predictions == 0)
            contingency_table.append([positive_count, negative_count])
        
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'positive_rates': positive_rates,
            'group_sizes': group_sizes,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'has_bias': p_value < self.significance_level
        }
    
    def calculate_disparate_impact(self, predictions, sensitive_attributes):
        """Calculate disparate impact ratio"""
        
        groups = np.unique(sensitive_attributes)
        
        if len(groups) != 2:
            return {'error': 'Disparate impact requires exactly 2 groups'}
        
        # Calculate positive rates
        positive_rates = {}
        for group in groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            positive_rates[group] = np.mean(group_predictions == 1)
        
        # Calculate disparate impact ratio
        rates = list(positive_rates.values())
        disparate_impact_ratio = min(rates) / max(rates)
        
        return {
            'positive_rates': positive_rates,
            'disparate_impact_ratio': disparate_impact_ratio,
            'is_fair': disparate_impact_ratio >= 0.8  # 80% rule
        }
```

### Bias Detection in Data

```python
# Data Bias Detection
class DataBiasDetection:
    def __init__(self):
        self.bias_indicators = {
            'representation_bias': self._detect_representation_bias,
            'measurement_bias': self._detect_measurement_bias,
            'label_bias': self._detect_label_bias,
            'historical_bias': self._detect_historical_bias
        }
    
    def detect_data_bias(self, data, sensitive_attributes, labels=None):
        """Comprehensive data bias detection"""
        
        bias_report = {}
        
        for bias_type, detection_func in self.bias_indicators.items():
            bias_report[bias_type] = detection_func(data, sensitive_attributes, labels)
        
        return bias_report
    
    def _detect_representation_bias(self, data, sensitive_attributes, labels=None):
        """Detect representation bias in data"""
        
        groups = np.unique(sensitive_attributes)
        representation_analysis = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            group_size = np.sum(group_mask)
            total_size = len(sensitive_attributes)
            
            representation_analysis[group] = {
                'count': group_size,
                'percentage': group_size / total_size * 100,
                'representation_ratio': group_size / total_size
            }
        
        # Calculate representation disparity
        percentages = [analysis['percentage'] for analysis in representation_analysis.values()]
        representation_disparity = max(percentages) - min(percentages)
        
        return {
            'representation_analysis': representation_analysis,
            'representation_disparity': representation_disparity,
            'has_bias': representation_disparity > 20  # 20% threshold
        }
    
    def _detect_measurement_bias(self, data, sensitive_attributes, labels=None):
        """Detect measurement bias in features"""
        
        # Analyze feature distributions across groups
        feature_bias = {}
        
        for feature_idx in range(data.shape[1]):
            feature_values = data[:, feature_idx]
            feature_bias[feature_idx] = {}
            
            groups = np.unique(sensitive_attributes)
            for group in groups:
                group_mask = sensitive_attributes == group
                group_values = feature_values[group_mask]
                
                feature_bias[feature_idx][group] = {
                    'mean': np.mean(group_values),
                    'std': np.std(group_values),
                    'median': np.median(group_values)
                }
        
        return feature_bias
    
    def _detect_label_bias(self, data, sensitive_attributes, labels):
        """Detect bias in labels"""
        
        if labels is None:
            return {'error': 'Labels required for label bias detection'}
        
        groups = np.unique(sensitive_attributes)
        label_analysis = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            group_labels = labels[group_mask]
            
            label_analysis[group] = {
                'positive_rate': np.mean(group_labels == 1),
                'negative_rate': np.mean(group_labels == 0),
                'label_distribution': np.bincount(group_labels)
            }
        
        return label_analysis
```

## Fairness Metrics and Algorithms

### Fairness-Aware Training

```python
# Fairness-Aware Training Algorithms
class FairnessAwareTraining:
    def __init__(self, fairness_constraint='demographic_parity'):
        self.fairness_constraint = fairness_constraint
        self.fairness_methods = {
            'preprocessing': self._preprocessing_fairness,
            'inprocessing': self._inprocessing_fairness,
            'postprocessing': self._postprocessing_fairness
        }
    
    def train_fair_model(self, X, y, sensitive_attributes, method='inprocessing'):
        """Train a fairness-aware model"""
        
        if method not in self.fairness_methods:
            raise ValueError(f"Unknown fairness method: {method}")
        
        return self.fairness_methods[method](X, y, sensitive_attributes)
    
    def _preprocessing_fairness(self, X, y, sensitive_attributes):
        """Preprocessing approach to fairness"""
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        # Reweigh the data to achieve fairness
        reweighed_data = self._reweigh_data(X, y, sensitive_attributes)
        
        # Train model on reweighed data
        model = LogisticRegression(random_state=42)
        model.fit(reweighed_data['X'], reweighed_data['y'], 
                 sample_weight=reweighed_data['weights'])
        
        return model
    
    def _reweigh_data(self, X, y, sensitive_attributes):
        """Reweigh data to achieve fairness"""
        
        groups = np.unique(sensitive_attributes)
        labels = np.unique(y)
        
        # Calculate base rates
        base_rates = {}
        for group in groups:
            for label in labels:
                mask = (sensitive_attributes == group) & (y == label)
                base_rates[(group, label)] = np.sum(mask) / len(y)
        
        # Calculate target rates (equal representation)
        target_rates = {}
        for group in groups:
            group_mask = sensitive_attributes == group
            target_rates[group] = np.sum(group_mask) / len(y)
        
        # Calculate weights
        weights = np.ones(len(y))
        for i, (group, label) in enumerate(zip(sensitive_attributes, y)):
            if base_rates[(group, label)] > 0:
                weights[i] = target_rates[group] / base_rates[(group, label)]
        
        return {'X': X, 'y': y, 'weights': weights}
    
    def _inprocessing_fairness(self, X, y, sensitive_attributes):
        """In-processing approach to fairness"""
        
        # Implement adversarial debiasing
        return self._adversarial_debiasing(X, y, sensitive_attributes)
    
    def _adversarial_debiasing(self, X, y, sensitive_attributes):
        """Adversarial debiasing implementation"""
        
        import torch
        import torch.nn as nn
        
        # Define predictor network
        class Predictor(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Define adversary network
        class Adversary(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        s_tensor = torch.FloatTensor(sensitive_attributes).unsqueeze(1)
        
        # Initialize networks
        predictor = Predictor(X.shape[1])
        adversary = Adversary(1)  # Predict sensitive attribute from predictor output
        
        # Training
        predictor_optimizer = torch.optim.Adam(predictor.parameters())
        adversary_optimizer = torch.optim.Adam(adversary.parameters())
        
        for epoch in range(100):
            # Train adversary
            adversary_optimizer.zero_grad()
            predictions = predictor(X_tensor)
            adversary_predictions = adversary(predictions.detach())
            adversary_loss = nn.BCELoss()(adversary_predictions, s_tensor)
            adversary_loss.backward()
            adversary_optimizer.step()
            
            # Train predictor
            predictor_optimizer.zero_grad()
            predictions = predictor(X_tensor)
            predictor_loss = nn.BCELoss()(predictions, y_tensor)
            
            # Adversarial loss
            adversary_predictions = adversary(predictions)
            adversarial_loss = nn.BCELoss()(adversary_predictions, s_tensor)
            
            # Total loss
            total_loss = predictor_loss - 0.1 * adversarial_loss
            total_loss.backward()
            predictor_optimizer.step()
        
        return predictor
    
    def _postprocessing_fairness(self, X, y, sensitive_attributes):
        """Post-processing approach to fairness"""
        
        # Train base model
        from sklearn.linear_model import LogisticRegression
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)
        
        # Get base predictions
        base_predictions = base_model.predict_proba(X)[:, 1]
        
        # Apply post-processing for fairness
        fair_predictions = self._apply_postprocessing(
            base_predictions, y, sensitive_attributes
        )
        
        return {
            'base_model': base_model,
            'fair_predictions': fair_predictions,
            'predict': lambda X: self._predict_with_postprocessing(
                base_model, X, sensitive_attributes
            )
        }
    
    def _apply_postprocessing(self, predictions, y, sensitive_attributes):
        """Apply post-processing for fairness"""
        
        # Equalized odds post-processing
        groups = np.unique(sensitive_attributes)
        thresholds = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            group_labels = y[group_mask]
            
            # Find optimal threshold for this group
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds_roc = roc_curve(group_labels, group_predictions)
            
            # Choose threshold that gives equal TPR and FPR
            optimal_idx = np.argmax(tpr - fpr)
            thresholds[group] = thresholds_roc[optimal_idx]
        
        # Apply group-specific thresholds
        fair_predictions = np.zeros_like(predictions)
        for group in groups:
            group_mask = sensitive_attributes == group
            fair_predictions[group_mask] = (
                predictions[group_mask] >= thresholds[group]
            ).astype(float)
        
        return fair_predictions
```

## Explainable AI Techniques

### Model Interpretability

```python
# Model Interpretability Techniques
class ModelInterpretability:
    def __init__(self):
        self.interpretability_methods = {
            'feature_importance': self.feature_importance,
            'shap_values': self.shap_values,
            'lime_explanation': self.lime_explanation,
            'partial_dependence': self.partial_dependence,
            'counterfactual_explanations': self.counterfactual_explanations
        }
    
    def explain_model(self, model, X, method='shap_values'):
        """Generate model explanations"""
        
        if method not in self.interpretability_methods:
            raise ValueError(f"Unknown interpretability method: {method}")
        
        return self.interpretability_methods[method](model, X)
    
    def feature_importance(self, model, X):
        """Calculate feature importance"""
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            return {
                'importance_scores': model.feature_importances_,
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'method': 'tree_importance'
            }
        elif hasattr(model, 'coef_'):
            # Linear models
            return {
                'importance_scores': np.abs(model.coef_[0]),
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'method': 'coefficient_magnitude'
            }
        else:
            # Permutation importance
            return self._permutation_importance(model, X)
    
    def _permutation_importance(self, model, X, n_repeats=5):
        """Calculate permutation importance"""
        
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X, np.zeros(X.shape[0]), 
            n_repeats=n_repeats, random_state=42
        )
        
        return {
            'importance_scores': result.importances_mean,
            'importance_std': result.importances_std,
            'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
            'method': 'permutation_importance'
        }
    
    def shap_values(self, model, X, background_data=None):
        """Calculate SHAP values"""
        
        try:
            import shap
            
            if background_data is None:
                background_data = X[:100]  # Use first 100 samples as background
            
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, background_data)
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, background_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            return {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value,
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'method': 'shap'
            }
        
        except ImportError:
            return {'error': 'SHAP library not available'}
    
    def lime_explanation(self, model, X, instance_idx=0):
        """Generate LIME explanation for a specific instance"""
        
        try:
            from lime import lime_tabular
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X, feature_names=[f'feature_{i}' for i in range(X.shape[1])],
                class_names=['negative', 'positive'], mode='classification'
            )
            
            # Generate explanation
            explanation = explainer.explain_instance(
                X[instance_idx], model.predict_proba,
                num_features=X.shape[1]
            )
            
            return {
                'explanation': explanation,
                'instance_idx': instance_idx,
                'method': 'lime'
            }
        
        except ImportError:
            return {'error': 'LIME library not available'}
    
    def partial_dependence(self, model, X, feature_idx=0):
        """Calculate partial dependence plot"""
        
        from sklearn.inspection import partial_dependence
        
        # Calculate partial dependence
        feature_values = np.linspace(
            np.min(X[:, feature_idx]), 
            np.max(X[:, feature_idx]), 
            50
        )
        
        pdp = partial_dependence(
            model, X, [feature_idx], 
            percentiles=(0.05, 0.95),
            grid=feature_values
        )
        
        return {
            'feature_values': feature_values,
            'partial_dependence': pdp[1][0],
            'feature_idx': feature_idx,
            'method': 'partial_dependence'
        }
    
    def counterfactual_explanations(self, model, X, target_class=1, n_counterfactuals=5):
        """Generate counterfactual explanations"""
        
        # Simple counterfactual generation
        counterfactuals = []
        
        for i in range(min(n_counterfactuals, len(X))):
            original = X[i].copy()
            original_pred = model.predict([original])[0]
            
            if original_pred != target_class:
                # Generate counterfactual by perturbing features
                counterfactual = self._generate_counterfactual(
                    model, original, target_class
                )
                counterfactuals.append({
                    'original': original,
                    'counterfactual': counterfactual,
                    'changes': counterfactual - original
                })
        
        return {
            'counterfactuals': counterfactuals,
            'target_class': target_class,
            'method': 'counterfactual_explanations'
        }
    
    def _generate_counterfactual(self, model, instance, target_class, max_iterations=100):
        """Generate a counterfactual explanation"""
        
        counterfactual = instance.copy()
        
        for iteration in range(max_iterations):
            pred = model.predict([counterfactual])[0]
            
            if pred == target_class:
                break
            
            # Perturb features randomly
            perturbation = np.random.normal(0, 0.1, counterfactual.shape)
            counterfactual += perturbation
            
            # Clip to valid range
            counterfactual = np.clip(counterfactual, 0, 1)
        
        return counterfactual
```

## Model Interpretability

### Global vs Local Interpretability

```python
# Global and Local Interpretability
class InterpretabilityAnalysis:
    def __init__(self):
        self.global_methods = {
            'feature_importance': self.global_feature_importance,
            'partial_dependence': self.global_partial_dependence,
            'model_structure': self.model_structure_analysis
        }
        
        self.local_methods = {
            'shap_values': self.local_shap_values,
            'lime_explanation': self.local_lime_explanation,
            'counterfactual': self.local_counterfactual
        }
    
    def global_interpretability(self, model, X, method='feature_importance'):
        """Global model interpretability"""
        
        if method not in self.global_methods:
            raise ValueError(f"Unknown global method: {method}")
        
        return self.global_methods[method](model, X)
    
    def local_interpretability(self, model, X, instance_idx, method='shap_values'):
        """Local model interpretability for specific instance"""
        
        if method not in self.local_methods:
            raise ValueError(f"Unknown local method: {method}")
        
        return self.local_methods[method](model, X, instance_idx)
    
    def global_feature_importance(self, model, X):
        """Global feature importance analysis"""
        
        # Get feature importance
        importance_analysis = ModelInterpretability().feature_importance(model, X)
        
        # Sort features by importance
        feature_importance = list(zip(
            importance_analysis['feature_names'],
            importance_analysis['importance_scores']
        ))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'feature_importance': feature_importance,
            'top_features': feature_importance[:10],
            'method': 'global_feature_importance'
        }
    
    def global_partial_dependence(self, model, X, top_features=5):
        """Global partial dependence analysis"""
        
        # Get top features
        importance_analysis = self.global_feature_importance(model, X)
        top_feature_indices = [
            int(f.split('_')[1]) for f, _ in importance_analysis['top_features'][:top_features]
        ]
        
        # Calculate partial dependence for top features
        pdp_results = {}
        for feature_idx in top_feature_indices:
            pdp = ModelInterpretability().partial_dependence(model, X, feature_idx)
            pdp_results[feature_idx] = pdp
        
        return {
            'partial_dependence': pdp_results,
            'top_features': top_feature_indices,
            'method': 'global_partial_dependence'
        }
    
    def model_structure_analysis(self, model, X):
        """Analyze model structure and complexity"""
        
        analysis = {
            'model_type': type(model).__name__,
            'parameters': self._count_parameters(model),
            'complexity': self._assess_complexity(model),
            'interpretability_score': self._calculate_interpretability_score(model)
        }
        
        return analysis
    
    def _count_parameters(self, model):
        """Count model parameters"""
        
        if hasattr(model, 'coef_'):
            return len(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            return len(model.feature_importances_)
        else:
            return 'unknown'
    
    def _assess_complexity(self, model):
        """Assess model complexity"""
        
        if hasattr(model, 'n_estimators'):
            return f"Ensemble with {model.n_estimators} estimators"
        elif hasattr(model, 'max_depth'):
            return f"Tree with max_depth={model.max_depth}"
        else:
            return "Linear model"
    
    def _calculate_interpretability_score(self, model):
        """Calculate interpretability score (0-1)"""
        
        # Simple scoring based on model type
        if hasattr(model, 'coef_'):
            return 1.0  # Linear models are highly interpretable
        elif hasattr(model, 'feature_importances_'):
            return 0.8  # Tree-based models are interpretable
        else:
            return 0.3  # Other models are less interpretable
```

## Fairness in Practice

### Fairness Monitoring

```python
# Fairness Monitoring System
class FairnessMonitoring:
    def __init__(self):
        self.monitoring_metrics = {
            'demographic_parity': self.monitor_demographic_parity,
            'equalized_odds': self.monitor_equalized_odds,
            'individual_fairness': self.monitor_individual_fairness
        }
    
    def setup_fairness_monitoring(self, model, baseline_data, sensitive_attributes):
        """Setup fairness monitoring system"""
        
        # Calculate baseline fairness metrics
        baseline_predictions = model.predict(baseline_data)
        baseline_metrics = self._calculate_baseline_metrics(
            baseline_predictions, baseline_data, sensitive_attributes
        )
        
        return {
            'baseline_metrics': baseline_metrics,
            'monitoring_config': self._create_monitoring_config(),
            'alert_thresholds': self._set_alert_thresholds()
        }
    
    def monitor_fairness(self, new_predictions, new_data, sensitive_attributes, baseline_metrics):
        """Monitor fairness in real-time"""
        
        current_metrics = {}
        alerts = []
        
        for metric_name, metric_func in self.monitoring_metrics.items():
            current_metrics[metric_name] = metric_func(
                new_predictions, new_data, sensitive_attributes
            )
            
            # Check for fairness drift
            drift_detected = self._detect_fairness_drift(
                baseline_metrics[metric_name],
                current_metrics[metric_name]
            )
            
            if drift_detected:
                alerts.append({
                    'metric': metric_name,
                    'drift_detected': True,
                    'severity': 'high' if drift_detected['magnitude'] > 0.2 else 'medium'
                })
        
        return {
            'current_metrics': current_metrics,
            'alerts': alerts,
            'overall_fairness_score': self._calculate_overall_fairness_score(current_metrics)
        }
    
    def _calculate_baseline_metrics(self, predictions, data, sensitive_attributes):
        """Calculate baseline fairness metrics"""
        
        baseline_metrics = {}
        
        for metric_name, metric_func in self.monitoring_metrics.items():
            baseline_metrics[metric_name] = metric_func(
                predictions, data, sensitive_attributes
            )
        
        return baseline_metrics
    
    def _detect_fairness_drift(self, baseline_metric, current_metric):
        """Detect fairness drift"""
        
        # Calculate drift magnitude
        if 'disparity' in baseline_metric and 'disparity' in current_metric:
            drift_magnitude = abs(
                current_metric['disparity'] - baseline_metric['disparity']
            )
            
            return {
                'drift_detected': drift_magnitude > 0.1,  # 10% threshold
                'magnitude': drift_magnitude,
                'baseline_disparity': baseline_metric['disparity'],
                'current_disparity': current_metric['disparity']
            }
        
        return {'drift_detected': False}
    
    def _calculate_overall_fairness_score(self, metrics):
        """Calculate overall fairness score"""
        
        # Simple average of fairness indicators
        fairness_scores = []
        
        for metric_name, metric_result in metrics.items():
            if 'is_fair' in metric_result:
                fairness_scores.append(1.0 if metric_result['is_fair'] else 0.0)
            elif 'disparity' in metric_result:
                # Convert disparity to score (lower disparity = higher score)
                score = max(0, 1 - metric_result['disparity'])
                fairness_scores.append(score)
        
        return np.mean(fairness_scores) if fairness_scores else 0.0
```

## Regulatory Compliance

### AI Ethics and Compliance

```python
# AI Ethics and Regulatory Compliance
class AIEthicsCompliance:
    def __init__(self):
        self.compliance_frameworks = {
            'eu_ai_act': self.eu_ai_act_compliance,
            'gdpr': self.gdpr_compliance,
            'algorithmic_accountability': self.algorithmic_accountability,
            'fair_credit_reporting': self.fair_credit_reporting
        }
    
    def assess_compliance(self, model, data, deployment_config):
        """Assess compliance with various frameworks"""
        
        compliance_report = {}
        
        for framework, assessment_func in self.compliance_frameworks.items():
            compliance_report[framework] = assessment_func(
                model, data, deployment_config
            )
        
        return compliance_report
    
    def eu_ai_act_compliance(self, model, data, deployment_config):
        """Assess EU AI Act compliance"""
        
        # EU AI Act requirements
        requirements = {
            'transparency': self._assess_transparency(model),
            'human_oversight': self._assess_human_oversight(deployment_config),
            'accuracy': self._assess_accuracy(model, data),
            'robustness': self._assess_robustness(model),
            'privacy': self._assess_privacy_compliance(data),
            'fairness': self._assess_fairness_compliance(model, data)
        }
        
        # Determine risk level
        risk_level = self._determine_risk_level(requirements)
        
        return {
            'requirements': requirements,
            'risk_level': risk_level,
            'compliance_score': self._calculate_compliance_score(requirements),
            'recommendations': self._generate_compliance_recommendations(requirements)
        }
    
    def gdpr_compliance(self, model, data, deployment_config):
        """Assess GDPR compliance"""
        
        gdpr_requirements = {
            'data_minimization': self._assess_data_minimization(data),
            'purpose_limitation': self._assess_purpose_limitation(deployment_config),
            'storage_limitation': self._assess_storage_limitation(deployment_config),
            'right_to_explanation': self._assess_explainability(model),
            'right_to_erasure': self._assess_erasure_capability(deployment_config),
            'consent_management': self._assess_consent_management(deployment_config)
        }
        
        return {
            'gdpr_requirements': gdpr_requirements,
            'compliance_score': self._calculate_gdpr_compliance_score(gdpr_requirements),
            'recommendations': self._generate_gdpr_recommendations(gdpr_requirements)
        }
    
    def _assess_transparency(self, model):
        """Assess model transparency"""
        
        transparency_score = 0.0
        
        # Check if model is interpretable
        if hasattr(model, 'coef_'):
            transparency_score += 0.4  # Linear models are transparent
        
        # Check if feature importance is available
        if hasattr(model, 'feature_importances_'):
            transparency_score += 0.3
        
        # Check if model documentation exists
        transparency_score += 0.3  # Assume documentation exists
        
        return {
            'score': transparency_score,
            'is_compliant': transparency_score >= 0.7,
            'recommendations': ['Improve model interpretability'] if transparency_score < 0.7 else []
        }
    
    def _assess_human_oversight(self, deployment_config):
        """Assess human oversight capabilities"""
        
        oversight_score = 0.0
        
        # Check for human review mechanisms
        if deployment_config.get('human_review_enabled', False):
            oversight_score += 0.5
        
        # Check for monitoring and alerting
        if deployment_config.get('monitoring_enabled', False):
            oversight_score += 0.3
        
        # Check for override capabilities
        if deployment_config.get('override_enabled', False):
            oversight_score += 0.2
        
        return {
            'score': oversight_score,
            'is_compliant': oversight_score >= 0.6,
            'recommendations': ['Implement human oversight mechanisms'] if oversight_score < 0.6 else []
        }
    
    def _determine_risk_level(self, requirements):
        """Determine AI system risk level"""
        
        # Count non-compliant requirements
        non_compliant_count = sum(
            1 for req in requirements.values() 
            if not req.get('is_compliant', True)
        )
        
        if non_compliant_count >= 3:
            return 'high_risk'
        elif non_compliant_count >= 1:
            return 'medium_risk'
        else:
            return 'low_risk'
```

## Practical Implementation

### Complete Fairness Pipeline

```python
# Complete Fairness and Explainability Pipeline
class CompleteFairnessPipeline:
    def __init__(self):
        self.fairness_detector = DataBiasDetection()
        self.fairness_trainer = FairnessAwareTraining()
        self.interpreter = ModelInterpretability()
        self.monitor = FairnessMonitoring()
        self.compliance = AIEthicsCompliance()
    
    def build_fair_model(self, X, y, sensitive_attributes, deployment_config):
        """Build a complete fair and explainable model"""
        
        # 1. Detect bias in data
        bias_report = self.fairness_detector.detect_data_bias(X, sensitive_attributes, y)
        
        # 2. Train fair model
        fair_model = self.fairness_trainer.train_fair_model(
            X, y, sensitive_attributes, method='inprocessing'
        )
        
        # 3. Generate explanations
        explanations = self.interpreter.explain_model(fair_model, X, method='shap_values')
        
        # 4. Setup monitoring
        monitoring_config = self.monitor.setup_fairness_monitoring(
            fair_model, X, sensitive_attributes
        )
        
        # 5. Assess compliance
        compliance_report = self.compliance.assess_compliance(
            fair_model, X, deployment_config
        )
        
        return {
            'model': fair_model,
            'bias_report': bias_report,
            'explanations': explanations,
            'monitoring_config': monitoring_config,
            'compliance_report': compliance_report
        }
    
    def deploy_fair_model(self, fair_model_config, deployment_environment):
        """Deploy fair model with monitoring"""
        
        # Setup deployment
        deployment = {
            'model': fair_model_config['model'],
            'monitoring': fair_model_config['monitoring_config'],
            'explanations': fair_model_config['explanations'],
            'compliance': fair_model_config['compliance_report']
        }
        
        # Deploy with fairness monitoring
        deployed_model = self._deploy_with_monitoring(deployment, deployment_environment)
        
        return deployed_model
    
    def _deploy_with_monitoring(self, deployment, environment):
        """Deploy model with fairness monitoring"""
        
        # Setup real-time monitoring
        monitoring_endpoint = self._setup_monitoring_endpoint(
            deployment['monitoring'], environment
        )
        
        # Setup explanation endpoint
        explanation_endpoint = self._setup_explanation_endpoint(
            deployment['explanations'], environment
        )
        
        # Setup compliance dashboard
        compliance_dashboard = self._setup_compliance_dashboard(
            deployment['compliance'], environment
        )
        
        return {
            'model_endpoint': f"{environment['base_url']}/predict",
            'monitoring_endpoint': monitoring_endpoint,
            'explanation_endpoint': explanation_endpoint,
            'compliance_dashboard': compliance_dashboard,
            'fairness_metrics': deployment['monitoring']
        }
```

## Exercises and Projects

### Exercise 1: Bias Detection Implementation

Implement comprehensive bias detection:

1. **Statistical Parity Testing**: Implement chi-square tests for fairness
2. **Disparate Impact Analysis**: Calculate disparate impact ratios
3. **Feature Bias Detection**: Analyze bias in individual features
4. **Label Bias Detection**: Detect bias in training labels

**Requirements:**
- Implement at least 3 bias detection methods
- Test on real-world datasets
- Generate bias reports with recommendations
- Visualize bias patterns

### Exercise 2: Fairness-Aware Model Training

Build fairness-aware models:

1. **Preprocessing Fairness**: Implement reweighing techniques
2. **In-processing Fairness**: Implement adversarial debiasing
3. **Post-processing Fairness**: Implement equalized odds post-processing
4. **Fairness Evaluation**: Compare different fairness approaches

**Implementation:**
```python
# Fairness-Aware Model Training
class FairnessAwareModelTraining:
    def __init__(self):
        self.fairness_methods = {
            'preprocessing': self.preprocessing_fairness,
            'inprocessing': self.inprocessing_fairness,
            'postprocessing': self.postprocessing_fairness
        }
    
    def compare_fairness_methods(self, X, y, sensitive_attributes):
        """Compare different fairness methods"""
        
        results = {}
        
        for method_name, method_func in self.fairness_methods.items():
            model = method_func(X, y, sensitive_attributes)
            fairness_metrics = self.evaluate_fairness(model, X, y, sensitive_attributes)
            results[method_name] = fairness_metrics
        
        return results
```

### Project: Fairness Monitoring System

Build a comprehensive fairness monitoring system:

1. **Real-time Monitoring**: Monitor fairness metrics in production
2. **Drift Detection**: Detect fairness drift over time
3. **Alert System**: Alert when fairness thresholds are violated
4. **Dashboard**: Visualize fairness metrics and trends
5. **Automated Mitigation**: Automatically apply fairness corrections

**Features:**
- Real-time fairness monitoring
- Automated bias detection
- Compliance reporting
- Explainable AI integration
- Regulatory compliance checking

### Project: Explainable AI Platform

Develop a complete explainable AI platform:

1. **Model Interpretability**: Multiple explanation methods
2. **Fairness Analysis**: Comprehensive fairness assessment
3. **Compliance Checking**: Regulatory compliance verification
4. **User Interface**: Intuitive dashboard for explanations
5. **API Integration**: RESTful API for explanations

**Deliverables:**
- Complete explainable AI platform
- Multiple explanation methods (SHAP, LIME, etc.)
- Fairness monitoring and reporting
- Compliance verification system
- User-friendly dashboard

## Summary

Model Fairness and Explainability covers essential concepts for building trustworthy AI:

- **Bias Detection**: Statistical methods for detecting various types of bias
- **Fairness Metrics**: Demographic parity, equalized odds, and other fairness measures
- **Explainable AI**: SHAP, LIME, and other interpretability techniques
- **Fairness-Aware Training**: Methods for training fair models
- **Monitoring and Compliance**: Real-time fairness monitoring and regulatory compliance

The practical implementation provides a foundation for building fair, explainable, and compliant AI systems that can be trusted in production environments.