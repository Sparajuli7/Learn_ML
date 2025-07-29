# Model Evaluation & Testing: Comprehensive Assessment Framework

*"In ML, evaluation is not just about accuracy—it's about trust, fairness, and real-world impact."*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Evaluation Metrics](#evaluation-metrics)
3. [A/B Testing for ML](#ab-testing-for-ml)
4. [Model Debugging](#model-debugging)
5. [Statistical Significance](#statistical-significance)
6. [Production Monitoring](#production-monitoring)
7. [Exercises and Projects](#exercises-and-projects)
8. [Further Reading](#further-reading)

---

## 🎯 Introduction

Model evaluation and testing form the backbone of trustworthy ML systems. Unlike traditional software testing, ML evaluation must account for data drift, concept drift, and the inherent uncertainty in predictions. This chapter covers comprehensive evaluation frameworks that go beyond simple accuracy metrics to ensure models perform reliably in production.

### Key Challenges in ML Evaluation

1. **Data Distribution Shift**: Training data ≠ production data
2. **Concept Drift**: Relationships between features and targets change over time
3. **Evaluation Metrics Mismatch**: Metrics that optimize for business outcomes
4. **Statistical Significance**: Ensuring improvements are real, not random
5. **Fairness and Bias**: Models must perform equitably across groups

### 2025 Trends in ML Evaluation

- **Automated Evaluation**: AI-powered evaluation of AI systems
- **Multi-objective Optimization**: Balancing accuracy, fairness, efficiency, and cost
- **Continuous Evaluation**: Real-time monitoring and adaptation
- **Explainable Evaluation**: Understanding why models fail
- **Regulatory Compliance**: EU AI Act evaluation requirements

---

## 📊 Evaluation Metrics

### Classification Metrics

#### Binary Classification

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_binary_evaluation(y_true, y_pred, y_pred_proba):
    """
    Comprehensive binary classification evaluation
    """
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'balanced_accuracy': (recall + specificity) / 2
    }
    
    return metrics

# Example usage
y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1]
y_pred_proba = [0.1, 0.9, 0.2, 0.8, 0.4, 0.1, 0.9, 0.2, 0.8, 0.9]

metrics = comprehensive_binary_evaluation(y_true, y_pred, y_pred_proba)
print("Binary Classification Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

#### Multi-class Classification

```python
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

def multiclass_evaluation(y_true, y_pred, classes):
    """
    Multi-class classification evaluation
    """
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Overall metrics
    metrics = {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics, report

# Example with 3 classes
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
y_pred = [0, 1, 2, 0, 1, 1, 0, 1, 2, 0]
classes = ['class_0', 'class_1', 'class_2']

metrics, report = multiclass_evaluation(y_true, y_pred, classes)
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_evaluation(y_true, y_pred):
    """
    Comprehensive regression evaluation
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Symmetric mean absolute percentage error
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'smape': smape
    }
    
    return metrics
```

### Custom Business Metrics

```python
def business_metrics_evaluation(y_true, y_pred, costs):
    """
    Business-focused evaluation metrics
    """
    # Cost matrix: [TN_cost, FP_cost, FN_cost, TP_cost]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = (tn * costs[0] + fp * costs[1] + fn * costs[2] + tp * costs[3])
    
    # Revenue metrics (example for fraud detection)
    revenue_saved = fn * costs[2]  # Money saved by catching fraud
    false_alarm_cost = fp * costs[1]  # Cost of false alarms
    
    net_benefit = revenue_saved - false_alarm_cost
    
    return {
        'total_cost': total_cost,
        'revenue_saved': revenue_saved,
        'false_alarm_cost': false_alarm_cost,
        'net_benefit': net_benefit,
        'roi': (net_benefit / total_cost) * 100 if total_cost > 0 else 0
    }

# Example: Fraud detection costs
# [TN_cost, FP_cost, FN_cost, TP_cost]
fraud_costs = [0, 10, 100, 0]  # False positive costs $10, missed fraud costs $100
```

---

## 🧪 A/B Testing for ML

### Statistical Framework

```python
import scipy.stats as stats
from scipy.stats import ttest_ind, chi2_contingency

class MLABTest:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power
    
    def calculate_sample_size(self, effect_size, baseline_metric):
        """
        Calculate required sample size for A/B test
        """
        # Using Cohen's d for effect size
        if effect_size == 'small':
            d = 0.2
        elif effect_size == 'medium':
            d = 0.5
        elif effect_size == 'large':
            d = 0.8
        else:
            d = effect_size
        
        # Calculate sample size using power analysis
        from statsmodels.stats.power import TTestPower
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=d, 
            alpha=self.alpha, 
            power=self.power
        )
        
        return int(sample_size * 2)  # *2 for two groups
    
    def run_ab_test(self, control_metrics, treatment_metrics, metric_type='continuous'):
        """
        Run A/B test and return statistical significance
        """
        if metric_type == 'continuous':
            # T-test for continuous metrics
            t_stat, p_value = ttest_ind(control_metrics, treatment_metrics)
            effect_size = (np.mean(treatment_metrics) - np.mean(control_metrics)) / np.std(control_metrics)
            
        elif metric_type == 'binary':
            # Chi-square test for binary metrics
            contingency_table = np.array([
                [np.sum(control_metrics == 0), np.sum(control_metrics == 1)],
                [np.sum(treatment_metrics == 0), np.sum(treatment_metrics == 1)]
            ])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            effect_size = None
        
        return {
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'control_mean': np.mean(control_metrics),
            'treatment_mean': np.mean(treatment_metrics),
            'improvement': ((np.mean(treatment_metrics) - np.mean(control_metrics)) / np.mean(control_metrics)) * 100
        }

# Example usage
ab_test = MLABTest(alpha=0.05, power=0.8)

# Calculate sample size needed
sample_size = ab_test.calculate_sample_size('medium', 0.5)
print(f"Required sample size per group: {sample_size}")

# Simulate A/B test results
np.random.seed(42)
control_accuracy = np.random.normal(0.85, 0.02, 1000)
treatment_accuracy = np.random.normal(0.87, 0.02, 1000)

results = ab_test.run_ab_test(control_accuracy, treatment_accuracy, 'continuous')
print(f"A/B Test Results: {results}")
```

### Multi-Armed Bandit Testing

```python
import numpy as np
from scipy.stats import beta

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        self.total_pulls = np.zeros(n_arms)
    
    def select_arm(self):
        """
        Select arm using Thompson sampling
        """
        samples = []
        for arm in range(self.n_arms):
            # Sample from Beta distribution
            sample = np.random.beta(
                self.successes[arm] + 1, 
                self.failures[arm] + 1
            )
            samples.append(sample)
        
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """
        Update arm statistics
        """
        self.total_pulls[arm] += 1
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
    
    def get_arm_stats(self):
        """
        Get current statistics for all arms
        """
        stats = {}
        for arm in range(self.n_arms):
            if self.total_pulls[arm] > 0:
                success_rate = self.successes[arm] / self.total_pulls[arm]
                confidence_interval = 1.96 * np.sqrt(
                    success_rate * (1 - success_rate) / self.total_pulls[arm]
                )
                stats[arm] = {
                    'success_rate': success_rate,
                    'confidence_interval': confidence_interval,
                    'total_pulls': self.total_pulls[arm]
                }
        return stats

# Example: Testing different ML model variants
n_models = 3
bandit = ThompsonSampling(n_models)

# Simulate testing
for _ in range(1000):
    selected_model = bandit.select_arm()
    
    # Simulate reward (1 for success, 0 for failure)
    if selected_model == 0:
        reward = np.random.binomial(1, 0.85)  # 85% success rate
    elif selected_model == 1:
        reward = np.random.binomial(1, 0.87)  # 87% success rate
    else:
        reward = np.random.binomial(1, 0.83)  # 83% success rate
    
    bandit.update(selected_model, reward)

print("Thompson Sampling Results:")
for arm, stats in bandit.get_arm_stats().items():
    print(f"Model {arm}: {stats}")
```

---

## 🔍 Model Debugging

### Error Analysis Framework

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelDebugger:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def analyze_errors(self, y_pred):
        """
        Comprehensive error analysis
        """
        errors = y_pred != self.y_test
        error_indices = np.where(errors)[0]
        
        # Error patterns
        error_analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(self.y_test),
            'error_by_class': self.y_test[error_indices].value_counts(),
            'predicted_when_wrong': y_pred[error_indices].value_counts()
        }
        
        return error_analysis, error_indices
    
    def feature_importance_analysis(self, feature_names):
        """
        Analyze feature importance for error cases
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def data_quality_analysis(self):
        """
        Analyze data quality issues
        """
        # Missing values
        missing_values = self.X_test.isnull().sum()
        
        # Outliers (using IQR method)
        outliers = {}
        for col in self.X_test.columns:
            Q1 = self.X_test[col].quantile(0.25)
            Q3 = self.X_test[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(self.X_test[
                (self.X_test[col] < Q1 - 1.5 * IQR) | 
                (self.X_test[col] > Q3 + 1.5 * IQR)
            ])
            outliers[col] = outlier_count
        
        # Data drift detection
        train_stats = self.X_train.describe()
        test_stats = self.X_test.describe()
        
        drift_scores = {}
        for col in self.X_train.columns:
            if col in self.X_test.columns:
                # Kolmogorov-Smirnov test for distribution drift
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(
                    self.X_train[col].dropna(), 
                    self.X_test[col].dropna()
                )
                drift_scores[col] = {
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }
        
        return {
            'missing_values': missing_values,
            'outliers': outliers,
            'drift_scores': drift_scores
        }
    
    def generate_debug_report(self, y_pred, feature_names):
        """
        Generate comprehensive debug report
        """
        error_analysis, error_indices = self.analyze_errors(y_pred)
        feature_importance = self.feature_importance_analysis(feature_names)
        data_quality = self.data_quality_analysis()
        
        report = {
            'error_analysis': error_analysis,
            'feature_importance': feature_importance,
            'data_quality': data_quality,
            'recommendations': self._generate_recommendations(
                error_analysis, feature_importance, data_quality
            )
        }
        
        return report
    
    def _generate_recommendations(self, error_analysis, feature_importance, data_quality):
        """
        Generate actionable recommendations
        """
        recommendations = []
        
        # Error rate recommendations
        if error_analysis['error_rate'] > 0.1:
            recommendations.append("High error rate detected. Consider model retraining or feature engineering.")
        
        # Data quality recommendations
        high_missing = data_quality['missing_values'][data_quality['missing_values'] > 0]
        if len(high_missing) > 0:
            recommendations.append(f"Missing values detected in: {list(high_missing.index)}")
        
        drift_detected = [col for col, stats in data_quality['drift_scores'].items() 
                         if stats['drift_detected']]
        if drift_detected:
            recommendations.append(f"Data drift detected in: {drift_detected}")
        
        return recommendations

# Example usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create debugger
debugger = ModelDebugger(model, X_train, y_train, X_test, y_test)

# Generate predictions
y_pred = model.predict(X_test)

# Generate debug report
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
report = debugger.generate_debug_report(y_pred, feature_names)

print("Debug Report:")
print(f"Error Rate: {report['error_analysis']['error_rate']:.4f}")
print(f"Recommendations: {report['recommendations']}")
```

### Model Interpretability Tools

```python
import shap
import lime
import lime.lime_tabular

class ModelInterpreter:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
    
    def shap_analysis(self, X_sample):
        """
        SHAP analysis for model interpretability
        """
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names)
            
            # Force plot for specific instance
            shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0])
            
            return shap_values
        else:
            print("Model doesn't support SHAP analysis")
            return None
    
    def lime_analysis(self, X_sample, instance_idx=0):
        """
        LIME analysis for local interpretability
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['class_0', 'class_1'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            X_sample.iloc[instance_idx].values,
            self.model.predict_proba,
            num_features=10
        )
        
        return exp
    
    def partial_dependence_analysis(self, feature_idx, X_sample):
        """
        Partial dependence plots
        """
        from sklearn.inspection import partial_dependence
        
        pdp = partial_dependence(
            self.model, X_sample, [feature_idx], percentiles=(0.05, 0.95)
        )
        
        return pdp

# Example usage
interpreter = ModelInterpreter(model, X_train, feature_names)

# SHAP analysis
shap_values = interpreter.shap_analysis(X_test[:10])

# LIME analysis
lime_exp = interpreter.lime_analysis(X_test, instance_idx=0)
print("LIME Explanation:")
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight:.4f}")
```

---

## 📈 Statistical Significance

### Power Analysis

```python
from statsmodels.stats.power import TTestPower, FTestAnovaPower
from statsmodels.stats.proportion import proportions_ztest

class StatisticalAnalyzer:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power
    
    def sample_size_power_analysis(self, effect_size, test_type='t'):
        """
        Calculate required sample size for desired power
        """
        if test_type == 't':
            power_analysis = TTestPower()
        elif test_type == 'f':
            power_analysis = FTestAnovaPower()
        
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=self.alpha,
            power=self.power
        )
        
        return int(sample_size)
    
    def effect_size_calculation(self, group1, group2):
        """
        Calculate Cohen's d effect size
        """
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + 
                             (len(group2) - 1) * np.var(group2)) / 
                            (len(group1) + len(group2) - 2))
        
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return effect_size
    
    def confidence_interval(self, data, confidence=0.95):
        """
        Calculate confidence interval
        """
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=std_err)
        
        return mean, ci
    
    def multiple_comparison_correction(self, p_values, method='bonferroni'):
        """
        Apply multiple comparison correction
        """
        from statsmodels.stats.multitest import multipletests
        
        if method == 'bonferroni':
            corrected_p_values = multipletests(p_values, method='bonferroni')[1]
        elif method == 'fdr_bh':
            corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
        elif method == 'holm':
            corrected_p_values = multipletests(p_values, method='holm')[1]
        
        return corrected_p_values

# Example usage
analyzer = StatisticalAnalyzer(alpha=0.05, power=0.8)

# Calculate effect size
group1 = np.random.normal(0.85, 0.02, 100)
group2 = np.random.normal(0.87, 0.02, 100)

effect_size = analyzer.effect_size_calculation(group1, group2)
print(f"Effect size (Cohen's d): {effect_size:.4f}")

# Calculate required sample size
sample_size = analyzer.sample_size_power_analysis(effect_size)
print(f"Required sample size per group: {sample_size}")

# Confidence interval
mean, ci = analyzer.confidence_interval(group1)
print(f"Mean: {mean:.4f}, CI: ({ci[0]:.4f}, {ci[1]:.4f})")
```

---

## 🔄 Production Monitoring

### Real-time Evaluation Framework

```python
import time
import threading
from collections import deque
import json

class ProductionMonitor:
    def __init__(self, model, evaluation_window=1000, alert_threshold=0.1):
        self.model = model
        self.evaluation_window = evaluation_window
        self.alert_threshold = alert_threshold
        
        # Rolling metrics
        self.predictions = deque(maxlen=evaluation_window)
        self.actuals = deque(maxlen=evaluation_window)
        self.timestamps = deque(maxlen=evaluation_window)
        
        # Performance tracking
        self.metrics_history = []
        self.alerts = []
        
        # Threading for real-time monitoring
        self.monitoring_thread = None
        self.is_monitoring = False
    
    def add_prediction(self, prediction, actual, timestamp=None):
        """
        Add new prediction to monitoring queue
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)
        
        # Check for alerts
        if len(self.predictions) >= self.evaluation_window:
            self._check_alerts()
    
    def _check_alerts(self):
        """
        Check for performance degradation
        """
        current_accuracy = np.mean(np.array(self.predictions) == np.array(self.actuals))
        
        if len(self.metrics_history) > 0:
            baseline_accuracy = np.mean([m['accuracy'] for m in self.metrics_history[-10:]])
            
            if current_accuracy < baseline_accuracy - self.alert_threshold:
                alert = {
                    'timestamp': time.time(),
                    'type': 'performance_degradation',
                    'current_accuracy': current_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'degradation': baseline_accuracy - current_accuracy
                }
                self.alerts.append(alert)
                print(f"ALERT: Performance degradation detected! {alert}")
        
        # Update metrics history
        self.metrics_history.append({
            'timestamp': time.time(),
            'accuracy': current_accuracy,
            'sample_size': len(self.predictions)
        })
    
    def get_current_metrics(self):
        """
        Get current performance metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)
        
        metrics = {
            'accuracy': np.mean(predictions == actuals),
            'total_predictions': len(predictions),
            'recent_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600])
        }
        
        return metrics
    
    def start_monitoring(self):
        """
        Start real-time monitoring
        """
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """
        Stop real-time monitoring
        """
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """
        Real-time monitoring loop
        """
        while self.is_monitoring:
            # Check for data drift
            if len(self.predictions) >= self.evaluation_window:
                self._detect_data_drift()
            
            # Generate periodic reports
            if len(self.metrics_history) % 100 == 0:
                self._generate_report()
            
            time.sleep(60)  # Check every minute
    
    def _detect_data_drift(self):
        """
        Detect data distribution drift
        """
        # Simple drift detection using statistical tests
        if len(self.metrics_history) > 20:
            recent_accuracy = [m['accuracy'] for m in self.metrics_history[-10:]]
            baseline_accuracy = [m['accuracy'] for m in self.metrics_history[-20:-10]]
            
            # T-test for drift detection
            t_stat, p_value = stats.ttest_ind(recent_accuracy, baseline_accuracy)
            
            if p_value < 0.05:
                drift_alert = {
                    'timestamp': time.time(),
                    'type': 'data_drift',
                    'p_value': p_value,
                    't_statistic': t_stat
                }
                self.alerts.append(drift_alert)
                print(f"ALERT: Data drift detected! p-value: {p_value:.4f}")
    
    def _generate_report(self):
        """
        Generate periodic monitoring report
        """
        report = {
            'timestamp': time.time(),
            'current_metrics': self.get_current_metrics(),
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'monitoring_duration': time.time() - self.metrics_history[0]['timestamp'] if self.metrics_history else 0
        }
        
        # Save report to file
        with open(f'monitoring_report_{int(time.time())}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

# Example usage
monitor = ProductionMonitor(model, evaluation_window=100, alert_threshold=0.05)

# Simulate production predictions
for i in range(200):
    # Simulate prediction and actual
    prediction = model.predict([X_test[i]])[0]
    actual = y_test[i]
    
    monitor.add_prediction(prediction, actual)
    
    if i % 50 == 0:
        metrics = monitor.get_current_metrics()
        print(f"Metrics at step {i}: {metrics}")

# Start monitoring
monitor.start_monitoring()
time.sleep(5)  # Let it run for 5 seconds
monitor.stop_monitoring()
```

---

## 🧪 Exercises and Projects

### Exercise 1: Comprehensive Model Evaluation

Create a comprehensive evaluation framework that includes:

1. **Multiple Metrics**: Accuracy, precision, recall, F1, AUC-ROC, business metrics
2. **Statistical Testing**: Confidence intervals, significance testing
3. **Error Analysis**: Detailed breakdown of where models fail
4. **Visualization**: Plots showing performance across different segments

```python
# Your implementation here
class ComprehensiveEvaluator:
    def __init__(self):
        pass
    
    def evaluate_model(self, model, X_test, y_test, business_costs=None):
        """
        Implement comprehensive evaluation
        """
        pass
    
    def generate_report(self):
        """
        Generate detailed evaluation report
        """
        pass
```

### Exercise 2: A/B Testing Framework

Build a complete A/B testing system for ML models:

1. **Sample Size Calculator**: Determine required sample size
2. **Statistical Tests**: T-tests, chi-square tests, multiple comparison corrections
3. **Multi-armed Bandits**: Thompson sampling for adaptive testing
4. **Results Visualization**: Clear presentation of test results

### Exercise 3: Model Debugging System

Create an automated debugging system:

1. **Error Pattern Detection**: Identify systematic errors
2. **Feature Analysis**: Understand which features contribute to errors
3. **Data Quality Checks**: Detect data drift and quality issues
4. **Recommendation Engine**: Suggest improvements based on analysis

### Project: Production ML Monitoring Dashboard

Build a real-time monitoring dashboard that includes:

1. **Real-time Metrics**: Accuracy, latency, throughput
2. **Alert System**: Automated alerts for performance degradation
3. **Drift Detection**: Statistical tests for data and concept drift
4. **Visualization**: Interactive charts and graphs
5. **Reporting**: Automated report generation

### Advanced Project: Automated Model Evaluation Pipeline

Create a complete evaluation pipeline that:

1. **Automatically Evaluates**: New model versions
2. **Compares Models**: Statistical significance testing
3. **Generates Reports**: Comprehensive evaluation reports
4. **Integrates with MLOps**: CI/CD pipeline integration
5. **Business Impact**: ROI calculations and business metrics

---

## 📖 Further Reading

### Essential Papers

1. **"A Survey of Methods for Explaining Black Box Models"** - Guidotti et al.
2. **"Practical Lessons from Predicting Clicks on Ads at Facebook"** - He et al.
3. **"Deep Learning with Differential Privacy"** - Abadi et al.
4. **"Model Cards for Model Reporting"** - Mitchell et al.

### Books

1. **"Evaluating Machine Learning Models"** - Alice Zheng
2. **"A/B Testing: The Most Powerful Way to Turn Clicks Into Customers"** - Dan Siroker
3. **"Trustworthy Machine Learning"** - Kush R. Varshney

### Online Resources

1. **Google's ML Testing Guide**: Comprehensive testing strategies
2. **Microsoft's Responsible AI**: Evaluation frameworks
3. **AWS Model Monitor**: Production monitoring tools
4. **Weights & Biases**: Experiment tracking and evaluation

### Tools and Frameworks

1. **SHAP**: Model interpretability
2. **LIME**: Local interpretability
3. **Evidently AI**: Data drift detection
4. **Great Expectations**: Data validation
5. **MLflow**: Experiment tracking
6. **Weights & Biases**: Comprehensive ML platform

---

## 🎯 Key Takeaways

1. **Comprehensive Evaluation**: Go beyond accuracy to include business metrics, fairness, and interpretability
2. **Statistical Rigor**: Use proper statistical tests and confidence intervals
3. **Continuous Monitoring**: Implement real-time monitoring for production systems
4. **Automated Debugging**: Build systems to automatically detect and diagnose issues
5. **Business Alignment**: Ensure evaluation metrics align with business objectives

---

*"The best model is not the one with the highest accuracy, but the one that creates the most value."*

**Next: [Model Deployment](ml_engineering/28_deployment.md) → Production serving, APIs, and containerization**