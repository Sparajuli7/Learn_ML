# Causal AI: Understanding Cause and Effect in Machine Learning

## 🎯 Learning Objectives
- Understand causal inference principles and methods
- Master causal discovery algorithms and techniques
- Implement causal machine learning models
- Build interpretable AI systems with causal reasoning
- Apply causal AI to real-world problems

## 📚 Prerequisites
- Statistical inference and probability theory
- Machine learning fundamentals
- Python programming with scientific libraries
- Understanding of graph theory and Bayesian networks

---

## 🚀 Module Overview

### 1. Causal Inference Fundamentals

#### 1.1 Causal Graphs and DAGs
```python
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.inference.CausalInference import CausalInference

class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = set()
        self.edges = []
    
    def add_node(self, node):
        """Add a node to the causal graph"""
        self.nodes.add(node)
        self.graph.add_node(node)
    
    def add_edge(self, from_node, to_node):
        """Add a directed edge representing causation"""
        self.edges.append((from_node, to_node))
        self.graph.add_edge(from_node, to_node)
    
    def get_parents(self, node):
        """Get parent nodes (causes) of a given node"""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node):
        """Get child nodes (effects) of a given node"""
        return list(self.graph.successors(node))
    
    def is_d_separated(self, x, y, z):
        """Check if x and y are d-separated given z"""
        return nx.d_separated(self.graph, {x}, {y}, set(z))
    
    def get_backdoor_paths(self, treatment, outcome):
        """Find backdoor paths between treatment and outcome"""
        paths = []
        for path in nx.all_simple_paths(self.graph, treatment, outcome):
            if self.is_backdoor_path(path):
                paths.append(path)
        return paths
```

#### 1.2 Potential Outcomes Framework
```python
class PotentialOutcomes:
    def __init__(self, data):
        self.data = data
        self.treatment = None
        self.outcome = None
    
    def set_treatment_outcome(self, treatment_col, outcome_col):
        """Set treatment and outcome variables"""
        self.treatment = treatment_col
        self.outcome = outcome_col
    
    def calculate_ate(self):
        """Calculate Average Treatment Effect"""
        treated = self.data[self.data[self.treatment] == 1]
        control = self.data[self.data[self.treatment] == 0]
        
        ate = treated[self.outcome].mean() - control[self.outcome].mean()
        return ate
    
    def calculate_att(self):
        """Calculate Average Treatment Effect on Treated"""
        treated = self.data[self.data[self.treatment] == 1]
        control = self.data[self.data[self.treatment] == 0]
        
        # Match treated to control units
        matched_control = self.match_treated_to_control(treated, control)
        att = treated[self.outcome].mean() - matched_control[self.outcome].mean()
        return att
    
    def match_treated_to_control(self, treated, control):
        """Simple matching for ATT calculation"""
        # Simplified matching - in practice, use more sophisticated methods
        return control.sample(len(treated))
```

### 2. Causal Discovery Algorithms

#### 2.1 PC Algorithm Implementation
```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

class CausalDiscovery:
    def __init__(self, data):
        self.data = data
        self.graph = None
    
    def run_pc_algorithm(self, alpha=0.05):
        """Run PC algorithm for causal discovery"""
        # Convert data to numpy array
        data_array = self.data.values
        
        # Run PC algorithm
        self.graph, sep_set = pc(data_array, alpha, fisherz)
        
        return self.graph, sep_set
    
    def run_fges_algorithm(self):
        """Run FGES algorithm for causal discovery"""
        from causallearn.search.ScoreBased.FGES import fges
        
        # Run FGES algorithm
        self.graph = fges(self.data.values)
        
        return self.graph
    
    def validate_causal_graph(self, true_graph):
        """Validate discovered causal graph against true graph"""
        # Calculate structural accuracy
        accuracy = self.calculate_structural_accuracy(self.graph, true_graph)
        
        # Calculate edge accuracy
        edge_accuracy = self.calculate_edge_accuracy(self.graph, true_graph)
        
        return {
            'structural_accuracy': accuracy,
            'edge_accuracy': edge_accuracy
        }
    
    def calculate_structural_accuracy(self, discovered, true):
        """Calculate structural accuracy of discovered graph"""
        # Simplified accuracy calculation
        discovered_edges = set(discovered.edges())
        true_edges = set(true.edges())
        
        correct_edges = discovered_edges.intersection(true_edges)
        accuracy = len(correct_edges) / len(true_edges) if true_edges else 0
        
        return accuracy
```

#### 2.2 Structural Causal Models
```python
class StructuralCausalModel:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.noise_distributions = {}
    
    def add_variable(self, name, function, noise_dist):
        """Add a variable to the SCM"""
        self.variables[name] = {
            'function': function,
            'noise_dist': noise_dist
        }
    
    def generate_data(self, n_samples=1000):
        """Generate data from the SCM"""
        data = {}
        
        # Generate noise for all variables
        for var_name, var_info in self.variables.items():
            noise = var_info['noise_dist'].rvs(n_samples)
            data[f'{var_name}_noise'] = noise
        
        # Generate variable values using functions
        for var_name, var_info in self.variables.items():
            function = var_info['function']
            var_value = function(data)
            data[var_name] = var_value
        
        return pd.DataFrame(data)
    
    def do_intervention(self, variable, value):
        """Perform do-intervention on a variable"""
        # Create intervened SCM
        intervened_scm = StructuralCausalModel()
        
        for var_name, var_info in self.variables.items():
            if var_name == variable:
                # Replace function with constant
                intervened_scm.add_variable(var_name, lambda data: value, var_info['noise_dist'])
            else:
                intervened_scm.add_variable(var_name, var_info['function'], var_info['noise_dist'])
        
        return intervened_scm
```

### 3. Causal Machine Learning

#### 3.1 Causal Forests
```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

class CausalForest:
    def __init__(self, n_estimators=100, min_samples_leaf=10):
        self.model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf),
            model_t=RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf),
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf
        )
    
    def fit(self, X, T, Y):
        """Fit causal forest model"""
        self.model.fit(Y, T, X=X)
        return self
    
    def estimate_cate(self, X):
        """Estimate Conditional Average Treatment Effect"""
        return self.model.effect(X)
    
    def estimate_ate(self, X):
        """Estimate Average Treatment Effect"""
        cate = self.estimate_cate(X)
        return np.mean(cate)
    
    def get_feature_importance(self):
        """Get feature importance for treatment effects"""
        return self.model.feature_importances_
```

#### 3.2 Double Machine Learning
```python
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV

class DoubleMachineLearning:
    def __init__(self):
        self.model = LinearDML(
            model_y=LassoCV(),
            model_t=LassoCV()
        )
    
    def fit(self, X, T, Y):
        """Fit double machine learning model"""
        self.model.fit(Y, T, X=X)
        return self
    
    def estimate_ate(self):
        """Estimate Average Treatment Effect"""
        return self.model.ate_
    
    def estimate_cate(self, X):
        """Estimate Conditional Average Treatment Effect"""
        return self.model.effect(X)
    
    def get_confidence_intervals(self, X):
        """Get confidence intervals for CATE"""
        return self.model.effect_interval(X)
```

### 4. Causal Inference Methods

#### 4.1 Propensity Score Methods
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class PropensityScoreMatching:
    def __init__(self, estimator='logistic'):
        if estimator == 'logistic':
            self.ps_model = LogisticRegression()
        elif estimator == 'random_forest':
            self.ps_model = RandomForestClassifier()
        else:
            raise ValueError("Estimator must be 'logistic' or 'random_forest'")
    
    def fit(self, X, T):
        """Fit propensity score model"""
        self.ps_model.fit(X, T)
        return self
    
    def predict_propensity_scores(self, X):
        """Predict propensity scores"""
        return self.ps_model.predict_proba(X)[:, 1]
    
    def match_treated_control(self, X, T, Y):
        """Match treated and control units using propensity scores"""
        ps_scores = self.predict_propensity_scores(X)
        
        treated_indices = np.where(T == 1)[0]
        control_indices = np.where(T == 0)[0]
        
        matched_pairs = []
        
        for treated_idx in treated_indices:
            treated_ps = ps_scores[treated_idx]
            
            # Find closest control unit
            control_ps = ps_scores[control_indices]
            closest_control_idx = control_indices[np.argmin(np.abs(control_ps - treated_ps))]
            
            matched_pairs.append((treated_idx, closest_control_idx))
        
        return matched_pairs
    
    def estimate_ate_matching(self, X, T, Y):
        """Estimate ATE using propensity score matching"""
        matched_pairs = self.match_treated_control(X, T, Y)
        
        treatment_effects = []
        for treated_idx, control_idx in matched_pairs:
            effect = Y[treated_idx] - Y[control_idx]
            treatment_effects.append(effect)
        
        return np.mean(treatment_effects)
```

#### 4.2 Instrumental Variables
```python
class InstrumentalVariables:
    def __init__(self):
        self.model = None
    
    def fit_2sls(self, X, Z, T, Y):
        """Fit Two-Stage Least Squares"""
        from sklearn.linear_model import LinearRegression
        
        # First stage: T = Z * gamma + X * delta + error
        first_stage = LinearRegression()
        first_stage.fit(np.column_stack([Z, X]), T)
        
        # Predict treatment
        T_pred = first_stage.predict(np.column_stack([Z, X]))
        
        # Second stage: Y = T_pred * beta + X * alpha + error
        second_stage = LinearRegression()
        second_stage.fit(np.column_stack([T_pred, X]), Y)
        
        self.model = {
            'first_stage': first_stage,
            'second_stage': second_stage
        }
        
        return self
    
    def estimate_ate_iv(self):
        """Estimate ATE using instrumental variables"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        return self.model['second_stage'].coef_[0]
    
    def test_exclusion_restriction(self, Z, T, Y):
        """Test exclusion restriction assumption"""
        from scipy import stats
        
        # Test if Z is correlated with Y after controlling for T
        residuals = self.get_residuals(Z, T, Y)
        
        # Correlation test
        correlation, p_value = stats.pearsonr(Z, residuals)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'exclusion_restriction_holds': p_value > 0.05
        }
```

### 5. Causal AI Applications

#### 5.1 Causal Recommendation Systems
```python
class CausalRecommender:
    def __init__(self):
        self.user_embeddings = None
        self.item_embeddings = None
        self.causal_model = None
    
    def fit_causal_model(self, user_features, item_features, interactions, confounders):
        """Fit causal model for recommendations"""
        # Create treatment (exposure) and outcome variables
        treatment = interactions['exposure']
        outcome = interactions['rating']
        
        # Fit causal forest
        self.causal_model = CausalForest()
        self.causal_model.fit(
            X=np.column_stack([user_features, item_features, confounders]),
            T=treatment,
            Y=outcome
        )
        
        return self
    
    def estimate_causal_effect(self, user_features, item_features, confounders):
        """Estimate causal effect of recommendation"""
        X = np.column_stack([user_features, item_features, confounders])
        return self.causal_model.estimate_cate(X)
    
    def recommend_with_causal_effects(self, user_id, candidate_items):
        """Generate recommendations using causal effects"""
        user_features = self.get_user_features(user_id)
        item_features = [self.get_item_features(item_id) for item_id in candidate_items]
        confounders = self.get_confounders(user_id, candidate_items)
        
        causal_effects = []
        for item_feat, conf in zip(item_features, confounders):
            effect = self.estimate_causal_effect(user_features, item_feat, conf)
            causal_effects.append(effect)
        
        # Rank items by causal effect
        ranked_items = [(item_id, effect) for item_id, effect in zip(candidate_items, causal_effects)]
        ranked_items.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_items
```

#### 5.2 Causal Fairness in AI
```python
class CausalFairness:
    def __init__(self):
        self.causal_model = None
    
    def detect_discrimination(self, data, protected_attributes, target_variable):
        """Detect discrimination using causal methods"""
        # Build causal graph
        causal_graph = self.build_causal_graph(data, protected_attributes, target_variable)
        
        # Identify discriminatory paths
        discriminatory_paths = self.find_discriminatory_paths(causal_graph, protected_attributes, target_variable)
        
        # Calculate causal effect of protected attributes
        causal_effects = self.calculate_causal_effects(data, protected_attributes, target_variable)
        
        return {
            'discriminatory_paths': discriminatory_paths,
            'causal_effects': causal_effects,
            'has_discrimination': len(discriminatory_paths) > 0
        }
    
    def mitigate_discrimination(self, data, protected_attributes, target_variable):
        """Mitigate discrimination using causal methods"""
        # Identify backdoor paths
        backdoor_paths = self.find_backdoor_paths(protected_attributes, target_variable)
        
        # Apply backdoor adjustment
        adjusted_data = self.apply_backdoor_adjustment(data, backdoor_paths)
        
        # Train fair model
        fair_model = self.train_fair_model(adjusted_data, target_variable)
        
        return fair_model
    
    def calculate_causal_effects(self, data, protected_attributes, target_variable):
        """Calculate causal effects of protected attributes"""
        effects = {}
        
        for attr in protected_attributes:
            # Use double machine learning
            dml = DoubleMachineLearning()
            dml.fit(data.drop([target_variable] + protected_attributes, axis=1), 
                   data[attr], data[target_variable])
            
            effects[attr] = dml.estimate_ate()
        
        return effects
```

### 6. Causal AI Production Systems

#### 6.1 Causal Model Monitoring
```python
class CausalModelMonitor:
    def __init__(self, causal_model):
        self.causal_model = causal_model
        self.metrics = {}
    
    def monitor_causal_effects(self, new_data):
        """Monitor causal effects over time"""
        # Calculate causal effects on new data
        new_effects = self.causal_model.estimate_cate(new_data)
        
        # Compare with historical effects
        drift_score = self.calculate_effect_drift(new_effects)
        
        # Check for violations of causal assumptions
        assumption_violations = self.check_causal_assumptions(new_data)
        
        return {
            'effect_drift': drift_score,
            'assumption_violations': assumption_violations,
            'needs_retraining': drift_score > 0.1
        }
    
    def calculate_effect_drift(self, new_effects):
        """Calculate drift in causal effects"""
        if 'historical_effects' not in self.metrics:
            self.metrics['historical_effects'] = new_effects
            return 0.0
        
        drift = np.mean(np.abs(new_effects - self.metrics['historical_effects']))
        self.metrics['historical_effects'] = new_effects
        
        return drift
    
    def check_causal_assumptions(self, data):
        """Check for violations of causal assumptions"""
        violations = []
        
        # Check for unmeasured confounding
        if self.detect_unmeasured_confounding(data):
            violations.append('unmeasured_confounding')
        
        # Check for selection bias
        if self.detect_selection_bias(data):
            violations.append('selection_bias')
        
        return violations
```

#### 6.2 Causal AI Pipeline
```python
class CausalAIPipeline:
    def __init__(self):
        self.causal_discovery = None
        self.causal_inference = None
        self.monitor = None
    
    def build_pipeline(self, data):
        """Build end-to-end causal AI pipeline"""
        # Step 1: Causal discovery
        self.causal_discovery = CausalDiscovery(data)
        causal_graph, sep_set = self.causal_discovery.run_pc_algorithm()
        
        # Step 2: Causal inference
        self.causal_inference = CausalForest()
        self.causal_inference.fit(data.drop('outcome', axis=1), 
                                data['treatment'], data['outcome'])
        
        # Step 3: Monitoring
        self.monitor = CausalModelMonitor(self.causal_inference)
        
        return {
            'causal_graph': causal_graph,
            'causal_model': self.causal_inference,
            'monitor': self.monitor
        }
    
    def predict_with_causal_effects(self, new_data):
        """Make predictions with causal effects"""
        # Get causal effects
        causal_effects = self.causal_inference.estimate_cate(new_data)
        
        # Combine with traditional predictions
        traditional_predictions = self.get_traditional_predictions(new_data)
        
        # Weighted combination
        final_predictions = 0.7 * traditional_predictions + 0.3 * causal_effects
        
        return final_predictions
    
    def explain_causal_effects(self, data_point):
        """Explain causal effects for a data point"""
        # Get causal path
        causal_path = self.get_causal_path(data_point)
        
        # Get feature importance
        feature_importance = self.causal_inference.get_feature_importance()
        
        # Generate explanation
        explanation = self.generate_causal_explanation(causal_path, feature_importance)
        
        return explanation
```

---

## 🎯 Key Takeaways

1. **Causal Understanding**: Causal AI provides deeper understanding of cause-effect relationships
2. **Robust Inference**: Causal methods are more robust to distribution shifts
3. **Fairness**: Causal methods help detect and mitigate discrimination
4. **Interpretability**: Causal models provide interpretable explanations
5. **Production Ready**: Deploy causal AI systems with proper monitoring

## 🚀 Next Steps

1. **Advanced Causal Discovery**: Explore more sophisticated causal discovery algorithms
2. **Causal Reinforcement Learning**: Study causal methods in RL
3. **Causal NLP**: Apply causal methods to natural language processing
4. **Causal Computer Vision**: Extend causal methods to computer vision
5. **Causal Meta-Learning**: Study causal methods in meta-learning

## 📚 Additional Resources

- **Causal Inference Book**: "Causal Inference: The Mixtape" by Scott Cunningham
- **Causal Discovery**: PC, FGES, and other discovery algorithms
- **Causal ML Libraries**: DoWhy, CausalML, EconML
- **Causal Fairness**: Methods for fair AI using causal inference
- **Causal Reinforcement Learning**: Causal methods in RL

---

*This module provides a comprehensive foundation in causal AI, enabling you to build AI systems that understand cause and effect!* 🚀 