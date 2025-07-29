# Causal AI & Bayesian Methods: Causal Inference and Probabilistic Programming

*"Understanding causality is the key to building AI systems that can reason about interventions and make reliable predictions in changing environments."*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Causal Inference Fundamentals](#causal-inference-fundamentals)
3. [Bayesian Methods](#bayesian-methods)
4. [Modern Causal Discovery](#modern-causal-discovery)
5. [Implementation](#implementation)
6. [Applications](#applications)
7. [Exercises and Projects](#exercises-and-projects)

---

## 🎯 Introduction

Causal AI represents a paradigm shift in artificial intelligence, moving beyond correlation to understand true cause-and-effect relationships. Combined with Bayesian methods for principled uncertainty quantification, these approaches enable AI systems that can reason about interventions, make reliable predictions, and understand the underlying mechanisms of complex systems.

### Key Concepts

1. **Causality vs Correlation**: Distinguishing between spurious correlations and true causal relationships
2. **Interventions**: Understanding how changing one variable affects others
3. **Counterfactuals**: Reasoning about "what if" scenarios
4. **Uncertainty Quantification**: Bayesian methods for principled uncertainty handling
5. **Causal Discovery**: Automatically learning causal structure from data

### 2025 Trends

- **Causal Reinforcement Learning**: RL agents that understand intervention effects
- **Causal Language Models**: LLMs that can reason about causality
- **Bayesian Neural Networks**: Neural networks with uncertainty quantification
- **Causal Fairness**: Ensuring AI systems are causally fair
- **Interventional AI**: Systems that can plan and execute interventions

---

## 🔗 Causal Inference Fundamentals

### Structural Causal Models (SCMs)

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
import networkx as nx

class StructuralCausalModel:
    """Structural Causal Model implementation"""
    
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.noise_distributions = {}
        self.graph = nx.DiGraph()
    
    def add_variable(self, name: str, function: Callable, noise_dist: Callable):
        """Add a variable with its structural equation"""
        self.variables[name] = {
            'function': function,
            'noise_dist': noise_dist
        }
        self.graph.add_node(name)
    
    def add_causal_relationship(self, cause: str, effect: str):
        """Add a causal relationship between variables"""
        self.graph.add_edge(cause, effect)
    
    def sample(self, n_samples: int = 1000) -> pd.DataFrame:
        """Sample from the SCM"""
        data = {}
        
        # Topological sort to ensure correct sampling order
        sorted_vars = list(nx.topological_sort(self.graph))
        
        for var in sorted_vars:
            if var in self.variables:
                # Sample noise
                noise = self.variables[var]['noise_dist'](n_samples)
                
                # Get parent values
                parents = list(self.graph.predecessors(var))
                parent_values = {parent: data[parent] for parent in parents}
                
                # Apply structural equation
                function = self.variables[var]['function']
                data[var] = function(parent_values, noise)
        
        return pd.DataFrame(data)
    
    def do_intervention(self, variable: str, value: float) -> 'StructuralCausalModel':
        """Perform do-intervention on a variable"""
        intervened_scm = StructuralCausalModel()
        
        # Copy all variables and relationships
        for var in self.variables:
            if var != variable:
                intervened_scm.variables[var] = self.variables[var].copy()
                intervened_scm.noise_distributions[var] = self.noise_distributions[var]
        
        # Add intervened variable with constant value
        intervened_scm.variables[variable] = {
            'function': lambda parents, noise: np.full(len(noise), value),
            'noise_dist': self.variables[variable]['noise_dist']
        }
        
        # Copy graph structure
        intervened_scm.graph = self.graph.copy()
        
        return intervened_scm
    
    def estimate_causal_effect(self, treatment: str, outcome: str, 
                             intervention_value: float = 1.0) -> float:
        """Estimate average causal effect"""
        # Sample from original model
        original_data = self.sample(1000)
        original_outcome = original_data[outcome].mean()
        
        # Sample from intervened model
        intervened_scm = self.do_intervention(treatment, intervention_value)
        intervened_data = intervened_scm.sample(1000)
        intervened_outcome = intervened_data[outcome].mean()
        
        return intervened_outcome - original_outcome

# Example usage
def create_example_scm():
    """Create a simple SCM example"""
    scm = StructuralCausalModel()
    
    # Add variables with structural equations
    scm.add_variable('X', 
                     lambda parents, noise: noise,  # X = noise
                     lambda n: np.random.normal(0, 1, n))
    
    scm.add_variable('Y', 
                     lambda parents, noise: 2 * parents['X'] + noise,  # Y = 2X + noise
                     lambda n: np.random.normal(0, 0.5, n))
    
    scm.add_variable('Z', 
                     lambda parents, noise: parents['X'] + parents['Y'] + noise,  # Z = X + Y + noise
                     lambda n: np.random.normal(0, 0.3, n))
    
    # Add causal relationships
    scm.add_causal_relationship('X', 'Y')
    scm.add_causal_relationship('X', 'Z')
    scm.add_causal_relationship('Y', 'Z')
    
    return scm

# scm = create_example_scm()
# data = scm.sample(1000)
# effect = scm.estimate_causal_effect('X', 'Y')
# print(f"Causal effect of X on Y: {effect:.3f}")
```

### Backdoor Adjustment

```python
class CausalInference:
    """Causal inference using backdoor adjustment"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.causal_graph = None
    
    def set_causal_graph(self, graph: nx.DiGraph):
        """Set the causal graph"""
        self.causal_graph = graph
    
    def get_backdoor_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Find backdoor adjustment set"""
        if self.causal_graph is None:
            raise ValueError("Causal graph must be set first")
        
        # Find all backdoor paths
        backdoor_paths = []
        for path in nx.all_simple_paths(self.causal_graph, treatment, outcome):
            if self._is_backdoor_path(path, treatment):
                backdoor_paths.append(path)
        
        # Find minimal adjustment set
        adjustment_set = set()
        for path in backdoor_paths:
            # Find variables that can block this path
            for var in path[1:-1]:  # Exclude treatment and outcome
                if var != treatment and var != outcome:
                    adjustment_set.add(var)
        
        return list(adjustment_set)
    
    def _is_backdoor_path(self, path: List[str], treatment: str) -> bool:
        """Check if path is a backdoor path"""
        if len(path) < 3:
            return False
        
        # Check if path has arrow pointing to treatment
        for i in range(len(path) - 1):
            if path[i+1] == treatment and self.causal_graph.has_edge(path[i], path[i+1]):
                return True
        
        return False
    
    def estimate_ate(self, treatment: str, outcome: str) -> float:
        """Estimate Average Treatment Effect using backdoor adjustment"""
        adjustment_set = self.get_backdoor_adjustment_set(treatment, outcome)
        
        if not adjustment_set:
            # No adjustment needed
            treated = self.data[self.data[treatment] == 1][outcome].mean()
            control = self.data[self.data[treatment] == 0][outcome].mean()
            return treated - control
        
        # Stratified analysis
        ate_estimates = []
        
        # Group by adjustment variables
        groups = self.data.groupby(adjustment_set)
        
        for name, group in groups:
            if len(group) > 0:
                treated = group[group[treatment] == 1][outcome].mean()
                control = group[group[treatment] == 0][outcome].mean()
                
                if not (np.isnan(treated) or np.isnan(control)):
                    ate_estimates.append(treated - control)
        
        return np.mean(ate_estimates) if ate_estimates else 0.0
    
    def estimate_conditional_ate(self, treatment: str, outcome: str, 
                               condition: str, condition_value: Any) -> float:
        """Estimate Conditional Average Treatment Effect"""
        # Filter data by condition
        conditional_data = self.data[self.data[condition] == condition_value]
        
        if len(conditional_data) == 0:
            return 0.0
        
        # Create temporary inference object
        temp_inference = CausalInference(conditional_data)
        temp_inference.set_causal_graph(self.causal_graph)
        
        return temp_inference.estimate_ate(treatment, outcome)
```

---

## 🎲 Bayesian Methods

### Bayesian Inference

```python
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

class BayesianInference:
    """Bayesian inference implementation"""
    
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
        self.data = []
    
    def set_prior(self, parameter: str, prior_dist: stats.rv_continuous):
        """Set prior distribution for a parameter"""
        self.priors[parameter] = prior_dist
    
    def add_data(self, data_point: float):
        """Add new data point"""
        self.data.append(data_point)
    
    def update_posterior(self, parameter: str, likelihood_func: Callable):
        """Update posterior distribution"""
        if parameter not in self.priors:
            raise ValueError(f"No prior set for parameter {parameter}")
        
        prior = self.priors[parameter]
        
        # Compute likelihood
        likelihood = 1.0
        for data_point in self.data:
            likelihood *= likelihood_func(data_point, parameter)
        
        # Posterior is proportional to prior * likelihood
        # For conjugate priors, we can compute analytically
        if isinstance(prior, stats.norm):
            # Normal-Normal conjugate pair
            mu_0, sigma_0 = prior.mean(), prior.std()
            n = len(self.data)
            x_bar = np.mean(self.data)
            
            # Assuming known variance for simplicity
            sigma = 1.0  # Known variance
            
            # Posterior parameters
            mu_post = (mu_0 / sigma_0**2 + n * x_bar / sigma**2) / (1 / sigma_0**2 + n / sigma**2)
            sigma_post = np.sqrt(1 / (1 / sigma_0**2 + n / sigma**2))
            
            self.posteriors[parameter] = stats.norm(mu_post, sigma_post)
        
        elif isinstance(prior, stats.beta):
            # Beta-Bernoulli conjugate pair
            alpha_0, beta_0 = prior.args
            
            # Count successes and failures
            successes = sum(1 for x in self.data if x == 1)
            failures = len(self.data) - successes
            
            alpha_post = alpha_0 + successes
            beta_post = beta_0 + failures
            
            self.posteriors[parameter] = stats.beta(alpha_post, beta_post)
    
    def get_posterior_mean(self, parameter: str) -> float:
        """Get posterior mean for a parameter"""
        if parameter not in self.posteriors:
            raise ValueError(f"No posterior computed for parameter {parameter}")
        
        return self.posteriors[parameter].mean()
    
    def get_posterior_interval(self, parameter: str, confidence: float = 0.95) -> Tuple[float, float]:
        """Get credible interval for a parameter"""
        if parameter not in self.posteriors:
            raise ValueError(f"No posterior computed for parameter {parameter}")
        
        alpha = 1 - confidence
        return self.posteriors[parameter].ppf([alpha/2, 1-alpha/2])
    
    def predict(self, n_samples: int = 1000) -> np.ndarray:
        """Make predictions using posterior samples"""
        if not self.posteriors:
            raise ValueError("No posteriors computed")
        
        predictions = []
        for _ in range(n_samples):
            # Sample from posteriors
            params = {}
            for param in self.posteriors:
                params[param] = self.posteriors[param].rvs()
            
            # Generate prediction (simplified)
            prediction = self._generate_prediction(params)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _generate_prediction(self, params: Dict[str, float]) -> float:
        """Generate prediction given parameters"""
        # Simplified prediction function
        # In practice, this would depend on your model
        return params.get('mu', 0.0) + np.random.normal(0, 0.1)

# Example usage
def bayesian_linear_regression_example():
    """Example of Bayesian linear regression"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.normal(0, 1, 100)
    y = 2 * X + 1 + np.random.normal(0, 0.5, 100)
    
    # Set up Bayesian inference
    bayes = BayesianInference()
    
    # Set priors for slope and intercept
    bayes.set_prior('slope', stats.norm(0, 2))
    bayes.set_prior('intercept', stats.norm(0, 2))
    
    # Add data
    for x, y_val in zip(X, y):
        bayes.add_data((x, y_val))
    
    # Update posteriors (simplified)
    # In practice, you'd use MCMC or variational inference
    
    return bayes

# bayes = bayesian_linear_regression_example()
```

### Markov Chain Monte Carlo (MCMC)

```python
import numpy as np
from typing import Callable, List, Tuple

class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler"""
    
    def __init__(self, target_distribution: Callable, proposal_distribution: Callable):
        self.target_dist = target_distribution
        self.proposal_dist = proposal_distribution
        self.samples = []
        self.acceptance_rate = 0.0
    
    def sample(self, n_samples: int, initial_state: float = 0.0) -> np.ndarray:
        """Generate samples using Metropolis-Hastings"""
        current_state = initial_state
        samples = [current_state]
        accepted = 0
        
        for _ in range(n_samples):
            # Propose new state
            proposed_state = self.proposal_dist(current_state)
            
            # Calculate acceptance probability
            alpha = min(1.0, 
                       self.target_dist(proposed_state) / self.target_dist(current_state))
            
            # Accept or reject
            if np.random.random() < alpha:
                current_state = proposed_state
                accepted += 1
            
            samples.append(current_state)
        
        self.samples = np.array(samples)
        self.acceptance_rate = accepted / n_samples
        
        return self.samples
    
    def get_acceptance_rate(self) -> float:
        """Get acceptance rate"""
        return self.acceptance_rate
    
    def get_effective_sample_size(self) -> int:
        """Estimate effective sample size"""
        # Simple autocorrelation-based estimate
        if len(self.samples) < 2:
            return len(self.samples)
        
        # Calculate autocorrelation
        autocorr = np.correlate(self.samples, self.samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first zero crossing
        zero_crossing = np.where(np.diff(np.sign(autocorr)))[0]
        if len(zero_crossing) > 0:
            lag = zero_crossing[0]
        else:
            lag = len(autocorr)
        
        return len(self.samples) // (2 * lag + 1)

# Example usage
def gaussian_target(x: float) -> float:
    """Target distribution: Gaussian"""
    return np.exp(-0.5 * (x - 2)**2)

def gaussian_proposal(current: float) -> float:
    """Proposal distribution: Gaussian random walk"""
    return np.random.normal(current, 0.5)

# mcmc = MetropolisHastings(gaussian_target, gaussian_proposal)
# samples = mcmc.sample(10000)
# print(f"Acceptance rate: {mcmc.get_acceptance_rate():.3f}")
# print(f"Effective sample size: {mcmc.get_effective_sample_size()}")
```

---

## 🔍 Modern Causal Discovery

### PC Algorithm

```python
class CausalDiscovery:
    """Causal discovery using the PC algorithm"""
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        self.data = data
        self.alpha = alpha
        self.skeleton = None
        self.causal_graph = None
    
    def pc_algorithm(self) -> nx.Graph:
        """Implement PC algorithm for causal discovery"""
        # Step 1: Start with complete undirected graph
        n_vars = len(self.data.columns)
        skeleton = nx.complete_graph(n_vars)
        
        # Map node indices to variable names
        var_names = list(self.data.columns)
        node_to_var = {i: var_names[i] for i in range(n_vars)}
        
        # Step 2: Remove edges based on conditional independence tests
        l = 0  # Size of conditioning set
        while True:
            edges_to_remove = []
            
            for edge in skeleton.edges():
                i, j = edge
                neighbors_i = list(skeleton.neighbors(i))
                neighbors_j = list(skeleton.neighbors(j))
                
                # Find common neighbors
                common_neighbors = set(neighbors_i) & set(neighbors_j)
                
                # Test all subsets of size l
                for subset in self._get_subsets(common_neighbors, l):
                    if self._test_conditional_independence(i, j, subset):
                        edges_to_remove.append((i, j))
                        break
            
            # Remove edges
            for edge in edges_to_remove:
                skeleton.remove_edge(*edge)
            
            # Check if any node has degree <= l
            max_degree = max(skeleton.degree(node) for node in skeleton.nodes())
            if max_degree <= l:
                break
            
            l += 1
        
        self.skeleton = skeleton
        return skeleton
    
    def _test_conditional_independence(self, i: int, j: int, subset: List[int]) -> bool:
        """Test conditional independence between variables i and j given subset"""
        var_names = list(self.data.columns)
        
        # Get variable names
        var_i = var_names[i]
        var_j = var_names[j]
        conditioning_vars = [var_names[k] for k in subset]
        
        # Perform conditional independence test
        # Using partial correlation for continuous variables
        if len(conditioning_vars) == 0:
            # Simple correlation test
            correlation = self.data[var_i].corr(self.data[var_j])
            p_value = self._correlation_p_value(correlation, len(self.data))
        else:
            # Partial correlation test
            partial_corr = self._partial_correlation(var_i, var_j, conditioning_vars)
            p_value = self._correlation_p_value(partial_corr, len(self.data))
        
        return p_value > self.alpha
    
    def _partial_correlation(self, var1: str, var2: str, conditioning_vars: List[str]) -> float:
        """Calculate partial correlation"""
        # Simplified implementation
        # In practice, use proper partial correlation calculation
        
        # For simplicity, use correlation of residuals
        from sklearn.linear_model import LinearRegression
        
        # Regress var1 on conditioning variables
        if conditioning_vars:
            X = self.data[conditioning_vars]
            y1 = self.data[var1]
            reg1 = LinearRegression().fit(X, y1)
            residuals1 = y1 - reg1.predict(X)
            
            # Regress var2 on conditioning variables
            y2 = self.data[var2]
            reg2 = LinearRegression().fit(X, y2)
            residuals2 = y2 - reg2.predict(X)
            
            # Correlation of residuals
            return np.corrcoef(residuals1, residuals2)[0, 1]
        else:
            return self.data[var1].corr(self.data[var2])
    
    def _correlation_p_value(self, correlation: float, n: int) -> float:
        """Calculate p-value for correlation test"""
        # Simplified p-value calculation
        # In practice, use proper statistical test
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        return 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    def _get_subsets(self, elements: set, size: int) -> List[List[int]]:
        """Get all subsets of given size"""
        from itertools import combinations
        return [list(combo) for combo in combinations(elements, size)]
    
    def orient_edges(self) -> nx.DiGraph:
        """Orient edges to create causal graph"""
        if self.skeleton is None:
            raise ValueError("Must run PC algorithm first")
        
        # Simplified orientation
        # In practice, use more sophisticated orientation rules
        
        causal_graph = nx.DiGraph()
        causal_graph.add_edges_from(self.skeleton.edges())
        
        # Apply orientation rules
        for edge in self.skeleton.edges():
            i, j = edge
            
            # Check for unshielded triples
            neighbors_i = set(self.skeleton.neighbors(i))
            neighbors_j = set(self.skeleton.neighbors(j))
            
            # Find common neighbors
            common = neighbors_i & neighbors_j
            
            for k in common:
                # Check if i-k-j is an unshielded triple
                if not self.skeleton.has_edge(i, k) and not self.skeleton.has_edge(j, k):
                    # Apply orientation rule
                    if self._test_conditional_independence(i, j, [k]):
                        # Orient as i -> k <- j
                        causal_graph.add_edge(i, k)
                        causal_graph.add_edge(j, k)
                        if causal_graph.has_edge(i, j):
                            causal_graph.remove_edge(i, j)
        
        self.causal_graph = causal_graph
        return causal_graph

# Example usage
def causal_discovery_example():
    """Example of causal discovery"""
    # Generate synthetic data with known causal structure
    np.random.seed(42)
    n = 1000
    
    # X -> Y -> Z
    X = np.random.normal(0, 1, n)
    Y = 2 * X + np.random.normal(0, 0.5, n)
    Z = 1.5 * Y + np.random.normal(0, 0.3, n)
    
    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    # Run causal discovery
    discovery = CausalDiscovery(data)
    skeleton = discovery.pc_algorithm()
    causal_graph = discovery.orient_edges()
    
    return discovery, data

# discovery, data = causal_discovery_example()
# print("Discovered causal graph:")
# print(list(discovery.causal_graph.edges()))
```

---

## 💻 Implementation

### Causal Effect Estimation

```python
class CausalEffectEstimator:
    """Comprehensive causal effect estimation"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.causal_graph = None
    
    def set_causal_graph(self, graph: nx.DiGraph):
        """Set the causal graph"""
        self.causal_graph = graph
    
    def estimate_ate(self, treatment: str, outcome: str, 
                    method: str = "backdoor") -> Dict[str, Any]:
        """Estimate Average Treatment Effect"""
        
        if method == "backdoor":
            return self._backdoor_adjustment(treatment, outcome)
        elif method == "instrumental_variable":
            return self._instrumental_variable(treatment, outcome)
        elif method == "front_door":
            return self._front_door_adjustment(treatment, outcome)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _backdoor_adjustment(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Estimate ATE using backdoor adjustment"""
        if self.causal_graph is None:
            raise ValueError("Causal graph must be set")
        
        # Find backdoor adjustment set
        adjustment_set = self._find_backdoor_adjustment_set(treatment, outcome)
        
        # Stratified analysis
        if not adjustment_set:
            # No adjustment needed
            treated = self.data[self.data[treatment] == 1][outcome].mean()
            control = self.data[self.data[treatment] == 0][outcome].mean()
            ate = treated - control
            
            return {
                "ate": ate,
                "method": "backdoor",
                "adjustment_set": [],
                "treated_mean": treated,
                "control_mean": control
            }
        
        # Stratified analysis
        groups = self.data.groupby(adjustment_set)
        ate_estimates = []
        weights = []
        
        for name, group in groups:
            if len(group) > 0:
                treated = group[group[treatment] == 1][outcome].mean()
                control = group[group[treatment] == 0][outcome].mean()
                
                if not (np.isnan(treated) or np.isnan(control)):
                    ate_estimates.append(treated - control)
                    weights.append(len(group))
        
        if ate_estimates:
            ate = np.average(ate_estimates, weights=weights)
        else:
            ate = 0.0
        
        return {
            "ate": ate,
            "method": "backdoor",
            "adjustment_set": adjustment_set,
            "strata_estimates": ate_estimates,
            "weights": weights
        }
    
    def _find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Find backdoor adjustment set"""
        if self.causal_graph is None:
            return []
        
        # Find all backdoor paths
        backdoor_paths = []
        for path in nx.all_simple_paths(self.causal_graph, treatment, outcome):
            if self._is_backdoor_path(path, treatment):
                backdoor_paths.append(path)
        
        # Find minimal adjustment set
        adjustment_set = set()
        for path in backdoor_paths:
            for var in path[1:-1]:  # Exclude treatment and outcome
                if var != treatment and var != outcome:
                    adjustment_set.add(var)
        
        return list(adjustment_set)
    
    def _is_backdoor_path(self, path: List[str], treatment: str) -> bool:
        """Check if path is a backdoor path"""
        if len(path) < 3:
            return False
        
        # Check if path has arrow pointing to treatment
        for i in range(len(path) - 1):
            if path[i+1] == treatment and self.causal_graph.has_edge(path[i], path[i+1]):
                return True
        
        return False
    
    def _instrumental_variable(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Estimate ATE using instrumental variable"""
        # Simplified IV estimation
        # In practice, use proper IV methods
        
        # Assume we have an instrument Z
        if 'Z' not in self.data.columns:
            raise ValueError("Instrumental variable Z not found in data")
        
        # First stage: regress treatment on instrument
        from sklearn.linear_model import LinearRegression
        
        X_iv = self.data[['Z']]
        y_treatment = self.data[treatment]
        
        first_stage = LinearRegression().fit(X_iv, y_treatment)
        treatment_pred = first_stage.predict(X_iv)
        
        # Second stage: regress outcome on predicted treatment
        X_second = treatment_pred.reshape(-1, 1)
        y_outcome = self.data[outcome]
        
        second_stage = LinearRegression().fit(X_second, y_outcome)
        ate = second_stage.coef_[0]
        
        return {
            "ate": ate,
            "method": "instrumental_variable",
            "first_stage_coef": first_stage.coef_[0],
            "second_stage_coef": second_stage.coef_[0]
        }
    
    def _front_door_adjustment(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Estimate ATE using front-door adjustment"""
        # Simplified front-door adjustment
        # In practice, use proper front-door methods
        
        # Assume we have a mediator M
        if 'M' not in self.data.columns:
            raise ValueError("Mediator M not found in data")
        
        # Step 1: Estimate effect of treatment on mediator
        treated_mediator = self.data[self.data[treatment] == 1]['M'].mean()
        control_mediator = self.data[self.data[treatment] == 0]['M'].mean()
        effect_treatment_mediator = treated_mediator - control_mediator
        
        # Step 2: Estimate effect of mediator on outcome
        # Stratify by treatment
        effect_mediator_outcome_treated = self._estimate_mediator_effect(1)
        effect_mediator_outcome_control = self._estimate_mediator_effect(0)
        
        # Step 3: Combine effects
        ate = (effect_treatment_mediator * effect_mediator_outcome_treated + 
               effect_treatment_mediator * effect_mediator_outcome_control) / 2
        
        return {
            "ate": ate,
            "method": "front_door",
            "effect_treatment_mediator": effect_treatment_mediator,
            "effect_mediator_outcome_treated": effect_mediator_outcome_treated,
            "effect_mediator_outcome_control": effect_mediator_outcome_control
        }
    
    def _estimate_mediator_effect(self, treatment_value: int) -> float:
        """Estimate effect of mediator on outcome for given treatment value"""
        # Simplified estimation
        # In practice, use proper mediation analysis
        
        subset = self.data[self.data['treatment'] == treatment_value]
        if len(subset) == 0:
            return 0.0
        
        # Simple correlation-based estimate
        return subset['M'].corr(subset['outcome'])

# Example usage
def causal_effect_example():
    """Example of causal effect estimation"""
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # Confounder
    U = np.random.normal(0, 1, n)
    
    # Treatment affected by confounder
    treatment = 0.5 * U + np.random.normal(0, 0.5, n)
    treatment = (treatment > 0).astype(int)
    
    # Outcome affected by treatment and confounder
    outcome = 2 * treatment + 1.5 * U + np.random.normal(0, 0.3, n)
    
    data = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'confounder': U
    })
    
    # Estimate causal effect
    estimator = CausalEffectEstimator(data)
    
    # Create causal graph
    graph = nx.DiGraph()
    graph.add_edge('confounder', 'treatment')
    graph.add_edge('confounder', 'outcome')
    graph.add_edge('treatment', 'outcome')
    
    estimator.set_causal_graph(graph)
    
    # Estimate ATE
    result = estimator.estimate_ate('treatment', 'outcome', 'backdoor')
    
    return result

# result = causal_effect_example()
# print(f"Estimated ATE: {result['ate']:.3f}")
```

---

## 🎯 Applications

### Causal Fairness

```python
class CausalFairness:
    """Causal fairness analysis"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sensitive_attributes = []
        self.outcome = None
        self.treatment = None
    
    def set_sensitive_attributes(self, attributes: List[str]):
        """Set sensitive attributes for fairness analysis"""
        self.sensitive_attributes = attributes
    
    def set_outcome(self, outcome: str):
        """Set outcome variable"""
        self.outcome = outcome
    
    def set_treatment(self, treatment: str):
        """Set treatment variable"""
        self.treatment = treatment
    
    def analyze_direct_effect(self) -> Dict[str, Any]:
        """Analyze direct effect of sensitive attributes on outcome"""
        results = {}
        
        for attr in self.sensitive_attributes:
            # Stratify by sensitive attribute
            groups = self.data.groupby(attr)
            
            group_effects = {}
            for name, group in groups:
                if self.treatment and self.outcome:
                    treated = group[group[self.treatment] == 1][self.outcome].mean()
                    control = group[group[self.treatment] == 0][self.outcome].mean()
                    effect = treated - control
                    group_effects[name] = effect
            
            results[attr] = group_effects
        
        return results
    
    def analyze_indirect_effect(self) -> Dict[str, Any]:
        """Analyze indirect effect through mediators"""
        results = {}
        
        for attr in self.sensitive_attributes:
            # Find mediators between sensitive attribute and outcome
            mediators = self._find_mediators(attr, self.outcome)
            
            indirect_effects = {}
            for mediator in mediators:
                effect = self._estimate_indirect_effect(attr, mediator, self.outcome)
                indirect_effects[mediator] = effect
            
            results[attr] = indirect_effects
        
        return results
    
    def _find_mediators(self, cause: str, effect: str) -> List[str]:
        """Find potential mediators between cause and effect"""
        # Simplified mediator detection
        # In practice, use more sophisticated methods
        
        potential_mediators = []
        for col in self.data.columns:
            if col not in [cause, effect]:
                # Check if variable is correlated with both cause and effect
                corr_cause = abs(self.data[cause].corr(self.data[col]))
                corr_effect = abs(self.data[col].corr(self.data[effect]))
                
                if corr_cause > 0.1 and corr_effect > 0.1:
                    potential_mediators.append(col)
        
        return potential_mediators
    
    def _estimate_indirect_effect(self, cause: str, mediator: str, effect: str) -> float:
        """Estimate indirect effect through mediator"""
        # Simplified indirect effect estimation
        # In practice, use proper mediation analysis
        
        # Effect of cause on mediator
        effect_cause_mediator = self.data[cause].corr(self.data[mediator])
        
        # Effect of mediator on effect (controlling for cause)
        from sklearn.linear_model import LinearRegression
        
        X = self.data[[cause, mediator]]
        y = self.data[effect]
        
        model = LinearRegression().fit(X, y)
        effect_mediator_outcome = model.coef_[1]  # Coefficient for mediator
        
        return effect_cause_mediator * effect_mediator_outcome
```

---

## 🧪 Exercises and Projects

### Exercise 1: Implement Causal Discovery

```python
# Your task: Implement a basic causal discovery algorithm

class CustomCausalDiscovery:
    def __init__(self, data: pd.DataFrame):
        """
        TODO: Implement this class
        
        Requirements:
        1. Implement skeleton discovery
        2. Implement edge orientation
        3. Handle different data types
        4. Provide confidence measures
        """
        pass

def test_causal_discovery():
    """Test the custom causal discovery implementation"""
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, n)
    Y = 2 * X + np.random.normal(0, 0.5, n)
    Z = 1.5 * Y + np.random.normal(0, 0.3, n)
    
    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    discovery = CustomCausalDiscovery(data)
    graph = discovery.discover_causal_structure()
    print(graph.edges())
```

### Exercise 2: Build Bayesian Network

```python
# Your task: Implement a Bayesian network

class CustomBayesianNetwork:
    def __init__(self):
        """
        TODO: Implement this class
        
        Requirements:
        1. Define network structure
        2. Learn conditional probabilities
        3. Perform inference
        4. Handle missing data
        """
        pass

def test_bayesian_network():
    """Test the custom Bayesian network implementation"""
    network = CustomBayesianNetwork()
    # Test inference capabilities
    pass
```

### Project: Causal Recommendation System

Build a recommendation system that understands causality:

- Identify causal relationships in user behavior
- Predict intervention effects
- Handle confounding variables
- Provide causally fair recommendations

**Implementation Steps:**
1. Build causal discovery for user behavior
2. Implement causal effect estimation
3. Create intervention planning system
4. Add fairness constraints
5. Develop evaluation metrics

### Project: Bayesian Neural Network

Create a neural network with uncertainty quantification:

- Implement Bayesian layers
- Add uncertainty estimation
- Handle out-of-distribution data
- Provide reliable predictions

**Features:**
- Monte Carlo dropout
- Variational inference
- Uncertainty calibration
- Out-of-distribution detection
- Reliable decision making

---

## 📖 Further Reading

### Essential Papers

1. **"Causal Inference in Statistics: A Primer"** (Pearl et al., 2016)
2. **"Elements of Causal Inference"** (Peters et al., 2017)
3. **"Bayesian Data Analysis"** (Gelman et al., 2013)
4. **"Causal Discovery with Reinforcement Learning"** (Zhu et al., 2020)

### Advanced Topics

1. **Causal Reinforcement Learning**: RL with causal understanding
2. **Causal Language Models**: LLMs that understand causality
3. **Bayesian Deep Learning**: Neural networks with uncertainty
4. **Causal Fairness**: Ensuring AI systems are causally fair
5. **Interventional AI**: Systems that can plan interventions

### Tools and Frameworks

1. **PyMC**: Probabilistic programming
2. **CausalML**: Causal inference library
3. **DoWhy**: Causal reasoning framework
4. **Pyro**: Deep probabilistic programming
5. **CausalDiscoveryToolbox**: Causal discovery algorithms

---

## 🎯 Key Takeaways

1. **Causal inference** goes beyond correlation to understand true cause-and-effect relationships.

2. **Bayesian methods** provide principled uncertainty quantification for AI systems.

3. **Causal discovery** enables automatic learning of causal structure from data.

4. **Interventional reasoning** allows AI systems to understand the effects of actions and interventions.

5. **Causal fairness** ensures AI systems are fair in terms of causal relationships rather than just correlations.

6. **Modern causal AI** combines causal understanding with machine learning for more reliable and interpretable systems.

---

*"Understanding causality is not just about making better predictions, but about building AI systems that can reason about interventions and make reliable decisions in changing environments."*

**Next: [Continual & Meta Learning](specialized_ml/20_continual_meta_learning.md) → Lifelong learning and adaptation**