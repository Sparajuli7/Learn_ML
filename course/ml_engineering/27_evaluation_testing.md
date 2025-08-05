# Model Evaluation & Testing: Comprehensive Assessment Framework

*"In ML, evaluation is not just about accuracy—it's about trust, fairness, and real-world impact."*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Statistical Foundations (2025)](#statistical-foundations)
   - [Hypothesis Testing](#hypothesis-testing)
   - [Sequential Analysis](#sequential-analysis)
   - [Multi-Armed Bandits](#multi-armed-bandits)
   - [Adaptive Experimentation](#adaptive-experimentation)
3. [Evaluation Metrics](#evaluation-metrics)
4. [A/B Testing for ML](#ab-testing-for-ml)
5. [Model Debugging](#model-debugging)
6. [Statistical Significance](#statistical-significance)
7. [Production Monitoring](#production-monitoring)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

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

## 📊 Statistical Foundations (2025)

### Hypothesis Testing Framework

```python
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple

class ModernHypothesisTesting:
    """Modern hypothesis testing framework with robust statistical guarantees"""
    
    def __init__(self,
                 alpha: float = 0.05,
                 power: float = 0.8,
                 effect_size: float = 0.1):
        """Initialize hypothesis testing framework"""
        self.alpha = alpha
        self.power = power
        self.effect_size = effect_size
        
    def calculate_sample_size(self,
                            baseline_metric: float,
                            variance: Optional[float] = None) -> int:
        """Calculate required sample size using power analysis"""
        if variance is None:
            # Estimate variance for binary metrics
            variance = baseline_metric * (1 - baseline_metric)
        
        # Calculate non-centrality parameter
        ncp = self.effect_size / np.sqrt(2 * variance)
        
        # Calculate sample size using power analysis
        from statsmodels.stats.power import TTestPower
        analysis = TTestPower()
        n = analysis.solve_power(
            effect_size=ncp,
            alpha=self.alpha,
            power=self.power
        )
        
        return int(np.ceil(n))
    
    def test_hypothesis(self,
                       control: np.ndarray,
                       treatment: np.ndarray,
                       method: str = 'robust') -> Dict:
        """Test hypothesis with robust methods"""
        if method == 'robust':
            # Use robust statistical tests
            statistic, p_value = stats.mannwhitneyu(
                control, treatment, alternative='two-sided'
            )
        elif method == 'bootstrap':
            # Bootstrap hypothesis test
            statistic, p_value = self._bootstrap_test(
                control, treatment
            )
        else:
            # Traditional t-test
            statistic, p_value = stats.ttest_ind(
                control, treatment
            )
        
        # Calculate effect size
        effect = np.mean(treatment) - np.mean(control)
        relative_effect = effect / np.mean(control)
        
        # Calculate confidence intervals
        ci = self._calculate_confidence_intervals(
            control, treatment
        )
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect': effect,
            'relative_effect': relative_effect,
            'confidence_interval': ci,
            'significant': p_value < self.alpha
        }
    
    def _bootstrap_test(self,
                       control: np.ndarray,
                       treatment: np.ndarray,
                       n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Perform bootstrap hypothesis test"""
        # Calculate observed difference
        observed_diff = np.mean(treatment) - np.mean(control)
        
        # Combine samples
        combined = np.concatenate([control, treatment])
        n_control = len(control)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = np.random.choice(
                combined, size=len(combined), replace=True
            )
            
            # Calculate difference
            diff = (np.mean(resampled[n_control:]) - 
                   np.mean(resampled[:n_control]))
            bootstrap_diffs.append(diff)
        
        # Calculate p-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= 
                         np.abs(observed_diff))
        
        return observed_diff, p_value
    
    def _calculate_confidence_intervals(self,
                                     control: np.ndarray,
                                     treatment: np.ndarray,
                                     confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals"""
        # Mean difference CI
        from scipy.stats import t
        
        # Calculate standard error
        se = np.sqrt(np.var(control)/len(control) + 
                    np.var(treatment)/len(treatment))
        
        # Calculate t-critical value
        df = len(control) + len(treatment) - 2
        t_crit = t.ppf((1 + confidence)/2, df)
        
        # Calculate CI
        diff = np.mean(treatment) - np.mean(control)
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se
        
        return {
            'lower': ci_lower,
            'upper': ci_upper,
            'confidence': confidence
        }

### Sequential Analysis Framework

```python
class SequentialAnalysis:
    """Sequential analysis framework with continuous monitoring"""
    
    def __init__(self,
                 alpha: float = 0.05,
                 beta: float = 0.2,
                 max_steps: int = 100):
        """Initialize sequential analysis"""
        self.alpha = alpha
        self.beta = beta
        self.max_steps = max_steps
        
        # Initialize boundaries
        self._setup_boundaries()
    
    def _setup_boundaries(self):
        """Setup sequential boundaries"""
        # Calculate boundaries using Wald's SPRT
        A = (1 - self.beta) / self.alpha
        B = self.beta / (1 - self.alpha)
        
        self.upper_bound = np.log(A)
        self.lower_bound = np.log(B)
        
        # Adjust for multiple testing
        self.adjusted_alpha = self._calculate_adjusted_alpha()
    
    def _calculate_adjusted_alpha(self) -> float:
        """Calculate alpha adjusted for sequential testing"""
        # Use O'Brien-Fleming spending function
        from scipy.stats import norm
        
        def spending_function(t):
            return 2 * (1 - norm.cdf(norm.ppf(1-self.alpha/2)/np.sqrt(t)))
        
        return spending_function
    
    def analyze_stream(self,
                      control_stream: np.ndarray,
                      treatment_stream: np.ndarray) -> Dict:
        """Analyze streaming data"""
        results = []
        decisions = []
        
        for t in range(len(control_stream)):
            # Calculate likelihood ratio
            lr = self._calculate_likelihood_ratio(
                control_stream[:t+1],
                treatment_stream[:t+1]
            )
            
            # Make decision
            decision = self._make_decision(lr, t)
            
            results.append({
                'time': t,
                'lr': lr,
                'decision': decision
            })
            
            if decision != 'continue':
                break
        
        return {
            'results': results,
            'final_decision': decision,
            'stopping_time': t
        }
    
    def _calculate_likelihood_ratio(self,
                                  control: np.ndarray,
                                  treatment: np.ndarray) -> float:
        """Calculate likelihood ratio statistic"""
        # Use robust likelihood ratio
        from scipy.stats import norm
        
        # Calculate means and standard errors
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        
        pooled_std = np.sqrt(
            (np.var(control) * (len(control) - 1) +
             np.var(treatment) * (len(treatment) - 1)) /
            (len(control) + len(treatment) - 2)
        )
        
        # Calculate likelihood ratio
        lr = ((treatment_mean - control_mean) /
              (pooled_std * np.sqrt(2/len(control))))
        
        return lr
    
    def _make_decision(self,
                      lr: float,
                      t: int) -> str:
        """Make sequential decision"""
        # Adjust boundaries for multiple testing
        alpha_t = self.adjusted_alpha(t/self.max_steps)
        
        if lr > self.upper_bound * np.sqrt(alpha_t):
            return 'reject_null'
        elif lr < self.lower_bound * np.sqrt(alpha_t):
            return 'accept_null'
        else:
            return 'continue'

### Multi-Armed Bandit Framework

```python
class AdaptiveBanditFramework:
    """Advanced multi-armed bandit framework with adaptive allocation"""
    
    def __init__(self,
                 n_arms: int,
                 horizon: int,
                 method: str = 'thompson'):
        """Initialize bandit framework"""
        self.n_arms = n_arms
        self.horizon = horizon
        self.method = method
        
        # Initialize statistics
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms)
        
        # Setup method-specific parameters
        self._setup_method()
    
    def _setup_method(self):
        """Setup method-specific parameters"""
        if self.method == 'thompson':
            # Thompson sampling parameters
            self.alpha = np.ones(self.n_arms)
            self.beta = np.ones(self.n_arms)
        elif self.method == 'ucb':
            # UCB parameters
            self.exploration_constant = np.sqrt(2)
        elif self.method == 'eps_greedy':
            # Epsilon-greedy parameters
            self.epsilon = 0.1
    
    def select_arm(self, t: int) -> int:
        """Select arm using specified method"""
        if self.method == 'thompson':
            # Thompson sampling
            samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
        elif self.method == 'ucb':
            # UCB
            if np.any(self.pulls == 0):
                return np.where(self.pulls == 0)[0][0]
            
            ucb_values = (self.successes / self.pulls +
                         self.exploration_constant *
                         np.sqrt(np.log(t) / self.pulls))
            return np.argmax(ucb_values)
        else:
            # Epsilon-greedy
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_arms)
            else:
                return np.argmax(self.successes / 
                               np.maximum(self.pulls, 1))
    
    def update(self, arm: int, reward: float):
        """Update statistics for selected arm"""
        self.pulls[arm] += 1
        
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
    
        # Update method-specific parameters
        if self.method == 'thompson':
            self.alpha[arm] = self.successes[arm] + 1
            self.beta[arm] = self.failures[arm] + 1
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'pulls': self.pulls,
            'success_rates': self.successes / np.maximum(self.pulls, 1),
            'confidence_intervals': self._calculate_confidence_intervals()
        }
    
    def _calculate_confidence_intervals(self) -> np.ndarray:
        """Calculate confidence intervals for each arm"""
        rates = self.successes / np.maximum(self.pulls, 1)
        ses = np.sqrt(
            rates * (1 - rates) / np.maximum(self.pulls, 1)
        )
        
        return np.vstack([
            rates - 1.96 * ses,
            rates + 1.96 * ses
        ])

### Adaptive Experimentation Framework

```python
class AdaptiveExperimentationFramework:
    """Advanced adaptive experimentation framework (2025)"""
    
    def __init__(self,
                 config: Dict,
                 optimization_method: str = 'auto'):
        """Initialize adaptive experimentation"""
        self.config = config
        self.optimization_method = optimization_method
        
        # Initialize components
        self.bandit = AdaptiveBanditFramework(
            n_arms=config['n_arms'],
            horizon=config['horizon']
        )
        
        self.sequential = SequentialAnalysis(
            alpha=config['alpha'],
            beta=config['beta']
        )
        
        # Setup optimization
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Setup optimization method"""
        if self.optimization_method == 'auto':
            # Auto-select based on problem characteristics
            if self.config['n_arms'] > 10:
                self.optimization_method = 'thompson'
            elif self.config['horizon'] > 1000:
                self.optimization_method = 'ucb'
            else:
                self.optimization_method = 'eps_greedy'
    
    def run_experiment(self,
                      experiment_fn: callable,
                      max_samples: int) -> Dict:
        """Run adaptive experiment"""
        results = []
        decisions = []
        
        for t in range(max_samples):
            # Select arm
            arm = self.bandit.select_arm(t)
            
            # Run experiment
            outcome = experiment_fn(arm)
            
            # Update statistics
            self.bandit.update(arm, outcome)
            
            # Perform sequential analysis
            analysis = self.sequential.analyze_stream(
                self.bandit.successes,
                self.bandit.failures
            )
            
            # Store results
            results.append({
                'time': t,
                'arm': arm,
                'outcome': outcome,
                'analysis': analysis
            })
            
            # Check for early stopping
            if analysis['final_decision'] != 'continue':
                break
        
        return {
            'results': results,
            'final_decision': analysis['final_decision'],
            'statistics': self.bandit.get_statistics(),
            'stopping_time': t
        }
    
    def get_optimal_allocation(self) -> np.ndarray:
        """Get optimal allocation ratios"""
        stats = self.bandit.get_statistics()
        rates = stats['success_rates']
        
        # Calculate optimal allocation using Thompson sampling
        samples = np.random.beta(
            self.bandit.successes + 1,
            self.bandit.failures + 1,
            size=(1000, self.config['n_arms'])
        )
        
        # Optimal allocation proportional to probability of being best
        allocation = np.mean(
            samples == np.max(samples, axis=1, keepdims=True),
            axis=0
        )
        
        return allocation

# Example usage
def demonstrate_adaptive_experimentation():
    """Demonstrate adaptive experimentation framework"""
    
    # Configuration
    config = {
        'n_arms': 3,
        'horizon': 1000,
        'alpha': 0.05,
        'beta': 0.2
    }
    
    # Initialize framework
    framework = AdaptiveExperimentationFramework(config)
    
    # Define experiment function
    def experiment_fn(arm):
        # Simulate experiment outcome
        if arm == 0:
            return np.random.binomial(1, 0.5)
        elif arm == 1:
            return np.random.binomial(1, 0.55)
        else:
            return np.random.binomial(1, 0.45)
    
    # Run experiment
    results = framework.run_experiment(
        experiment_fn,
        max_samples=1000
    )
    
    # Get optimal allocation
    allocation = framework.get_optimal_allocation()
    
    return results, allocation

# Run demonstration
results, allocation = demonstrate_adaptive_experimentation()
print(f"Optimal allocation: {allocation}")
print(f"Final decision: {results['final_decision']}")
print(f"Stopping time: {results['stopping_time']}")
```

---

## 🚀 Exercises and Projects

### Project 1: Statistical Testing Framework
Build a comprehensive statistical testing framework implementing:
- Modern hypothesis testing with multiple correction methods
- Sequential analysis with adaptive stopping rules
- Multi-armed bandit optimization
- Power analysis and sample size calculation

### Project 2: Comprehensive Model Evaluation
Design and implement a model evaluation system featuring:
- Functional, performance, robustness, and fairness testing
- Automated bias detection across protected attributes
- Production testing with A/B testing integration
- Real-time monitoring and alerting

### Assessment Questions

1. **Statistical Theory**: Explain the mathematical foundations of sequential analysis and how it differs from traditional hypothesis testing.

2. **Practical Implementation**: Design a testing strategy for a recommendation system used in healthcare, considering fairness, robustness, and regulatory requirements.

3. **Production Scenarios**: How would you implement continuous evaluation for a fraud detection model that needs to adapt to new attack patterns?

---

## 📚 Further Reading

### Academic Papers
- "Sequential Analysis and Optimal Design" by Siegmund
- "Fairness in Machine Learning" by Barocas et al.
- "Testing Machine Learning Systems" by Breck et al.

### Industry Resources
- Google's ML Testing Best Practices
- Facebook's Responsible AI Testing Framework
- Microsoft's Fairness Assessment Guide

### Tools and Frameworks
- **Testing**: pytest-ml, mltest, great-expectations
- **Statistics**: scipy.stats, statsmodels, pymc3
- **Fairness**: fairlearn, aif360, what-if-tool
- **Production**: mlflow, weights-biases, evidently

---

*Next Chapter: [Model Deployment](31_deployment.md)*