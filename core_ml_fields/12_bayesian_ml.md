# Bayesian Machine Learning

## Overview
Bayesian Machine Learning provides a probabilistic framework for learning from data with explicit uncertainty quantification. This guide covers probabilistic models, inference methods, and uncertainty quantification for 2025.

## Table of Contents
1. [Bayesian Fundamentals](#bayesian-fundamentals)
2. [Probabilistic Models](#probabilistic-models)
3. [Inference Methods](#inference-methods)
4. [Gaussian Processes](#gaussian-processes)
5. [Variational Inference](#variational-inference)
6. [Production Applications](#production-applications)

## Bayesian Fundamentals

### Basic Bayesian Framework
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianModel:
    def __init__(self, prior_params: Dict[str, float]):
        self.prior_params = prior_params
        self.posterior_params = None
        
    def update_posterior(self, data: np.ndarray):
        """Update posterior given new data"""
        # Simplified conjugate prior update
        n = len(data)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        # Prior parameters
        mu_0 = self.prior_params['mu_0']
        sigma_0 = self.prior_params['sigma_0']
        nu_0 = self.prior_params['nu_0']
        sigma_0_sq = self.prior_params['sigma_0_sq']
        
        # Posterior parameters
        mu_n = (nu_0 * mu_0 + n * sample_mean) / (nu_0 + n)
        nu_n = nu_0 + n
        sigma_n_sq = (nu_0 * sigma_0_sq + (n-1) * sample_var + 
                     (n * nu_0 / (nu_0 + n)) * (sample_mean - mu_0)**2) / nu_n
        
        self.posterior_params = {
            'mu_n': mu_n,
            'nu_n': nu_n,
            'sigma_n_sq': sigma_n_sq
        }
        
        return self.posterior_params
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        if self.posterior_params is None:
            raise ValueError("Model not trained yet")
        
        # Predictive distribution
        mu_pred = self.posterior_params['mu_n']
        sigma_pred = np.sqrt(self.posterior_params['sigma_n_sq'])
        
        predictions = np.full_like(x, mu_pred)
        uncertainties = np.full_like(x, sigma_pred)
        
        return predictions, uncertainties

# Example usage
prior_params = {
    'mu_0': 0.0,
    'sigma_0': 1.0,
    'nu_0': 1.0,
    'sigma_0_sq': 1.0
}

model = BayesianModel(prior_params)

# Generate some data
np.random.seed(42)
data = np.random.normal(2.0, 1.5, 100)

# Update posterior
posterior = model.update_posterior(data)
print(f"Posterior mean: {posterior['mu_n']:.3f}")
print(f"Posterior variance: {posterior['sigma_n_sq']:.3f}")
```

### Conjugate Priors
```python
class ConjugatePriors:
    """Common conjugate prior-posterior pairs"""
    
    @staticmethod
    def normal_normal(prior_mu: float, prior_sigma: float, 
                     data: np.ndarray, sigma_known: float) -> Dict[str, float]:
        """Normal-Normal conjugate pair"""
        n = len(data)
        sample_mean = np.mean(data)
        
        # Posterior parameters
        posterior_precision = 1/prior_sigma**2 + n/sigma_known**2
        posterior_mu = (prior_mu/prior_sigma**2 + n*sample_mean/sigma_known**2) / posterior_precision
        posterior_sigma = np.sqrt(1/posterior_precision)
        
        return {
            'mu': posterior_mu,
            'sigma': posterior_sigma
        }
    
    @staticmethod
    def beta_binomial(prior_alpha: float, prior_beta: float, 
                     successes: int, trials: int) -> Dict[str, float]:
        """Beta-Binomial conjugate pair"""
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + (trials - successes)
        
        return {
            'alpha': posterior_alpha,
            'beta': posterior_beta
        }
    
    @staticmethod
    def gamma_poisson(prior_alpha: float, prior_beta: float, 
                     data: np.ndarray) -> Dict[str, float]:
        """Gamma-Poisson conjugate pair"""
        n = len(data)
        data_sum = np.sum(data)
        
        posterior_alpha = prior_alpha + data_sum
        posterior_beta = prior_beta + n
        
        return {
            'alpha': posterior_alpha,
            'beta': posterior_beta
        }

# Example usage
# Normal-Normal
data = np.random.normal(5.0, 2.0, 50)
posterior = ConjugatePriors.normal_normal(
    prior_mu=0.0, prior_sigma=10.0, 
    data=data, sigma_known=2.0
)
print(f"Normal posterior: {posterior}")

# Beta-Binomial
successes, trials = 15, 20
posterior = ConjugatePriors.beta_binomial(
    prior_alpha=1.0, prior_beta=1.0,
    successes=successes, trials=trials
)
print(f"Beta posterior: {posterior}")
```

## Probabilistic Models

### Bayesian Linear Regression
```python
class BayesianLinearRegression:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Bayesian Linear Regression with Normal-Gamma prior
        
        Args:
            alpha: Prior precision for weights
            beta: Prior precision for noise
        """
        self.alpha = alpha
        self.beta = beta
        self.posterior_params = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model"""
        n, d = X.shape
        
        # Prior parameters
        m_0 = np.zeros(d)  # Prior mean for weights
        S_0 = (1/self.alpha) * np.eye(d)  # Prior covariance for weights
        a_0 = 1.0  # Prior shape for noise precision
        b_0 = 1.0  # Prior rate for noise precision
        
        # Posterior parameters
        S_n_inv = np.linalg.inv(S_0) + self.beta * X.T @ X
        S_n = np.linalg.inv(S_n_inv)
        m_n = S_n @ (np.linalg.inv(S_0) @ m_0 + self.beta * X.T @ y)
        
        a_n = a_0 + n/2
        b_n = b_0 + 0.5 * (y.T @ y - m_n.T @ S_n_inv @ m_n + m_0.T @ np.linalg.inv(S_0) @ m_0)
        
        self.posterior_params = {
            'm_n': m_n,
            'S_n': S_n,
            'a_n': a_n,
            'b_n': b_n
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        if self.posterior_params is None:
            raise ValueError("Model not fitted yet")
        
        m_n = self.posterior_params['m_n']
        S_n = self.posterior_params['S_n']
        a_n = self.posterior_params['a_n']
        b_n = self.posterior_params['b_n']
        
        # Predictive mean
        y_pred = X @ m_n
        
        # Predictive variance
        noise_var = b_n / (a_n - 1)  # Expected noise variance
        weights_var = np.diag(X @ S_n @ X.T)  # Uncertainty from weights
        total_var = noise_var + weights_var
        
        return y_pred, total_var
    
    def sample_predictions(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Sample from predictive distribution"""
        if self.posterior_params is None:
            raise ValueError("Model not fitted yet")
        
        m_n = self.posterior_params['m_n']
        S_n = self.posterior_params['S_n']
        a_n = self.posterior_params['a_n']
        b_n = self.posterior_params['b_n']
        
        # Sample noise precision
        noise_precision = np.random.gamma(a_n, 1/b_n, n_samples)
        noise_var = 1/noise_precision
        
        # Sample weights
        weights = np.random.multivariate_normal(m_n, S_n, n_samples)
        
        # Generate predictions
        predictions = []
        for i in range(n_samples):
            pred = X @ weights[i] + np.random.normal(0, np.sqrt(noise_var[i]))
            predictions.append(pred)
        
        return np.array(predictions)

# Example usage
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3)
true_weights = np.array([1.5, -0.8, 0.3])
y = X @ true_weights + np.random.normal(0, 0.5, n_samples)

# Fit Bayesian model
blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
blr.fit(X, y)

# Make predictions
X_test = np.random.randn(10, 3)
y_pred, y_var = blr.predict(X_test)
print(f"Predictions: {y_pred}")
print(f"Uncertainties: {y_var}")

# Sample predictions
samples = blr.sample_predictions(X_test, n_samples=50)
print(f"Sample predictions shape: {samples.shape}")
```

### Bayesian Neural Networks
```python
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 prior_std: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights with prior
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with prior distribution"""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(module.weight, 0, self.prior_std)
            nn.init.normal_(module.bias, 0, self.prior_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def log_prior(self) -> torch.Tensor:
        """Compute log prior probability"""
        log_prior = 0
        for module in [self.fc1, self.fc2, self.fc3]:
            log_prior += torch.sum(stats.norm.logpdf(module.weight, 0, self.prior_std))
            log_prior += torch.sum(stats.norm.logpdf(module.bias, 0, self.prior_std))
        return log_prior
    
    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log likelihood"""
        y_pred = self.forward(x)
        return torch.sum(stats.norm.logpdf(y, y_pred, 1.0))
    
    def log_posterior(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log posterior (up to constant)"""
        return self.log_prior() + self.log_likelihood(x, y)

class BayesianNNTrainer:
    def __init__(self, model: BayesianNeuralNetwork, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Negative log posterior (loss function)
        loss = -self.model.log_posterior(x, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty using MC dropout"""
        self.model.train()  # Enable dropout for uncertainty
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_pred, uncertainty

# Example usage
# Generate data
X = torch.randn(100, 2)
y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + 0.1 * torch.randn(100, 1)

# Create and train model
bnn = BayesianNeuralNetwork(input_dim=2, hidden_dim=20, output_dim=1)
trainer = BayesianNNTrainer(bnn)

# Training
for epoch in range(1000):
    loss = trainer.train_step(X, y)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Predictions with uncertainty
X_test = torch.randn(10, 2)
mean_pred, uncertainty = trainer.predict_with_uncertainty(X_test)
print(f"Mean predictions: {mean_pred}")
print(f"Uncertainties: {uncertainty}")
```

## Inference Methods

### Markov Chain Monte Carlo (MCMC)
```python
import pymc3 as pm
import arviz as az

class MCMCInference:
    def __init__(self):
        self.trace = None
        self.model = None
    
    def fit_linear_regression(self, X: np.ndarray, y: np.ndarray, 
                            n_samples: int = 2000) -> az.InferenceData:
        """Fit linear regression using MCMC"""
        
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sd=1)
            
            # Likelihood
            mu = alpha + pm.math.dot(X, beta)
            likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=y)
            
            # MCMC sampling
            trace = pm.sample(n_samples, tune=1000, return_inferencedata=True)
        
        self.trace = trace
        self.model = model
        
        return trace
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        if self.trace is None:
            raise ValueError("Model not fitted yet")
        
        with self.model:
            pm_pred = pm.sample_posterior_predictive(
                self.trace, samples=1000, X=X
            )
        
        predictions = pm_pred['likelihood']
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def plot_posterior(self):
        """Plot posterior distributions"""
        if self.trace is None:
            raise ValueError("Model not fitted yet")
        
        az.plot_posterior(self.trace)
        plt.show()

# Example usage
np.random.seed(42)
X = np.random.randn(100, 2)
true_weights = np.array([1.5, -0.8])
y = X @ true_weights + np.random.normal(0, 0.5, 100)

# Fit with MCMC
mcmc = MCMCInference()
trace = mcmc.fit_linear_regression(X, y)

# Predictions
X_test = np.random.randn(10, 2)
mean_pred, uncertainty = mcmc.predict_with_uncertainty(X_test)
print(f"Mean predictions: {mean_pred}")
print(f"Uncertainties: {uncertainty}")
```

### Variational Inference
```python
class VariationalInference:
    def __init__(self, model: nn.Module, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples
        self.optimizer = None
        
    def fit(self, x: torch.Tensor, y: torch.Tensor, 
            n_epochs: int = 1000, learning_rate: float = 0.001):
        """Fit model using variational inference"""
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # Sample from variational posterior
            loss = 0
            for _ in range(self.n_samples):
                # Forward pass
                y_pred = self.model(x)
                
                # Compute loss (negative ELBO)
                log_likelihood = -0.5 * torch.sum((y - y_pred)**2)
                kl_divergence = self._compute_kl_divergence()
                
                loss += -(log_likelihood - kl_divergence)
            
            loss = loss / self.n_samples
            loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def _compute_kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational posterior and prior"""
        kl_div = 0
        for module in [self.model.fc1, self.model.fc2, self.model.fc3]:
            # Simplified KL divergence computation
            kl_div += torch.sum(module.weight**2) / (2 * self.model.prior_std**2)
            kl_div += torch.sum(module.bias**2) / (2 * self.model.prior_std**2)
        return kl_div
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty"""
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.n_samples):
            pred = self.model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_pred, uncertainty

# Example usage
# Generate data
X = torch.randn(100, 2)
y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + 0.1 * torch.randn(100, 1)

# Create model and fit with VI
bnn = BayesianNeuralNetwork(input_dim=2, hidden_dim=20, output_dim=1)
vi = VariationalInference(bnn)
vi.fit(X, y)

# Predictions
X_test = torch.randn(10, 2)
mean_pred, uncertainty = vi.predict_with_uncertainty(X_test)
print(f"Mean predictions: {mean_pred}")
print(f"Uncertainties: {uncertainty}")
```

## Gaussian Processes

### Gaussian Process Regression
```python
import GPy

class GaussianProcessRegression:
    def __init__(self, kernel_type: str = 'RBF'):
        self.kernel_type = kernel_type
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process"""
        if self.kernel_type == 'RBF':
            kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        elif self.kernel_type == 'Matern':
            kernel = GPy.kern.Matern52(input_dim=X.shape[1])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        self.model = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel)
        self.model.optimize()
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        mean, variance = self.model.predict(X)
        return mean.flatten(), variance.flatten()
    
    def sample_predictions(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Sample from predictive distribution"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        samples = self.model.posterior_samples_f(X, size=n_samples)
        return samples.reshape(n_samples, -1)

# Example usage
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X.flatten()) + 0.1 * np.random.randn(100)

# Fit GP
gp = GaussianProcessRegression(kernel_type='RBF')
gp.fit(X, y)

# Predictions
X_test = np.linspace(0, 10, 50).reshape(-1, 1)
mean_pred, variance = gp.predict(X_test)
print(f"Mean predictions: {mean_pred}")
print(f"Variances: {variance}")

# Sample predictions
samples = gp.sample_predictions(X_test, n_samples=10)
print(f"Sample predictions shape: {samples.shape}")
```

## Production Applications

### Bayesian Optimization
```python
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, objective_func, bounds: List[Tuple[float, float]], 
                 n_initial_points: int = 5):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.X = []
        self.y = []
        
    def optimize(self, n_iterations: int = 20) -> Tuple[np.ndarray, float]:
        """Optimize using Bayesian optimization"""
        
        # Initial random points
        for _ in range(self.n_initial_points):
            x = np.random.uniform([b[0] for b in self.bounds], 
                                [b[1] for b in self.bounds])
            y = self.objective_func(x)
            self.X.append(x)
            self.y.append(y)
        
        # Bayesian optimization loop
        for i in range(n_iterations):
            # Fit GP
            X = np.array(self.X)
            y = np.array(self.y)
            
            gp = GaussianProcessRegression()
            gp.fit(X, y)
            
            # Acquisition function (Expected Improvement)
            def acquisition_function(x):
                mean, variance = gp.predict(x.reshape(1, -1))
                current_best = np.max(self.y)
                
                # Expected Improvement
                improvement = mean - current_best
                z = improvement / np.sqrt(variance + 1e-8)
                ei = improvement * stats.norm.cdf(z) + np.sqrt(variance) * stats.norm.pdf(z)
                return -ei  # Minimize negative EI
            
            # Find next point
            result = minimize(acquisition_function, 
                           x0=np.random.uniform([b[0] for b in self.bounds], 
                                               [b[1] for b in self.bounds]),
                           bounds=self.bounds,
                           method='L-BFGS-B')
            
            # Evaluate objective
            x_next = result.x
            y_next = self.objective_func(x_next)
            
            self.X.append(x_next)
            self.y.append(y_next)
            
            print(f"Iteration {i+1}: x={x_next}, y={y_next:.4f}")
        
        # Return best result
        best_idx = np.argmax(self.y)
        return np.array(self.X[best_idx]), self.y[best_idx]

# Example usage
def objective_function(x):
    """Example objective function"""
    return -(x[0]**2 + x[1]**2)  # Maximize negative distance from origin

bounds = [(-5, 5), (-5, 5)]
optimizer = BayesianOptimizer(objective_function, bounds)
best_x, best_y = optimizer.optimize(n_iterations=10)
print(f"Best x: {best_x}, Best y: {best_y}")
```

## Conclusion

Bayesian Machine Learning provides a principled framework for uncertainty quantification and probabilistic modeling. Key areas include:

1. **Probabilistic Models**: Bayesian linear regression, neural networks, and Gaussian processes
2. **Inference Methods**: MCMC, variational inference, and conjugate priors
3. **Uncertainty Quantification**: Predictive distributions and confidence intervals
4. **Production Applications**: Bayesian optimization and decision making

The field continues to evolve with new methods for more efficient inference and broader applications.

## Resources

- [PyMC3 Documentation](https://docs.pymc.io/)
- [GPy Documentation](https://gpy.readthedocs.io/)
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)