# Uncertainty Quantification: Measuring ML Confidence

## Table of Contents
1. [Introduction](#introduction)
2. [Bayesian Neural Networks](#bayesian-neural-networks)
3. [Ensemble Methods](#ensemble-methods)
4. [Monte Carlo Methods](#monte-carlo-methods)
5. [Conformal Prediction](#conformal-prediction)
6. [Uncertainty Calibration](#uncertainty-calibration)
7. [Practical Implementation](#practical-implementation)
8. [Exercises and Projects](#exercises-and-projects)

## Introduction

Uncertainty quantification is crucial for building reliable ML systems. This chapter covers techniques for estimating and calibrating uncertainty in model predictions, enabling better decision-making in production environments.

### Key Learning Objectives
- Understand Bayesian approaches to uncertainty quantification
- Implement ensemble methods for uncertainty estimation
- Apply Monte Carlo techniques for uncertainty sampling
- Use conformal prediction for reliable uncertainty bounds
- Calibrate uncertainty estimates for real-world applications

## Bayesian Neural Networks

### Variational Inference

```python
# Bayesian Neural Networks with Variational Inference
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_samples=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        # Variational parameters for weights
        self.w1_mu = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w1_logvar = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.b1_mu = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.b1_logvar = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        
        self.w2_mu = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        self.w2_logvar = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        self.b2_mu = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.b2_logvar = nn.Parameter(torch.randn(output_dim) * 0.1)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, sample_weights=True):
        """Forward pass with optional weight sampling"""
        
        if sample_weights:
            # Sample weights from variational distribution
            w1 = self.reparameterize(self.w1_mu, self.w1_logvar)
            b1 = self.reparameterize(self.b1_mu, self.b1_logvar)
            w2 = self.reparameterize(self.w2_mu, self.w2_logvar)
            b2 = self.reparameterize(self.b2_mu, self.b2_logvar)
        else:
            # Use mean weights
            w1, b1 = self.w1_mu, self.b1_mu
            w2, b2 = self.w2_mu, self.b2_mu
        
        # Forward pass
        h = F.relu(F.linear(x, w1, b1))
        output = F.linear(h, w2, b2)
        
        return output
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Make predictions with uncertainty estimates"""
        
        predictions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x, sample_weights=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'predictions': predictions,
            'confidence': 1.96 * std_pred  # 95% confidence interval
        }
    
    def elbo_loss(self, x, y, num_samples=10):
        """Evidence Lower BOund (ELBO) loss"""
        
        # Reconstruction loss
        recon_loss = 0
        for _ in range(num_samples):
            pred = self.forward(x, sample_weights=True)
            recon_loss += F.mse_loss(pred, y)
        recon_loss /= num_samples
        
        # KL divergence for weights
        kl_w1 = -0.5 * torch.sum(1 + self.w1_logvar - self.w1_mu.pow(2) - self.w1_logvar.exp())
        kl_b1 = -0.5 * torch.sum(1 + self.b1_logvar - self.b1_mu.pow(2) - self.b1_logvar.exp())
        kl_w2 = -0.5 * torch.sum(1 + self.w2_logvar - self.w2_mu.pow(2) - self.w2_logvar.exp())
        kl_b2 = -0.5 * torch.sum(1 + self.b2_logvar - self.b2_mu.pow(2) - self.b2_logvar.exp())
        
        kl_loss = kl_w1 + kl_b1 + kl_w2 + kl_b2
        
        return recon_loss + kl_loss
```

### MCMC Sampling

```python
# Markov Chain Monte Carlo for Bayesian Inference
class MCMCSampler:
    def __init__(self, model, step_size=0.01):
        self.model = model
        self.step_size = step_size
        self.samples = []
    
    def metropolis_hastings(self, x, y, num_samples=1000, burn_in=100):
        """Metropolis-Hastings sampling"""
        
        # Initialize parameters
        current_params = self._get_model_parameters()
        current_log_prob = self._log_posterior(current_params, x, y)
        
        accepted = 0
        
        for i in range(num_samples + burn_in):
            # Propose new parameters
            proposed_params = self._propose_parameters(current_params)
            proposed_log_prob = self._log_posterior(proposed_params, x, y)
            
            # Accept/reject
            log_acceptance_ratio = proposed_log_prob - current_log_prob
            if torch.rand(1) < torch.exp(log_acceptance_ratio):
                current_params = proposed_params
                current_log_prob = proposed_log_prob
                accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                self.samples.append(current_params.copy())
        
        acceptance_rate = accepted / (num_samples + burn_in)
        
        return {
            'samples': self.samples,
            'acceptance_rate': acceptance_rate
        }
    
    def _get_model_parameters(self):
        """Get current model parameters"""
        
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.data.clone()
        
        return params
    
    def _propose_parameters(self, current_params):
        """Propose new parameters using random walk"""
        
        proposed_params = {}
        for name, param in current_params.items():
            # Add Gaussian noise
            noise = torch.randn_like(param) * self.step_size
            proposed_params[name] = param + noise
        
        return proposed_params
    
    def _log_posterior(self, params, x, y):
        """Calculate log posterior probability"""
        
        # Set model parameters
        for name, param in params.items():
            getattr(self.model, name).data = param
        
        # Calculate likelihood
        pred = self.model(x, sample_weights=False)
        log_likelihood = -F.mse_loss(pred, y)
        
        # Calculate prior (assuming Gaussian prior)
        log_prior = 0
        for name, param in params.items():
            log_prior += -0.5 * torch.sum(param ** 2)  # Gaussian prior
        
        return log_likelihood + log_prior
    
    def predict_with_mcmc_samples(self, x):
        """Make predictions using MCMC samples"""
        
        predictions = []
        
        for sample_params in self.samples:
            # Set model parameters
            for name, param in sample_params.items():
                getattr(self.model, name).data = param
            
            # Make prediction
            with torch.no_grad():
                pred = self.model(x, sample_weights=False)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'predictions': predictions,
            'confidence': 1.96 * std_pred
        }
```

## Ensemble Methods

### Deep Ensembles

```python
# Deep Ensembles for Uncertainty Quantification
class DeepEnsemble:
    def __init__(self, model_class, num_models=5, **model_kwargs):
        self.model_class = model_class
        self.num_models = num_models
        self.model_kwargs = model_kwargs
        self.models = []
    
    def train_ensemble(self, train_loader, epochs=100, lr=0.001):
        """Train ensemble of models"""
        
        for i in range(self.num_models):
            print(f"Training model {i+1}/{self.num_models}")
            
            # Initialize model
            model = self.model_class(**self.model_kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Train model
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    
                    loss.backward()
                    optimizer.step()
            
            self.models.append(model)
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Make predictions with uncertainty estimates"""
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'predictions': predictions,
            'confidence': 1.96 * std_pred
        }
    
    def epistemic_aleatoric_uncertainty(self, x):
        """Separate epistemic and aleatoric uncertainty"""
        
        # Get ensemble predictions
        ensemble_preds = self.predict_with_uncertainty(x)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = ensemble_preds['std'] ** 2
        
        # Aleatoric uncertainty (data uncertainty)
        # This is a simplified approach - in practice, you'd need to model the noise
        aleatoric_uncertainty = torch.ones_like(epistemic_uncertainty) * 0.1
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'total': total_uncertainty
        }
```

### Bootstrap Ensembles

```python
# Bootstrap Ensembles for Uncertainty Estimation
class BootstrapEnsemble:
    def __init__(self, model_class, num_models=10, bootstrap_ratio=0.8, **model_kwargs):
        self.model_class = model_class
        self.num_models = num_models
        self.bootstrap_ratio = bootstrap_ratio
        self.model_kwargs = model_kwargs
        self.models = []
        self.bootstrap_indices = []
    
    def create_bootstrap_samples(self, dataset_size):
        """Create bootstrap samples"""
        
        for _ in range(self.num_models):
            # Sample with replacement
            indices = torch.randint(0, dataset_size, (int(dataset_size * self.bootstrap_ratio),))
            self.bootstrap_indices.append(indices)
    
    def train_ensemble(self, dataset, epochs=100, lr=0.001):
        """Train ensemble with bootstrap sampling"""
        
        dataset_size = len(dataset)
        self.create_bootstrap_samples(dataset_size)
        
        for i in range(self.num_models):
            print(f"Training bootstrap model {i+1}/{self.num_models}")
            
            # Create bootstrap dataset
            bootstrap_indices = self.bootstrap_indices[i]
            bootstrap_dataset = torch.utils.data.Subset(dataset, bootstrap_indices)
            bootstrap_loader = torch.utils.data.DataLoader(bootstrap_dataset, batch_size=32, shuffle=True)
            
            # Initialize and train model
            model = self.model_class(**self.model_kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in bootstrap_loader:
                    optimizer.zero_grad()
                    
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    
                    loss.backward()
                    optimizer.step()
            
            self.models.append(model)
    
    def predict_with_uncertainty(self, x):
        """Make predictions with bootstrap uncertainty"""
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Bootstrap confidence intervals
        sorted_preds = torch.sort(predictions, dim=0)[0]
        lower_ci = sorted_preds[int(0.025 * self.num_models)]
        upper_ci = sorted_preds[int(0.975 * self.num_models)]
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'predictions': predictions,
            'confidence_interval': (lower_ci, upper_ci),
            'confidence': upper_ci - lower_ci
        }
```

## Monte Carlo Methods

### Monte Carlo Dropout

```python
# Monte Carlo Dropout for Uncertainty Estimation
class MCDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, enable_dropout=True):
        """Forward pass with optional dropout"""
        
        if enable_dropout:
            x = self.dropout1(F.relu(self.fc1(x)))
            x = self.dropout2(F.relu(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        
        return self.fc3(x)
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Make predictions with MC dropout uncertainty"""
        
        predictions = []
        
        self.train()  # Enable dropout during inference
        
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x, enable_dropout=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'predictions': predictions,
            'confidence': 1.96 * std_pred
        }
    
    def epistemic_uncertainty(self, x, num_samples=100):
        """Estimate epistemic uncertainty using MC dropout"""
        
        # Get predictions with dropout
        mc_predictions = self.predict_with_uncertainty(x, num_samples)
        
        # Epistemic uncertainty is the variance of predictions
        epistemic_uncertainty = mc_predictions['std'] ** 2
        
        return epistemic_uncertainty
```

### Hamiltonian Monte Carlo

```python
# Hamiltonian Monte Carlo for Bayesian Inference
class HamiltonianMCMC:
    def __init__(self, model, step_size=0.01, num_steps=10):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        self.samples = []
    
    def hamiltonian_monte_carlo(self, x, y, num_samples=1000, burn_in=100):
        """Hamiltonian Monte Carlo sampling"""
        
        # Initialize parameters and momenta
        current_params = self._get_model_parameters()
        current_momenta = self._initialize_momenta(current_params)
        
        accepted = 0
        
        for i in range(num_samples + burn_in):
            # Propose new state using HMC
            proposed_params, proposed_momenta = self._leapfrog_step(
                current_params, current_momenta, x, y
            )
            
            # Accept/reject based on Hamiltonian
            current_hamiltonian = self._hamiltonian(current_params, current_momenta, x, y)
            proposed_hamiltonian = self._hamiltonian(proposed_params, proposed_momenta, x, y)
            
            log_acceptance_ratio = current_hamiltonian - proposed_hamiltonian
            
            if torch.rand(1) < torch.exp(log_acceptance_ratio):
                current_params = proposed_params
                current_momenta = proposed_momenta
                accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                self.samples.append(current_params.copy())
        
        acceptance_rate = accepted / (num_samples + burn_in)
        
        return {
            'samples': self.samples,
            'acceptance_rate': acceptance_rate
        }
    
    def _initialize_momenta(self, params):
        """Initialize momenta for HMC"""
        
        momenta = {}
        for name, param in params.items():
            momenta[name] = torch.randn_like(param)
        
        return momenta
    
    def _leapfrog_step(self, params, momenta, x, y):
        """Leapfrog integration step"""
        
        # Half step for momenta
        momenta = self._update_momenta(params, momenta, x, y, -0.5 * self.step_size)
        
        # Full step for parameters
        params = self._update_parameters(params, momenta, self.step_size)
        
        # Half step for momenta
        momenta = self._update_momenta(params, momenta, x, y, -0.5 * self.step_size)
        
        return params, momenta
    
    def _update_parameters(self, params, momenta, step_size):
        """Update parameters using momenta"""
        
        updated_params = {}
        for name, param in params.items():
            updated_params[name] = param + step_size * momenta[name]
        
        return updated_params
    
    def _update_momenta(self, params, momenta, x, y, step_size):
        """Update momenta using gradients"""
        
        # Set model parameters
        for name, param in params.items():
            getattr(self.model, name).data = param
        
        # Calculate gradients
        pred = self.model(x, sample_weights=False)
        loss = F.mse_loss(pred, y)
        loss.backward()
        
        updated_momenta = {}
        for name, momentum in momenta.items():
            param = getattr(self.model, name)
            gradient = param.grad
            updated_momenta[name] = momentum + step_size * gradient
        
        return updated_momenta
    
    def _hamiltonian(self, params, momenta, x, y):
        """Calculate Hamiltonian (energy)"""
        
        # Kinetic energy
        kinetic_energy = 0
        for momentum in momenta.values():
            kinetic_energy += 0.5 * torch.sum(momentum ** 2)
        
        # Potential energy (negative log posterior)
        for name, param in params.items():
            getattr(self.model, name).data = param
        
        pred = self.model(x, sample_weights=False)
        potential_energy = F.mse_loss(pred, y)
        
        # Add prior energy (assuming Gaussian prior)
        prior_energy = 0
        for param in params.values():
            prior_energy += 0.5 * torch.sum(param ** 2)
        
        return kinetic_energy + potential_energy + prior_energy
```

## Conformal Prediction

### Conformal Prediction Implementation

```python
# Conformal Prediction for Reliable Uncertainty Quantification
class ConformalPrediction:
    def __init__(self, model, calibration_data, alpha=0.1):
        self.model = model
        self.calibration_data = calibration_data
        self.alpha = alpha  # Significance level
        self.conformity_scores = []
    
    def fit(self):
        """Fit conformal predictor using calibration data"""
        
        self.conformity_scores = []
        
        for x, y in self.calibration_data:
            # Get model prediction
            with torch.no_grad():
                pred = self.model(x.unsqueeze(0))
            
            # Calculate conformity score (negative absolute error)
            conformity_score = -torch.abs(pred - y).item()
            self.conformity_scores.append(conformity_score)
        
        # Sort conformity scores
        self.conformity_scores.sort()
        
        # Calculate threshold for (1-alpha) coverage
        n_calibration = len(self.conformity_scores)
        threshold_idx = int((1 - self.alpha) * n_calibration)
        self.threshold = self.conformity_scores[threshold_idx]
    
    def predict_with_intervals(self, x):
        """Make predictions with conformal prediction intervals"""
        
        with torch.no_grad():
            pred = self.model(x.unsqueeze(0))
        
        # Calculate prediction interval
        interval_width = -self.threshold  # Convert back from conformity score
        
        lower_bound = pred - interval_width
        upper_bound = pred + interval_width
        
        return {
            'prediction': pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': interval_width,
            'coverage': 1 - self.alpha
        }
    
    def adaptive_conformal_prediction(self, x, calibration_data):
        """Adaptive conformal prediction"""
        
        # Use local calibration data
        local_scores = []
        
        for cal_x, cal_y in calibration_data:
            # Calculate distance to query point
            distance = torch.norm(cal_x - x)
            
            # Weight by distance
            weight = torch.exp(-distance)
            
            # Get prediction and conformity score
            with torch.no_grad():
                pred = self.model(cal_x.unsqueeze(0))
            
            conformity_score = -torch.abs(pred - cal_y).item()
            local_scores.append((conformity_score, weight))
        
        # Sort by weighted scores
        local_scores.sort(key=lambda x: x[0])
        
        # Calculate weighted threshold
        total_weight = sum(weight for _, weight in local_scores)
        target_weight = (1 - self.alpha) * total_weight
        
        cumulative_weight = 0
        threshold = None
        
        for score, weight in local_scores:
            cumulative_weight += weight
            if cumulative_weight >= target_weight:
                threshold = score
                break
        
        # Make prediction
        with torch.no_grad():
            pred = self.model(x.unsqueeze(0))
        
        interval_width = -threshold
        lower_bound = pred - interval_width
        upper_bound = pred + interval_width
        
        return {
            'prediction': pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': interval_width,
            'coverage': 1 - self.alpha
        }
```

## Uncertainty Calibration

### Calibration Methods

```python
# Uncertainty Calibration
class UncertaintyCalibration:
    def __init__(self):
        self.calibration_methods = {
            'temperature_scaling': self.temperature_scaling,
            'platt_scaling': self.platt_scaling,
            'isotonic_regression': self.isotonic_regression
        }
    
    def calibrate_uncertainty(self, model, calibration_data, method='temperature_scaling'):
        """Calibrate uncertainty estimates"""
        
        if method not in self.calibration_methods:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return self.calibration_methods[method](model, calibration_data)
    
    def temperature_scaling(self, model, calibration_data):
        """Temperature scaling for uncertainty calibration"""
        
        # Add temperature parameter
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01)
        
        def loss_fn():
            optimizer.zero_grad()
            total_loss = 0
            
            for x, y in calibration_data:
                with torch.no_grad():
                    logits = model(x.unsqueeze(0))
                
                # Scale logits by temperature
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=1)
                
                # Calculate negative log likelihood
                loss = -torch.log(probs[0, y.long()])
                total_loss += loss
            
            total_loss.backward()
            return total_loss
        
        # Optimize temperature
        optimizer.step(loss_fn)
        
        return {
            'temperature': temperature.item(),
            'calibrated_model': lambda x: model(x) / temperature
        }
    
    def platt_scaling(self, model, calibration_data):
        """Platt scaling for uncertainty calibration"""
        
        # Add scaling parameters
        a = nn.Parameter(torch.zeros(1))
        b = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([a, b], lr=0.01)
        
        def loss_fn():
            optimizer.zero_grad()
            total_loss = 0
            
            for x, y in calibration_data:
                with torch.no_grad():
                    logits = model(x.unsqueeze(0))
                
                # Apply Platt scaling
                scaled_logits = a * logits + b
                probs = torch.sigmoid(scaled_logits)
                
                # Calculate binary cross entropy
                loss = F.binary_cross_entropy(probs, y.float().unsqueeze(0))
                total_loss += loss
            
            total_loss.backward()
            return total_loss
        
        # Optimize parameters
        for _ in range(100):
            optimizer.step(loss_fn)
        
        return {
            'a': a.item(),
            'b': b.item(),
            'calibrated_model': lambda x: torch.sigmoid(a * model(x) + b)
        }
    
    def isotonic_regression(self, model, calibration_data):
        """Isotonic regression for uncertainty calibration"""
        
        from sklearn.isotonic import IsotonicRegression
        
        # Get predictions and true labels
        predictions = []
        true_labels = []
        
        for x, y in calibration_data:
            with torch.no_grad():
                pred = model(x.unsqueeze(0))
                predictions.append(pred.item())
                true_labels.append(y.item())
        
        # Fit isotonic regression
        isotonic = IsotonicRegression(out_of_bounds='clip')
        isotonic.fit(predictions, true_labels)
        
        return {
            'isotonic_model': isotonic,
            'calibrated_model': lambda x: torch.tensor(isotonic.predict(model(x).detach().numpy()))
        }
    
    def evaluate_calibration(self, model, test_data):
        """Evaluate calibration quality"""
        
        # Calculate reliability diagram
        confidences = []
        accuracies = []
        
        num_bins = 10
        bin_size = 1.0 / num_bins
        
        for i in range(num_bins):
            lower_conf = i * bin_size
            upper_conf = (i + 1) * bin_size
            
            # Find samples in this confidence bin
            bin_samples = []
            bin_labels = []
            
            for x, y in test_data:
                with torch.no_grad():
                    pred = model(x.unsqueeze(0))
                    confidence = torch.max(F.softmax(pred, dim=1)).item()
                
                if lower_conf <= confidence < upper_conf:
                    bin_samples.append(pred)
                    bin_labels.append(y)
            
            if len(bin_samples) > 0:
                # Calculate accuracy for this bin
                bin_samples = torch.stack(bin_samples)
                bin_labels = torch.stack(bin_labels)
                
                _, predicted = torch.max(bin_samples, 1)
                accuracy = (predicted == bin_labels).float().mean().item()
                
                confidences.append((lower_conf + upper_conf) / 2)
                accuracies.append(accuracy)
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(np.array(confidences) - np.array(accuracies)))
        
        return {
            'confidences': confidences,
            'accuracies': accuracies,
            'calibration_error': calibration_error
        }
```

## Practical Implementation

### Complete Uncertainty Quantification Pipeline

```python
# Complete Uncertainty Quantification Pipeline
class CompleteUncertaintyPipeline:
    def __init__(self):
        self.bayesian_nn = BayesianNeuralNetwork
        self.deep_ensemble = DeepEnsemble
        self.mc_dropout = MCDropout
        self.conformal_prediction = ConformalPrediction
        self.calibration = UncertaintyCalibration
    
    def build_uncertainty_model(self, input_dim, hidden_dim, output_dim, method='ensemble'):
        """Build uncertainty quantification model"""
        
        if method == 'bayesian':
            model = self.bayesian_nn(input_dim, hidden_dim, output_dim)
        elif method == 'ensemble':
            model = self.deep_ensemble(self.bayesian_nn, num_models=5, 
                                     input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        elif method == 'mc_dropout':
            model = self.mc_dropout(input_dim, hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
        
        return model
    
    def train_with_uncertainty(self, model, train_loader, method='ensemble'):
        """Train model with uncertainty quantification"""
        
        if method == 'ensemble':
            model.train_ensemble(train_loader)
        else:
            # Standard training for other methods
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(100):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    if method == 'bayesian':
                        loss = model.elbo_loss(batch_x, batch_y)
                    else:
                        pred = model(batch_x)
                        loss = criterion(pred, batch_y)
                    
                    loss.backward()
                    optimizer.step()
        
        return model
    
    def predict_with_uncertainty(self, model, x, method='ensemble', calibration_data=None):
        """Make predictions with uncertainty estimates"""
        
        if method == 'ensemble':
            return model.predict_with_uncertainty(x)
        elif method == 'bayesian':
            return model.predict_with_uncertainty(x)
        elif method == 'mc_dropout':
            return model.predict_with_uncertainty(x)
        elif method == 'conformal':
            conformal_model = self.conformal_prediction(model, calibration_data)
            conformal_model.fit()
            return conformal_model.predict_with_intervals(x)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def calibrate_uncertainty(self, model, calibration_data, method='temperature_scaling'):
        """Calibrate uncertainty estimates"""
        
        calibrator = self.calibration()
        return calibrator.calibrate_uncertainty(model, calibration_data, method)
    
    def evaluate_uncertainty_quality(self, model, test_data, method='ensemble'):
        """Evaluate uncertainty quantification quality"""
        
        # Calculate various uncertainty metrics
        predictions = []
        uncertainties = []
        true_values = []
        
        for x, y in test_data:
            pred_result = self.predict_with_uncertainty(model, x, method)
            
            predictions.append(pred_result['mean'])
            uncertainties.append(pred_result['std'])
            true_values.append(y)
        
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        true_values = torch.stack(true_values)
        
        # Calculate metrics
        mse = F.mse_loss(predictions, true_values)
        mae = F.l1_loss(predictions, true_values)
        
        # Calibration metrics
        calibration_error = self._calculate_calibration_error(predictions, uncertainties, true_values)
        
        # Sharpness (lower is better)
        sharpness = torch.mean(uncertainties)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'calibration_error': calibration_error,
            'sharpness': sharpness.item(),
            'predictions': predictions,
            'uncertainties': uncertainties
        }
    
    def _calculate_calibration_error(self, predictions, uncertainties, true_values):
        """Calculate calibration error"""
        
        # Calculate empirical coverage
        z_scores = (predictions - true_values) / uncertainties
        empirical_coverage = torch.mean((z_scores.abs() <= 1.96).float())
        
        # Expected coverage is 0.95 for 95% confidence intervals
        expected_coverage = 0.95
        calibration_error = abs(empirical_coverage - expected_coverage)
        
        return calibration_error.item()
```

## Exercises and Projects

### Exercise 1: Bayesian Neural Network Implementation

Implement a complete Bayesian neural network:

1. **Variational Inference**: Implement ELBO loss and reparameterization
2. **MCMC Sampling**: Implement Metropolis-Hastings and HMC
3. **Uncertainty Estimation**: Compare different sampling methods
4. **Calibration**: Calibrate uncertainty estimates

**Requirements:**
- Implement variational inference with ELBO
- Compare with MCMC sampling
- Evaluate uncertainty quality
- Visualize uncertainty estimates

### Exercise 2: Ensemble Methods for Uncertainty

Build ensemble-based uncertainty quantification:

1. **Deep Ensembles**: Train multiple models with different initializations
2. **Bootstrap Ensembles**: Use bootstrap sampling for diversity
3. **Bagging and Boosting**: Implement different ensemble strategies
4. **Uncertainty Decomposition**: Separate epistemic and aleatoric uncertainty

**Implementation:**
```python
# Ensemble Uncertainty Quantification
class EnsembleUncertaintyQuantification:
    def __init__(self, base_model_class, num_models=10):
        self.base_model_class = base_model_class
        self.num_models = num_models
        self.models = []
    
    def train_ensemble(self, train_data, **training_kwargs):
        """Train ensemble of models"""
        
        for i in range(self.num_models):
            # Create model with different initialization
            model = self.base_model_class()
            
            # Train model
            self._train_model(model, train_data, **training_kwargs)
            
            self.models.append(model)
    
    def predict_with_uncertainty(self, x):
        """Make predictions with ensemble uncertainty"""
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate uncertainty metrics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred ** 2
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictions': predictions
        }
```

### Project: Reliable Uncertainty Quantification System

Build a complete uncertainty quantification system:

1. **Multiple Methods**: Implement Bayesian, ensemble, and conformal methods
2. **Uncertainty Calibration**: Calibrate uncertainty estimates
3. **Quality Assessment**: Evaluate uncertainty quality
4. **Visualization**: Create uncertainty visualization tools
5. **Production Deployment**: Deploy with uncertainty monitoring

**Features:**
- Multiple uncertainty quantification methods
- Uncertainty calibration and validation
- Real-time uncertainty monitoring
- Uncertainty-aware decision making
- Comprehensive evaluation metrics

### Project: Uncertainty-Aware ML Platform

Develop an uncertainty-aware ML platform:

1. **Model Training**: Train models with uncertainty quantification
2. **Inference Pipeline**: Make predictions with uncertainty estimates
3. **Decision Making**: Use uncertainty for better decisions
4. **Monitoring**: Monitor uncertainty in production
5. **User Interface**: Visualize uncertainty for users

**Deliverables:**
- Complete uncertainty-aware ML platform
- Multiple uncertainty quantification methods
- Uncertainty calibration and validation
- Real-time uncertainty monitoring
- User-friendly uncertainty visualization

## Summary

Uncertainty Quantification covers essential techniques for reliable ML:

- **Bayesian Methods**: Variational inference and MCMC sampling
- **Ensemble Methods**: Deep ensembles and bootstrap sampling
- **Monte Carlo Methods**: MC dropout and HMC
- **Conformal Prediction**: Reliable uncertainty bounds
- **Calibration**: Accurate uncertainty estimates

The practical implementation provides a foundation for building reliable, uncertainty-aware ML systems that can make better decisions in production environments.