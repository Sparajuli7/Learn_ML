# AI Security Fundamentals: Protecting ML Systems

## Table of Contents
1. [Introduction](#introduction)
2. [AI Security Threats](#ai-security-threats)
3. [Adversarial Attacks](#adversarial-attacks)
4. [Model Inversion Attacks](#model-inversion-attacks)
5. [Data Poisoning](#data-poisoning)
6. [Model Extraction](#model-extraction)
7. [Privacy Attacks](#privacy-attacks)
8. [Defense Mechanisms](#defense-mechanisms)
9. [Secure ML Development](#secure-ml-development)
10. [Practical Implementation](#practical-implementation)
11. [Exercises and Projects](#exercises-and-projects)

## Introduction

AI Security focuses on protecting machine learning systems from various attacks and vulnerabilities. This chapter covers fundamental security concepts, attack vectors, and defense strategies for ML systems.

### Key Learning Objectives
- Understand AI security threats and attack vectors
- Implement defense mechanisms against adversarial attacks
- Protect ML models from extraction and inversion attacks
- Develop secure ML systems and pipelines
- Apply privacy-preserving techniques

## AI Security Threats

### Threat Landscape

```python
# AI Security Threat Classification
class AISecurityThreats:
    def __init__(self):
        self.threat_categories = {
            'adversarial_attacks': {
                'evasion_attacks': 'Manipulate input to cause misclassification',
                'poisoning_attacks': 'Corrupt training data to compromise model',
                'extraction_attacks': 'Steal model architecture and parameters',
                'inversion_attacks': 'Reconstruct training data from model'
            },
            'privacy_attacks': {
                'membership_inference': 'Determine if data point was in training set',
                'model_inversion': 'Reconstruct sensitive training data',
                'attribute_inference': 'Extract sensitive attributes from model'
            },
            'system_attacks': {
                'model_stealing': 'Copy model functionality',
                'backdoor_attacks': 'Insert malicious functionality',
                'data_exfiltration': 'Steal sensitive training data'
            }
        }
    
    def assess_threat_level(self, model_config, deployment_environment):
        """Assess security threat level for ML system"""
        
        threat_assessment = {
            'adversarial_risk': self._assess_adversarial_risk(model_config),
            'privacy_risk': self._assess_privacy_risk(model_config),
            'system_risk': self._assess_system_risk(deployment_environment),
            'overall_risk': 'high'  # Will be calculated
        }
        
        # Calculate overall risk
        risk_scores = {
            'adversarial_risk': self._score_risk(threat_assessment['adversarial_risk']),
            'privacy_risk': self._score_risk(threat_assessment['privacy_risk']),
            'system_risk': self._score_risk(threat_assessment['system_risk'])
        }
        
        overall_score = sum(risk_scores.values()) / len(risk_scores)
        threat_assessment['overall_risk'] = self._categorize_risk(overall_score)
        
        return threat_assessment
```

### Security Risk Assessment

```python
# ML Security Risk Assessment
class MLSecurityRiskAssessment:
    def __init__(self):
        self.risk_factors = {
            'model_complexity': 'Complex models are harder to defend',
            'data_sensitivity': 'Sensitive data increases privacy risks',
            'deployment_environment': 'Public APIs increase attack surface',
            'access_controls': 'Weak access controls increase risk',
            'monitoring_capabilities': 'Lack of monitoring increases risk'
        }
    
    def assess_model_security(self, model_config, data_config, deployment_config):
        """Comprehensive security assessment of ML system"""
        
        assessment = {
            'model_security': self._assess_model_security(model_config),
            'data_security': self._assess_data_security(data_config),
            'deployment_security': self._assess_deployment_security(deployment_config),
            'recommendations': []
        }
        
        # Generate security recommendations
        assessment['recommendations'] = self._generate_security_recommendations(assessment)
        
        return assessment
    
    def _assess_model_security(self, model_config):
        """Assess model-specific security risks"""
        
        risks = []
        
        # Check model complexity
        if model_config.get('complexity', 'low') in ['high', 'very_high']:
            risks.append('High model complexity increases adversarial risk')
        
        # Check model interpretability
        if not model_config.get('interpretable', False):
            risks.append('Black-box models are harder to defend')
        
        # Check model versioning
        if not model_config.get('version_control', False):
            risks.append('Lack of version control increases security risk')
        
        return {
            'risk_level': 'high' if len(risks) > 2 else 'medium' if len(risks) > 0 else 'low',
            'risks': risks
        }
```

## Adversarial Attacks

### Evasion Attacks

```python
# Adversarial Evasion Attacks
import numpy as np
from typing import Tuple, List

class AdversarialEvasionAttack:
    def __init__(self, model, epsilon=0.1, max_iterations=100):
        self.model = model
        self.epsilon = epsilon
        self.max_iterations = max_iterations
    
    def fast_gradient_sign_method(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast Gradient Sign Method (FGSM) attack"""
        
        # Compute gradient of loss with respect to input
        x_tensor = torch.tensor(x, requires_grad=True)
        y_tensor = torch.tensor(y)
        
        # Forward pass
        output = self.model(x_tensor)
        loss = torch.nn.functional.cross_entropy(output, y_tensor)
        
        # Backward pass
        loss.backward()
        
        # Create adversarial example
        perturbation = self.epsilon * torch.sign(x_tensor.grad)
        adversarial_x = x_tensor + perturbation
        
        return adversarial_x.detach().numpy()
    
    def projected_gradient_descent(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Projected Gradient Descent (PGD) attack"""
        
        x_adv = x.copy()
        
        for iteration in range(self.max_iterations):
            # Compute gradient
            x_tensor = torch.tensor(x_adv, requires_grad=True)
            y_tensor = torch.tensor(y)
            
            output = self.model(x_tensor)
            loss = torch.nn.functional.cross_entropy(output, y_tensor)
            loss.backward()
            
            # Update adversarial example
            perturbation = self.epsilon / self.max_iterations * torch.sign(x_tensor.grad)
            x_adv = x_adv + perturbation.detach().numpy()
            
            # Project to epsilon ball
            delta = x_adv - x
            delta = np.clip(delta, -self.epsilon, self.epsilon)
            x_adv = x + delta
        
        return x_adv
    
    def carlini_wagner_attack(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Carlini & Wagner L2 attack"""
        
        # Initialize adversarial example
        x_adv = x.copy()
        
        # Binary search for optimal c
        c_low, c_high = 0, 1e10
        
        for binary_search_step in range(10):
            c = (c_low + c_high) / 2
            
            # Optimize adversarial example
            x_adv = self._optimize_cw_attack(x, y, c)
            
            # Check if attack succeeded
            if self._is_attack_successful(x_adv, y):
                c_high = c
            else:
                c_low = c
        
        return x_adv
    
    def _optimize_cw_attack(self, x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
        """Optimize Carlini & Wagner attack for given c"""
        
        x_adv = torch.tensor(x, requires_grad=True)
        
        optimizer = torch.optim.Adam([x_adv], lr=0.01)
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Compute loss
            output = self.model(x_adv)
            loss = self._cw_loss(output, y, x_adv, x, c)
            
            loss.backward()
            optimizer.step()
            
            # Project to valid range
            x_adv.data = torch.clamp(x_adv.data, 0, 1)
        
        return x_adv.detach().numpy()
    
    def _cw_loss(self, output, y, x_adv, x_orig, c):
        """Carlini & Wagner loss function"""
        
        # Distance loss
        distance_loss = torch.norm(x_adv - x_orig, p=2)
        
        # Classification loss
        target_logit = output[range(len(output)), y]
        other_logits = output[range(len(output)), :]
        other_logits[range(len(output)), y] = -float('inf')
        max_other_logit = torch.max(other_logits, dim=1)[0]
        
        classification_loss = torch.clamp(max_other_logit - target_logit, min=0)
        
        return distance_loss + c * classification_loss
```

### Defense Against Adversarial Attacks

```python
# Adversarial Defense Mechanisms
class AdversarialDefense:
    def __init__(self, model):
        self.model = model
        self.defense_methods = {
            'adversarial_training': self.adversarial_training,
            'defensive_distillation': self.defensive_distillation,
            'input_preprocessing': self.input_preprocessing,
            'ensemble_defense': self.ensemble_defense
        }
    
    def adversarial_training(self, training_data, training_labels, attack_method):
        """Adversarial training defense"""
        
        # Generate adversarial examples
        adversarial_data = []
        adversarial_labels = []
        
        for x, y in zip(training_data, training_labels):
            adv_x = attack_method(x, y)
            adversarial_data.append(adv_x)
            adversarial_labels.append(y)
        
        # Combine original and adversarial data
        combined_data = np.concatenate([training_data, adversarial_data])
        combined_labels = np.concatenate([training_labels, adversarial_labels])
        
        # Retrain model on combined dataset
        self.model.fit(combined_data, combined_labels)
        
        return self.model
    
    def defensive_distillation(self, training_data, training_labels, temperature=100):
        """Defensive distillation defense"""
        
        # Train teacher model
        teacher_model = self._train_teacher_model(training_data, training_labels)
        
        # Generate soft labels
        soft_labels = self._generate_soft_labels(teacher_model, training_data, temperature)
        
        # Train student model on soft labels
        student_model = self._train_student_model(training_data, soft_labels, temperature)
        
        return student_model
    
    def input_preprocessing(self, x, preprocessing_method='gaussian_noise'):
        """Input preprocessing defense"""
        
        if preprocessing_method == 'gaussian_noise':
            noise = np.random.normal(0, 0.1, x.shape)
            return np.clip(x + noise, 0, 1)
        
        elif preprocessing_method == 'bit_depth_reduction':
            # Reduce bit depth to remove adversarial perturbations
            return np.round(x * 255) / 255
        
        elif preprocessing_method == 'spatial_smoothing':
            # Apply spatial smoothing
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(x, sigma=0.5)
        
        else:
            return x
    
    def ensemble_defense(self, models, x):
        """Ensemble defense using multiple models"""
        
        predictions = []
        for model in models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Aggregate predictions (majority voting for classification)
        ensemble_prediction = self._aggregate_predictions(predictions)
        
        return ensemble_prediction
    
    def _aggregate_predictions(self, predictions):
        """Aggregate predictions from ensemble"""
        
        # For classification, use majority voting
        if len(predictions[0].shape) == 1:
            # Single class predictions
            return np.argmax(np.bincount(predictions))
        else:
            # Probability predictions
            return np.mean(predictions, axis=0)
```

## Model Inversion Attacks

### Model Inversion Implementation

```python
# Model Inversion Attack
class ModelInversionAttack:
    def __init__(self, target_model, target_class, optimization_method='gradient'):
        self.target_model = target_model
        self.target_class = target_class
        self.optimization_method = optimization_method
    
    def invert_model(self, initial_guess=None, max_iterations=1000):
        """Perform model inversion attack"""
        
        if initial_guess is None:
            # Initialize with random data
            initial_guess = np.random.rand(1, self.target_model.input_shape[1])
        
        if self.optimization_method == 'gradient':
            return self._gradient_based_inversion(initial_guess, max_iterations)
        elif self.optimization_method == 'genetic':
            return self._genetic_algorithm_inversion(initial_guess, max_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _gradient_based_inversion(self, initial_guess, max_iterations):
        """Gradient-based model inversion"""
        
        x = torch.tensor(initial_guess, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.01)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.target_model(x)
            
            # Loss function for inversion
            loss = self._inversion_loss(output, x)
            
            loss.backward()
            optimizer.step()
            
            # Project to valid range
            x.data = torch.clamp(x.data, 0, 1)
        
        return x.detach().numpy()
    
    def _inversion_loss(self, output, x):
        """Loss function for model inversion"""
        
        # Target class confidence
        target_confidence = output[0, self.target_class]
        
        # Regularization terms
        l2_regularization = torch.norm(x, p=2)
        tv_regularization = self._total_variation(x)
        
        # Total loss
        loss = -target_confidence + 0.1 * l2_regularization + 0.01 * tv_regularization
        
        return loss
    
    def _total_variation(self, x):
        """Total variation regularization"""
        
        # For 2D images
        if len(x.shape) == 4:  # (batch, channels, height, width)
            tv_h = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
            tv_v = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
            return tv_h + tv_v
        else:
            return torch.sum(torch.abs(x[:, 1:] - x[:, :-1]))
    
    def _genetic_algorithm_inversion(self, initial_guess, max_iterations):
        """Genetic algorithm for model inversion"""
        
        population_size = 50
        population = [initial_guess.copy() for _ in range(population_size)]
        
        for generation in range(max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                output = self.target_model(torch.tensor(individual))
                fitness = output[0, self.target_class].item()
                fitness_scores.append(fitness)
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(population_size):
                parent1, parent2 = self._select_parents(selected)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]
```

## Data Poisoning

### Data Poisoning Attacks

```python
# Data Poisoning Attack Implementation
class DataPoisoningAttack:
    def __init__(self, target_model, poisoning_ratio=0.1):
        self.target_model = target_model
        self.poisoning_ratio = poisoning_ratio
    
    def label_flipping_attack(self, training_data, training_labels, target_class):
        """Label flipping poisoning attack"""
        
        # Select samples to poison
        n_samples = len(training_data)
        n_poison = int(n_samples * self.poisoning_ratio)
        
        # Randomly select samples to flip
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Create poisoned dataset
        poisoned_data = training_data.copy()
        poisoned_labels = training_labels.copy()
        
        for idx in poison_indices:
            # Flip label to target class
            poisoned_labels[idx] = target_class
        
        return poisoned_data, poisoned_labels
    
    def backdoor_attack(self, training_data, training_labels, trigger_pattern, target_class):
        """Backdoor poisoning attack"""
        
        # Select samples to poison
        n_samples = len(training_data)
        n_poison = int(n_samples * self.poisoning_ratio)
        
        # Randomly select samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Create poisoned dataset
        poisoned_data = training_data.copy()
        poisoned_labels = training_labels.copy()
        
        for idx in poison_indices:
            # Add trigger pattern to data
            poisoned_data[idx] = self._add_trigger(poisoned_data[idx], trigger_pattern)
            # Set target label
            poisoned_labels[idx] = target_class
        
        return poisoned_data, poisoned_labels
    
    def _add_trigger(self, data, trigger_pattern):
        """Add trigger pattern to data"""
        
        if len(data.shape) == 3:  # Image data
            # Add trigger pattern to image
            trigger_mask = trigger_pattern > 0
            data[trigger_mask] = trigger_pattern[trigger_mask]
        else:
            # Add trigger to feature vector
            data = data + trigger_pattern
        
        return data
    
    def clean_label_attack(self, training_data, training_labels, target_class, epsilon=0.1):
        """Clean label poisoning attack"""
        
        # Select samples to poison
        n_samples = len(training_data)
        n_poison = int(n_samples * self.poisoning_ratio)
        
        # Find samples close to decision boundary
        boundary_samples = self._find_boundary_samples(training_data, training_labels)
        poison_indices = boundary_samples[:n_poison]
        
        # Create poisoned dataset
        poisoned_data = training_data.copy()
        poisoned_labels = training_labels.copy()
        
        for idx in poison_indices:
            # Add small perturbation to move sample across boundary
            perturbation = np.random.normal(0, epsilon, poisoned_data[idx].shape)
            poisoned_data[idx] = np.clip(poisoned_data[idx] + perturbation, 0, 1)
            # Change label to target class
            poisoned_labels[idx] = target_class
        
        return poisoned_data, poisoned_labels
    
    def _find_boundary_samples(self, data, labels):
        """Find samples close to decision boundary"""
        
        # Train a simple model to find boundary
        from sklearn.svm import SVC
        boundary_model = SVC(probability=True)
        boundary_model.fit(data, labels)
        
        # Get prediction probabilities
        probabilities = boundary_model.predict_proba(data)
        
        # Find samples with low confidence (close to boundary)
        confidence = np.max(probabilities, axis=1)
        boundary_indices = np.argsort(confidence)[:len(data)//4]  # Bottom 25%
        
        return boundary_indices
```

### Defense Against Data Poisoning

```python
# Data Poisoning Defense
class DataPoisoningDefense:
    def __init__(self):
        self.defense_methods = {
            'outlier_detection': self.outlier_detection,
            'robust_aggregation': self.robust_aggregation,
            'data_validation': self.data_validation,
            'ensemble_defense': self.ensemble_defense
        }
    
    def outlier_detection(self, training_data, training_labels):
        """Detect and remove poisoned samples using outlier detection"""
        
        # Use isolation forest for outlier detection
        from sklearn.ensemble import IsolationForest
        
        # Train outlier detector
        outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = outlier_detector.fit_predict(training_data)
        
        # Remove outliers (poisoned samples)
        clean_indices = outlier_labels == 1
        clean_data = training_data[clean_indices]
        clean_labels = training_labels[clean_indices]
        
        return clean_data, clean_labels
    
    def robust_aggregation(self, models, aggregation_method='median'):
        """Robust aggregation of multiple models"""
        
        if aggregation_method == 'median':
            return self._median_aggregation(models)
        elif aggregation_method == 'trimmed_mean':
            return self._trimmed_mean_aggregation(models)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _median_aggregation(self, models):
        """Median aggregation of model predictions"""
        
        def median_predict(x):
            predictions = [model.predict(x) for model in models]
            return np.median(predictions, axis=0)
        
        return median_predict
    
    def _trimmed_mean_aggregation(self, models, trim_ratio=0.1):
        """Trimmed mean aggregation of model predictions"""
        
        def trimmed_mean_predict(x):
            predictions = [model.predict(x) for model in models]
            predictions = np.array(predictions)
            
            # Sort predictions along axis 0
            sorted_predictions = np.sort(predictions, axis=0)
            
            # Trim top and bottom predictions
            trim_count = int(len(models) * trim_ratio)
            trimmed_predictions = sorted_predictions[trim_count:-trim_count]
            
            return np.mean(trimmed_predictions, axis=0)
        
        return trimmed_mean_predict
    
    def data_validation(self, training_data, training_labels, validation_rules):
        """Validate training data against predefined rules"""
        
        valid_indices = []
        
        for i, (data, label) in enumerate(zip(training_data, training_labels)):
            is_valid = True
            
            for rule in validation_rules:
                if not rule(data, label):
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
        
        valid_data = training_data[valid_indices]
        valid_labels = training_labels[valid_indices]
        
        return valid_data, valid_labels
```

## Model Extraction

### Model Extraction Attacks

```python
# Model Extraction Attack
class ModelExtractionAttack:
    def __init__(self, target_model, query_budget=10000):
        self.target_model = target_model
        self.query_budget = query_budget
        self.extracted_model = None
    
    def extract_model(self, extraction_method='active_learning'):
        """Extract target model using various methods"""
        
        if extraction_method == 'active_learning':
            return self._active_learning_extraction()
        elif extraction_method == 'synthetic_data':
            return self._synthetic_data_extraction()
        elif extraction_method == 'adversarial_examples':
            return self._adversarial_examples_extraction()
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
    
    def _active_learning_extraction(self):
        """Extract model using active learning approach"""
        
        # Initialize synthetic dataset
        synthetic_data = self._generate_initial_data()
        synthetic_labels = []
        
        # Query target model
        for data_point in synthetic_data:
            label = self.target_model.predict(data_point.reshape(1, -1))[0]
            synthetic_labels.append(label)
        
        # Train surrogate model
        surrogate_model = self._train_surrogate_model(synthetic_data, synthetic_labels)
        
        # Active learning loop
        for iteration in range(self.query_budget // 100):
            # Generate new queries using uncertainty sampling
            new_queries = self._generate_uncertainty_queries(surrogate_model, synthetic_data)
            
            # Query target model
            new_labels = []
            for query in new_queries:
                label = self.target_model.predict(query.reshape(1, -1))[0]
                new_labels.append(label)
            
            # Add to training set
            synthetic_data = np.vstack([synthetic_data, new_queries])
            synthetic_labels.extend(new_labels)
            
            # Retrain surrogate model
            surrogate_model = self._train_surrogate_model(synthetic_data, synthetic_labels)
        
        self.extracted_model = surrogate_model
        return surrogate_model
    
    def _synthetic_data_extraction(self):
        """Extract model using synthetic data generation"""
        
        # Generate synthetic dataset
        synthetic_data = self._generate_synthetic_data()
        synthetic_labels = []
        
        # Query target model
        for data_point in synthetic_data:
            label = self.target_model.predict(data_point.reshape(1, -1))[0]
            synthetic_labels.append(label)
        
        # Train surrogate model
        surrogate_model = self._train_surrogate_model(synthetic_data, synthetic_labels)
        
        self.extracted_model = surrogate_model
        return surrogate_model
    
    def _adversarial_examples_extraction(self):
        """Extract model using adversarial examples"""
        
        # Generate adversarial examples
        adversarial_data = self._generate_adversarial_examples()
        adversarial_labels = []
        
        # Query target model
        for data_point in adversarial_data:
            label = self.target_model.predict(data_point.reshape(1, -1))[0]
            adversarial_labels.append(label)
        
        # Train surrogate model
        surrogate_model = self._train_surrogate_model(adversarial_data, adversarial_labels)
        
        self.extracted_model = surrogate_model
        return surrogate_model
    
    def _generate_initial_data(self):
        """Generate initial synthetic data"""
        
        # Generate random data
        n_samples = 1000
        n_features = self.target_model.n_features_in_
        
        synthetic_data = np.random.rand(n_samples, n_features)
        
        return synthetic_data
    
    def _generate_uncertainty_queries(self, surrogate_model, existing_data):
        """Generate queries using uncertainty sampling"""
        
        # Generate candidate queries
        n_candidates = 1000
        n_features = existing_data.shape[1]
        
        candidates = np.random.rand(n_candidates, n_features)
        
        # Calculate uncertainty for each candidate
        uncertainties = []
        for candidate in candidates:
            # Get prediction probabilities
            probs = surrogate_model.predict_proba(candidate.reshape(1, -1))[0]
            # Calculate entropy (uncertainty)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            uncertainties.append(entropy)
        
        # Select most uncertain candidates
        n_queries = 100
        uncertain_indices = np.argsort(uncertainties)[-n_queries:]
        
        return candidates[uncertain_indices]
    
    def _train_surrogate_model(self, data, labels):
        """Train surrogate model on synthetic data"""
        
        from sklearn.ensemble import RandomForestClassifier
        
        surrogate_model = RandomForestClassifier(n_estimators=100, random_state=42)
        surrogate_model.fit(data, labels)
        
        return surrogate_model
```

## Privacy Attacks

### Membership Inference Attack

```python
# Membership Inference Attack
class MembershipInferenceAttack:
    def __init__(self, target_model, shadow_models=None):
        self.target_model = target_model
        self.shadow_models = shadow_models or []
        self.attack_model = None
    
    def perform_membership_inference(self, data_samples, true_membership):
        """Perform membership inference attack"""
        
        # Extract features from target model
        attack_features = self._extract_attack_features(data_samples)
        
        # Train attack model
        self.attack_model = self._train_attack_model(attack_features, true_membership)
        
        # Perform attack
        predictions = self.attack_model.predict(attack_features)
        probabilities = self.attack_model.predict_proba(attack_features)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'accuracy': np.mean(predictions == true_membership)
        }
    
    def _extract_attack_features(self, data_samples):
        """Extract features for membership inference attack"""
        
        attack_features = []
        
        for sample in data_samples:
            # Get model prediction
            prediction = self.target_model.predict(sample.reshape(1, -1))[0]
            prediction_proba = self.target_model.predict_proba(sample.reshape(1, -1))[0]
            
            # Get prediction confidence
            confidence = np.max(prediction_proba)
            
            # Get prediction entropy
            entropy = -np.sum(prediction_proba * np.log(prediction_proba + 1e-10))
            
            # Get prediction margin
            sorted_proba = np.sort(prediction_proba)
            margin = sorted_proba[-1] - sorted_proba[-2]
            
            # Combine features
            features = [
                prediction,
                confidence,
                entropy,
                margin,
                *prediction_proba  # All class probabilities
            ]
            
            attack_features.append(features)
        
        return np.array(attack_features)
    
    def _train_attack_model(self, features, labels):
        """Train attack model for membership inference"""
        
        from sklearn.ensemble import RandomForestClassifier
        
        attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
        attack_model.fit(features, labels)
        
        return attack_model
    
    def evaluate_attack_performance(self, test_features, test_labels):
        """Evaluate membership inference attack performance"""
        
        predictions = self.attack_model.predict(test_features)
        probabilities = self.attack_model.predict_proba(test_features)
        
        # Calculate metrics
        accuracy = np.mean(predictions == test_labels)
        precision = np.mean(predictions[test_labels == 1] == 1)
        recall = np.mean(test_labels[predictions == 1] == 1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        }
```

## Defense Mechanisms

### Privacy-Preserving ML

```python
# Privacy-Preserving Machine Learning
class PrivacyPreservingML:
    def __init__(self):
        self.privacy_methods = {
            'differential_privacy': self.differential_privacy,
            'federated_learning': self.federated_learning,
            'secure_multiparty_computation': self.secure_mpc,
            'homomorphic_encryption': self.homomorphic_encryption
        }
    
    def differential_privacy(self, model, epsilon=1.0, delta=1e-5):
        """Apply differential privacy to model training"""
        
        # Add noise to gradients during training
        def dp_gradient_function(gradients, epsilon, delta):
            # Calculate noise scale
            noise_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            
            # Add noise to gradients
            noise = np.random.normal(0, noise_scale, gradients.shape)
            dp_gradients = gradients + noise
            
            return dp_gradients
        
        return dp_gradient_function
    
    def federated_learning(self, clients, aggregation_method='fedavg'):
        """Federated learning with privacy preservation"""
        
        if aggregation_method == 'fedavg':
            return self._federated_averaging(clients)
        elif aggregation_method == 'secure_aggregation':
            return self._secure_aggregation(clients)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _federated_averaging(self, clients):
        """Federated averaging aggregation"""
        
        # Collect model parameters from clients
        client_parameters = []
        client_weights = []
        
        for client in clients:
            params = client.get_model_parameters()
            weight = client.get_data_size()
            
            client_parameters.append(params)
            client_weights.append(weight)
        
        # Weighted average of parameters
        total_weight = sum(client_weights)
        aggregated_params = {}
        
        for param_name in client_parameters[0].keys():
            weighted_sum = sum(
                client_params[param_name] * weight 
                for client_params, weight in zip(client_parameters, client_weights)
            )
            aggregated_params[param_name] = weighted_sum / total_weight
        
        return aggregated_params
    
    def _secure_aggregation(self, clients):
        """Secure aggregation with cryptographic techniques"""
        
        # Implement secure aggregation using homomorphic encryption
        # This is a simplified version
        encrypted_parameters = []
        
        for client in clients:
            # Encrypt client parameters
            encrypted_params = self._encrypt_parameters(client.get_model_parameters())
            encrypted_parameters.append(encrypted_params)
        
        # Aggregate encrypted parameters
        aggregated_encrypted = self._aggregate_encrypted(encrypted_parameters)
        
        # Decrypt aggregated parameters
        aggregated_params = self._decrypt_parameters(aggregated_encrypted)
        
        return aggregated_params
    
    def secure_mpc(self, parties, computation_function):
        """Secure multiparty computation"""
        
        # Split data among parties
        data_shares = self._split_data(parties)
        
        # Perform secure computation
        result_shares = []
        for party in parties:
            share = party.compute_share(data_shares[party.id], computation_function)
            result_shares.append(share)
        
        # Reconstruct result
        final_result = self._reconstruct_result(result_shares)
        
        return final_result
    
    def homomorphic_encryption(self, model, encryption_scheme='paillier'):
        """Homomorphic encryption for privacy-preserving ML"""
        
        if encryption_scheme == 'paillier':
            return self._paillier_encryption(model)
        else:
            raise ValueError(f"Unknown encryption scheme: {encryption_scheme}")
    
    def _paillier_encryption(self, model):
        """Paillier homomorphic encryption implementation"""
        
        # Generate key pair
        public_key, private_key = self._generate_paillier_keys()
        
        # Encrypt model parameters
        encrypted_model = {}
        for param_name, param_value in model.get_parameters().items():
            encrypted_param = self._encrypt_with_paillier(param_value, public_key)
            encrypted_model[param_name] = encrypted_param
        
        return {
            'encrypted_model': encrypted_model,
            'public_key': public_key,
            'private_key': private_key
        }
```

## Secure ML Development

### Secure ML Pipeline

```python
# Secure ML Development Pipeline
class SecureMLPipeline:
    def __init__(self):
        self.security_checks = {
            'data_security': self.check_data_security,
            'model_security': self.check_model_security,
            'deployment_security': self.check_deployment_security,
            'privacy_compliance': self.check_privacy_compliance
        }
    
    def secure_ml_development(self, project_config):
        """Implement secure ML development process"""
        
        # Security assessment
        security_assessment = self._conduct_security_assessment(project_config)
        
        # Implement security measures
        security_measures = self._implement_security_measures(security_assessment)
        
        # Security testing
        security_testing = self._conduct_security_testing(security_measures)
        
        # Compliance verification
        compliance_verification = self._verify_compliance(security_measures)
        
        return {
            'assessment': security_assessment,
            'measures': security_measures,
            'testing': security_testing,
            'compliance': compliance_verification
        }
    
    def check_data_security(self, data_config):
        """Check data security measures"""
        
        security_checks = {
            'encryption_at_rest': data_config.get('encryption_at_rest', False),
            'encryption_in_transit': data_config.get('encryption_in_transit', False),
            'access_controls': data_config.get('access_controls', False),
            'data_anonymization': data_config.get('data_anonymization', False),
            'audit_logging': data_config.get('audit_logging', False)
        }
        
        return {
            'checks': security_checks,
            'score': sum(security_checks.values()) / len(security_checks),
            'recommendations': self._generate_data_security_recommendations(security_checks)
        }
    
    def check_model_security(self, model_config):
        """Check model security measures"""
        
        security_checks = {
            'model_encryption': model_config.get('model_encryption', False),
            'model_signing': model_config.get('model_signing', False),
            'adversarial_defense': model_config.get('adversarial_defense', False),
            'privacy_preservation': model_config.get('privacy_preservation', False),
            'model_watermarking': model_config.get('model_watermarking', False)
        }
        
        return {
            'checks': security_checks,
            'score': sum(security_checks.values()) / len(security_checks),
            'recommendations': self._generate_model_security_recommendations(security_checks)
        }
    
    def check_deployment_security(self, deployment_config):
        """Check deployment security measures"""
        
        security_checks = {
            'network_security': deployment_config.get('network_security', False),
            'container_security': deployment_config.get('container_security', False),
            'api_security': deployment_config.get('api_security', False),
            'monitoring_security': deployment_config.get('monitoring_security', False),
            'incident_response': deployment_config.get('incident_response', False)
        }
        
        return {
            'checks': security_checks,
            'score': sum(security_checks.values()) / len(security_checks),
            'recommendations': self._generate_deployment_security_recommendations(security_checks)
        }
    
    def check_privacy_compliance(self, privacy_config):
        """Check privacy compliance"""
        
        compliance_checks = {
            'gdpr_compliance': privacy_config.get('gdpr_compliance', False),
            'data_minimization': privacy_config.get('data_minimization', False),
            'consent_management': privacy_config.get('consent_management', False),
            'right_to_forget': privacy_config.get('right_to_forget', False),
            'data_portability': privacy_config.get('data_portability', False)
        }
        
        return {
            'checks': compliance_checks,
            'score': sum(compliance_checks.values()) / len(compliance_checks),
            'recommendations': self._generate_privacy_recommendations(compliance_checks)
        }
```

## Practical Implementation

### Complete AI Security System

```python
# Complete AI Security System
class CompleteAISecuritySystem:
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.defense_system = DefenseSystem()
        self.monitoring_system = SecurityMonitoring()
        self.incident_response = IncidentResponse()
    
    def setup_security_system(self, ml_system_config):
        """Setup comprehensive AI security system"""
        
        # Setup threat detection
        threat_detection = self.threat_detector.setup_detection(ml_system_config)
        
        # Setup defense mechanisms
        defense_mechanisms = self.defense_system.setup_defenses(ml_system_config)
        
        # Setup monitoring
        monitoring = self.monitoring_system.setup_monitoring(ml_system_config)
        
        # Setup incident response
        incident_response = self.incident_response.setup_response(ml_system_config)
        
        return {
            'threat_detection': threat_detection,
            'defense_mechanisms': defense_mechanisms,
            'monitoring': monitoring,
            'incident_response': incident_response
        }
    
    def monitor_and_respond(self, security_events):
        """Monitor security events and respond accordingly"""
        
        responses = []
        
        for event in security_events:
            # Analyze event
            threat_level = self.threat_detector.analyze_threat(event)
            
            # Determine response
            if threat_level == 'high':
                response = self.incident_response.emergency_response(event)
            elif threat_level == 'medium':
                response = self.defense_system.activate_defense(event)
            else:
                response = self.monitoring_system.log_event(event)
            
            responses.append({
                'event': event,
                'threat_level': threat_level,
                'response': response
            })
        
        return responses
```

## Exercises and Projects

### Exercise 1: Adversarial Attack Implementation

Implement adversarial attacks and defenses:

1. **FGSM Attack**: Implement Fast Gradient Sign Method
2. **PGD Attack**: Implement Projected Gradient Descent
3. **Adversarial Training**: Train model with adversarial examples
4. **Defense Evaluation**: Test effectiveness of defenses

**Requirements:**
- Implement at least 3 attack methods
- Implement 2 defense mechanisms
- Evaluate attack/defense effectiveness
- Document findings and recommendations

### Exercise 2: Privacy Attack Simulation

Simulate privacy attacks on ML models:

1. **Membership Inference**: Implement membership inference attack
2. **Model Inversion**: Implement model inversion attack
3. **Data Extraction**: Implement data extraction attack
4. **Privacy Defense**: Implement privacy-preserving techniques

**Implementation:**
```python
# Privacy Attack Simulation
class PrivacyAttackSimulation:
    def __init__(self, target_model):
        self.target_model = target_model
        self.attack_results = {}
    
    def run_privacy_attacks(self, test_data, test_labels):
        """Run comprehensive privacy attack simulation"""
        
        # Membership inference attack
        membership_attack = MembershipInferenceAttack(self.target_model)
        membership_results = membership_attack.perform_membership_inference(
            test_data, test_labels
        )
        
        # Model inversion attack
        inversion_attack = ModelInversionAttack(self.target_model, target_class=0)
        inversion_results = inversion_attack.invert_model()
        
        # Store results
        self.attack_results = {
            'membership_inference': membership_results,
            'model_inversion': inversion_results
        }
        
        return self.attack_results
```

### Project: Secure ML Platform

Build a secure ML platform with:

1. **Threat Detection**: Real-time threat detection system
2. **Defense Mechanisms**: Multiple defense strategies
3. **Privacy Protection**: Privacy-preserving ML techniques
4. **Security Monitoring**: Comprehensive security monitoring
5. **Incident Response**: Automated incident response system

**Features:**
- Real-time attack detection
- Automated defense activation
- Privacy compliance monitoring
- Security audit logging
- Incident response automation

### Project: AI Security Assessment Tool

Develop a comprehensive AI security assessment tool:

1. **Vulnerability Scanning**: Automated vulnerability detection
2. **Attack Simulation**: Simulate various attack scenarios
3. **Defense Evaluation**: Test defense effectiveness
4. **Compliance Checking**: Verify regulatory compliance
5. **Risk Assessment**: Comprehensive risk analysis

**Deliverables:**
- Automated security assessment tool
- Attack simulation framework
- Defense evaluation metrics
- Compliance verification system
- Risk assessment reports

## Summary

AI Security Fundamentals covers essential concepts for protecting ML systems:

- **Threat Landscape**: Understanding various attack vectors
- **Adversarial Attacks**: Evasion, poisoning, and extraction attacks
- **Privacy Attacks**: Membership inference and model inversion
- **Defense Mechanisms**: Adversarial training, input preprocessing, and privacy preservation
- **Secure Development**: Security-first ML development practices
- **Monitoring and Response**: Real-time security monitoring and incident response

The practical implementation provides a foundation for building secure ML systems that can withstand various attacks while protecting user privacy and data integrity.