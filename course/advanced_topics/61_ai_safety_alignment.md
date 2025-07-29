# AI Safety and Alignment: Building Trustworthy AI Systems

## 🎯 Learning Objectives
- Understand AI safety principles and alignment challenges
- Master robustness and adversarial defense techniques
- Implement interpretable and explainable AI systems
- Build aligned AI systems that follow human values
- Deploy safe AI systems in production environments

## 📚 Prerequisites
- Deep learning and machine learning fundamentals
- Understanding of AI ethics and bias
- Python programming with security libraries
- Knowledge of adversarial attacks and defenses

---

## 🚀 Module Overview

### 1. AI Safety Fundamentals

#### 1.1 Safety Principles and Frameworks
```python
class AISafetyFramework:
    def __init__(self):
        self.safety_principles = {
            'robustness': 'AI systems should be robust to distribution shifts',
            'interpretability': 'AI systems should be interpretable and explainable',
            'alignment': 'AI systems should be aligned with human values',
            'privacy': 'AI systems should protect user privacy',
            'fairness': 'AI systems should be fair and unbiased'
        }
        self.safety_metrics = {}
    
    def assess_safety(self, model, test_data):
        """Comprehensive safety assessment"""
        safety_scores = {}
        
        # Robustness assessment
        safety_scores['robustness'] = self.assess_robustness(model, test_data)
        
        # Interpretability assessment
        safety_scores['interpretability'] = self.assess_interpretability(model, test_data)
        
        # Alignment assessment
        safety_scores['alignment'] = self.assess_alignment(model, test_data)
        
        # Privacy assessment
        safety_scores['privacy'] = self.assess_privacy(model, test_data)
        
        # Fairness assessment
        safety_scores['fairness'] = self.assess_fairness(model, test_data)
        
        return safety_scores
    
    def assess_robustness(self, model, test_data):
        """Assess model robustness"""
        # Test against adversarial examples
        adversarial_score = self.test_adversarial_robustness(model, test_data)
        
        # Test against distribution shifts
        distribution_score = self.test_distribution_robustness(model, test_data)
        
        # Test against noise
        noise_score = self.test_noise_robustness(model, test_data)
        
        return (adversarial_score + distribution_score + noise_score) / 3
    
    def assess_interpretability(self, model, test_data):
        """Assess model interpretability"""
        # Test explanation quality
        explanation_score = self.test_explanation_quality(model, test_data)
        
        # Test feature importance consistency
        feature_score = self.test_feature_importance(model, test_data)
        
        # Test decision transparency
        transparency_score = self.test_decision_transparency(model, test_data)
        
        return (explanation_score + feature_score + transparency_score) / 3
```

#### 1.2 Safety Monitoring and Alerting
```python
class AISafetyMonitor:
    def __init__(self, model, safety_thresholds):
        self.model = model
        self.safety_thresholds = safety_thresholds
        self.safety_history = []
        self.alerts = []
    
    def monitor_prediction(self, input_data, prediction):
        """Monitor individual prediction for safety issues"""
        safety_issues = []
        
        # Check for adversarial inputs
        if self.detect_adversarial_input(input_data):
            safety_issues.append('adversarial_input')
        
        # Check for out-of-distribution inputs
        if self.detect_ood_input(input_data):
            safety_issues.append('out_of_distribution')
        
        # Check for biased predictions
        if self.detect_bias(prediction, input_data):
            safety_issues.append('biased_prediction')
        
        # Check for privacy violations
        if self.detect_privacy_violation(input_data):
            safety_issues.append('privacy_violation')
        
        # Log safety issues
        if safety_issues:
            self.log_safety_issue(input_data, prediction, safety_issues)
        
        return {
            'safety_issues': safety_issues,
            'confidence': self.model.get_confidence(input_data),
            'should_block': len(safety_issues) > 0
        }
    
    def detect_adversarial_input(self, input_data):
        """Detect potential adversarial inputs"""
        # Check for unusual patterns
        pattern_score = self.calculate_pattern_score(input_data)
        
        # Check for gradient-based attacks
        gradient_score = self.calculate_gradient_score(input_data)
        
        return pattern_score > self.safety_thresholds['adversarial'] or \
               gradient_score > self.safety_thresholds['gradient']
    
    def detect_ood_input(self, input_data):
        """Detect out-of-distribution inputs"""
        # Calculate distance from training distribution
        distance = self.calculate_distribution_distance(input_data)
        
        return distance > self.safety_thresholds['ood']
    
    def log_safety_issue(self, input_data, prediction, issues):
        """Log safety issues for analysis"""
        issue_log = {
            'timestamp': datetime.now(),
            'input_hash': hash(str(input_data)),
            'prediction': prediction,
            'issues': issues,
            'confidence': self.model.get_confidence(input_data)
        }
        
        self.safety_history.append(issue_log)
        
        # Trigger alerts for critical issues
        if 'adversarial_input' in issues:
            self.trigger_alert('CRITICAL: Adversarial input detected', issue_log)
```

### 2. Robustness and Adversarial Defense

#### 2.1 Adversarial Training
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialTrainer:
    def __init__(self, model, epsilon=0.1, alpha=0.01, steps=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def generate_adversarial_examples(self, inputs, targets):
        """Generate adversarial examples using PGD"""
        inputs_adv = inputs.clone().detach().requires_grad_(True)
        
        for step in range(self.steps):
            # Forward pass
            outputs = self.model(inputs_adv)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial inputs
            with torch.no_grad():
                inputs_adv = inputs_adv + self.alpha * inputs_adv.grad.sign()
                inputs_adv = torch.clamp(inputs_adv, inputs - self.epsilon, inputs + self.epsilon)
                inputs_adv = torch.clamp(inputs_adv, 0, 1)
            
            inputs_adv.grad.zero_()
        
        return inputs_adv
    
    def train_with_adversarial(self, train_loader, epochs=10):
        """Train model with adversarial examples"""
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Generate adversarial examples
                inputs_adv = self.generate_adversarial_examples(inputs, targets)
                
                # Combine clean and adversarial data
                combined_inputs = torch.cat([inputs, inputs_adv], dim=0)
                combined_targets = torch.cat([targets, targets], dim=0)
                
                # Forward pass
                outputs = self.model(combined_inputs)
                loss = F.cross_entropy(outputs, combined_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

#### 2.2 Certified Robustness
```python
class CertifiedRobustness:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
    
    def get_certified_bounds(self, inputs):
        """Get certified robustness bounds"""
        # Use interval bound propagation
        bounds = self.interval_bound_propagation(inputs)
        
        return bounds
    
    def interval_bound_propagation(self, inputs):
        """Compute interval bounds for robustness certification"""
        # Initialize bounds
        lower_bounds = inputs - self.epsilon
        upper_bounds = inputs + self.epsilon
        
        # Propagate bounds through network
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                lower_bounds, upper_bounds = self.propagate_linear_bounds(
                    layer, lower_bounds, upper_bounds
                )
            elif isinstance(layer, nn.ReLU):
                lower_bounds, upper_bounds = self.propagate_relu_bounds(
                    lower_bounds, upper_bounds
                )
        
        return lower_bounds, upper_bounds
    
    def is_certified_robust(self, inputs, targets):
        """Check if prediction is certified robust"""
        lower_bounds, upper_bounds = self.get_certified_bounds(inputs)
        
        # Check if target class has highest lower bound
        target_lower = lower_bounds[:, targets]
        other_upper = upper_bounds[:, [i for i in range(upper_bounds.shape[1]) if i != targets]]
        
        return torch.all(target_lower > torch.max(other_upper, dim=1)[0])
    
    def calculate_certified_accuracy(self, test_loader):
        """Calculate certified accuracy on test set"""
        certified_correct = 0
        total = 0
        
        for inputs, targets in test_loader:
            for i in range(inputs.shape[0]):
                if self.is_certified_robust(inputs[i:i+1], targets[i:i+1]):
                    certified_correct += 1
                total += 1
        
        return certified_correct / total
```

### 3. Interpretability and Explainability

#### 3.1 Model Interpretability Techniques
```python
class ModelInterpreter:
    def __init__(self, model):
        self.model = model
        self.interpretation_methods = {}
    
    def add_interpretation_method(self, name, method):
        """Add an interpretation method"""
        self.interpretation_methods[name] = method
    
    def explain_prediction(self, input_data, method='integrated_gradients'):
        """Explain model prediction"""
        if method not in self.interpretation_methods:
            raise ValueError(f"Method {method} not available")
        
        explanation = self.interpretation_methods[method](input_data)
        return explanation
    
    def integrated_gradients(self, input_data, baseline=None, steps=50):
        """Compute integrated gradients"""
        if baseline is None:
            baseline = torch.zeros_like(input_data)
        
        # Generate interpolated inputs
        interpolated_inputs = []
        for step in range(steps):
            alpha = step / steps
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = torch.stack(interpolated_inputs)
        
        # Compute gradients
        interpolated_inputs.requires_grad_(True)
        outputs = self.model(interpolated_inputs)
        
        # Compute gradients with respect to inputs
        gradients = torch.autograd.grad(
            outputs.sum(), interpolated_inputs,
            create_graph=True
        )[0]
        
        # Average gradients
        avg_gradients = gradients.mean(dim=0)
        
        # Compute integrated gradients
        integrated_gradients = (input_data - baseline) * avg_gradients
        
        return integrated_gradients
    
    def shap_values(self, input_data, background_data):
        """Compute SHAP values"""
        import shap
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(self.model, background_data)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(input_data)
        
        return shap_values
    
    def lime_explanation(self, input_data, num_samples=1000):
        """Generate LIME explanation"""
        import lime
        import lime.lime_tabular
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names
        )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            input_data.cpu().numpy(),
            self.model.predict_proba,
            num_samples=num_samples
        )
        
        return explanation
```

#### 3.2 Decision Trees and Rule Extraction
```python
class RuleExtractor:
    def __init__(self, model):
        self.model = model
        self.rules = []
    
    def extract_rules(self, training_data, max_depth=5):
        """Extract decision rules from neural network"""
        # Train surrogate decision tree
        surrogate_tree = self.train_surrogate_tree(training_data, max_depth)
        
        # Extract rules from tree
        rules = self.extract_rules_from_tree(surrogate_tree)
        
        return rules
    
    def train_surrogate_tree(self, training_data, max_depth):
        """Train surrogate decision tree"""
        from sklearn.tree import DecisionTreeClassifier
        
        # Get model predictions
        predictions = self.model.predict(training_data)
        
        # Train decision tree
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(training_data, predictions)
        
        return tree
    
    def extract_rules_from_tree(self, tree):
        """Extract rules from decision tree"""
        rules = []
        
        def extract_rules_recursive(node, rule):
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                # Leaf node
                prediction = tree.tree_.value[node].argmax()
                rules.append((rule, prediction))
            else:
                # Internal node
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                
                # Left child (feature <= threshold)
                left_rule = rule + [(feature, '<=', threshold)]
                extract_rules_recursive(tree.tree_.children_left[node], left_rule)
                
                # Right child (feature > threshold)
                right_rule = rule + [(feature, '>', threshold)]
                extract_rules_recursive(tree.tree_.children_right[node], right_rule)
        
        extract_rules_recursive(0, [])
        return rules
    
    def apply_rules(self, input_data, rules):
        """Apply extracted rules to new data"""
        predictions = []
        
        for data_point in input_data:
            for rule, prediction in rules:
                if self.satisfies_rule(data_point, rule):
                    predictions.append(prediction)
                    break
            else:
                # Default prediction if no rule matches
                predictions.append(0)
        
        return predictions
    
    def satisfies_rule(self, data_point, rule):
        """Check if data point satisfies a rule"""
        for feature, operator, threshold in rule:
            if operator == '<=':
                if data_point[feature] > threshold:
                    return False
            elif operator == '>':
                if data_point[feature] <= threshold:
                    return False
        
        return True
```

### 4. AI Alignment and Value Learning

#### 4.1 Reward Modeling and Alignment
```python
class RewardModel:
    def __init__(self, state_dim, action_dim):
        self.reward_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def predict_reward(self, state, action):
        """Predict reward for state-action pair"""
        inputs = torch.cat([state, action], dim=-1)
        reward = self.reward_network(inputs)
        return reward
    
    def train_on_human_feedback(self, demonstrations, preferences):
        """Train reward model on human feedback"""
        optimizer = torch.optim.Adam(self.reward_network.parameters())
        
        for demo, pref in zip(demonstrations, preferences):
            # Compute rewards for demonstration
            demo_rewards = []
            for state, action in demo:
                reward = self.predict_reward(state, action)
                demo_rewards.append(reward)
            
            # Compute preference loss
            loss = self.compute_preference_loss(demo_rewards, pref)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def compute_preference_loss(self, rewards, preference):
        """Compute loss based on human preferences"""
        # Bradley-Terry model for preferences
        logits = torch.stack(rewards)
        preference_probs = torch.softmax(logits, dim=0)
        
        # Cross-entropy loss
        loss = F.cross_entropy(preference_probs, preference)
        return loss
```

#### 4.2 Constitutional AI and Safety Filters
```python
class ConstitutionalAI:
    def __init__(self, base_model, constitution):
        self.base_model = base_model
        self.constitution = constitution
        self.safety_filters = []
    
    def add_safety_filter(self, filter_func):
        """Add a safety filter"""
        self.safety_filters.append(filter_func)
    
    def generate_safe_response(self, prompt):
        """Generate response that follows constitutional principles"""
        # Generate initial response
        initial_response = self.base_model.generate(prompt)
        
        # Apply safety filters
        safe_response = self.apply_safety_filters(initial_response, prompt)
        
        return safe_response
    
    def apply_safety_filters(self, response, prompt):
        """Apply safety filters to response"""
        filtered_response = response
        
        for filter_func in self.safety_filters:
            filtered_response = filter_func(filtered_response, prompt)
            
            # If response is blocked, generate alternative
            if filtered_response is None:
                filtered_response = self.generate_alternative_response(prompt)
        
        return filtered_response
    
    def check_constitutional_compliance(self, response, prompt):
        """Check if response complies with constitution"""
        compliance_scores = {}
        
        for principle, check_func in self.constitution.items():
            score = check_func(response, prompt)
            compliance_scores[principle] = score
        
        return compliance_scores
    
    def generate_alternative_response(self, prompt):
        """Generate alternative response when safety filters block"""
        # Use constitutional principles to guide generation
        constitutional_prompt = self.add_constitutional_guidance(prompt)
        
        return self.base_model.generate(constitutional_prompt)
    
    def add_constitutional_guidance(self, prompt):
        """Add constitutional guidance to prompt"""
        guidance = "Please respond in a way that is helpful, harmless, and honest."
        return f"{prompt}\n\n{guidance}"
```

### 5. Privacy-Preserving AI

#### 5.1 Differential Privacy
```python
class DifferentialPrivacy:
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise_to_gradients(self, gradients, sensitivity):
        """Add noise to gradients for differential privacy"""
        # Calculate noise scale
        noise_scale = sensitivity / self.epsilon
        
        # Add Gaussian noise
        noise = torch.randn_like(gradients) * noise_scale
        
        return gradients + noise
    
    def train_with_dp(self, model, train_loader, epochs=10):
        """Train model with differential privacy"""
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Add noise to gradients
                for param in model.parameters():
                    if param.grad is not None:
                        sensitivity = self.calculate_sensitivity(param.grad)
                        param.grad = self.add_noise_to_gradients(param.grad, sensitivity)
                
                optimizer.step()
    
    def calculate_sensitivity(self, gradients):
        """Calculate sensitivity of gradients"""
        # L2 sensitivity
        sensitivity = torch.norm(gradients, p=2)
        return sensitivity
```

#### 5.2 Federated Learning with Privacy
```python
class FederatedLearning:
    def __init__(self, global_model, clients):
        self.global_model = global_model
        self.clients = clients
        self.dp_mechanism = DifferentialPrivacy()
    
    def federated_training_round(self):
        """Perform one round of federated training"""
        client_models = []
        
        # Train on each client
        for client in self.clients:
            client_model = self.train_client(client)
            client_models.append(client_model)
        
        # Aggregate models with privacy
        aggregated_model = self.aggregate_models(client_models)
        
        # Update global model
        self.update_global_model(aggregated_model)
    
    def train_client(self, client):
        """Train model on client data"""
        # Copy global model to client
        client_model = copy.deepcopy(self.global_model)
        
        # Train with differential privacy
        self.dp_mechanism.train_with_dp(client_model, client.data_loader)
        
        return client_model
    
    def aggregate_models(self, client_models):
        """Aggregate client models with privacy"""
        # Simple averaging with noise
        aggregated_params = {}
        
        for param_name in self.global_model.state_dict().keys():
            param_values = [model.state_dict()[param_name] for model in client_models]
            avg_param = torch.stack(param_values).mean(dim=0)
            
            # Add noise for privacy
            noise = torch.randn_like(avg_param) * 0.01
            aggregated_params[param_name] = avg_param + noise
        
        return aggregated_params
    
    def update_global_model(self, aggregated_params):
        """Update global model with aggregated parameters"""
        self.global_model.load_state_dict(aggregated_params)
```

### 6. AI Safety Production Systems

#### 6.1 Safety Monitoring Pipeline
```python
class AISafetyPipeline:
    def __init__(self, model, safety_config):
        self.model = model
        self.safety_config = safety_config
        self.monitor = AISafetyMonitor(model, safety_config['thresholds'])
        self.interpreter = ModelInterpreter(model)
    
    def safe_predict(self, input_data):
        """Make safe prediction with monitoring"""
        # Pre-flight safety checks
        pre_checks = self.monitor.monitor_prediction(input_data, None)
        
        if pre_checks['should_block']:
            return {
                'prediction': None,
                'blocked': True,
                'reason': pre_checks['safety_issues']
            }
        
        # Make prediction
        prediction = self.model(input_data)
        
        # Post-flight safety checks
        post_checks = self.monitor.monitor_prediction(input_data, prediction)
        
        # Generate explanation
        explanation = self.interpreter.explain_prediction(input_data)
        
        return {
            'prediction': prediction,
            'confidence': self.model.get_confidence(input_data),
            'explanation': explanation,
            'safety_checks': post_checks
        }
    
    def continuous_monitoring(self, data_stream):
        """Continuously monitor model safety"""
        safety_metrics = {
            'adversarial_detected': 0,
            'ood_detected': 0,
            'bias_detected': 0,
            'privacy_violations': 0
        }
        
        for batch in data_stream:
            for input_data in batch:
                result = self.safe_predict(input_data)
                
                if result['blocked']:
                    for issue in result['reason']:
                        if issue in safety_metrics:
                            safety_metrics[issue] += 1
        
        return safety_metrics
```

#### 6.2 Safety Testing Framework
```python
class AISafetyTester:
    def __init__(self, model):
        self.model = model
        self.test_suites = {}
    
    def add_test_suite(self, name, test_suite):
        """Add a test suite"""
        self.test_suites[name] = test_suite
    
    def run_safety_tests(self):
        """Run comprehensive safety tests"""
        test_results = {}
        
        for suite_name, test_suite in self.test_suites.items():
            results = test_suite.run_tests(self.model)
            test_results[suite_name] = results
        
        return test_results
    
    def adversarial_test_suite(self):
        """Create adversarial test suite"""
        class AdversarialTestSuite:
            def __init__(self):
                self.attack_methods = [
                    'fgsm', 'pgd', 'carlini_wagner'
                ]
            
            def run_tests(self, model):
                results = {}
                
                for attack in self.attack_methods:
                    success_rate = self.test_attack(model, attack)
                    results[attack] = success_rate
                
                return results
            
            def test_attack(self, model, attack_name):
                # Simplified attack testing
                return 0.1  # 10% success rate
        
        return AdversarialTestSuite()
    
    def bias_test_suite(self):
        """Create bias test suite"""
        class BiasTestSuite:
            def __init__(self):
                self.protected_attributes = ['gender', 'race', 'age']
            
            def run_tests(self, model):
                results = {}
                
                for attr in self.protected_attributes:
                    bias_score = self.test_bias(model, attr)
                    results[attr] = bias_score
                
                return results
            
            def test_bias(self, model, attribute):
                # Simplified bias testing
                return 0.05  # 5% bias score
        
        return BiasTestSuite()
```

---

## 🎯 Key Takeaways

1. **Safety First**: AI safety should be built into systems from the ground up
2. **Robustness**: AI systems must be robust to adversarial attacks and distribution shifts
3. **Interpretability**: AI systems should be interpretable and explainable
4. **Alignment**: AI systems should be aligned with human values and preferences
5. **Privacy**: AI systems should protect user privacy and data

## 🚀 Next Steps

1. **Advanced Safety Techniques**: Explore more sophisticated safety methods
2. **AI Alignment Research**: Study advanced alignment techniques
3. **Privacy-Preserving ML**: Deep dive into federated learning and differential privacy
4. **Safety Testing**: Develop comprehensive safety testing frameworks
5. **Production Safety**: Deploy safe AI systems at scale

## 📚 Additional Resources

- **AI Safety Papers**: Latest research in AI safety and alignment
- **Robustness Libraries**: Foolbox, Advertorch, CleverHans
- **Interpretability Tools**: SHAP, LIME, Captum
- **Privacy Libraries**: PySyft, TensorFlow Privacy
- **Safety Frameworks**: AI Safety Gridworlds, Safety Gym

---

*This module provides a comprehensive foundation in AI safety and alignment, enabling you to build trustworthy AI systems!* 🚀 