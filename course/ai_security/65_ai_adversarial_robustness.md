# AI Adversarial Robustness

## 🛡️ Overview
Techniques for defending AI systems against adversarial attacks and ensuring model robustness. This comprehensive guide covers attack methods, defense strategies, and practical implementations for securing ML systems.

---

## ⚔️ Adversarial Attack Methods

### Understanding Adversarial Attacks
Adversarial attacks manipulate input data to cause ML models to make incorrect predictions while appearing normal to humans.

#### Fast Gradient Sign Method (FGSM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class AdversarialAttacks:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon
        self.device = next(model.parameters()).device
    
    def fgsm_attack(self, images, labels):
        """Fast Gradient Sign Method attack"""
        
        # Set requires_grad to True for images
        images.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Get gradients with respect to images
        image_gradients = images.grad.data
        
        # Create adversarial examples
        adversarial_images = images + self.epsilon * image_gradients.sign()
        
        # Clip to valid range [0, 1]
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        
        return adversarial_images.detach()
    
    def pgd_attack(self, images, labels, alpha=0.01, num_steps=40):
        """Projected Gradient Descent attack"""
        
        # Initialize adversarial examples
        adversarial_images = images.clone()
        
        for step in range(num_steps):
            # Set requires_grad
            adversarial_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(adversarial_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                adversarial_images = adversarial_images + alpha * adversarial_images.grad.sign()
                
                # Project to epsilon ball around original images
                delta = adversarial_images - images
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adversarial_images = torch.clamp(images + delta, 0, 1)
        
        return adversarial_images.detach()
    
    def carlini_wagner_attack(self, images, labels, c=1.0, num_steps=1000, lr=0.01):
        """Carlini & Wagner L2 attack"""
        
        # Initialize perturbation
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        for step in range(num_steps):
            # Create adversarial examples
            adversarial_images = images + delta
            adversarial_images = torch.clamp(adversarial_images, 0, 1)
            
            # Forward pass
            outputs = self.model(adversarial_images)
            
            # Calculate loss
            l2_loss = torch.norm(delta.view(delta.size(0), -1), dim=1)
            ce_loss = F.cross_entropy(outputs, labels)
            total_loss = l2_loss.mean() + c * ce_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Check if attack succeeded
            with torch.no_grad():
                pred_labels = outputs.argmax(dim=1)
                if (pred_labels != labels).all():
                    break
        
        return (images + delta).detach()
    
    def evaluate_attack_success(self, original_images, adversarial_images, labels):
        """Evaluate attack success rate"""
        
        with torch.no_grad():
            # Get original predictions
            original_outputs = self.model(original_images)
            original_preds = original_outputs.argmax(dim=1)
            
            # Get adversarial predictions
            adversarial_outputs = self.model(adversarial_images)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            # Calculate metrics
            original_accuracy = (original_preds == labels).float().mean()
            adversarial_accuracy = (adversarial_preds == labels).float().mean()
            attack_success_rate = 1 - adversarial_accuracy
            
            # Calculate perturbation magnitude
            perturbation = torch.norm((adversarial_images - original_images).view(adversarial_images.size(0), -1), dim=1)
            avg_perturbation = perturbation.mean()
            
            return {
                'original_accuracy': original_accuracy.item(),
                'adversarial_accuracy': adversarial_accuracy.item(),
                'attack_success_rate': attack_success_rate.item(),
                'avg_perturbation': avg_perturbation.item()
            }
```

---

## 🛡️ Defense Mechanisms and Strategies

### Adversarial Training and Robust Models
Implementing defenses against adversarial attacks through robust training and model hardening.

#### Adversarial Training Implementation

```python
class AdversarialTraining:
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_steps=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = next(model.parameters()).device
    
    def generate_adversarial_batch(self, images, labels):
        """Generate adversarial examples for training"""
        
        # Initialize adversarial examples
        adversarial_images = images.clone()
        
        for step in range(self.num_steps):
            adversarial_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(adversarial_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                adversarial_images = adversarial_images + self.alpha * adversarial_images.grad.sign()
                delta = adversarial_images - images
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adversarial_images = torch.clamp(images + delta, 0, 1)
        
        return adversarial_images.detach()
    
    def adversarial_training_step(self, images, labels, optimizer):
        """Single training step with adversarial examples"""
        
        # Generate adversarial examples
        adversarial_images = self.generate_adversarial_batch(images, labels)
        
        # Combine original and adversarial examples
        combined_images = torch.cat([images, adversarial_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        
        # Forward pass
        outputs = self.model(combined_images)
        loss = F.cross_entropy(outputs, combined_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_with_adversarial_training(self, train_loader, num_epochs=10, lr=0.001):
        """Train model with adversarial training"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                loss = self.adversarial_training_step(images, labels, optimizer)
                total_loss += loss
                num_batches += 1
            
            scheduler.step()
            
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        return self.model

class DefensiveDistillation:
    def __init__(self, teacher_model, temperature=10.0):
        self.teacher_model = teacher_model
        self.temperature = temperature
    
    def distill_model(self, student_model, train_loader, num_epochs=10, lr=0.001):
        """Train student model using defensive distillation"""
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(next(student_model.parameters()).device)
                
                # Get teacher predictions with temperature
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                    teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
                
                # Student forward pass
                student_outputs = student_model(images)
                student_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
                
                # KL divergence loss
                loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        return student_model
```

---

## 🔒 Robust Training Techniques

### Advanced Defense Strategies
Implementing sophisticated defense mechanisms against various attack types.

#### Input Preprocessing Defenses

```python
class InputPreprocessing:
    def __init__(self):
        self.defense_methods = {}
    
    def spatial_smoothing(self, images, kernel_size=3):
        """Apply spatial smoothing to reduce adversarial perturbations"""
        
        # Apply average pooling
        smoothed = F.avg_pool2d(images, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        return smoothed
    
    def bit_depth_reduction(self, images, bit_depth=4):
        """Reduce bit depth to remove subtle perturbations"""
        
        # Quantize to specified bit depth
        max_val = 2**bit_depth - 1
        quantized = torch.round(images * max_val) / max_val
        
        return quantized
    
    def jpeg_compression(self, images, quality=75):
        """Apply JPEG compression to remove adversarial perturbations"""
        
        # Convert to PIL images
        pil_images = []
        for i in range(images.size(0)):
            img = transforms.ToPILImage()(images[i])
            pil_images.append(img)
        
        # Apply JPEG compression
        compressed_images = []
        for img in pil_images:
            # Save and reload with JPEG compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            compressed_images.append(transforms.ToTensor()(compressed_img))
        
        return torch.stack(compressed_images)
    
    def random_crop_and_resize(self, images, crop_size=0.9):
        """Apply random crop and resize to break adversarial patterns"""
        
        batch_size = images.size(0)
        cropped_images = []
        
        for i in range(batch_size):
            # Random crop
            h, w = images[i].shape[1], images[i].shape[2]
            crop_h, crop_w = int(h * crop_size), int(w * crop_size)
            
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
            
            cropped = images[i][:, top:top+crop_h, left:left+crop_w]
            
            # Resize back to original size
            resized = F.interpolate(cropped.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
            cropped_images.append(resized.squeeze(0))
        
        return torch.stack(cropped_images)

class FeatureDenoising:
    def __init__(self, denoising_method='gaussian'):
        self.denoising_method = denoising_method
    
    def denoise_features(self, features, sigma=0.1):
        """Apply feature denoising to remove adversarial perturbations"""
        
        if self.denoising_method == 'gaussian':
            # Add Gaussian noise and denoise
            noisy_features = features + torch.randn_like(features) * sigma
            denoised_features = self.gaussian_denoising(noisy_features, sigma)
        
        elif self.denoising_method == 'median':
            # Apply median filtering
            denoised_features = self.median_filtering(features)
        
        else:
            denoised_features = features
        
        return denoised_features
    
    def gaussian_denoising(self, features, sigma):
        """Gaussian denoising using low-pass filter"""
        
        # Apply Gaussian blur
        kernel_size = 3
        sigma_kernel = sigma * kernel_size
        kernel = self.create_gaussian_kernel(kernel_size, sigma_kernel)
        
        # Apply convolution
        denoised = F.conv2d(features, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
        
        return denoised
    
    def median_filtering(self, features):
        """Apply median filtering to features"""
        
        # Reshape for median filtering
        batch_size, channels, height, width = features.shape
        features_flat = features.view(batch_size * channels, height, width)
        
        # Apply median filter
        denoised_flat = torch.zeros_like(features_flat)
        for i in range(features_flat.size(0)):
            denoised_flat[i] = self.median_filter_2d(features_flat[i])
        
        return denoised_flat.view(batch_size, channels, height, width)
    
    def create_gaussian_kernel(self, size, sigma):
        """Create Gaussian kernel for denoising"""
        
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g
```

---

## 🎯 Adversarial Example Generation

### Creating and Analyzing Adversarial Examples
Understanding how adversarial examples are generated and their properties.

#### Adversarial Example Analysis

```python
class AdversarialExampleAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def generate_adversarial_examples(self, images, labels, attack_method='fgsm'):
        """Generate adversarial examples using different methods"""
        
        if attack_method == 'fgsm':
            attacker = AdversarialAttacks(self.model, epsilon=0.3)
            adversarial_images = attacker.fgsm_attack(images, labels)
        
        elif attack_method == 'pgd':
            attacker = AdversarialAttacks(self.model, epsilon=0.3)
            adversarial_images = attacker.pgd_attack(images, labels)
        
        elif attack_method == 'carlini_wagner':
            attacker = AdversarialAttacks(self.model, epsilon=0.3)
            adversarial_images = attacker.carlini_wagner_attack(images, labels)
        
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        return adversarial_images
    
    def analyze_adversarial_examples(self, original_images, adversarial_images, labels):
        """Analyze properties of adversarial examples"""
        
        with torch.no_grad():
            # Get predictions
            original_outputs = self.model(original_images)
            adversarial_outputs = self.model(adversarial_images)
            
            original_preds = original_outputs.argmax(dim=1)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            # Calculate perturbation statistics
            perturbation = adversarial_images - original_images
            l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), dim=1)
            linf_norm = torch.max(torch.abs(perturbation.view(perturbation.size(0), -1)), dim=1)[0]
            
            # Calculate confidence changes
            original_confidence = F.softmax(original_outputs, dim=1).max(dim=1)[0]
            adversarial_confidence = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]
            
            analysis = {
                'attack_success_rate': (adversarial_preds != labels).float().mean().item(),
                'avg_l2_perturbation': l2_norm.mean().item(),
                'avg_linf_perturbation': linf_norm.mean().item(),
                'confidence_drop': (original_confidence - adversarial_confidence).mean().item(),
                'perturbation_visibility': self.calculate_visibility(original_images, adversarial_images)
            }
        
        return analysis
    
    def calculate_visibility(self, original_images, adversarial_images):
        """Calculate perceptual similarity between original and adversarial images"""
        
        # Convert to grayscale for structural similarity
        original_gray = self.rgb_to_grayscale(original_images)
        adversarial_gray = self.rgb_to_grayscale(adversarial_images)
        
        # Calculate SSIM (Structural Similarity Index)
        ssim_scores = []
        for i in range(original_gray.size(0)):
            ssim = self.calculate_ssim(original_gray[i], adversarial_gray[i])
            ssim_scores.append(ssim)
        
        return torch.tensor(ssim_scores).mean().item()
    
    def rgb_to_grayscale(self, images):
        """Convert RGB images to grayscale"""
        
        # Use standard RGB to grayscale conversion
        grayscale = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        return grayscale.unsqueeze(1)
    
    def calculate_ssim(self, img1, img2, window_size=11):
        """Calculate Structural Similarity Index"""
        
        # Simplified SSIM calculation
        mu1 = F.avg_pool2d(img1.unsqueeze(0), window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2.unsqueeze(0), window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1.unsqueeze(0).pow(2), window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2.unsqueeze(0).pow(2), window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1.unsqueeze(0) * img2.unsqueeze(0), window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        c1, c2 = 0.01**2, 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean().item()
    
    def visualize_adversarial_examples(self, original_images, adversarial_images, labels, num_examples=5):
        """Visualize adversarial examples"""
        
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
        
        for i in range(num_examples):
            # Original image
            axes[i, 0].imshow(original_images[i].permute(1, 2, 0).cpu())
            axes[i, 0].set_title(f'Original (Label: {labels[i]})')
            axes[i, 0].axis('off')
            
            # Adversarial image
            axes[i, 1].imshow(adversarial_images[i].permute(1, 2, 0).cpu())
            axes[i, 1].set_title('Adversarial')
            axes[i, 1].axis('off')
            
            # Perturbation
            perturbation = adversarial_images[i] - original_images[i]
            perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
            axes[i, 2].imshow(perturbation.permute(1, 2, 0).cpu())
            axes[i, 2].set_title('Perturbation (Magnified)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
```

---

## 🏗️ Model Hardening Approaches

### Advanced Defense Architectures
Implementing robust model architectures and training strategies.

#### Robust Model Architectures

```python
class RobustModelArchitecture:
    def __init__(self):
        self.defense_components = {}
    
    def create_robust_cnn(self, num_classes=10):
        """Create CNN with built-in robustness features"""
        
        class RobustCNN(nn.Module):
            def __init__(self, num_classes):
                super(RobustCNN, self).__init__()
                
                # Feature extraction with regularization
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(0.2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(0.3),
                    
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                # Classification head with uncertainty estimation
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
                
                # Uncertainty estimation
                self.uncertainty_head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.features(x)
                features = features.view(features.size(0), -1)
                
                # Classification output
                logits = self.classifier(features)
                
                # Uncertainty output
                uncertainty = self.uncertainty_head(features)
                
                return logits, uncertainty
        
        return RobustCNN(num_classes)
    
    def create_ensemble_model(self, models, weights=None):
        """Create ensemble of models for improved robustness"""
        
        class EnsembleModel(nn.Module):
            def __init__(self, models, weights=None):
                super(EnsembleModel, self).__init__()
                self.models = nn.ModuleList(models)
                self.weights = weights if weights is not None else torch.ones(len(models))
                self.weights = self.weights / self.weights.sum()
            
            def forward(self, x):
                outputs = []
                uncertainties = []
                
                for model in self.models:
                    if hasattr(model, 'uncertainty_head'):
                        logits, uncertainty = model(x)
                        outputs.append(logits)
                        uncertainties.append(uncertainty)
                    else:
                        logits = model(x)
                        outputs.append(logits)
                        uncertainties.append(torch.zeros(x.size(0), 1).to(x.device))
                
                # Weighted ensemble
                weighted_outputs = torch.stack(outputs, dim=0) * self.weights.view(-1, 1, 1)
                ensemble_logits = weighted_outputs.sum(dim=0)
                
                # Average uncertainty
                ensemble_uncertainty = torch.stack(uncertainties, dim=0).mean(dim=0)
                
                return ensemble_logits, ensemble_uncertainty
        
        return EnsembleModel(models, weights)
    
    def create_certified_robust_model(self, base_model, radius=0.5):
        """Create model with certified robustness guarantees"""
        
        class CertifiedRobustModel(nn.Module):
            def __init__(self, base_model, radius):
                super(CertifiedRobustModel, self).__init__()
                self.base_model = base_model
                self.radius = radius
            
            def forward(self, x):
                # Apply randomized smoothing
                batch_size = x.size(0)
                noise = torch.randn_like(x) * self.radius
                noisy_x = x + noise
                noisy_x = torch.clamp(noisy_x, 0, 1)
                
                # Get predictions
                outputs = self.base_model(noisy_x)
                
                return outputs
            
            def certify(self, x, num_samples=1000):
                """Certify robustness using randomized smoothing"""
                
                predictions = []
                for _ in range(num_samples):
                    noise = torch.randn_like(x) * self.radius
                    noisy_x = x + noise
                    noisy_x = torch.clamp(noisy_x, 0, 1)
                    
                    with torch.no_grad():
                        outputs = self.base_model(noisy_x)
                        pred = outputs.argmax(dim=1)
                        predictions.append(pred)
                
                predictions = torch.stack(predictions, dim=0)
                
                # Calculate certification
                majority_pred = predictions.mode(dim=0)[0]
                confidence = (predictions == majority_pred.unsqueeze(0)).float().mean(dim=0)
                
                # Certify if confidence > 0.5
                certified = confidence > 0.5
                
                return majority_pred, confidence, certified
        
        return CertifiedRobustModel(base_model, radius)
```

---

## 📊 Evaluation of Robustness

### Comprehensive Robustness Assessment
Evaluating model robustness against various attack types and scenarios.

#### Robustness Evaluation Framework

```python
class RobustnessEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
    
    def evaluate_robustness(self, attack_methods=['fgsm', 'pgd', 'carlini_wagner']):
        """Evaluate model robustness against multiple attack methods"""
        
        results = {}
        
        for attack_method in attack_methods:
            print(f"Evaluating robustness against {attack_method}...")
            
            attack_results = self.evaluate_single_attack(attack_method)
            results[attack_method] = attack_results
        
        return results
    
    def evaluate_single_attack(self, attack_method):
        """Evaluate robustness against a single attack method"""
        
        attacker = AdversarialAttacks(self.model)
        
        total_correct = 0
        total_adversarial_correct = 0
        total_samples = 0
        
        perturbation_norms = []
        confidence_drops = []
        
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Generate adversarial examples
            if attack_method == 'fgsm':
                adversarial_images = attacker.fgsm_attack(images, labels)
            elif attack_method == 'pgd':
                adversarial_images = attacker.pgd_attack(images, labels)
            elif attack_method == 'carlini_wagner':
                adversarial_images = attacker.carlini_wagner_attack(images, labels)
            
            # Evaluate
            with torch.no_grad():
                # Original predictions
                original_outputs = self.model(images)
                original_preds = original_outputs.argmax(dim=1)
                original_correct = (original_preds == labels).sum()
                
                # Adversarial predictions
                adversarial_outputs = self.model(adversarial_images)
                adversarial_preds = adversarial_outputs.argmax(dim=1)
                adversarial_correct = (adversarial_preds == labels).sum()
                
                # Calculate metrics
                total_correct += original_correct.item()
                total_adversarial_correct += adversarial_correct.item()
                total_samples += images.size(0)
                
                # Perturbation analysis
                perturbation = adversarial_images - images
                l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), dim=1)
                perturbation_norms.extend(l2_norm.cpu().numpy())
                
                # Confidence analysis
                original_confidence = F.softmax(original_outputs, dim=1).max(dim=1)[0]
                adversarial_confidence = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]
                confidence_drop = original_confidence - adversarial_confidence
                confidence_drops.extend(confidence_drop.cpu().numpy())
        
        # Calculate final metrics
        clean_accuracy = total_correct / total_samples
        adversarial_accuracy = total_adversarial_correct / total_samples
        robustness = adversarial_accuracy / clean_accuracy
        
        avg_perturbation = np.mean(perturbation_norms)
        avg_confidence_drop = np.mean(confidence_drops)
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'robustness_ratio': robustness,
            'avg_perturbation_l2': avg_perturbation,
            'avg_confidence_drop': avg_confidence_drop
        }
    
    def evaluate_certified_robustness(self, radius_values=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """Evaluate certified robustness using randomized smoothing"""
        
        results = {}
        
        for radius in radius_values:
            print(f"Evaluating certified robustness with radius {radius}...")
            
            certified_correct = 0
            total_samples = 0
            
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Certify robustness
                majority_pred, confidence, certified = self.certify_robustness(images, radius)
                
                # Count correctly classified and certified samples
                correct_and_certified = (majority_pred == labels) & certified
                certified_correct += correct_and_certified.sum().item()
                total_samples += images.size(0)
            
            certified_accuracy = certified_correct / total_samples
            results[radius] = certified_accuracy
        
        return results
    
    def certify_robustness(self, x, radius, num_samples=1000):
        """Certify robustness using randomized smoothing"""
        
        predictions = []
        for _ in range(num_samples):
            noise = torch.randn_like(x) * radius
            noisy_x = x + noise
            noisy_x = torch.clamp(noisy_x, 0, 1)
            
            with torch.no_grad():
                outputs = self.model(noisy_x)
                pred = outputs.argmax(dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate certification
        majority_pred = predictions.mode(dim=0)[0]
        confidence = (predictions == majority_pred.unsqueeze(0)).float().mean(dim=0)
        
        # Certify if confidence > 0.5
        certified = confidence > 0.5
        
        return majority_pred, confidence, certified
    
    def generate_robustness_report(self, results):
        """Generate comprehensive robustness report"""
        
        report = {
            'summary': {
                'total_attacks_evaluated': len(results),
                'average_robustness': np.mean([r['robustness_ratio'] for r in results.values()]),
                'best_defense': max(results.items(), key=lambda x: x[1]['robustness_ratio'])[0]
            },
            'detailed_results': results,
            'recommendations': self.generate_recommendations(results)
        }
        
        return report
    
    def generate_recommendations(self, results):
        """Generate recommendations based on robustness evaluation"""
        
        recommendations = []
        
        # Analyze weakest attack
        weakest_attack = min(results.items(), key=lambda x: x[1]['adversarial_accuracy'])[0]
        recommendations.append(f"Focus defense efforts on {weakest_attack} attacks")
        
        # Analyze perturbation sensitivity
        high_perturbation_attacks = [k for k, v in results.items() if v['avg_perturbation_l2'] > 0.5]
        if high_perturbation_attacks:
            recommendations.append(f"Consider input preprocessing for {', '.join(high_perturbation_attacks)}")
        
        # Analyze confidence drops
        high_confidence_drops = [k for k, v in results.items() if v['avg_confidence_drop'] > 0.3]
        if high_confidence_drops:
            recommendations.append(f"Implement uncertainty estimation for {', '.join(high_confidence_drops)}")
        
        return recommendations
```

---

## 🌍 Real-World Attack Scenarios

### Practical Adversarial Attack Applications
Understanding real-world scenarios where adversarial attacks pose significant threats.

#### Real-World Attack Scenarios

```python
class RealWorldAttackScenarios:
    def __init__(self):
        self.scenarios = {}
    
    def autonomous_vehicle_attack(self):
        """Adversarial attack on autonomous vehicle perception"""
        
        scenario = {
            'target_system': 'Autonomous Vehicle Perception',
            'attack_vector': 'Road sign manipulation',
            'attack_method': 'Physical adversarial patches',
            'impact': 'Traffic sign misclassification',
            'defense_strategies': [
                'Multi-sensor fusion',
                'Temporal consistency checks',
                'Physical security measures',
                'Robust training with physical perturbations'
            ],
            'implementation': '''
            # Physical adversarial patch attack
            def create_physical_adversarial_patch():
                # Design patch that looks like stop sign
                # but classified as speed limit sign
                patch = create_adversarial_patch()
                return patch
            
            # Defense: Multi-sensor validation
            def validate_traffic_sign(camera_pred, lidar_data, gps_context):
                if camera_pred == 'stop_sign':
                    # Verify with other sensors
                    if not lidar_confirms_stop_sign():
                        return 'unknown'
                return camera_pred
            '''
        }
        
        return scenario
    
    def facial_recognition_attack(self):
        """Adversarial attack on facial recognition systems"""
        
        scenario = {
            'target_system': 'Facial Recognition',
            'attack_vector': 'Adversarial glasses or makeup',
            'attack_method': 'Physical adversarial examples',
            'impact': 'Identity spoofing or evasion',
            'defense_strategies': [
                'Liveness detection',
                'Multi-angle verification',
                'Temporal consistency',
                'Robust feature extraction'
            ],
            'implementation': '''
            # Adversarial makeup attack
            def apply_adversarial_makeup(face_image):
                # Apply makeup that fools face recognition
                # while appearing natural to humans
                makeup = generate_adversarial_makeup()
                return apply_makeup(face_image, makeup)
            
            # Defense: Liveness detection
            def detect_liveness(face_sequence):
                # Check for natural eye blinking
                # Verify head movement patterns
                return analyze_movement_patterns(face_sequence)
            '''
        }
        
        return scenario
    
    def medical_imaging_attack(self):
        """Adversarial attack on medical imaging systems"""
        
        scenario = {
            'target_system': 'Medical Imaging AI',
            'attack_vector': 'Image manipulation',
            'attack_method': 'Digital adversarial examples',
            'impact': 'Misdiagnosis or false positives',
            'defense_strategies': [
                'Multi-expert validation',
                'Image integrity verification',
                'Robust training with medical perturbations',
                'Human oversight requirements'
            ],
            'implementation': '''
            # Medical image adversarial attack
            def create_medical_adversarial_example(x_ray_image):
                # Create perturbation that changes diagnosis
                # from normal to pneumonia
                perturbation = generate_medical_adversarial_perturbation()
                return x_ray_image + perturbation
            
            # Defense: Multi-expert validation
            def validate_medical_diagnosis(ai_prediction, image_metadata):
                # Require human radiologist review
                # Check image integrity and provenance
                if ai_prediction.confidence < 0.9:
                    return 'require_human_review'
                return ai_prediction
            '''
        }
        
        return scenario
    
    def financial_fraud_attack(self):
        """Adversarial attack on financial fraud detection"""
        
        scenario = {
            'target_system': 'Financial Fraud Detection',
            'attack_vector': 'Transaction pattern manipulation',
            'attack_method': 'Feature space attacks',
            'impact': 'Fraud evasion or false accusations',
            'defense_strategies': [
                'Multi-factor authentication',
                'Behavioral analysis',
                'Temporal pattern analysis',
                'Robust feature engineering'
            ],
            'implementation': '''
            # Financial transaction adversarial attack
            def create_adversarial_transaction(legitimate_transaction):
                # Modify transaction features to appear legitimate
                # while actually being fraudulent
                adversarial_features = modify_transaction_features(legitimate_transaction)
                return create_transaction(adversarial_features)
            
            # Defense: Multi-factor validation
            def validate_transaction(transaction_features, user_behavior, device_info):
                # Combine multiple signals for robust detection
                risk_score = calculate_risk_score(transaction_features, user_behavior, device_info)
                return risk_score < threshold
            '''
        }
        
        return scenario
    
    def cybersecurity_attack(self):
        """Adversarial attack on cybersecurity systems"""
        
        scenario = {
            'target_system': 'Cybersecurity AI',
            'attack_vector': 'Malware obfuscation',
            'attack_method': 'Code and behavior manipulation',
            'impact': 'Malware evasion or false positives',
            'defense_strategies': [
                'Multi-layer detection',
                'Behavioral analysis',
                'Sandbox execution',
                'Robust feature extraction'
            ],
            'implementation': '''
            # Malware adversarial attack
            def create_adversarial_malware(original_malware):
                # Modify malware to evade detection
                # while maintaining malicious functionality
                obfuscated_code = obfuscate_malware_code(original_malware)
                return create_malware_variant(obfuscated_code)
            
            # Defense: Multi-layer detection
            def detect_malware(file_features, behavior_patterns, network_activity):
                # Combine static and dynamic analysis
                static_score = analyze_static_features(file_features)
                dynamic_score = analyze_behavior(behavior_patterns)
                network_score = analyze_network_activity(network_activity)
                
                return combine_scores(static_score, dynamic_score, network_score)
            '''
        }
        
        return scenario
```

---

## 🚀 Implementation Best Practices

### Comprehensive Adversarial Defense System

```python
class ComprehensiveAdversarialDefense:
    def __init__(self, model, defense_config):
        self.model = model
        self.config = defense_config
        self.defense_layers = []
        
    def build_defense_system(self):
        """Build comprehensive adversarial defense system"""
        
        defense_system = {
            'input_preprocessing': {
                'spatial_smoothing': True,
                'bit_depth_reduction': True,
                'jpeg_compression': True,
                'random_crop_resize': True
            },
            'model_robustness': {
                'adversarial_training': True,
                'defensive_distillation': True,
                'ensemble_methods': True,
                'certified_robustness': True
            },
            'detection_monitoring': {
                'anomaly_detection': True,
                'uncertainty_estimation': True,
                'confidence_thresholding': True,
                'temporal_consistency': True
            },
            'response_mechanisms': {
                'automatic_rejection': True,
                'human_oversight': True,
                'fallback_systems': True,
                'alert_generation': True
            }
        }
        
        return defense_system
    
    def implement_defense_pipeline(self, input_data):
        """Implement complete defense pipeline"""
        
        # Input preprocessing
        processed_data = self.preprocess_input(input_data)
        
        # Model prediction with uncertainty
        predictions, uncertainty = self.model(processed_data)
        
        # Anomaly detection
        is_anomalous = self.detect_anomalies(processed_data, predictions)
        
        # Confidence thresholding
        is_uncertain = uncertainty > self.config.uncertainty_threshold
        
        # Decision making
        if is_anomalous or is_uncertain:
            return self.handle_suspicious_input(input_data, predictions, uncertainty)
        else:
            return predictions
    
    def preprocess_input(self, input_data):
        """Apply input preprocessing defenses"""
        
        preprocessed = input_data
        
        if self.config.spatial_smoothing:
            preprocessed = self.spatial_smoothing(preprocessed)
        
        if self.config.bit_depth_reduction:
            preprocessed = self.bit_depth_reduction(preprocessed)
        
        if self.config.jpeg_compression:
            preprocessed = self.jpeg_compression(preprocessed)
        
        if self.config.random_crop_resize:
            preprocessed = self.random_crop_resize(preprocessed)
        
        return preprocessed
    
    def detect_anomalies(self, input_data, predictions):
        """Detect anomalous inputs"""
        
        # Multiple anomaly detection methods
        feature_anomaly = self.detect_feature_anomalies(input_data)
        prediction_anomaly = self.detect_prediction_anomalies(predictions)
        temporal_anomaly = self.detect_temporal_anomalies(input_data)
        
        return feature_anomaly or prediction_anomaly or temporal_anomaly
    
    def handle_suspicious_input(self, input_data, predictions, uncertainty):
        """Handle suspicious or uncertain inputs"""
        
        response = {
            'action': 'reject',
            'reason': 'high_uncertainty_or_anomaly',
            'confidence': 1 - uncertainty.item(),
            'recommendation': 'human_review_required',
            'fallback_prediction': predictions.argmax().item()
        }
        
        # Log suspicious input
        self.log_suspicious_input(input_data, response)
        
        # Generate alert
        self.generate_alert(response)
        
        return response
```

### Key Considerations

1. **Multi-Layer Defense**
   - Input preprocessing for noise reduction
   - Model-level robustness through training
   - Post-processing for confidence estimation
   - Human oversight for critical decisions

2. **Performance Trade-offs**
   - Defense effectiveness vs computational cost
   - False positive rate vs security
   - Usability vs security measures
   - Real-time requirements vs thorough analysis

3. **Continuous Monitoring**
   - Real-time attack detection
   - Model performance monitoring
   - Adversarial example collection
   - Defense strategy updates

4. **Regulatory Compliance**
   - Privacy protection in defense systems
   - Audit trail maintenance
   - Explainable defense decisions
   - Compliance with industry standards

This comprehensive guide covers the essential aspects of AI adversarial robustness, from attack methods to defense strategies and real-world applications. 