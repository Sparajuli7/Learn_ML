# Cursor Advanced: Extensions, Debugging, and ML Workflows

*"Master the advanced features that make Cursor the ultimate ML development environment"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Advanced Extensions](#advanced-extensions)
3. [Debugging Techniques](#debugging-techniques)
4. [ML-Specific Workflows](#ml-specific-workflows)
5. [Performance Optimization](#performance-optimization)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction

While the basic Cursor IDE features provide excellent AI assistance, advanced users can unlock even more powerful capabilities through extensions, custom configurations, and specialized workflows. This guide covers the advanced techniques that separate expert ML developers from beginners.

### Advanced Features Overview

- **Custom Extensions**: Domain-specific tools for ML development
- **Advanced Debugging**: Complex debugging scenarios in ML pipelines
- **Workflow Automation**: Streamlined development processes
- **Performance Tuning**: Optimizing Cursor for large ML projects
- **Team Collaboration**: Advanced features for team development

### 2025 Advanced Trends

- **Multi-modal Debugging**: Debugging across code, data, and visualizations
- **AI-Powered Profiling**: Automatic performance analysis and optimization
- **Collaborative AI**: Team-based AI assistance and knowledge sharing
- **Custom AI Models**: Domain-specific AI assistants for ML teams

---

## 🔧 Advanced Extensions

### 1. ML-Specific Extensions

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.flake8",
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-keymap",
    "ms-toolsai.jupyter-renderers",
    "ms-toolsai.vscode-jupyter-cell-tags",
    "ms-toolsai.vscode-jupyter-slideshow",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-docker",
    "ms-azuretools.vscode-docker",
    "ms-vscode-remote.remote-containers",
    "ms-vscode-remote.remote-ssh",
    "ms-vscode-remote.remote-wsl"
  ]
}
```

### 2. Custom Extension Development

```typescript
// extensions/ml-debugger/package.json
{
  "name": "ml-debugger",
  "displayName": "ML Debugger",
  "description": "Advanced debugging for ML pipelines",
  "version": "0.0.1",
  "engines": {
    "vscode": "^1.60.0"
  },
  "categories": ["Other"],
  "activationEvents": [
    "onCommand:ml-debugger.startDebugging"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "ml-debugger.startDebugging",
        "title": "Start ML Debugging"
      }
    ],
    "configuration": {
      "title": "ML Debugger",
      "properties": {
        "mlDebugger.enableAutoProfiling": {
          "type": "boolean",
          "default": true,
          "description": "Enable automatic performance profiling"
        }
      }
    }
  }
}
```

### 3. Extension Configuration

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "jupyter.askForKernelRestart": false,
  "jupyter.enableAutoMatcher": true,
  "jupyter.enableAutoStartingServer": true,
  "jupyter.interactiveWindow.textEditor.executeSelection": "sendToInteractiveWindow",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "jupyter.outputTiming": true,
  "jupyter.showCellInputOnExecutionCount": true,
  "jupyter.askForKernelRestart": false,
  "jupyter.enableAutoMatcher": true,
  "jupyter.enableAutoStartingServer": true,
  "jupyter.interactiveWindow.textEditor.executeSelection": "sendToInteractiveWindow",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "jupyter.outputTiming": true,
  "jupyter.showCellInputOnExecutionCount": true
}
```

---

## 🐛 Debugging Techniques

### 1. Advanced Breakpoint Configuration

```python
# Advanced debugging with conditional breakpoints
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_complex_model(X, y, model_type='random_forest'):
    """
    Train a complex ML model with advanced debugging
    """
    # Set conditional breakpoint: only when data size > 10000
    if len(X) > 10000:  # Breakpoint condition
        print(f"Large dataset detected: {len(X)} samples")
    
    # Set breakpoint on specific data conditions
    if np.isnan(X).any():  # Breakpoint on NaN detection
        print("NaN values detected in features")
    
    # Set breakpoint on model performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'neural_network':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
    
    # Set breakpoint on training start
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Set breakpoint on evaluation
    score = model.score(X_test, y_test)
    if score < 0.7:  # Breakpoint on poor performance
        print(f"Poor model performance: {score:.3f}")
    
    return model, score
```

### 2. Custom Debugging Extensions

```python
# custom_debugger.py
import sys
import traceback
from typing import Any, Dict, List
import logging

class MLDebugger:
    """Advanced debugging for ML pipelines"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.debug_history = []
        
    def debug_data_pipeline(self, data: Any, stage: str) -> None:
        """Debug data at different pipeline stages"""
        self.logger.info(f"Debugging {stage} stage")
        
        if hasattr(data, 'shape'):
            self.logger.info(f"Data shape: {data.shape}")
        if hasattr(data, 'dtypes'):
            self.logger.info(f"Data types: {data.dtypes}")
        if hasattr(data, 'isnull'):
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                self.logger.warning(f"Null values found: {null_counts}")
        
        self.debug_history.append({
            'stage': stage,
            'data_info': str(data),
            'timestamp': pd.Timestamp.now()
        })
    
    def debug_model_training(self, model: Any, X: Any, y: Any) -> Dict:
        """Debug model training process"""
        self.logger.info("Debugging model training")
        
        debug_info = {
            'model_type': type(model).__name__,
            'training_samples': len(X),
            'features': X.shape[1] if hasattr(X, 'shape') else 'unknown',
            'target_distribution': y.value_counts().to_dict() if hasattr(y, 'value_counts') else 'unknown'
        }
        
        try:
            # Monitor training progress
            if hasattr(model, 'fit'):
                model.fit(X, y)
                debug_info['training_successful'] = True
            else:
                debug_info['training_successful'] = False
                self.logger.error("Model does not have fit method")
                
        except Exception as e:
            debug_info['training_successful'] = False
            debug_info['error'] = str(e)
            self.logger.error(f"Training failed: {e}")
            traceback.print_exc()
        
        return debug_info
    
    def get_debug_summary(self) -> Dict:
        """Get summary of all debugging information"""
        return {
            'total_stages': len(self.debug_history),
            'stages': [h['stage'] for h in self.debug_history],
            'last_debug': self.debug_history[-1] if self.debug_history else None
        }

# Usage example
debugger = MLDebugger()
```

### 3. Visual Debugging with Jupyter

```python
# visual_debugging.ipynb
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

class VisualDebugger:
    """Visual debugging for ML pipelines"""
    
    def __init__(self):
        self.figures = []
        
    def debug_data_distribution(self, data, title="Data Distribution"):
        """Visualize data distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Histogram
        axes[0, 0].hist(data.values.flatten(), bins=50, alpha=0.7)
        axes[0, 0].set_title("Histogram")
        
        # Box plot
        axes[0, 1].boxplot(data.values)
        axes[0, 1].set_title("Box Plot")
        
        # Correlation heatmap
        if data.shape[1] > 1:
            sns.heatmap(data.corr(), ax=axes[1, 0], cmap='coolwarm')
            axes[1, 0].set_title("Correlation Matrix")
        
        # Missing values
        missing_data = data.isnull().sum()
        axes[1, 1].bar(range(len(missing_data)), missing_data.values)
        axes[1, 1].set_title("Missing Values")
        axes[1, 1].set_xticks(range(len(missing_data)))
        axes[1, 1].set_xticklabels(missing_data.index, rotation=45)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
        
        self.figures.append(fig)
    
    def debug_model_performance(self, y_true, y_pred, title="Model Performance"):
        """Visualize model performance"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title("Confusion Matrix")
        
        # ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        
        # Precision-Recall curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        axes[2].plot(recall, precision, color='green', lw=2)
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
        
        self.figures.append(fig)

# Usage
visual_debugger = VisualDebugger()
```

---

## 🔄 ML-Specific Workflows

### 1. Automated ML Pipeline Debugging

```python
# ml_pipeline_debugger.py
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

class MLPipelineDebugger:
    """Comprehensive debugging for ML pipelines"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.debug_results = {}
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def debug_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality debugging"""
        self.logger.info("Starting data quality debugging")
        
        quality_report = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'null_counts': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            outliers = {}
            for col in numeric_cols.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | 
                               (data[col] > (Q3 + 1.5 * IQR))).sum()
                outliers[col] = outlier_count
            quality_report['outliers'] = outliers
        
        self.debug_results['data_quality'] = quality_report
        return quality_report
    
    def debug_feature_engineering(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Debug feature engineering process"""
        self.logger.info("Debugging feature engineering")
        
        feature_report = {
            'feature_count': X.shape[1],
            'sample_count': X.shape[0],
            'feature_correlation': X.corr().abs().mean().mean(),
            'feature_variance': X.var().to_dict(),
            'target_correlation': X.corrwith(y).abs().to_dict()
        }
        
        # Check for multicollinearity
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > 0.8]
            if not high_corr.empty:
                for idx in high_corr.index:
                    high_corr_pairs.append((column, idx, high_corr[idx]))
        
        feature_report['high_correlation_pairs'] = high_corr_pairs
        
        self.debug_results['feature_engineering'] = feature_report
        return feature_report
    
    def debug_model_training(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Debug model training process"""
        self.logger.info("Debugging model training")
        
        training_report = {
            'model_type': type(model).__name__,
            'training_samples': X.shape[0],
            'feature_count': X.shape[1],
            'target_distribution': y.value_counts().to_dict()
        }
        
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5)
            training_report['cv_mean'] = cv_scores.mean()
            training_report['cv_std'] = cv_scores.std()
            training_report['cv_scores'] = cv_scores.tolist()
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                training_report['feature_importance'] = dict(zip(X.columns, model.feature_importances_))
            
        except Exception as e:
            training_report['error'] = str(e)
            self.logger.error(f"Training debugging failed: {e}")
        
        self.debug_results['model_training'] = training_report
        return training_report
    
    def debug_model_evaluation(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Debug model evaluation"""
        self.logger.info("Debugging model evaluation")
        
        evaluation_report = {}
        
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Classification metrics
            evaluation_report['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            evaluation_report['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
            # Additional metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            evaluation_report['accuracy'] = accuracy_score(y_test, y_pred)
            evaluation_report['precision'] = precision_score(y_test, y_pred, average='weighted')
            evaluation_report['recall'] = recall_score(y_test, y_pred, average='weighted')
            evaluation_report['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            
            # Prediction distribution
            evaluation_report['prediction_distribution'] = pd.Series(y_pred).value_counts().to_dict()
            
        except Exception as e:
            evaluation_report['error'] = str(e)
            self.logger.error(f"Evaluation debugging failed: {e}")
        
        self.debug_results['model_evaluation'] = evaluation_report
        return evaluation_report
    
    def generate_debug_report(self) -> str:
        """Generate comprehensive debug report"""
        report = "# ML Pipeline Debug Report\n\n"
        
        for section, data in self.debug_results.items():
            report += f"## {section.replace('_', ' ').title()}\n\n"
            report += f"```json\n{data}\n```\n\n"
        
        return report

# Usage example
debugger = MLPipelineDebugger()
```

### 2. Custom Keyboard Shortcuts

```json
// .vscode/keybindings.json
[
  {
    "key": "ctrl+shift+d",
    "command": "workbench.action.debug.start",
    "when": "editorTextFocus"
  },
  {
    "key": "ctrl+shift+r",
    "command": "jupyter.runAllCells",
    "when": "notebookEditorFocused"
  },
  {
    "key": "ctrl+shift+c",
    "command": "jupyter.clearAllOutputs",
    "when": "notebookEditorFocused"
  },
  {
    "key": "ctrl+shift+p",
    "command": "python.startREPL",
    "when": "python"
  },
  {
    "key": "ctrl+shift+t",
    "command": "python.runPythonFileInTerminal",
    "when": "python"
  }
]
```

---

## ⚡ Performance Optimization

### 1. Cursor Performance Settings

```json
// .vscode/settings.json
{
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/node_modules": true,
    "**/.git": true,
    "**/venv": true,
    "**/env": true,
    "**/.env": true,
    "**/data": true,
    "**/models": true,
    "**/logs": true
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/bower_components": true,
    "**/*.code-search": true,
    "**/venv": true,
    "**/env": true,
    "**/data": true,
    "**/models": true
  },
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/venv/**": true,
    "**/env/**": true,
    "**/data/**": true,
    "**/models/**": true,
    "**/logs/**": true
  },
  "python.analysis.extraPaths": [
    "./src",
    "./lib"
  ],
  "python.analysis.autoImportCompletions": true,
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoSearchPaths": true,
  "python.analysis.diagnosticMode": "workspace",
  "python.analysis.stubPath": "./typings",
  "python.analysis.autoComplete.addBrackets": true,
  "python.analysis.autoComplete.includeFunctionParens": true
}
```

### 2. Memory Optimization

```python
# memory_optimizer.py
import psutil
import gc
import logging
from typing import Any, Dict

class MemoryOptimizer:
    """Memory optimization for large ML projects"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'used': memory.used / (1024**3),  # GB
            'percent': memory.percent
        }
    
    def optimize_memory(self) -> None:
        """Perform memory optimization"""
        memory_info = self.check_memory_usage()
        
        if memory_info['percent'] > self.memory_threshold * 100:
            self.logger.warning(f"High memory usage: {memory_info['percent']:.1f}%")
            
            # Force garbage collection
            gc.collect()
            
            # Clear IPython cache if available
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                if ipython is not None:
                    ipython.magic('reset -f')
            except ImportError:
                pass
            
            self.logger.info("Memory optimization completed")
        else:
            self.logger.info(f"Memory usage normal: {memory_info['percent']:.1f}%")
    
    def monitor_memory(self, func):
        """Decorator to monitor memory usage of functions"""
        def wrapper(*args, **kwargs):
            before = self.check_memory_usage()
            result = func(*args, **kwargs)
            after = self.check_memory_usage()
            
            memory_diff = after['used'] - before['used']
            self.logger.info(f"Function {func.__name__} used {memory_diff:.2f} GB")
            
            return result
        return wrapper

# Usage
memory_optimizer = MemoryOptimizer()
```

---

## 🧪 Exercises and Projects

### Exercise 1: Advanced Debugging Setup

1. Create a custom debugging extension for ML pipelines
2. Implement conditional breakpoints for data quality issues
3. Set up visual debugging with matplotlib/seaborn
4. Create automated debugging reports

### Exercise 2: Performance Optimization

1. Profile a large ML project for performance bottlenecks
2. Implement memory optimization techniques
3. Set up automated performance monitoring
4. Create performance benchmarking tools

### Exercise 3: Custom Workflows

1. Design custom keyboard shortcuts for ML development
2. Create automated testing workflows
3. Implement continuous integration for ML projects
4. Build custom extensions for specific ML tasks

### Project: Advanced ML Development Environment

**Objective**: Build a comprehensive ML development environment with advanced debugging and optimization

**Requirements**:
- Custom debugging extensions for ML pipelines
- Performance monitoring and optimization tools
- Automated testing and quality assurance
- Custom workflows for different ML tasks
- Team collaboration features

**Deliverables**:
- Custom Cursor extensions for ML development
- Performance optimization toolkit
- Automated debugging framework
- Custom workflow templates
- Team collaboration setup

---

## 📖 Further Reading

### Essential Resources

1. **Advanced Cursor Features**
   - [Cursor Advanced Documentation](https://cursor.sh/docs/advanced)
   - [Extension Development Guide](https://code.visualstudio.com/api)
   - [Debugging Best Practices](https://cursor.sh/docs/debugging)

2. **ML-Specific Tools**
   - [Jupyter Debugging](https://jupyter.org/)
   - [MLflow Debugging](https://mlflow.org/)
   - [Weights & Biases](https://wandb.ai/)

3. **Performance Optimization**
   - [Python Profiling](https://docs.python.org/3/library/profile.html)
   - [Memory Optimization](https://pythonspeed.com/)
   - [Large Scale ML](https://www.oreilly.com/library/view/large-scale-machine/9781491962289/)

### Advanced Topics

- **Custom AI Models**: Training domain-specific AI assistants
- **Multi-modal Debugging**: Debugging across code, data, and visualizations
- **Distributed Debugging**: Debugging distributed ML systems
- **Real-time Monitoring**: Live debugging of production ML systems

### 2025 Trends to Watch

- **AI-Powered Debugging**: Automatic bug detection and fixes
- **Visual Programming**: Drag-and-drop ML pipeline debugging
- **Collaborative Debugging**: Team-based debugging sessions
- **Predictive Debugging**: AI anticipating and preventing bugs

---

## 🎯 Key Takeaways

1. **Advanced Extensions**: Custom extensions can significantly enhance ML development
2. **Comprehensive Debugging**: Multi-layered debugging approaches improve code quality
3. **Performance Monitoring**: Continuous performance optimization is essential for large ML projects
4. **Custom Workflows**: Tailored workflows can dramatically improve productivity
5. **Team Collaboration**: Advanced features enable better team coordination

---

*"The best debugging tool is a good night's sleep."*

**Next: [Python Ecosystem](tools_and_ides/34_python_ecosystem.md) → NumPy, Pandas, and visualization tools**