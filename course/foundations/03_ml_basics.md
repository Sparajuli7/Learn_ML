# Machine Learning Basics: From Theory to Practice

*"Understanding the fundamental principles that power modern AI systems"*

---

## 📚 Table of Contents

1. [What is Machine Learning?](#what-is-machine-learning)
2. [ML vs. Traditional Programming](#ml-vs-traditional-programming)
3. [Types of Machine Learning](#types-of-machine-learning)
4. [Core Algorithms](#core-algorithms)
5. [Data Handling](#data-handling)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Practical Implementation](#practical-implementation)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

---

## 🤖 What is Machine Learning?

### Definition

**Machine Learning (ML)** is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Instead of following rigid rules, ML systems identify patterns in data to make predictions or decisions.

### Core Concept: Learning from Data

```
Traditional Programming:
Input + Rules → Output

Machine Learning:
Input + Output → Rules (Model)
```

### Key Characteristics

1. **Data-Driven**: Learns patterns from historical data
2. **Adaptive**: Improves performance with more data
3. **Predictive**: Makes predictions on new, unseen data
4. **Generalizable**: Applies learned patterns to similar problems

---

## 🔄 ML vs. Traditional Programming

### Traditional Programming Example

```python
# Traditional rule-based email spam detection
def is_spam_traditional(email):
    """Rule-based spam detection"""
    spam_indicators = [
        'free money',
        'urgent action required',
        'click here',
        'limited time offer'
    ]
    
    email_lower = email.lower()
    spam_score = 0
    
    for indicator in spam_indicators:
        if indicator in email_lower:
            spam_score += 1
    
    return spam_score >= 2  # If 2+ indicators found
```

### Machine Learning Example

```python
# ML-based email spam detection
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class SpamDetectorML:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.is_trained = False
    
    def prepare_features(self, emails):
        """Convert emails to numerical features"""
        return self.vectorizer.fit_transform(emails)
    
    def train(self, emails, labels):
        """Train the model on labeled data"""
        X = self.prepare_features(emails)
        self.model.fit(X, labels)
        self.is_trained = True
    
    def predict(self, email):
        """Predict if email is spam"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X = self.vectorizer.transform([email])
        return self.model.predict(X)[0]

# Usage example
emails = [
    "Get rich quick! Click here for free money!",
    "Meeting tomorrow at 3pm in conference room",
    "URGENT: Your account has been suspended",
    "Project update: Q4 results are in"
]

labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

detector = SpamDetectorML()
detector.train(emails, labels)

# Test on new email
new_email = "Limited time offer! Don't miss out!"
prediction = detector.predict(new_email)
print(f"Spam prediction: {prediction}")
```

### Comparison Table

| Aspect | Traditional Programming | Machine Learning |
|--------|------------------------|------------------|
| **Approach** | Rule-based logic | Pattern recognition |
| **Flexibility** | Fixed behavior | Adapts to data |
| **Maintenance** | Manual rule updates | Retrain with new data |
| **Performance** | Limited by human knowledge | Improves with more data |
| **Transparency** | Explicit rules | Black box (usually) |

---

## 🎯 Types of Machine Learning

### 1. Supervised Learning

**Definition**: Learning from labeled examples to predict outcomes

**Types**:
- **Classification**: Predict discrete categories
- **Regression**: Predict continuous values

```python
# Supervised Learning Example: House Price Prediction
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'square_feet': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'bathrooms': [1, 2, 2, 3, 3],
    'price': [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make prediction
new_house = [[1800, 3, 2]]  # 1800 sq ft, 3 bed, 2 bath
predicted_price = model.predict(new_house)[0]
print(f"Predicted price: ${predicted_price:,.0f}")
```

### 2. Unsupervised Learning

**Definition**: Finding patterns in data without labeled examples

**Types**:
- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Reduce data complexity
- **Association**: Find relationships between variables

```python
# Unsupervised Learning Example: Customer Segmentation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample customer data
customer_data = np.array([
    [25, 50000, 2],   # Age, Income, Purchases
    [35, 75000, 5],
    [45, 120000, 8],
    [28, 45000, 1],
    [52, 150000, 12],
    [30, 60000, 3]
])

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Analyze clusters
for i in range(3):
    cluster_customers = customer_data[clusters == i]
    print(f"Cluster {i}: {len(cluster_customers)} customers")
    print(f"Average age: {cluster_customers[:, 0].mean():.1f}")
    print(f"Average income: ${cluster_customers[:, 1].mean():,.0f}")
    print(f"Average purchases: {cluster_customers[:, 2].mean():.1f}")
    print()
```

### 3. Reinforcement Learning

**Definition**: Learning through interaction with environment to maximize rewards

```python
# Simplified Reinforcement Learning Example: Q-Learning
import numpy as np

class SimpleEnvironment:
    def __init__(self):
        self.states = 4  # 0, 1, 2, 3
        self.actions = 2  # 0: left, 1: right
        self.current_state = 0
        self.goal_state = 3
        
        # Reward structure
        self.rewards = {
            (0, 1): 1,   # State 0, action 1 (right) gets reward
            (1, 1): 1,   # State 1, action 1 (right) gets reward
            (2, 1): 10,  # State 2, action 1 (right) gets big reward
        }
    
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def step(self, action):
        # Simple state transition
        if action == 0:  # left
            next_state = max(0, self.current_state - 1)
        else:  # right
            next_state = min(3, self.current_state + 1)
        
        # Get reward
        reward = self.rewards.get((self.current_state, action), 0)
        
        # Check if done
        done = next_state == self.goal_state
        
        self.current_state = next_state
        return next_state, reward, done

class QLearningAgent:
    def __init__(self, states, actions, learning_rate=0.1, discount=0.9, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action
    
    def learn(self, state, action, reward, next_state):
        """Update Q-values using Q-learning formula"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# Train the agent
env = SimpleEnvironment()
agent = QLearningAgent(states=4, actions=2)

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

print("Final Q-table:")
print(agent.q_table)
```

---

## 🔧 Core Algorithms

### 1. Linear Regression

**Purpose**: Predict continuous values

**Mathematical Foundation**:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

```python
# Linear Regression Implementation
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot results
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.show()

print(f"Slope: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
```

### 2. Logistic Regression

**Purpose**: Binary classification

**Mathematical Foundation**:
```
P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

```python
# Logistic Regression Example
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42, n_clusters_per_class=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### 3. Decision Trees

**Purpose**: Classification and regression with interpretable rules

```python
# Decision Tree Example
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.show()

# Feature importance
feature_importance = tree.feature_importances_
for feature, importance in zip(iris.feature_names, feature_importance):
    print(f"{feature}: {importance:.3f}")
```

### 4. Random Forest

**Purpose**: Ensemble method combining multiple decision trees

```python
# Random Forest Example
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Train random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

---

## 📊 Data Handling

### Data Preprocessing

```python
# Comprehensive Data Preprocessing Example
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Sample dataset with various data types
data = {
    'age': [25, 30, np.nan, 35, 28],
    'income': [50000, 75000, 60000, np.nan, 45000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'High School'],
    'employed': [True, True, False, True, False],
    'target': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
    
    def preprocess(self, df, target_column=None):
        """Complete preprocessing pipeline"""
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        if target_column:
            numeric_columns = numeric_columns.drop(target_column)
        
        df_processed[numeric_columns] = self.imputer.fit_transform(df_processed[numeric_columns])
        
        # Encode categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Scale numerical features
        if len(numeric_columns) > 0:
            df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])
        
        return df_processed

# Apply preprocessing
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess(df, target_column='target')

print("Original Data:")
print(df)
print("\nProcessed Data:")
print(df_processed)
```

### Feature Engineering

```python
# Feature Engineering Examples
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100)
        self.pca = PCA(n_components=2)
    
    def create_time_features(self, df, date_column):
        """Extract time-based features"""
        df[date_column] = pd.to_datetime(df[date_column])
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        return df
    
    def create_interaction_features(self, df, feature1, feature2):
        """Create interaction features"""
        df[f'{feature1}_{feature2}_interaction'] = df[feature1] * df[feature2]
        return df
    
    def create_polynomial_features(self, df, feature, degree=2):
        """Create polynomial features"""
        for i in range(2, degree + 1):
            df[f'{feature}_power_{i}'] = df[feature] ** i
        return df
    
    def extract_text_features(self, text_series):
        """Extract features from text data"""
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(text_series)
        
        # Basic text features
        text_features = pd.DataFrame({
            'char_count': text_series.str.len(),
            'word_count': text_series.str.split().str.len(),
            'avg_word_length': text_series.str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
        })
        
        return tfidf_features, text_features

# Example usage
sample_data = {
    'date': ['2023-01-15', '2023-02-20', '2023-03-10'],
    'price': [100, 150, 200],
    'quantity': [5, 3, 8],
    'text': ['great product', 'excellent service', 'amazing quality']
}

df = pd.DataFrame(sample_data)
engineer = FeatureEngineer()

# Apply feature engineering
df = engineer.create_time_features(df, 'date')
df = engineer.create_interaction_features(df, 'price', 'quantity')
df = engineer.create_polynomial_features(df, 'price', degree=2)

print("Enhanced Features:")
print(df)
```

---

## 📈 Evaluation Metrics

### Classification Metrics

```python
# Comprehensive Classification Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_true, y_pred, y_prob=None):
    """Comprehensive classification evaluation"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # ROC-AUC if probabilities available
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print(f"ROC-AUC: {roc_auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### Regression Metrics

```python
# Regression Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation"""
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Regression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"R² Score: {r2:.3f}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
```

---

## 🧪 Exercises and Projects

### Exercise 1: Email Classification

```python
# TODO: Build an email classifier
# Dataset: Spam vs. Ham emails
# Features: Word frequency, email length, presence of links, etc.

def build_email_classifier():
    """
    Build a complete email classification system
    
    Steps:
    1. Load and preprocess email data
    2. Extract features (TF-IDF, email characteristics)
    3. Train multiple models (Naive Bayes, SVM, Random Forest)
    4. Evaluate and compare performance
    5. Deploy best model
    """
    pass
```

### Exercise 2: House Price Prediction

```python
# TODO: Predict house prices using multiple features
# Features: Square footage, bedrooms, bathrooms, location, year built

def house_price_predictor():
    """
    Build a house price prediction model
    
    Steps:
    1. Load housing dataset
    2. Handle missing values and outliers
    3. Create feature interactions
    4. Train regression models
    5. Evaluate and select best model
    """
    pass
```

### Exercise 3: Customer Segmentation

```python
# TODO: Segment customers based on purchasing behavior
# Features: Purchase frequency, amount, product categories

def customer_segmentation():
    """
    Perform customer segmentation analysis
    
    Steps:
    1. Load customer transaction data
    2. Create customer features (RFM analysis)
    3. Apply clustering algorithms
    4. Analyze and interpret segments
    5. Create marketing strategies for each segment
    """
    pass
```

---

## 📖 Further Reading

### Essential Books
- "Introduction to Machine Learning with Python" by Andreas Müller and Sarah Guido
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Key Papers
- "A Few Useful Things to Know About Machine Learning" by Pedro Domingos
- "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, Jerome Friedman

### Online Resources
- [Scikit-learn Documentation](https://scikit-learn.org/) - Comprehensive ML library
- [Kaggle](https://www.kaggle.com/) - Competitions and datasets
- [Coursera ML Course](https://www.coursera.org/learn/machine-learning) - Andrew Ng's course

### Next Steps
- **[Deep Learning Basics](04_deep_learning_basics.md)**: Neural networks and deep learning
- **[NLP Fundamentals](core_ml_fields/06_nlp_fundamentals.md)**: Natural language processing
- **[Computer Vision Basics](core_ml_fields/08_computer_vision_basics.md)**: Image processing

---

## 🎯 Key Takeaways

1. **ML Definition**: Learning patterns from data to make predictions
2. **Three Types**: Supervised, unsupervised, and reinforcement learning
3. **Core Algorithms**: Linear/logistic regression, decision trees, random forests
4. **Data Handling**: Preprocessing, feature engineering, evaluation
5. **Practical Skills**: Implementation, evaluation, deployment
6. **Best Practices**: Cross-validation, hyperparameter tuning, model selection

---

*"Machine learning is not just about algorithms—it's about understanding data and extracting meaningful insights."*

**Next: [Deep Learning Basics](04_deep_learning_basics.md) → Neural networks and modern AI** 