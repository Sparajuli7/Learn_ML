# Unsupervised Learning and Clustering
## Advanced Techniques for Pattern Discovery and Data Exploration

*"Finding hidden patterns in data without labels - the art of discovery"*

---

## 🎯 Learning Objectives

By the end of this guide, you will:

- **Master clustering algorithms**: K-means, hierarchical, DBSCAN, and spectral clustering
- **Understand dimensionality reduction**: PCA, t-SNE, UMAP, and autoencoders
- **Explore association rules**: Market basket analysis and frequent pattern mining
- **Apply density-based methods**: Outlier detection and anomaly identification
- **Build practical systems**: Customer segmentation, recommendation engines, data preprocessing
- **Stay current with 2025 trends**: Self-supervised learning, contrastive learning, and modern clustering

### Prerequisites

- **Mathematics**: Linear algebra, probability theory, optimization
- **Programming**: Python, NumPy, scikit-learn, matplotlib
- **ML Basics**: Understanding of supervised learning concepts
- **Data Analysis**: Experience with data preprocessing and visualization

---

## 📚 Table of Contents

1. [Fundamentals of Unsupervised Learning](#fundamentals)
2. [Clustering Algorithms](#clustering)
3. [Dimensionality Reduction](#dimensionality)
4. [Association Rule Mining](#association)
5. [Density-Based Methods](#density)
6. [Self-Supervised Learning](#self-supervised)
7. [Practical Applications](#applications)
8. [Advanced Topics](#advanced)
9. [Tools and Frameworks](#tools)
10. [Exercises and Projects](#exercises)

---

## 🔍 Fundamentals of Unsupervised Learning

### What is Unsupervised Learning?

Unsupervised learning discovers hidden patterns in data without predefined labels or targets. Unlike supervised learning, there's no "correct" answer - the goal is to find meaningful structure.

```python
# Key characteristics of unsupervised learning
characteristics = {
    "no_labels": "Data comes without target variables",
    "pattern_discovery": "Find hidden structures and relationships",
    "data_exploration": "Understand data distribution and properties",
    "preprocessing": "Prepare data for supervised learning",
    "dimensionality": "Reduce complexity while preserving information"
}
```

### Types of Unsupervised Learning

1. **Clustering**: Group similar data points together
2. **Dimensionality Reduction**: Reduce feature space while preserving structure
3. **Association Rule Mining**: Find relationships between variables
4. **Density Estimation**: Model data distribution
5. **Anomaly Detection**: Identify unusual patterns

### Evaluation Challenges

Unlike supervised learning, evaluation is more subjective:

```python
# Common evaluation metrics for clustering
clustering_metrics = {
    "silhouette_score": "Measures cluster cohesion and separation",
    "calinski_harabasz": "Ratio of between-cluster to within-cluster variance",
    "davies_bouldin": "Average similarity measure of clusters",
    "inertia": "Sum of squared distances to cluster centers (K-means)",
    "modularity": "Quality measure for community detection"
}
```

---

## 🎯 Clustering Algorithms

### K-Means Clustering

The most popular clustering algorithm, K-means partitions data into K clusters by minimizing within-cluster variance.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=300, c='red', marker='x')
plt.title('K-Means Clustering')
plt.show()
```

**K-Means Algorithm Steps:**

1. **Initialize**: Randomly choose K cluster centers
2. **Assign**: Assign each point to nearest center
3. **Update**: Recalculate centers as mean of assigned points
4. **Repeat**: Until convergence or max iterations

**Choosing K:**

```python
# Elbow method for optimal K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### Hierarchical Clustering

Builds a tree of clusters by successively merging or splitting groups.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
labels = hierarchical.fit_predict(X)

# Create linkage matrix for dendrogram
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

**Linkage Methods:**

- **Ward**: Minimizes variance within clusters
- **Complete**: Maximum distance between clusters
- **Average**: Average distance between clusters
- **Single**: Minimum distance between clusters

### DBSCAN (Density-Based Spatial Clustering)

Discovers clusters of arbitrary shape based on density.

```python
from sklearn.cluster import DBSCAN

# DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# Check number of clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

**DBSCAN Parameters:**

- **eps**: Maximum distance between points to be considered neighbors
- **min_samples**: Minimum points to form a core point

### Spectral Clustering

Uses eigenvalues of similarity matrix to perform dimensionality reduction before clustering.

```python
from sklearn.cluster import SpectralClustering

# Spectral clustering
spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')
labels = spectral.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Clustering')
plt.show()
```

### Gaussian Mixture Models (GMM)

Models data as mixture of Gaussian distributions.

```python
from sklearn.mixture import GaussianMixture

# GMM clustering
gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')
plt.show()
```

---

## 📉 Dimensionality Reduction

### Principal Component Analysis (PCA)

Reduces dimensionality by finding directions of maximum variance.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize results
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization')
plt.show()

# Explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
```

**PCA Applications:**

- **Data visualization**: Reduce to 2-3 dimensions for plotting
- **Noise reduction**: Remove low-variance components
- **Feature engineering**: Create new features from principal components
- **Compression**: Reduce storage and computation requirements

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction for visualization.

```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('t-SNE Visualization')
plt.show()
```

**t-SNE Characteristics:**

- **Non-linear**: Preserves local structure better than PCA
- **Stochastic**: Results vary with random seed
- **Computational cost**: O(n²) complexity
- **Parameter sensitivity**: Perplexity affects results

### UMAP (Uniform Manifold Approximation and Projection)

Modern dimensionality reduction with better scalability.

```python
import umap

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title('UMAP Visualization')
plt.show()
```

**UMAP Advantages:**

- **Scalability**: O(n log n) complexity
- **Preserves structure**: Both local and global relationships
- **Flexible**: Works with various distance metrics
- **Reproducible**: Deterministic results

### Autoencoders

Neural networks for unsupervised dimensionality reduction.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Simple autoencoder
input_dim = X.shape[1]
encoding_dim = 2

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=32, verbose=0)

# Get encoded representation
X_encoded = encoder.predict(X)

plt.scatter(X_encoded[:, 0], X_encoded[:, 1])
plt.title('Autoencoder Visualization')
plt.show()
```

---

## 🔗 Association Rule Mining

### Apriori Algorithm

Finds frequent itemsets and generates association rules.

```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample transaction data
transactions = [
    ['bread', 'milk', 'eggs'],
    ['bread', 'diapers', 'beer', 'eggs'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['bread', 'milk', 'diapers', 'beer'],
    ['bread', 'milk', 'diapers', 'cola']
]

# Create transaction matrix
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules)
```

**Association Rule Metrics:**

- **Support**: Frequency of itemset in transactions
- **Confidence**: Probability of consequent given antecedent
- **Lift**: Ratio of observed support to expected support
- **Conviction**: Measure of dependence between items

### Market Basket Analysis Example

```python
# Real-world market basket analysis
def analyze_market_basket(transactions, min_support=0.01, min_confidence=0.5):
    """
    Analyze market basket data for retail insights
    """
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", 
                            min_threshold=min_confidence)
    
    return rules.sort_values('lift', ascending=False)

# Example usage
retail_transactions = [
    ['laptop', 'mouse', 'keyboard'],
    ['laptop', 'mouse', 'headphones'],
    ['laptop', 'keyboard', 'monitor'],
    ['mouse', 'keyboard', 'mousepad'],
    ['laptop', 'mouse', 'keyboard', 'monitor']
]

rules = analyze_market_basket(retail_transactions)
print("Top association rules:")
print(rules.head())
```

---

## 🎯 Density-Based Methods

### Outlier Detection

Identify unusual data points that don't fit the expected pattern.

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Generate data with outliers
X_clean, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.5, random_state=0)
outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_with_outliers = np.vstack([X_clean, outliers])

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_labels = iso_forest.fit_predict(X_with_outliers)

# Local Outlier Factor
lof = LocalOutlierFactor(contamination=0.1)
lof_labels = lof.fit_predict(X_with_outliers)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Isolation Forest
ax1.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], 
           c=iso_labels, cmap='viridis')
ax1.set_title('Isolation Forest')

# Local Outlier Factor
ax2.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], 
           c=lof_labels, cmap='viridis')
ax2.set_title('Local Outlier Factor')

plt.tight_layout()
plt.show()
```

### Anomaly Detection Applications

```python
# Credit card fraud detection example
def detect_fraud(transactions, contamination=0.01):
    """
    Detect fraudulent credit card transactions
    """
    # Features: amount, time, location, merchant_category, etc.
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    fraud_labels = iso_forest.fit_predict(transactions)
    
    # Return suspicious transactions
    suspicious_indices = np.where(fraud_labels == -1)[0]
    return suspicious_indices, fraud_labels

# Network intrusion detection
def detect_intrusion(network_data, contamination=0.05):
    """
    Detect network intrusions using anomaly detection
    """
    lof = LocalOutlierFactor(contamination=contamination)
    intrusion_labels = lof.fit_predict(network_data)
    
    return intrusion_labels
```

---

## 🧠 Self-Supervised Learning

### Contrastive Learning

Learn representations by comparing similar and dissimilar pairs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

def contrastive_loss(embeddings, labels, temperature=0.5):
    """
    Compute contrastive loss for self-supervised learning
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Create positive mask
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    # Compute loss
    exp_sim = torch.exp(similarity_matrix / temperature)
    log_prob = similarity_matrix / temperature - torch.log(exp_sim.sum(1, keepdim=True))
    
    mean_log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
    loss = -mean_log_prob.mean()
    
    return loss

# Example usage
def train_contrastive_model(X, labels, epochs=100):
    """
    Train a contrastive learning model
    """
    model = ContrastiveLearningModel(X.shape[1], 128, 64)
    optimizer = torch.optim.Adam(model.parameters())
    
    X_tensor = torch.FloatTensor(X)
    labels_tensor = torch.LongTensor(labels)
    
    for epoch in range(epochs):
        embeddings = model(X_tensor)
        loss = contrastive_loss(embeddings, labels_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model
```

### Autoencoder Variants

```python
# Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    """
    VAE loss: reconstruction + KL divergence
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss
```

---

## 🚀 Practical Applications

### Customer Segmentation

```python
def customer_segmentation(customer_data, n_clusters=5):
    """
    Segment customers based on behavior and demographics
    """
    # Preprocess data
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data)
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    segments = kmeans.fit_predict(customer_data_scaled)
    
    # Analyze segments
    customer_data['segment'] = segments
    
    segment_analysis = customer_data.groupby('segment').mean()
    
    return segments, segment_analysis

# Example customer features
customer_features = [
    'age', 'income', 'spending_score', 'frequency', 
    'recency', 'total_purchases', 'avg_order_value'
]

# Usage
# segments, analysis = customer_segmentation(customer_df[customer_features])
```

### Recommendation Systems

```python
def collaborative_filtering(user_item_matrix, n_components=50):
    """
    Build recommendation system using matrix factorization
    """
    # Apply SVD for dimensionality reduction
    from sklearn.decomposition import TruncatedSVD
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T
    
    # Reconstruct ratings
    predicted_ratings = user_factors @ item_factors.T
    
    return predicted_ratings, user_factors, item_factors

def content_based_filtering(item_features, user_profile):
    """
    Content-based filtering using item features
    """
    # Calculate similarity between user profile and items
    similarities = np.dot(item_features, user_profile) / (
        np.linalg.norm(item_features, axis=1) * np.linalg.norm(user_profile)
    )
    
    return similarities
```

### Image Clustering

```python
def cluster_images(image_features, n_clusters=10):
    """
    Cluster images based on extracted features
    """
    # Apply dimensionality reduction
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(image_features)
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_reduced)
    
    return clusters

def visualize_image_clusters(images, clusters, n_samples=5):
    """
    Visualize image clusters
    """
    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(15, 3*n_clusters))
    
    for i in range(n_clusters):
        cluster_images = images[clusters == i]
        for j in range(min(n_samples, len(cluster_images))):
            axes[i, j].imshow(cluster_images[j])
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'Cluster {i}')
    
    plt.tight_layout()
    plt.show()
```

### Document Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_documents(documents, n_clusters=5):
    """
    Cluster documents using TF-IDF features
    """
    # Extract features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    features = vectorizer.fit_transform(documents)
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Analyze clusters
    cluster_docs = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_docs:
            cluster_docs[cluster] = []
        cluster_docs[cluster].append(documents[i])
    
    return clusters, cluster_docs

# Example usage
documents = [
    "Machine learning algorithms for data analysis",
    "Deep learning neural networks and AI",
    "Data science and statistical analysis",
    "Artificial intelligence and machine learning",
    "Big data processing and analytics",
    "Natural language processing techniques",
    "Computer vision and image recognition",
    "Reinforcement learning and robotics"
]

clusters, cluster_docs = cluster_documents(documents)
for cluster_id, docs in cluster_docs.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
```

---

## 🔬 Advanced Topics

### Spectral Clustering for Complex Networks

```python
import networkx as nx
from sklearn.cluster import SpectralClustering

def community_detection(adjacency_matrix, n_communities=3):
    """
    Detect communities in networks using spectral clustering
    """
    # Apply spectral clustering
    spectral = SpectralClustering(n_clusters=n_communities, 
                                 affinity='precomputed',
                                 random_state=42)
    communities = spectral.fit_predict(adjacency_matrix)
    
    return communities

# Example: Social network analysis
def analyze_social_network(edge_list):
    """
    Analyze social network structure
    """
    # Create graph
    G = nx.from_edgelist(edge_list)
    
    # Get adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    
    # Detect communities
    communities = community_detection(adjacency_matrix)
    
    # Analyze network properties
    network_analysis = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'communities': communities
    }
    
    return network_analysis
```

### Time Series Clustering

```python
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

def cluster_time_series(time_series_data, n_clusters=5):
    """
    Cluster time series using dynamic time warping
    """
    # Calculate pairwise distances
    distances = pdist(time_series_data, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Apply hierarchical clustering
    linkage_matrix = linkage(distances, method='ward')
    
    # Get clusters
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    return clusters, linkage_matrix

def visualize_time_series_clusters(time_series, clusters):
    """
    Visualize time series clusters
    """
    unique_clusters = np.unique(clusters)
    
    fig, axes = plt.subplots(len(unique_clusters), 1, figsize=(12, 3*len(unique_clusters)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_series = time_series[clusters == cluster]
        
        for series in cluster_series:
            axes[i].plot(series, alpha=0.5)
        
        axes[i].set_title(f'Cluster {cluster}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
```

### Multi-View Clustering

```python
def multi_view_clustering(views, n_clusters=3):
    """
    Cluster data with multiple views/representations
    """
    from sklearn.cluster import SpectralClustering
    
    # Combine views (simple concatenation)
    combined_features = np.hstack(views)
    
    # Apply spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                 affinity='nearest_neighbors',
                                 random_state=42)
    clusters = spectral.fit_predict(combined_features)
    
    return clusters

# Example: Multi-modal data clustering
def cluster_multimodal_data(text_features, image_features, n_clusters=5):
    """
    Cluster data with both text and image features
    """
    # Normalize features
    scaler_text = StandardScaler()
    scaler_image = StandardScaler()
    
    text_scaled = scaler_text.fit_transform(text_features)
    image_scaled = scaler_image.fit_transform(image_features)
    
    # Combine features
    combined = np.hstack([text_scaled, image_scaled])
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(combined)
    
    return clusters
```

---

## 🛠️ Tools and Frameworks

### Popular Libraries

```python
# Core ML libraries
import sklearn
import scipy
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Deep learning
import torch
import tensorflow as tf

# Specialized clustering
import hdbscan
import umap
import tsne

# Network analysis
import networkx as nx

# Time series
import tslearn
import pyts
```

### Evaluation Metrics

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score

def evaluate_clustering(X, labels):
    """
    Comprehensive clustering evaluation
    """
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }
    
    return metrics

def compare_clustering_algorithms(X, algorithms):
    """
    Compare different clustering algorithms
    """
    results = {}
    
    for name, algorithm in algorithms.items():
        labels = algorithm.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        results[name] = metrics
    
    return pd.DataFrame(results).T
```

### Best Practices

```python
# Data preprocessing checklist
def preprocess_for_clustering(data):
    """
    Prepare data for clustering analysis
    """
    # 1. Handle missing values
    data = data.dropna()  # or impute
    
    # 2. Scale features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 3. Remove outliers (optional)
    iso_forest = IsolationForest(contamination=0.1)
    outlier_labels = iso_forest.fit_predict(data_scaled)
    data_clean = data_scaled[outlier_labels == 1]
    
    # 4. Reduce dimensionality if needed
    if data_clean.shape[1] > 50:
        pca = PCA(n_components=50)
        data_reduced = pca.fit_transform(data_clean)
    else:
        data_reduced = data_clean
    
    return data_reduced

# Parameter tuning
def tune_clustering_parameters(X, algorithm='kmeans'):
    """
    Find optimal parameters for clustering
    """
    if algorithm == 'kmeans':
        inertias = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        from kneed import KneeLocator
        kn = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
        optimal_k = kn.elbow
        
        return optimal_k
    
    elif algorithm == 'dbscan':
        # Grid search for eps and min_samples
        best_score = -1
        best_params = {}
        
        for eps in [0.1, 0.3, 0.5, 0.7, 1.0]:
            for min_samples in [3, 5, 7, 10]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                if len(set(labels)) > 1:  # At least 2 clusters
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        
        return best_params
```

---

## 🎯 Exercises and Projects

### Exercise 1: Customer Segmentation

```python
# Create synthetic customer data
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    'age': np.random.normal(35, 10, n_customers),
    'income': np.random.normal(50000, 20000, n_customers),
    'spending_score': np.random.uniform(0, 100, n_customers),
    'frequency': np.random.poisson(5, n_customers),
    'recency': np.random.exponential(30, n_customers)
})

# Task: Segment customers into 5 groups
# 1. Preprocess the data
# 2. Apply K-means clustering
# 3. Analyze each segment
# 4. Create customer personas
```

### Exercise 2: Document Clustering

```python
# Sample documents
documents = [
    "Machine learning algorithms for predictive modeling",
    "Deep learning neural networks for image recognition",
    "Data science techniques for business analytics",
    "Artificial intelligence applications in healthcare",
    "Natural language processing for text analysis",
    "Computer vision algorithms for object detection",
    "Reinforcement learning for autonomous systems",
    "Big data processing with distributed computing",
    "Statistical analysis and hypothesis testing",
    "Database management and SQL optimization"
]

# Task: Cluster documents by topic
# 1. Extract TF-IDF features
# 2. Apply hierarchical clustering
# 3. Visualize dendrogram
# 4. Identify main topics
```

### Exercise 3: Anomaly Detection

```python
# Generate data with anomalies
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.uniform(-5, 5, (50, 2))
data_with_anomalies = np.vstack([normal_data, anomaly_data])

# Task: Detect anomalies
# 1. Apply Isolation Forest
# 2. Apply Local Outlier Factor
# 3. Compare results
# 4. Visualize detected anomalies
```

### Project: Recommendation System

```python
# Build a movie recommendation system
def build_movie_recommender(ratings_matrix, movie_features=None):
    """
    Build a hybrid recommendation system
    """
    # Collaborative filtering
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(ratings_matrix)
    item_factors = svd.components_.T
    
    # Content-based filtering (if features available)
    if movie_features is not None:
        # Combine collaborative and content-based
        pass
    
    return user_factors, item_factors

def recommend_movies(user_id, user_factors, item_factors, n_recommendations=5):
    """
    Generate movie recommendations for a user
    """
    user_vector = user_factors[user_id]
    scores = np.dot(item_factors, user_vector)
    
    # Get top recommendations
    top_indices = np.argsort(scores)[::-1][:n_recommendations]
    
    return top_indices, scores[top_indices]
```

### Advanced Project: Multi-Modal Clustering

```python
def multimodal_clustering(text_data, image_data, n_clusters=5):
    """
    Cluster data with both text and image features
    """
    # Extract text features
    text_vectorizer = TfidfVectorizer(max_features=1000)
    text_features = text_vectorizer.fit_transform(text_data)
    
    # Extract image features (pretrained CNN)
    # image_features = extract_image_features(image_data)
    
    # Combine features
    combined_features = np.hstack([text_features.toarray(), image_features])
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(combined_features)
    
    return clusters
```

---

## 📈 2025 Trends and Future Directions

### Self-Supervised Learning Advances

- **Contrastive learning**: SimCLR, MoCo, BYOL
- **Masked modeling**: BERT-style pretraining for images
- **Multi-modal self-supervision**: CLIP, DALL-E, GPT-4V

### Scalable Clustering

- **Streaming algorithms**: Process data incrementally
- **Distributed clustering**: Scale to big data
- **GPU acceleration**: Fast clustering on modern hardware

### Interpretable Clustering

- **Explainable AI**: Understand cluster assignments
- **Feature importance**: Identify key clustering factors
- **Interactive visualization**: Explore clusters dynamically

### Domain-Specific Applications

- **Healthcare**: Patient stratification, drug discovery
- **Finance**: Risk assessment, fraud detection
- **Manufacturing**: Quality control, predictive maintenance
- **Retail**: Customer behavior, inventory optimization

---

## 🎓 Summary

Unsupervised learning is a powerful tool for discovering hidden patterns in data. Key takeaways:

1. **Clustering algorithms** help group similar data points
2. **Dimensionality reduction** preserves structure while reducing complexity
3. **Association rules** find relationships between variables
4. **Anomaly detection** identifies unusual patterns
5. **Self-supervised learning** learns representations without labels

### Next Steps

- **Practice**: Implement clustering algorithms from scratch
- **Explore**: Try different algorithms on your datasets
- **Specialize**: Focus on applications in your domain
- **Stay current**: Follow research in self-supervised learning

---

*"The goal of unsupervised learning is to find the hidden structure in data - to discover what we didn't know we were looking for."*

**Ready to explore patterns in your data? Start with clustering and work your way up to advanced techniques!** 