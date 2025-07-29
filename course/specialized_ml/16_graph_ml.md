# Graph Machine Learning: Networks, GNNs, and Beyond

*"Everything is connected - understanding relationships through graph intelligence"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Graph Theory Foundations](#graph-theory-foundations)
3. [Graph Neural Networks](#graph-neural-networks)
4. [Practical Implementation](#practical-implementation)
5. [Real-World Applications](#real-world-applications)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction

Graph Machine Learning represents one of the most exciting frontiers in AI, where we model complex relationships and interactions between entities. From social networks to molecular structures, from recommendation systems to knowledge graphs, graphs provide a natural way to represent and learn from interconnected data.

### Historical Context

Graph theory dates back to the 18th century with Euler's solution to the Königsberg bridge problem. The field gained momentum in the 20th century with applications in computer science, social network analysis, and biology. The 2010s saw the emergence of Graph Neural Networks (GNNs), and the 2020s have brought transformer-based graph models and large-scale graph applications.

### Current State (2025)

- **Large-Scale Graphs**: Billion-node graphs with efficient processing
- **Heterogeneous Graphs**: Multiple node and edge types
- **Temporal Graphs**: Dynamic networks evolving over time
- **Multimodal Graphs**: Integrating text, images, and structured data
- **Graph Foundation Models**: Pre-trained models for graph tasks
- **Graph Reasoning**: Complex logical inference on graphs

---

## 🧮 Graph Theory Foundations

### Basic Definitions

A graph G = (V, E) consists of:
- **V**: Set of vertices (nodes)
- **E**: Set of edges (connections)

**Types of Graphs**:
- **Undirected**: Edges have no direction
- **Directed**: Edges have direction (arrows)
- **Weighted**: Edges have numerical weights
- **Heterogeneous**: Multiple node/edge types

### Graph Representations

**Adjacency Matrix**:
```
A[i,j] = 1 if edge exists between nodes i and j, 0 otherwise
```

**Adjacency List**:
```
For each node, list its neighbors
```

**Edge List**:
```
List of (source, target) pairs
```

### Graph Properties

**Degree**: Number of edges connected to a node
```
deg(v) = Σᵢ A[v,i]
```

**Density**: Ratio of actual edges to maximum possible edges
```
density = |E| / (|V| × (|V|-1)/2)
```

**Clustering Coefficient**: Measure of local clustering
```
C(v) = 2 × triangles(v) / (deg(v) × (deg(v)-1))
```

### Centrality Measures

**Degree Centrality**:
```
C_degree(v) = deg(v) / (|V| - 1)
```

**Betweenness Centrality**:
```
C_betweenness(v) = Σₛₜ σₛₜ(v) / σₛₜ
```

**Eigenvector Centrality**:
```
C_eigenvector(v) = (1/λ) × Σᵤ A[v,u] × C_eigenvector(u)
```

### Graph Algorithms

**Breadth-First Search (BFS)**:
```python
def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**Depth-First Search (DFS)**:
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

**Shortest Path (Dijkstra)**:
```python
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
            
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distances
```

---

## 🧠 Graph Neural Networks

### Message Passing Framework

GNNs operate through message passing between nodes:

**Message Function**:
```
mᵥᵤ = M(hᵥ, hᵤ, eᵥᵤ)
```

**Aggregation Function**:
```
hᵥ' = U(hᵥ, AGG({mᵥᵤ : u ∈ N(v)}))
```

Where:
- `hᵥ`: Node embedding
- `eᵥᵤ`: Edge features
- `N(v)`: Neighbors of node v

### Graph Convolutional Networks (GCN)

**GCN Layer**:
```
H⁽ˡ⁺¹⁾ = σ(D̃⁻¹/² Ã D̃⁻¹/² H⁽ˡ⁾ W⁽ˡ⁾)
```

Where:
- `Ã = A + I` (adjacency matrix with self-loops)
- `D̃` is the degree matrix of `Ã`
- `W⁽ˡ⁾` are learnable weights

**Implementation**:
```python
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GCNLayer, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        H, A = inputs
        # Normalize adjacency matrix
        D = tf.reduce_sum(A, axis=1)
        D_inv_sqrt = tf.pow(D, -0.5)
        D_inv_sqrt = tf.where(tf.math.is_inf(D_inv_sqrt), 0., D_inv_sqrt)
        D_inv_sqrt = tf.linalg.diag(D_inv_sqrt)
        
        # Graph convolution
        H_new = tf.matmul(D_inv_sqrt, tf.matmul(A, tf.matmul(D_inv_sqrt, H)))
        H_new = tf.matmul(H_new, self.W)
        
        return tf.nn.relu(H_new)
```

### Graph Attention Networks (GAT)

**Attention Mechanism**:
```
αᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ || Whⱼ]))
```

**GAT Layer**:
```
hᵢ' = σ(Σⱼ αᵢⱼ Whⱼ)
```

**Multi-head Attention**:
```
hᵢ' = σ(1/K Σᵏ αᵢⱼᵏ Wᵏhⱼ)
```

**Implementation**:
```python
class GATLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads=8):
        super(GATLayer, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[1], self.output_dim * self.num_heads),
            initializer='glorot_uniform',
            trainable=True
        )
        self.attention = self.add_weight(
            shape=(2 * self.output_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        H, A = inputs
        batch_size = tf.shape(H)[0]
        num_nodes = tf.shape(H)[1]
        
        # Linear transformation
        H_transformed = tf.matmul(H, self.W)
        H_transformed = tf.reshape(H_transformed, 
                                 [batch_size, num_nodes, self.num_heads, -1])
        
        # Prepare for attention
        H_i = tf.expand_dims(H_transformed, 2)  # [batch, nodes, 1, heads, dim]
        H_j = tf.expand_dims(H_transformed, 1)  # [batch, 1, nodes, heads, dim]
        
        # Concatenate for attention
        H_concat = tf.concat([H_i, H_j], axis=-1)
        
        # Calculate attention scores
        attention_scores = tf.matmul(H_concat, self.attention)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        
        # Apply mask for non-existent edges
        mask = tf.expand_dims(A, axis=-1)
        attention_scores = tf.where(mask > 0, attention_scores, -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=2)
        
        # Apply attention
        H_attended = tf.matmul(attention_weights, H_transformed)
        H_attended = tf.reshape(H_attended, [batch_size, num_nodes, -1])
        
        return tf.nn.relu(H_attended)
```

### GraphSAGE

**Neighbor Sampling**:
```
N(v) = SAMPLE(neighbors(v), k)
```

**Aggregation Functions**:
- **Mean**: `AGG = mean({hᵤ : u ∈ N(v)})`
- **Max**: `AGG = max({hᵤ : u ∈ N(v)})`
- **LSTM**: `AGG = LSTM([hᵤ : u ∈ N(v)])`

**GraphSAGE Layer**:
```
hᵥ' = σ(W · CONCAT(hᵥ, AGG({hᵤ : u ∈ N(v)})))
```

### Graph Transformer

**Self-Attention on Graphs**:
```
Q = H⁽ˡ⁾WQ, K = H⁽ˡ⁾WK, V = H⁽ˡ⁾WV
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

**Positional Encoding**:
```
PE(v) = [sin(pos/10000^(2i/d)), cos(pos/10000^(2i/d))]
```

---

## 💻 Practical Implementation

### Setting Up the Environment

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
```

### Graph Data Structures

```python
class GraphDataset:
    def __init__(self, nodes, edges, features=None, labels=None):
        self.nodes = nodes
        self.edges = edges
        self.features = features
        self.labels = labels
        
    def to_networkx(self):
        """Convert to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G
    
    def to_pytorch_geometric(self):
        """Convert to PyTorch Geometric format"""
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(self.features, dtype=torch.float) if self.features is not None else None
        y = torch.tensor(self.labels, dtype=torch.long) if self.labels is not None else None
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def get_adjacency_matrix(self):
        """Get adjacency matrix"""
        n = len(self.nodes)
        adj = np.zeros((n, n))
        for i, j in self.edges:
            adj[i, j] = 1
            adj[j, i] = 1  # Undirected graph
        return adj

def create_synthetic_graph(num_nodes=100, edge_prob=0.1):
    """Create synthetic graph for testing"""
    # Create random graph
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    # Extract components
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    # Create random features
    features = np.random.randn(num_nodes, 16)
    
    # Create random labels (for node classification)
    labels = np.random.randint(0, 7, num_nodes)
    
    return GraphDataset(nodes, edges, features, labels)
```

### GCN Implementation

```python
class GCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(hidden_dim)
        self.gcn2 = GCNLayer(output_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, inputs, training=False):
        H, A = inputs
        H = self.gcn1([H, A])
        H = self.dropout(H, training=training)
        H = self.gcn2([H, A])
        return H

class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GCNLayer, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        H, A = inputs
        # Add self-loops
        A = A + tf.eye(tf.shape(A)[0])
        
        # Normalize adjacency matrix
        D = tf.reduce_sum(A, axis=1)
        D_inv_sqrt = tf.pow(D, -0.5)
        D_inv_sqrt = tf.where(tf.math.is_inf(D_inv_sqrt), 0., D_inv_sqrt)
        D_inv_sqrt = tf.linalg.diag(D_inv_sqrt)
        
        # Graph convolution
        H_new = tf.matmul(D_inv_sqrt, tf.matmul(A, tf.matmul(D_inv_sqrt, H)))
        H_new = tf.matmul(H_new, self.W)
        
        return tf.nn.relu(H_new)
```

### GAT Implementation

```python
class GAT(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(hidden_dim, num_heads)
        self.gat2 = GATLayer(output_dim, 1)  # Single head for output
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, inputs, training=False):
        H, A = inputs
        H = self.gat1([H, A])
        H = self.dropout(H, training=training)
        H = self.gat2([H, A])
        return H

class GATLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads=8):
        super(GATLayer, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[1], self.output_dim * self.num_heads),
            initializer='glorot_uniform',
            trainable=True
        )
        self.attention = self.add_weight(
            shape=(2 * self.output_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        H, A = inputs
        num_nodes = tf.shape(H)[0]
        
        # Linear transformation
        H_transformed = tf.matmul(H, self.W)
        H_transformed = tf.reshape(H_transformed, 
                                 [num_nodes, self.num_heads, -1])
        
        # Prepare for attention
        H_i = tf.expand_dims(H_transformed, 1)  # [nodes, 1, heads, dim]
        H_j = tf.expand_dims(H_transformed, 0)  # [1, nodes, heads, dim]
        
        # Concatenate for attention
        H_concat = tf.concat([H_i, H_j], axis=-1)
        
        # Calculate attention scores
        attention_scores = tf.matmul(H_concat, self.attention)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        
        # Apply mask for non-existent edges
        mask = tf.expand_dims(A, axis=-1)
        attention_scores = tf.where(mask > 0, attention_scores, -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention
        H_attended = tf.matmul(attention_weights, H_transformed)
        H_attended = tf.reshape(H_attended, [num_nodes, -1])
        
        return tf.nn.relu(H_attended)
```

### Graph Classification

```python
class GraphClassifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(GraphClassifier, self).__init__()
        self.gcn1 = GCNLayer(hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
    def call(self, inputs, training=False):
        H, A = inputs
        H = self.gcn1([H, A])
        H = tf.nn.relu(H)
        H = self.gcn2([H, A])
        H = tf.nn.relu(H)
        
        # Global pooling
        H = tf.expand_dims(H, 0)  # Add batch dimension
        H = self.pool(H)
        
        # Classification
        output = self.classifier(H, training=training)
        return output
```

### Node Classification Example

```python
def node_classification_example():
    """Complete example of node classification with GCN"""
    
    # Create synthetic graph
    dataset = create_synthetic_graph(num_nodes=200, edge_prob=0.1)
    
    # Split data
    num_nodes = len(dataset.nodes)
    train_mask = np.random.choice([True, False], num_nodes, p=[0.8, 0.2])
    val_mask = ~train_mask
    
    # Prepare data
    H = tf.constant(dataset.features, dtype=tf.float32)
    A = tf.constant(dataset.get_adjacency_matrix(), dtype=tf.float32)
    labels = tf.constant(dataset.labels, dtype=tf.int32)
    
    # Create model
    model = GCN(
        input_dim=dataset.features.shape[1],
        hidden_dim=64,
        output_dim=7,  # num_classes
        dropout=0.5
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = model([H, A], training=True)
            loss = loss_fn(labels[train_mask], logits[train_mask])
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if epoch % 20 == 0:
            # Validation
            val_logits = model([H, A], training=False)
            val_loss = loss_fn(labels[val_mask], val_logits[val_mask])
            val_acc = tf.reduce_mean(
                tf.cast(tf.argmax(val_logits[val_mask], axis=1) == labels[val_mask], tf.float32)
            )
            print(f"Epoch {epoch}: Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Final evaluation
    test_logits = model([H, A], training=False)
    test_acc = tf.reduce_mean(
        tf.cast(tf.argmax(test_logits[val_mask], axis=1) == labels[val_mask], tf.float32)
    )
    print(f"Final Test Accuracy: {test_acc:.4f}")

# Run the example
if __name__ == "__main__":
    node_classification_example()
```

### Graph Visualization

```python
def visualize_graph(dataset, node_colors=None, title="Graph Visualization"):
    """Visualize graph with NetworkX"""
    G = dataset.to_networkx()
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    if node_colors is None:
        node_colors = range(len(G.nodes()))
    
    nx.draw(G, pos, 
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=8,
            cmap=plt.cm.Set3)
    
    plt.title(title)
    plt.show()

def plot_embeddings(embeddings, labels, title="Node Embeddings"):
    """Plot 2D embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()
```

---

## 🎯 Real-World Applications

### 1. Social Network Analysis

**Community Detection**:
- Identifying groups of users with similar interests
- Influencer detection and viral content prediction
- Recommendation systems based on social connections

**Information Diffusion**:
- Modeling how information spreads through networks
- Predicting viral content and trends
- Understanding cascade effects

### 2. Biological Networks

**Protein-Protein Interaction Networks**:
- Predicting protein functions
- Drug target identification
- Disease mechanism understanding

**Gene Regulatory Networks**:
- Understanding gene expression patterns
- Identifying regulatory pathways
- Drug discovery applications

### 3. Knowledge Graphs

**Entity Relationship Modeling**:
- Building knowledge bases
- Question answering systems
- Semantic search and recommendation

**Graph Reasoning**:
- Logical inference on knowledge graphs
- Multi-hop reasoning
- Explainable AI applications

### 4. Computer Vision

**Scene Graphs**:
- Understanding visual relationships
- Image captioning and VQA
- Visual reasoning tasks

**Point Cloud Processing**:
- 3D object detection and segmentation
- Autonomous driving applications
- Robotics and manipulation

### 5. Natural Language Processing

**Dependency Parsing**:
- Syntactic analysis of sentences
- Grammar understanding
- Machine translation

**Semantic Role Labeling**:
- Understanding sentence structure
- Information extraction
- Question answering

### 6. Recommendation Systems

**Graph-based Recommendations**:
- Collaborative filtering with graph structure
- Heterogeneous information networks
- Multi-modal recommendations

**Session-based Recommendations**:
- Modeling user behavior sequences
- Next-item prediction
- Real-time recommendations

---

## 🧪 Exercises and Projects

### Beginner Exercises

1. **Graph Construction**
   ```python
   # Build graphs from different data sources
   # Social network from friendship data
   # Citation network from academic papers
   # Protein interaction network
   ```

2. **Centrality Analysis**
   ```python
   # Implement different centrality measures
   # Compare their effectiveness on different networks
   # Visualize centrality distributions
   ```

3. **Community Detection**
   ```python
   # Implement Louvain and Label Propagation algorithms
   # Evaluate community quality using modularity
   # Compare different community detection methods
   ```

### Intermediate Projects

1. **Graph Embedding Methods**
   - Implement Node2Vec, DeepWalk, and GraphSAGE
   - Compare embedding quality on downstream tasks
   - Visualize embeddings using t-SNE

2. **Heterogeneous Graph Neural Networks**
   - Build models for graphs with multiple node/edge types
   - Implement attention mechanisms for different edge types
   - Apply to knowledge graph completion tasks

3. **Temporal Graph Networks**
   - Model graphs that evolve over time
   - Implement temporal attention mechanisms
   - Predict future graph structure

### Advanced Projects

1. **Large-Scale Graph Processing**
   - Implement efficient graph sampling strategies
   - Build distributed graph neural networks
   - Handle billion-node graphs

2. **Graph Foundation Models**
   - Pre-train graph neural networks on large datasets
   - Implement few-shot learning for graph tasks
   - Build transfer learning capabilities

3. **Graph Reasoning Systems**
   - Implement logical inference on graphs
   - Build explainable graph neural networks
   - Create multi-hop reasoning systems

### Quiz Questions

1. **Conceptual Questions**
   - What is the difference between a graph and a tree?
   - How do GNNs differ from traditional neural networks?
   - What are the advantages of attention mechanisms in graphs?

2. **Mathematical Questions**
   - Derive the GCN convolution formula
   - Explain the message passing framework
   - Calculate graph centrality measures

3. **Implementation Questions**
   - How would you handle graphs with different sizes?
   - What are the trade-offs between different GNN architectures?
   - How do you validate graph neural networks?

---

## 📖 Further Reading

### Essential Papers

1. **"Semi-Supervised Classification with Graph Convolutional Networks"** - Kipf & Welling (2017)
2. **"Graph Attention Networks"** - Veličković et al. (2018)
3. **"Inductive Representation Learning on Large Graphs"** - Hamilton et al. (2017)
4. **"Attention Is All You Need"** - Vaswani et al. (2017)

### Books

1. **"Networks: An Introduction"** - Newman
2. **"Graph Neural Networks: A Review of Methods and Applications"** - Wu et al.
3. **"Deep Learning on Graphs"** - Zhang et al.

### Online Resources

1. **Libraries**: PyTorch Geometric, DGL, NetworkX
2. **Datasets**: OGB, TUDatasets, SNAP
3. **Competitions**: OGB Leaderboard, Graph ML challenges

### Next Steps

1. **Advanced Topics**: Graph transformers, foundation models
2. **Production Systems**: Large-scale graph processing, distributed training
3. **Domain Specialization**: Bioinformatics, social networks, knowledge graphs

---

## 🎯 Key Takeaways

1. **Graph Structure**: Leverage relational information for better predictions
2. **Message Passing**: Core mechanism for information flow in GNNs
3. **Scalability**: Consider computational efficiency for large graphs
4. **Interpretability**: Graph structure provides natural explainability
5. **Multi-modal**: Integrate graphs with other data types

---

*"In a connected world, understanding relationships is the key to intelligence."*

**Next: [Speech & Audio Processing](specialized_ml/14_speech_audio_processing.md) → ASR, TTS, and music generation**