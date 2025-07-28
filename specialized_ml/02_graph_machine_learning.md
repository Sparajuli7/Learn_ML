# Graph Machine Learning

## Overview
Graph Machine Learning (GML) is a specialized field that deals with data structured as graphs, where entities are represented as nodes and relationships as edges. It's essential for social networks, molecular biology, recommendation systems, and knowledge graphs.

## Graph Fundamentals

### Graph Representation
```python
import networkx as nx
import numpy as np

# Create a simple graph
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4)])

# Adjacency matrix
adj_matrix = nx.adjacency_matrix(G).toarray()

# Node features
node_features = np.random.rand(4, 10)  # 4 nodes, 10 features each
```

### Graph Properties
- **Degree**: Number of connections per node
- **Clustering Coefficient**: Measure of local clustering
- **Centrality**: Importance of nodes in the network
- **Community Detection**: Finding groups of densely connected nodes

## Traditional Graph Algorithms

### 1. PageRank
```python
def pagerank(graph, damping=0.85, max_iter=100):
    n = len(graph)
    ranks = np.ones(n) / n
    
    for _ in range(max_iter):
        new_ranks = (1 - damping) / n + damping * np.dot(graph, ranks)
        if np.allclose(ranks, new_ranks):
            break
        ranks = new_ranks
    
    return ranks
```

### 2. Community Detection
```python
from community import community_louvain

# Louvain method for community detection
communities = community_louvain.best_partition(G)
```

### 3. Shortest Path
```python
# Dijkstra's algorithm
shortest_paths = nx.single_source_dijkstra_path(G, source=1)
```

## Graph Neural Networks (GNNs)

### 1. Graph Convolutional Networks (GCN)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        # First graph convolution layer
        x = F.relu(self.conv1(torch.mm(adj, x)))
        # Second graph convolution layer
        x = self.conv2(torch.mm(adj, x))
        return F.log_softmax(x, dim=1)
```

### 2. Graph Attention Networks (GAT)
```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=8, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Linear transformation for each attention head
        self.W = nn.Linear(input_dim, output_dim * n_heads)
        self.attention = nn.Linear(output_dim * 2, 1)
        
    def forward(self, x, adj):
        batch_size, n_nodes, input_dim = x.size()
        
        # Linear transformation
        x = self.W(x)  # [batch_size, n_nodes, output_dim * n_heads]
        x = x.view(batch_size, n_nodes, self.n_heads, -1)
        x = x.transpose(1, 2)  # [batch_size, n_heads, n_nodes, output_dim]
        
        # Compute attention scores
        attention_input = torch.cat([
            x.repeat_interleave(n_nodes, dim=2),
            x.repeat(1, 1, n_nodes, 1)
        ], dim=-1)
        attention_input = attention_input.view(batch_size, self.n_heads, n_nodes, n_nodes, -1)
        
        attention_scores = self.attention(attention_input).squeeze(-1)
        attention_scores = attention_scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attention_scores, x)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, n_nodes, -1)
        
        return F.elu(out)
```

### 3. GraphSAGE
```python
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.aggregator = aggregator
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## Message Passing Neural Networks (MPNN)

### Basic MPNN Implementation
```python
class MPNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MPNNLayer, self).__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, node_features, edge_features, edge_index):
        # Message passing
        row, col = edge_index
        messages = torch.cat([
            node_features[row],
            node_features[col],
            edge_features
        ], dim=1)
        messages = self.message_mlp(messages)
        
        # Aggregate messages
        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.index_add_(0, row, messages)
        
        # Update node features
        updated_features = torch.cat([node_features, aggregated_messages], dim=1)
        updated_features = self.update_mlp(updated_features)
        
        return updated_features
```

## Graph Pooling

### 1. Top-K Pooling
```python
class TopKPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(TopKPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = nn.Linear(in_channels, 1)
    
    def forward(self, x, edge_index, batch=None):
        # Compute node scores
        scores = self.score_layer(x).squeeze()
        
        # Select top-k nodes
        num_nodes = x.size(0)
        k = int(self.ratio * num_nodes)
        _, indices = torch.topk(scores, k)
        
        # Filter nodes and edges
        x = x[indices]
        edge_index = edge_index[:, 
                               torch.isin(edge_index[0], indices) & 
                               torch.isin(edge_index[1], indices)]
        
        return x, edge_index, batch[indices] if batch is not None else None
```

### 2. DiffPool
```python
class DiffPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, pooling_ratio):
        super(DiffPool, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.assign_matrix = nn.Linear(in_channels, int(in_channels * pooling_ratio))
    
    def forward(self, x, edge_index, batch=None):
        # Compute assignment matrix
        S = torch.softmax(self.assign_matrix(x), dim=0)
        
        # Compute new node features
        x_new = torch.mm(S.t(), x)
        
        # Compute new adjacency matrix
        A = torch.sparse_coo_tensor(edge_index, 
                                   torch.ones(edge_index.size(1)),
                                   size=(x.size(0), x.size(0))).to_dense()
        A_new = torch.mm(torch.mm(S.t(), A), S)
        
        return x_new, A_new
```

## Graph Classification

### 1. Graph-Level Pooling
```python
class GraphLevelPooling(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim):
        super(GraphLevelPooling, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
```

### 2. Graph Isomorphism Network (GIN)
```python
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
```

## Applications

### 1. Social Network Analysis
```python
def social_network_analysis(graph, node_features):
    # Node classification
    model = GCN(input_dim=node_features.shape[1], 
                hidden_dim=64, 
                output_dim=7)  # 7 classes
    
    # Link prediction
    def link_prediction_loss(pos_edge_index, neg_edge_index, node_embeddings):
        pos_score = torch.sum(node_embeddings[pos_edge_index[0]] * 
                             node_embeddings[pos_edge_index[1]], dim=1)
        neg_score = torch.sum(node_embeddings[neg_edge_index[0]] * 
                             node_embeddings[neg_edge_index[1]], dim=1)
        
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + \
               F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return loss
```

### 2. Molecular Property Prediction
```python
class MolecularGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(MolecularGNN, self).__init__()
        self.conv1 = MPNNLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = MPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.pool = global_mean_pool
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features, edge_features, edge_index, batch):
        # Message passing
        x = self.conv1(node_features, edge_features, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_features, edge_index)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Property prediction
        x = self.classifier(x)
        return x
```

### 3. Knowledge Graph Completion
```python
class KnowledgeGraphEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(KnowledgeGraphEmbedding, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        # TransE scoring function
        score = h + r - t
        return torch.norm(score, p=2, dim=1)
```

## Training and Evaluation

### 1. Node Classification Training
```python
def train_node_classifier(model, data, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            val_acc = evaluate(model, data)
            print(f'Epoch {epoch}: Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}')
```

### 2. Graph Classification Training
```python
def train_graph_classifier(model, train_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}')
```

## Advanced Techniques

### 1. Graph Contrastive Learning
```python
class GraphCL(nn.Module):
    def __init__(self, encoder, projection_head):
        super(GraphCL, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        # Encode two views
        z1 = self.encoder(x1, edge_index1, batch1)
        z2 = self.encoder(x2, edge_index2, batch2)
        
        # Project to representation space
        z1 = self.projection_head(z1)
        z2 = self.projection_head(z2)
        
        return z1, z2
```

### 2. Graph Autoencoders
```python
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
```

## Tools and Libraries

- **PyTorch Geometric**: Main library for GNNs
- **NetworkX**: Graph manipulation and algorithms
- **DGL**: Deep Graph Library
- **Spektral**: Keras-based GNN library
- **StellarGraph**: Graph machine learning library

## Best Practices

1. **Graph Preprocessing**: Normalize node features and handle missing edges
2. **Architecture Choice**: Select appropriate GNN architecture for your task
3. **Regularization**: Use dropout and batch normalization
4. **Evaluation**: Use appropriate metrics for graph tasks
5. **Scalability**: Consider graph sampling for large graphs

## Next Steps

1. **Heterogeneous Graphs**: Handle different types of nodes and edges
2. **Temporal Graphs**: Model time-evolving graphs
3. **Graph Generation**: Generate realistic graphs
4. **Graph Explainability**: Understand GNN predictions
5. **Large-Scale Graphs**: Scale to billion-node graphs

---

*Graph Machine Learning combines the power of neural networks with the rich structure of graphs, enabling powerful models for relational data. From social networks to molecular structures, GNNs are revolutionizing how we analyze connected data.* 