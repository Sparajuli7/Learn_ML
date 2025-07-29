# Recommender Systems

## Overview
Recommender systems help users discover relevant items by predicting their preferences. This guide covers collaborative filtering, content-based methods, and hybrid approaches for 2025.

## Table of Contents
1. [Recommender System Fundamentals](#recommender-system-fundamentals)
2. [Collaborative Filtering](#collaborative-filtering)
3. [Content-Based Filtering](#content-based-filtering)
4. [Matrix Factorization](#matrix-factorization)
5. [Deep Learning Approaches](#deep-learning-approaches)
6. [Production Systems](#production-systems)

## Recommender System Fundamentals

### Basic Setup
```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecommenderSystem:
    def __init__(self, method: str = 'collaborative'):
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.model = None
        
    def fit(self, user_item_data: pd.DataFrame):
        """Fit the recommender system"""
        if self.method == 'collaborative':
            self._fit_collaborative(user_item_data)
        elif self.method == 'content':
            self._fit_content(user_item_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate recommendations for a user"""
        if self.method == 'collaborative':
            return self._recommend_collaborative(user_id, n_recommendations)
        elif self.method == 'content':
            return self._recommend_content(user_id, n_recommendations)
        else:
            raise ValueError(f"Unknown method: {self.method}")
```

## Collaborative Filtering

### User-Based Collaborative Filtering
```python
class UserBasedCF:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.user_similarity = None
        
    def fit(self, user_item_data: pd.DataFrame):
        """Fit user-based collaborative filtering"""
        # Create user-item matrix
        self.user_item_matrix = user_item_data.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if user_id not in self.user_item_matrix.index:
            return 0.0
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Find similar users
        user_similarities = self.user_similarity[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:self.n_neighbors+1]
        
        # Calculate weighted average
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_idx in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            similarity = user_similarities[similar_user_idx]
            
            if item_id in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[similar_user_id, item_id]
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
        
        if similarity_sum > 0:
            return weighted_sum / similarity_sum
        else:
            return 0.0
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        # Predict ratings for unrated items
        predictions = {}
        for item_id in self.user_item_matrix.columns:
            if item_id not in rated_items:
                pred_rating = self.predict_rating(user_id, item_id)
                predictions[item_id] = pred_rating
        
        # Return top recommendations
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]

# Example usage
# Generate sample data
np.random.seed(42)
n_users, n_items = 100, 50
user_item_data = pd.DataFrame({
    'user_id': np.random.randint(0, n_users, 1000),
    'item_id': np.random.randint(0, n_items, 1000),
    'rating': np.random.randint(1, 6, 1000)
})

# Fit user-based CF
ubcf = UserBasedCF(n_neighbors=5)
ubcf.fit(user_item_data)

# Generate recommendations
recommendations = ubcf.recommend(user_id=0, n_recommendations=5)
print(f"Recommendations for user 0: {recommendations}")
```

### Item-Based Collaborative Filtering
```python
class ItemBasedCF:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.item_similarity = None
        
    def fit(self, user_item_data: pd.DataFrame):
        """Fit item-based collaborative filtering"""
        # Create user-item matrix
        self.user_item_matrix = user_item_data.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Compute item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return 0.0
        
        user_ratings = self.user_item_matrix.loc[user_id]
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Find similar items that user has rated
        item_similarities = self.item_similarity[item_idx]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # Calculate weighted average
        weighted_sum = 0
        similarity_sum = 0
        
        for rated_item_id in rated_items.index:
            rated_item_idx = self.user_item_matrix.columns.get_loc(rated_item_id)
            similarity = item_similarities[rated_item_idx]
            
            if similarity > 0:
                weighted_sum += similarity * rated_items[rated_item_id]
                similarity_sum += similarity
        
        if similarity_sum > 0:
            return weighted_sum / similarity_sum
        else:
            return 0.0
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        # Predict ratings for unrated items
        predictions = {}
        for item_id in self.user_item_matrix.columns:
            if item_id not in rated_items:
                pred_rating = self.predict_rating(user_id, item_id)
                predictions[item_id] = pred_rating
        
        # Return top recommendations
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]

# Example usage
ibcf = ItemBasedCF(n_neighbors=5)
ibcf.fit(user_item_data)

recommendations = ibcf.recommend(user_id=0, n_recommendations=5)
print(f"Item-based recommendations for user 0: {recommendations}")
```

## Content-Based Filtering

### Content-Based Recommender
```python
class ContentBasedRecommender:
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.item_features = None
        self.user_profiles = None
        self.vectorizer = None
        
    def fit(self, item_data: pd.DataFrame, user_item_data: pd.DataFrame):
        """Fit content-based recommender"""
        # Extract item features
        if 'description' in self.feature_columns:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            descriptions = item_data['description'].fillna('')
            tfidf_features = self.vectorizer.fit_transform(descriptions)
            
            # Combine with other features
            other_features = item_data[self.feature_columns].drop('description', axis=1, errors='ignore')
            if not other_features.empty:
                self.item_features = np.hstack([tfidf_features.toarray(), other_features.values])
            else:
                self.item_features = tfidf_features.toarray()
        else:
            self.item_features = item_data[self.feature_columns].values
        
        # Create user profiles
        self._create_user_profiles(user_item_data, item_data)
        
    def _create_user_profiles(self, user_item_data: pd.DataFrame, item_data: pd.DataFrame):
        """Create user profiles based on rated items"""
        self.user_profiles = {}
        
        for user_id in user_item_data['user_id'].unique():
            user_ratings = user_item_data[user_item_data['user_id'] == user_id]
            
            if len(user_ratings) == 0:
                continue
            
            # Get items rated by user
            rated_items = user_ratings['item_id'].values
            ratings = user_ratings['rating'].values
            
            # Find item features for rated items
            item_indices = item_data[item_data['item_id'].isin(rated_items)].index
            if len(item_indices) == 0:
                continue
            
            # Calculate weighted average of item features
            user_features = np.zeros(self.item_features.shape[1])
            total_weight = 0
            
            for item_id, rating in zip(rated_items, ratings):
                item_idx = item_data[item_data['item_id'] == item_id].index
                if len(item_idx) > 0:
                    item_feature_idx = item_idx[0]
                    if item_feature_idx < len(self.item_features):
                        user_features += rating * self.item_features[item_feature_idx]
                        total_weight += rating
            
            if total_weight > 0:
                self.user_profiles[user_id] = user_features / total_weight
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate content-based recommendations"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity with all items
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get items not rated by user
        user_rated_items = set(user_item_data[user_item_data['user_id'] == user_id]['item_id'])
        all_items = set(item_data['item_id'])
        unrated_items = all_items - user_rated_items
        
        # Filter similarities for unrated items
        item_scores = []
        for item_id in unrated_items:
            item_idx = item_data[item_data['item_id'] == item_id].index
            if len(item_idx) > 0:
                item_feature_idx = item_idx[0]
                if item_feature_idx < len(similarities):
                    item_scores.append((item_id, similarities[item_feature_idx]))
        
        # Return top recommendations
        sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]

# Example usage
# Generate sample item data
item_data = pd.DataFrame({
    'item_id': range(50),
    'description': [f'Item {i} description' for i in range(50)],
    'category': np.random.choice(['A', 'B', 'C'], 50),
    'price': np.random.uniform(10, 100, 50)
})

# Fit content-based recommender
cbr = ContentBasedRecommender(feature_columns=['description', 'category'])
cbr.fit(item_data, user_item_data)

recommendations = cbr.recommend(user_id=0, n_recommendations=5)
print(f"Content-based recommendations for user 0: {recommendations}")
```

## Matrix Factorization

### Matrix Factorization Model
```python
class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 50):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # User and item embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Initialize embeddings
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        user_embeddings = self.user_factors(user_ids)
        item_embeddings = self.item_factors(item_ids)
        
        # Dot product
        predictions = torch.sum(user_embeddings * item_embeddings, dim=1)
        return predictions
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long)
            item_tensor = torch.tensor([item_id], dtype=torch.long)
            prediction = self.forward(user_tensor, item_tensor)
            return prediction.item()

class MFRecommender:
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def fit(self, user_item_data: pd.DataFrame, n_epochs: int = 100):
        """Fit matrix factorization model"""
        # Create mappings
        unique_users = user_item_data['user_id'].unique()
        unique_items = user_item_data['item_id'].unique()
        
        for i, user_id in enumerate(unique_users):
            self.user_mapping[user_id] = i
            self.reverse_user_mapping[i] = user_id
        
        for i, item_id in enumerate(unique_items):
            self.item_mapping[item_id] = i
            self.reverse_item_mapping[i] = item_id
        
        # Create model
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.model = MatrixFactorization(n_users, n_items, self.n_factors)
        
        # Prepare training data
        user_indices = [self.user_mapping[uid] for uid in user_item_data['user_id']]
        item_indices = [self.item_mapping[iid] for iid in user_item_data['item_id']]
        ratings = user_item_data['rating'].values
        
        # Convert to tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long)
        item_tensor = torch.tensor(item_indices, dtype=torch.long)
        rating_tensor = torch.tensor(ratings, dtype=torch.float)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            predictions = self.model(user_tensor, item_tensor)
            loss = criterion(predictions, rating_tensor)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate recommendations for a user"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Predict ratings for all items
        predictions = []
        for item_id in self.item_mapping.keys():
            item_idx = self.item_mapping[item_id]
            pred_rating = self.model.predict(user_idx, item_idx)
            predictions.append((item_id, pred_rating))
        
        # Return top recommendations
        sorted_items = sorted(predictions, key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]

# Example usage
mf_rec = MFRecommender(n_factors=20)
mf_rec.fit(user_item_data, n_epochs=50)

recommendations = mf_rec.recommend(user_id=0, n_recommendations=5)
print(f"Matrix factorization recommendations for user 0: {recommendations}")
```

## Deep Learning Approaches

### Neural Collaborative Filtering
```python
class NeuralCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 50, 
                 layers: List[int] = [100, 50, 20]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        self.layers = []
        input_dim = 2 * n_factors
        
        for layer_size in layers:
            self.layers.append(nn.Linear(input_dim, layer_size))
            self.layers.append(nn.ReLU())
            input_dim = layer_size
        
        self.layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*self.layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        # Pass through MLP
        output = self.mlp(concat_embeddings)
        return output.squeeze()
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long)
            item_tensor = torch.tensor([item_id], dtype=torch.long)
            prediction = self.forward(user_tensor, item_tensor)
            return prediction.item()

class NeuralCFRecommender:
    def __init__(self, n_factors: int = 50, layers: List[int] = [100, 50, 20], 
                 learning_rate: float = 0.001):
        self.n_factors = n_factors
        self.layers = layers
        self.learning_rate = learning_rate
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        
    def fit(self, user_item_data: pd.DataFrame, n_epochs: int = 100):
        """Fit neural collaborative filtering model"""
        # Create mappings
        unique_users = user_item_data['user_id'].unique()
        unique_items = user_item_data['item_id'].unique()
        
        for i, user_id in enumerate(unique_users):
            self.user_mapping[user_id] = i
        
        for i, item_id in enumerate(unique_items):
            self.item_mapping[item_id] = i
        
        # Create model
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.model = NeuralCF(n_users, n_items, self.n_factors, self.layers)
        
        # Prepare training data
        user_indices = [self.user_mapping[uid] for uid in user_item_data['user_id']]
        item_indices = [self.item_mapping[iid] for iid in user_item_data['item_id']]
        ratings = user_item_data['rating'].values
        
        # Convert to tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long)
        item_tensor = torch.tensor(item_indices, dtype=torch.long)
        rating_tensor = torch.tensor(ratings, dtype=torch.float)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            predictions = self.model(user_tensor, item_tensor)
            loss = criterion(predictions, rating_tensor)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate recommendations for a user"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Predict ratings for all items
        predictions = []
        for item_id in self.item_mapping.keys():
            item_idx = self.item_mapping[item_id]
            pred_rating = self.model.predict(user_idx, item_idx)
            predictions.append((item_id, pred_rating))
        
        # Return top recommendations
        sorted_items = sorted(predictions, key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]

# Example usage
ncf_rec = NeuralCFRecommender(n_factors=20, layers=[100, 50, 20])
ncf_rec.fit(user_item_data, n_epochs=50)

recommendations = ncf_rec.recommend(user_id=0, n_recommendations=5)
print(f"Neural CF recommendations for user 0: {recommendations}")
```

## Production Systems

### Hybrid Recommender System
```python
class HybridRecommender:
    def __init__(self, collaborative_weight: float = 0.6, content_weight: float = 0.4):
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.collaborative_model = None
        self.content_model = None
        
    def fit(self, user_item_data: pd.DataFrame, item_data: pd.DataFrame = None):
        """Fit hybrid recommender"""
        # Fit collaborative filtering
        self.collaborative_model = ItemBasedCF(n_neighbors=5)
        self.collaborative_model.fit(user_item_data)
        
        # Fit content-based filtering if item data available
        if item_data is not None:
            self.content_model = ContentBasedRecommender(feature_columns=['description'])
            self.content_model.fit(item_data, user_item_data)
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate hybrid recommendations"""
        collaborative_recs = []
        content_recs = []
        
        # Get collaborative recommendations
        if self.collaborative_model:
            collaborative_recs = self.collaborative_model.recommend(user_id, n_recommendations)
        
        # Get content-based recommendations
        if self.content_model:
            content_recs = self.content_model.recommend(user_id, n_recommendations)
        
        # Combine recommendations
        if collaborative_recs and content_recs:
            # Weighted combination
            all_items = set(collaborative_recs + content_recs)
            scores = {}
            
            for item_id in all_items:
                collab_score = collaborative_recs.index(item_id) if item_id in collaborative_recs else len(collaborative_recs)
                content_score = content_recs.index(item_id) if item_id in content_recs else len(content_recs)
                
                # Normalize scores (lower is better)
                collab_score = 1 - (collab_score / len(collaborative_recs))
                content_score = 1 - (content_score / len(content_recs))
                
                # Weighted average
                total_score = (self.collaborative_weight * collab_score + 
                             self.content_weight * content_score)
                scores[item_id] = total_score
            
            # Return top recommendations
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in sorted_items[:n_recommendations]]
        
        elif collaborative_recs:
            return collaborative_recs
        elif content_recs:
            return content_recs
        else:
            return []

# Example usage
hybrid_rec = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
hybrid_rec.fit(user_item_data, item_data)

recommendations = hybrid_rec.recommend(user_id=0, n_recommendations=5)
print(f"Hybrid recommendations for user 0: {recommendations}")
```

## Conclusion

Recommender systems provide powerful methods for personalized recommendations. Key areas include:

1. **Collaborative Filtering**: User-based and item-based approaches
2. **Content-Based Filtering**: Feature-based recommendations
3. **Matrix Factorization**: Latent factor models
4. **Deep Learning**: Neural collaborative filtering
5. **Hybrid Systems**: Combining multiple approaches

The field continues to evolve with new methods for more accurate and diverse recommendations.

## Resources

- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6)
- [Surprise Library](https://surpriselib.com/)
- [LightFM](https://github.com/lyst/lightfm)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)