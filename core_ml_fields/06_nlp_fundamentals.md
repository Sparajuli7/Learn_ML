# Natural Language Processing Fundamentals: From Text to Understanding

*"Language is the foundation of human intelligence - teaching machines to understand it unlocks the future of AI"*

---

## 📚 Table of Contents

1. [Introduction to NLP](#introduction-to-nlp)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Text Processing Pipeline](#text-processing-pipeline)
4. [Word Embeddings and Representations](#word-embeddings-and-representations)
5. [Language Models and Sentiment Analysis](#language-models-and-sentiment-analysis)
6. [Real-World Applications](#real-world-applications)
7. [Exercises and Projects](#exercises-and-projects)
8. [Further Reading](#further-reading)

---

## 🎯 Introduction to NLP

### What is Natural Language Processing?

Natural Language Processing (NLP) is the branch of artificial intelligence that enables computers to understand, interpret, and generate human language. It sits at the intersection of linguistics, computer science, and machine learning, creating systems that can process text and speech in ways that mimic human understanding.

### Historical Evolution

| Era | Key Development | Impact |
|-----|----------------|---------|
| **1950s** | Machine Translation (Georgetown Experiment) | First attempts at language processing |
| **1960s** | ELIZA chatbot | Demonstrated conversational AI |
| **1980s** | Statistical NLP | Probabilistic approaches to language |
| **2000s** | Machine Learning NLP | SVM, Naive Bayes for text classification |
| **2010s** | Word Embeddings | Word2Vec, GloVe revolutionized representations |
| **2018+** | Transformer Revolution | BERT, GPT, and modern language models |
| **2025** | Multimodal LLMs | Text + vision + audio understanding |

### Core NLP Tasks

#### 1. **Text Classification**
- Sentiment analysis
- Topic classification
- Intent detection
- Spam detection

#### 2. **Information Extraction**
- Named Entity Recognition (NER)
- Relation extraction
- Key phrase extraction
- Event detection

#### 3. **Text Generation**
- Machine translation
- Summarization
- Question answering
- Dialogue systems

#### 4. **Language Understanding**
- Semantic similarity
- Text entailment
- Coreference resolution
- Discourse analysis

### 2025 NLP Landscape

The field has evolved dramatically with the rise of large language models:

- **Scale**: Models with 100B+ parameters
- **Multimodality**: Text + image + video understanding
- **Efficiency**: Smaller, faster models for deployment
- **Specialization**: Domain-specific models for healthcare, legal, etc.
- **Regulation**: EU AI Act compliance for language systems

---

## 🧮 Mathematical Foundations

### Text Representation Mathematics

#### 1. **Bag of Words (BoW)**

The simplest text representation converts documents to vectors:

```
Document: "The cat sat on the mat"
Vocabulary: ["the", "cat", "sat", "on", "mat"]
BoW Vector: [2, 1, 1, 1, 1, 0, 0, ...]
```

Mathematical formulation:
```
V(d) = [f(w₁,d), f(w₂,d), ..., f(wₙ,d)]
```

Where `f(wᵢ,d)` is the frequency of word `wᵢ` in document `d`.

#### 2. **TF-IDF (Term Frequency-Inverse Document Frequency)**

Addresses the limitation of BoW by weighting terms:

```
TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in d)

IDF(t,D) = log(Total number of documents / Number of documents containing term t)

TF-IDF(t,d) = TF(t,d) × IDF(t,D)
```

#### 3. **Word Embeddings Mathematics**

Word embeddings map words to dense vectors in a continuous space:

```
Word: "king" → Vector: [0.2, -0.1, 0.8, ...]
Word: "queen" → Vector: [0.1, 0.9, 0.7, ...]
```

The cosine similarity between vectors captures semantic relationships:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

#### 4. **Language Model Probability**

Language models predict the probability of the next word:

```
P(wₜ|w₁, w₂, ..., wₜ₋₁) = P(wₜ|context)
```

For n-gram models:
```
P(wₜ|wₜ₋ₙ₊₁, ..., wₜ₋₁) = count(wₜ₋ₙ₊₁, ..., wₜ) / count(wₜ₋ₙ₊₁, ..., wₜ₋₁)
```

### Statistical Foundations

#### 1. **Perplexity**

Measures how well a language model predicts a sequence:

```
Perplexity = 2^(-1/N × Σ log₂ P(wᵢ))
```

Where N is the total number of words.

#### 2. **BLEU Score**

For machine translation evaluation:

```
BLEU = BP × exp(Σ wₙ log pₙ)
```

Where:
- `BP` = Brevity penalty
- `pₙ` = n-gram precision
- `wₙ` = weights for different n-grams

#### 3. **ROUGE Score**

For summarization evaluation:

```
ROUGE-N = Σₛ∈S Σₙ-gram∈s Count_match(n-gram) / Σₛ∈S Σₙ-gram∈s Count(n-gram)
```

---

## 💻 Text Processing Pipeline

### Complete NLP Pipeline Implementation

```python
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline"""
        # 1. Text cleaning
        text = self.clean_text(text)
        
        # 2. Tokenization
        tokens = word_tokenize(text.lower())
        
        # 3. Remove stopwords and punctuation
        tokens = [token for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        # 4. Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def clean_text(self, text):
        """Remove special characters and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, texts):
        """Extract TF-IDF features from text corpus"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        return vectorizer.fit_transform(texts)
    
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from text corpus"""
        all_tokens = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        
        # Count frequencies
        word_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.most_common())
                     if count >= min_freq}
        
        return vocabulary
    
    def text_to_sequence(self, text, vocabulary):
        """Convert text to sequence of indices"""
        tokens = self.preprocess_text(text)
        sequence = [vocabulary.get(token, 0) for token in tokens]
        return sequence

# Example usage
processor = NLPProcessor()

# Sample texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is fascinating and complex.",
    "Machine learning algorithms can understand text patterns.",
    "Deep learning has revolutionized NLP in recent years."
]

# Process texts
processed_texts = []
for text in texts:
    tokens = processor.preprocess_text(text)
    processed_texts.append(' '.join(tokens))
    print(f"Original: {text}")
    print(f"Processed: {' '.join(tokens)}\n")

# Extract features
features = processor.extract_features(processed_texts)
print(f"Feature matrix shape: {features.shape}")

# Build vocabulary
vocabulary = processor.build_vocabulary(texts)
print(f"Vocabulary size: {len(vocabulary)}")
```

### Advanced Text Processing

```python
import spacy
from textblob import TextBlob
import pandas as pd

class AdvancedNLPProcessor:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_dependencies(self, text):
        """Extract syntactic dependencies"""
        doc = self.nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'token': token.text,
                'dep': token.dep_,
                'head': token.head.text,
                'pos': token.pos_
            })
        
        return dependencies
    
    def sentiment_analysis(self, text):
        """Perform sentiment analysis using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_keyphrases(self, text, top_k=5):
        """Extract key phrases using TextRank-like approach"""
        doc = self.nlp(text)
        
        # Extract noun chunks and named entities
        keyphrases = []
        
        # Noun chunks
        for chunk in doc.noun_chunks:
            keyphrases.append(chunk.text)
        
        # Named entities
        for ent in doc.ents:
            keyphrases.append(ent.text)
        
        # Remove duplicates and return top k
        unique_phrases = list(set(keyphrases))
        return unique_phrases[:top_k]

# Example usage
advanced_processor = AdvancedNLPProcessor()

sample_text = "Apple Inc. CEO Tim Cook announced new iPhone features at the WWDC conference in San Francisco."

# Extract entities
entities = advanced_processor.extract_entities(sample_text)
print("Named Entities:")
for entity in entities:
    print(f"- {entity['text']} ({entity['label']})")

# Extract dependencies
deps = advanced_processor.extract_dependencies(sample_text)
print("\nDependencies:")
for dep in deps[:5]:  # Show first 5
    print(f"- {dep['token']} ({dep['dep']}) → {dep['head']}")

# Sentiment analysis
sentiment = advanced_processor.sentiment_analysis(sample_text)
print(f"\nSentiment: Polarity={sentiment['polarity']:.2f}, Subjectivity={sentiment['subjectivity']:.2f}")

# Key phrases
keyphrases = advanced_processor.extract_keyphrases(sample_text)
print(f"\nKey Phrases: {keyphrases}")
```

---

## 🔤 Word Embeddings and Representations

### Word2Vec Implementation

```python
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100, window_size=2, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        
    def softmax(self, x):
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, target_word_idx, context_word_idx):
        """Forward pass"""
        # Input layer
        h = self.W1[target_word_idx]
        
        # Output layer
        u = np.dot(self.W2.T, h)
        y_hat = self.softmax(u)
        
        return h, u, y_hat
    
    def backward(self, target_word_idx, context_word_idx, h, u, y_hat):
        """Backward pass"""
        # Create one-hot vector for context word
        y_true = np.zeros(self.vocab_size)
        y_true[context_word_idx] = 1
        
        # Compute gradients
        e = y_hat - y_true
        
        # Gradients for W2
        dW2 = np.outer(h, e)
        
        # Gradients for W1
        dh = np.dot(self.W2, e)
        dW1 = np.zeros_like(self.W1)
        dW1[target_word_idx] = dh
        
        return dW1, dW2
    
    def train(self, training_data, epochs=100):
        """Train the model"""
        for epoch in range(epochs):
            total_loss = 0
            
            for target_word, context_word in training_data:
                # Forward pass
                h, u, y_hat = self.forward(target_word, context_word)
                
                # Compute loss
                loss = -np.log(y_hat[context_word] + 1e-8)
                total_loss += loss
                
                # Backward pass
                dW1, dW2 = self.backward(target_word, context_word, h, u, y_hat)
                
                # Update weights
                self.W1 -= self.learning_rate * dW1
                self.W2 -= self.learning_rate * dW2
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def get_embedding(self, word_idx):
        """Get word embedding"""
        return self.W1[word_idx]
    
    def find_similar_words(self, word_idx, top_k=5):
        """Find most similar words"""
        target_embedding = self.W1[word_idx]
        
        similarities = []
        for i in range(self.vocab_size):
            if i != word_idx:
                similarity = np.dot(target_embedding, self.W1[i]) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(self.W1[i])
                )
                similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Example usage
def create_training_data(texts, window_size=2):
    """Create training data for Word2Vec"""
    # Simple tokenization
    all_words = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
    
    # Create vocabulary
    word_to_idx = {word: idx for idx, word in enumerate(set(all_words))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Create training pairs
    training_data = []
    for i, word in enumerate(all_words):
        word_idx = word_to_idx[word]
        
        # Get context words
        start = max(0, i - window_size)
        end = min(len(all_words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_word = all_words[j]
                context_idx = word_to_idx[context_word]
                training_data.append((word_idx, context_idx))
    
    return training_data, word_to_idx, idx_to_word

# Sample texts
sample_texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the bird flew over the tree",
    "the fish swam in the pond"
]

# Create training data
training_data, word_to_idx, idx_to_word = create_training_data(sample_texts)
vocab_size = len(word_to_idx)

print(f"Vocabulary size: {vocab_size}")
print(f"Training pairs: {len(training_data)}")

# Train Word2Vec
word2vec = Word2Vec(vocab_size, embedding_dim=50)
word2vec.train(training_data, epochs=50)

# Visualize embeddings
def visualize_embeddings(word2vec, word_to_idx, idx_to_word):
    """Visualize word embeddings using PCA"""
    # Get all embeddings
    embeddings = []
    words = []
    
    for word, idx in word_to_idx.items():
        embeddings.append(word2vec.get_embedding(idx))
        words.append(word)
    
    embeddings = np.array(embeddings)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('Word Embeddings Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Visualize embeddings
visualize_embeddings(word2vec, word_to_idx, idx_to_word)
```

### Modern Embeddings with Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class ModernEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embeddings(self, texts):
        """Get embeddings for a list of texts"""
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        
        return embeddings
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def semantic_similarity(self, text1, text2):
        """Compute semantic similarity between two texts"""
        embeddings = self.get_embeddings([text1, text2])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.dot(embeddings[0], embeddings[1])
        return similarity.item()

# Example usage
modern_embeddings = ModernEmbeddings()

# Test semantic similarity
text1 = "The cat sat on the mat"
text2 = "A feline rested on the carpet"
text3 = "The weather is sunny today"

similarity_1_2 = modern_embeddings.semantic_similarity(text1, text2)
similarity_1_3 = modern_embeddings.semantic_similarity(text1, text3)

print(f"Similarity between '{text1}' and '{text2}': {similarity_1_2:.4f}")
print(f"Similarity between '{text1}' and '{text3}': {similarity_1_3:.4f}")
```

---

## 🤖 Language Models and Sentiment Analysis

### N-gram Language Model

```python
from collections import defaultdict, Counter
import random

class NGramLanguageModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        
    def train(self, texts):
        """Train the n-gram model"""
        for text in texts:
            # Tokenize
            tokens = text.lower().split()
            tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            # Create n-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                next_token = tokens[i+self.n] if i+self.n < len(tokens) else '<END>'
                
                self.ngrams[ngram][next_token] += 1
                self.vocab.add(next_token)
    
    def generate_text(self, max_length=50, temperature=1.0):
        """Generate text using the trained model"""
        # Start with start tokens
        current = ['<START>'] * (self.n - 1)
        generated = []
        
        for _ in range(max_length):
            # Get current n-gram
            ngram = tuple(current[-(self.n-1):])
            
            if ngram not in self.ngrams:
                break
            
            # Get next token probabilities
            next_token_probs = self.ngrams[ngram]
            
            # Apply temperature
            if temperature != 1.0:
                probs = {token: count ** (1/temperature) 
                        for token, count in next_token_probs.items()}
                total = sum(probs.values())
                next_token_probs = {token: prob/total 
                                  for token, prob in probs.items()}
            
            # Sample next token
            tokens = list(next_token_probs.keys())
            weights = list(next_token_probs.values())
            
            if not tokens:
                break
                
            next_token = random.choices(tokens, weights=weights)[0]
            
            if next_token == '<END>':
                break
                
            generated.append(next_token)
            current.append(next_token)
        
        return ' '.join(generated)

# Example usage
training_texts = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps in the sun",
    "the brown fox is quick and clever",
    "the dog and fox are friends"
]

# Train model
ngram_model = NGramLanguageModel(n=3)
ngram_model.train(training_texts)

# Generate text
generated_text = ngram_model.generate_text(max_length=20, temperature=0.8)
print(f"Generated text: {generated_text}")
```

### Sentiment Analysis with Deep Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Embeddings
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Global average pooling
        mask = attention_mask.unsqueeze(-1).expand(lstm_out.size())
        masked_output = lstm_out * mask
        pooled = masked_output.sum(dim=1) / mask.sum(dim=1)
        
        # Classification
        output = self.dropout(pooled)
        output = self.fc(output)
        
        return output

def train_sentiment_model(texts, labels, tokenizer, epochs=10):
    """Train sentiment analysis model"""
    # Create dataset
    dataset = SentimentDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = SentimentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(set(labels))
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                batch['input_ids'],
                batch['attention_mask']
            )
            
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# Example usage
sample_texts = [
    "I love this product! It's amazing.",
    "This is terrible. Worst purchase ever.",
    "The product is okay, nothing special.",
    "Absolutely fantastic experience!",
    "Disappointed with the quality.",
    "Great service and fast delivery."
]

sample_labels = [2, 0, 1, 2, 0, 2]  # 0: negative, 1: neutral, 2: positive

# Simple tokenizer for demonstration
class SimpleTokenizer:
    def __init__(self, texts):
        words = set()
        for text in texts:
            words.update(text.lower().split())
        self.vocab = {word: idx for idx, word in enumerate(words)}
        self.vocab_size = len(self.vocab)
    
    def __call__(self, text, **kwargs):
        tokens = text.lower().split()
        input_ids = [self.vocab.get(token, 0) for token in tokens]
        
        # Pad or truncate
        max_length = kwargs.get('max_length', 128)
        if len(input_ids) < max_length:
            input_ids += [0] * (max_length - len(input_ids))
        else:
            input_ids = input_ids[:max_length]
        
        attention_mask = [1 if token != 0 else 0 for token in input_ids]
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

# Train model
tokenizer = SimpleTokenizer(sample_texts)
model = train_sentiment_model(sample_texts, sample_labels, tokenizer, epochs=5)

# Test prediction
def predict_sentiment(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text)
        output = model(encoding['input_ids'], encoding['attention_mask'])
        prediction = torch.argmax(output, dim=1).item()
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map[prediction]

# Test predictions
test_texts = [
    "This is wonderful!",
    "I hate this.",
    "It's fine, I guess."
]

for text in test_texts:
    sentiment = predict_sentiment(text, model, tokenizer)
    print(f"Text: '{text}' → Sentiment: {sentiment}")
```

---

## 🎯 Real-World Applications

### 1. **Customer Service Chatbots**

Modern NLP powers intelligent customer service:

```python
class CustomerServiceBot:
    def __init__(self):
        self.intent_classifier = self.load_intent_classifier()
        self.response_generator = self.load_response_generator()
        
    def process_message(self, user_message):
        # 1. Intent classification
        intent = self.intent_classifier.predict(user_message)
        
        # 2. Entity extraction
        entities = self.extract_entities(user_message)
        
        # 3. Generate response
        response = self.response_generator.generate(intent, entities)
        
        return response
    
    def extract_entities(self, message):
        """Extract relevant entities from user message"""
        entities = {}
        
        # Extract product names
        product_keywords = ['order', 'product', 'item', 'purchase']
        for keyword in product_keywords:
            if keyword in message.lower():
                entities['product_mentioned'] = True
        
        # Extract urgency
        urgency_words = ['urgent', 'asap', 'immediately', 'emergency']
        for word in urgency_words:
            if word in message.lower():
                entities['urgency'] = 'high'
        
        return entities
```

### 2. **Content Recommendation Systems**

NLP enables personalized content recommendations:

```python
class ContentRecommender:
    def __init__(self):
        self.content_embeddings = {}
        self.user_profiles = {}
        
    def compute_content_similarity(self, content1, content2):
        """Compute semantic similarity between content"""
        embedding1 = self.get_embedding(content1)
        embedding2 = self.get_embedding(content2)
        
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return similarity
    
    def recommend_content(self, user_id, user_history, top_k=5):
        """Recommend content based on user history"""
        # Get user profile
        user_profile = self.build_user_profile(user_history)
        
        # Find similar content
        recommendations = []
        for content_id, content_embedding in self.content_embeddings.items():
            if content_id not in user_history:
                similarity = np.dot(user_profile, content_embedding)
                recommendations.append((content_id, similarity))
        
        # Sort and return top k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
```

### 3. **Social Media Analysis**

NLP for social media monitoring and analysis:

```python
class SocialMediaAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = self.load_sentiment_model()
        self.topic_classifier = self.load_topic_classifier()
        
    def analyze_posts(self, posts):
        """Analyze social media posts"""
        results = []
        
        for post in posts:
            analysis = {
                'text': post['text'],
                'sentiment': self.sentiment_analyzer.analyze(post['text']),
                'topics': self.topic_classifier.classify(post['text']),
                'engagement_score': self.calculate_engagement(post),
                'virality_potential': self.predict_virality(post)
            }
            results.append(analysis)
        
        return results
    
    def detect_trends(self, posts, time_window='24h'):
        """Detect trending topics and hashtags"""
        # Extract hashtags
        hashtags = []
        for post in posts:
            hashtags.extend(self.extract_hashtags(post['text']))
        
        # Count frequencies
        hashtag_counts = Counter(hashtags)
        
        # Identify trends
        trends = []
        for hashtag, count in hashtag_counts.most_common(10):
            if count > 5:  # Minimum threshold
                trends.append({
                    'hashtag': hashtag,
                    'count': count,
                    'growth_rate': self.calculate_growth_rate(hashtag)
                })
        
        return trends
```

### 4. **Legal Document Analysis**

NLP for legal document processing:

```python
class LegalDocumentProcessor:
    def __init__(self):
        self.ner_model = self.load_ner_model()
        self.contract_analyzer = self.load_contract_analyzer()
        
    def extract_legal_entities(self, document):
        """Extract legal entities from document"""
        entities = self.ner_model.extract(document)
        
        legal_entities = {
            'parties': [],
            'dates': [],
            'amounts': [],
            'obligations': [],
            'penalties': []
        }
        
        for entity in entities:
            if entity['type'] == 'PERSON' or entity['type'] == 'ORG':
                legal_entities['parties'].append(entity)
            elif entity['type'] == 'DATE':
                legal_entities['dates'].append(entity)
            elif entity['type'] == 'MONEY':
                legal_entities['amounts'].append(entity)
        
        return legal_entities
    
    def analyze_contract_risk(self, contract_text):
        """Analyze contract for potential risks"""
        risks = []
        
        # Check for missing clauses
        required_clauses = ['termination', 'liability', 'confidentiality']
        for clause in required_clauses:
            if clause not in contract_text.lower():
                risks.append(f"Missing {clause} clause")
        
        # Check for unfavorable terms
        unfavorable_patterns = [
            r'unlimited liability',
            r'one-sided termination',
            r'excessive penalties'
        ]
        
        for pattern in unfavorable_patterns:
            if re.search(pattern, contract_text, re.IGNORECASE):
                risks.append(f"Unfavorable term detected: {pattern}")
        
        return risks
```

---

## 🧪 Exercises and Projects

### Exercise 1: Build a Simple Chatbot

```python
# TODO: Implement a rule-based chatbot that can:
# 1. Recognize greetings and respond appropriately
# 2. Answer questions about weather (mock data)
# 3. Provide basic information about a company
# 4. Handle unknown queries gracefully

class SimpleChatbot:
    def __init__(self):
        self.greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        self.weather_keywords = ['weather', 'temperature', 'forecast']
        self.company_keywords = ['company', 'about', 'information']
        
    def respond(self, user_input):
        user_input = user_input.lower()
        
        # Check for greetings
        if any(greeting in user_input for greeting in self.greetings):
            return "Hello! How can I help you today?"
        
        # Check for weather queries
        if any(keyword in user_input for keyword in self.weather_keywords):
            return "The weather today is sunny with a temperature of 22°C."
        
        # Check for company information
        if any(keyword in user_input for keyword in self.company_keywords):
            return "We are a technology company founded in 2020, specializing in AI solutions."
        
        # Default response
        return "I'm not sure how to help with that. Could you rephrase your question?"

# Test the chatbot
chatbot = SimpleChatbot()
test_inputs = [
    "Hello there!",
    "What's the weather like?",
    "Tell me about your company",
    "What's the meaning of life?"
]

for user_input in test_inputs:
    response = chatbot.respond(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
```

### Exercise 2: Text Classification Pipeline

```python
# TODO: Build a complete text classification pipeline that can:
# 1. Preprocess text data
# 2. Extract features using TF-IDF
# 3. Train a classifier (SVM, Random Forest, or Neural Network)
# 4. Evaluate performance
# 5. Make predictions on new data

def build_text_classifier():
    # Your implementation here
    pass
```

### Exercise 3: Named Entity Recognition

```python
# TODO: Implement a simple NER system that can identify:
# 1. Person names
# 2. Organization names
# 3. Locations
# 4. Dates
# 5. Money amounts

def extract_entities(text):
    # Your implementation here
    pass
```

### Project 1: News Summarization System

Build a system that can:
- Scrape news articles from RSS feeds
- Extract key information using NLP
- Generate summaries using extractive methods
- Categorize articles by topic
- Provide sentiment analysis

### Project 2: Language Learning Assistant

Create an NLP-powered language learning tool that:
- Analyzes user's writing for grammar errors
- Suggests vocabulary improvements
- Provides pronunciation feedback
- Tracks learning progress
- Generates personalized exercises

### Project 3: Social Media Sentiment Dashboard

Build a dashboard that:
- Monitors social media mentions
- Analyzes sentiment trends
- Identifies influential posts
- Tracks brand mentions
- Provides real-time alerts

---

## 📖 Further Reading

### Essential Papers
- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Vaswani, A., et al. (2017). "Attention Is All You Need"

### Books
- "Speech and Language Processing" by Dan Jurafsky and James H. Martin
- "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper
- "Transformers for Natural Language Processing" by Denis Rothman

### Online Resources
- [Hugging Face Course](https://huggingface.co/course)
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)
- [NLTK Documentation](https://www.nltk.org/)

### Next Steps
- **[NLP Advanced](07_nlp_advanced.md)**: Transformers, BERT, GPT, and fine-tuning
- **[Computer Vision Basics](08_computer_vision_basics.md)**: Image processing and CNNs
- **[ML Engineering](21_data_engineering.md)**: Production NLP systems

---

## 🎯 Key Takeaways

1. **Text Processing**: Foundation of all NLP tasks through tokenization, normalization, and feature extraction
2. **Word Embeddings**: Dense vector representations that capture semantic relationships
3. **Language Models**: Statistical and neural approaches to understanding language patterns
4. **Sentiment Analysis**: Practical application for understanding text polarity and subjectivity
5. **Real-World Impact**: NLP powers chatbots, recommendation systems, and content analysis
6. **2025 Relevance**: Integration with multimodal AI and large language models

---

*"Language is the most powerful tool humans have created. Teaching machines to understand it unlocks infinite possibilities."*

**Next: [NLP Advanced](07_nlp_advanced.md) → Transformers, BERT, GPT, and modern language models** 