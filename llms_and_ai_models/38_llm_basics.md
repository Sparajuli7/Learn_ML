# LLM Basics: Transformer Architecture and Pre-training

*"Master the foundation of modern language models"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Transformer Architecture](#transformer-architecture)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Pre-training Strategies](#pre-training-strategies)
5. [Tokenization](#tokenization)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Introduction

Large Language Models (LLMs) have revolutionized natural language processing through the transformer architecture. Understanding the fundamentals of transformers, attention mechanisms, and pre-training strategies is essential for working with modern language models in 2025.

### LLM Evolution Timeline

| Model | Year | Parameters | Key Innovation |
|-------|------|------------|----------------|
| **GPT** | 2018 | 117M | Transformer decoder |
| **BERT** | 2018 | 340M | Bidirectional attention |
| **GPT-2** | 2019 | 1.5B | Larger scale |
| **GPT-3** | 2020 | 175B | Few-shot learning |
| **GPT-4** | 2023 | ~1T | Multimodal capabilities |
| **Claude-3** | 2024 | ~200B | Constitutional AI |
| **Gemini** | 2024 | ~1T | Multimodal reasoning |

### 2025 LLM Trends

- **Efficiency**: Smaller, faster models with similar performance
- **Multimodal**: Text, image, video, audio integration
- **Reasoning**: Chain-of-thought and logical inference
- **Safety**: Alignment and robustness improvements

---

## 🔄 Transformer Architecture

### 1. Core Transformer Components

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention"""
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerModel(nn.Module):
    """Complete transformer model"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Output layer
        output = self.output_layer(x)
        
        return output

# Usage
model = TransformerModel(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048
)
```

### 2. Advanced Transformer Variants

```python
class AdvancedTransformer(nn.Module):
    """Advanced transformer with modern improvements"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Pre-norm transformer blocks
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        return self.output_layer(x)

class PreNormTransformerBlock(nn.Module):
    """Pre-norm transformer block for better training stability"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)[0]
        
        # Pre-norm feed-forward
        x = x + self.feed_forward(self.norm2(x))
        
        return x

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Generate rotation matrices
        position = torch.arange(0, max_len, dtype=torch.float)
        freqs = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                         (-math.log(10000.0) / d_model))
        
        angles = position.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())
    
    def forward(self, x, seq_len):
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        
        # Apply rotary embeddings
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
        x_rot[:, :, 1::2] = x[:, :, 0::2] * sin + x[:, :, 1::2] * cos
        
        return x_rot

# Usage
advanced_model = AdvancedTransformer(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048
)
```

---

## 🧠 Attention Mechanisms

### 1. Different Attention Types

```python
class AttentionVariants:
    """Different attention mechanism variants"""
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
    
    def causal_attention(self, Q, K, V):
        """Causal attention for autoregressive models"""
        
        batch_size, num_heads, seq_len, d_k = Q.size()
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + mask
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def relative_position_attention(self, Q, K, V, relative_positions):
        """Relative positional attention"""
        
        # Add relative position embeddings
        Q = Q + relative_positions
        K = K + relative_positions
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def sparse_attention(self, Q, K, V, sparsity_pattern):
        """Sparse attention for efficiency"""
        
        # Apply sparsity pattern
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores * sparsity_pattern
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def local_attention(self, Q, K, V, window_size=64):
        """Local attention with sliding window"""
        
        batch_size, num_heads, seq_len, d_k = Q.size()
        
        # Create local attention mask
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, :start] = 0
            mask[i, end:] = 0
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Usage
attention_variants = AttentionVariants(d_model=512, num_heads=8)
```

### 2. Attention Visualization

```python
class AttentionVisualizer:
    """Visualize attention patterns"""
    
    def __init__(self):
        self.attention_maps = {}
    
    def extract_attention_weights(self, model, input_ids):
        """Extract attention weights from model"""
        
        attention_weights = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attention_weights.append(output[1])
            else:
                attention_weights.append(output)
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def visualize_attention(self, attention_weights, tokens, layer_idx=0, head_idx=0):
        """Visualize attention weights"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract attention weights for specific layer and head
        attention_map = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_map, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues',
                   annot=True,
                   fmt='.2f')
        plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return attention_map
    
    def analyze_attention_patterns(self, attention_weights):
        """Analyze attention patterns"""
        
        patterns = {
            'diagonal': 0,
            'local': 0,
            'global': 0,
            'sparse': 0
        }
        
        for layer_weights in attention_weights:
            for head_weights in layer_weights[0]:
                weights = head_weights.cpu().numpy()
                
                # Analyze patterns
                if self._is_diagonal_pattern(weights):
                    patterns['diagonal'] += 1
                elif self._is_local_pattern(weights):
                    patterns['local'] += 1
                elif self._is_global_pattern(weights):
                    patterns['global'] += 1
                elif self._is_sparse_pattern(weights):
                    patterns['sparse'] += 1
        
        return patterns
    
    def _is_diagonal_pattern(self, weights):
        """Check if attention follows diagonal pattern"""
        seq_len = weights.shape[0]
        diagonal_strength = np.trace(weights) / seq_len
        return diagonal_strength > 0.3
    
    def _is_local_pattern(self, weights):
        """Check if attention is local"""
        seq_len = weights.shape[0]
        local_window = 10
        local_strength = 0
        
        for i in range(seq_len):
            start = max(0, i - local_window)
            end = min(seq_len, i + local_window)
            local_strength += np.sum(weights[i, start:end])
        
        return local_strength / seq_len > 0.5
    
    def _is_global_pattern(self, weights):
        """Check if attention is global"""
        return np.std(weights) < 0.1
    
    def _is_sparse_pattern(self, weights):
        """Check if attention is sparse"""
        sparsity = np.sum(weights < 0.01) / weights.size
        return sparsity > 0.8

# Usage
attention_viz = AttentionVisualizer()
```

---

## 🎯 Pre-training Strategies

### 1. Masked Language Modeling (MLM)

```python
class MaskedLanguageModeling:
    """Masked Language Modeling implementation"""
    
    def __init__(self, vocab_size, mask_token_id, mask_prob=0.15):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
    
    def create_mlm_data(self, input_ids):
        """Create masked language modeling data"""
        
        batch_size, seq_len = input_ids.shape
        masked_ids = input_ids.clone()
        labels = input_ids.clone()
        
        # Create mask
        mask = torch.rand(input_ids.shape) < self.mask_prob
        mask = mask & (input_ids != 0)  # Don't mask padding tokens
        
        # Apply masking strategies
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    rand = torch.rand(1).item()
                    
                    if rand < 0.8:
                        # 80% of the time, replace with [MASK]
                        masked_ids[i, j] = self.mask_token_id
                    elif rand < 0.9:
                        # 10% of the time, replace with random token
                        masked_ids[i, j] = torch.randint(1, self.vocab_size, (1,))
                    else:
                        # 10% of the time, keep original token
                        pass
                    
                    labels[i, j] = input_ids[i, j]
                else:
                    labels[i, j] = -100  # Ignore in loss calculation
        
        return masked_ids, labels
    
    def mlm_loss(self, predictions, labels):
        """Calculate MLM loss"""
        
        # Only compute loss on masked positions
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        loss = F.cross_entropy(predictions, labels)
        return loss

class MLMTrainer:
    """MLM training utilities"""
    
    def __init__(self, model, tokenizer, mask_prob=0.15):
        self.model = model
        self.tokenizer = tokenizer
        self.mlm = MaskedLanguageModeling(
            vocab_size=len(tokenizer),
            mask_token_id=tokenizer.mask_token_id,
            mask_prob=mask_prob
        )
    
    def train_step(self, batch):
        """Single MLM training step"""
        
        input_ids = batch['input_ids']
        masked_ids, labels = self.mlm.create_mlm_data(input_ids)
        
        # Forward pass
        outputs = self.model(masked_ids)
        loss = self.mlm.mlm_loss(outputs.logits, labels)
        
        return loss
    
    def evaluate_mlm(self, eval_data):
        """Evaluate MLM performance"""
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in eval_data:
                input_ids = batch['input_ids']
                masked_ids, labels = self.mlm.create_mlm_data(input_ids)
                
                outputs = self.model(masked_ids)
                loss = self.mlm.mlm_loss(outputs.logits, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                mask = labels != -100
                correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                total_predictions += mask.sum().item()
        
        return {
            'loss': total_loss / len(eval_data),
            'accuracy': correct_predictions / total_predictions
        }

# Usage
mlm_trainer = MLMTrainer(model, tokenizer)
```

### 2. Next Sentence Prediction (NSP)

```python
class NextSentencePrediction:
    """Next Sentence Prediction implementation"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
    
    def create_nsp_data(self, sentences):
        """Create NSP training data"""
        
        nsp_data = []
        
        for i in range(0, len(sentences) - 1, 2):
            # Positive example (consecutive sentences)
            if i + 1 < len(sentences):
                sentence_a = sentences[i]
                sentence_b = sentences[i + 1]
                is_next = 1
                nsp_data.append((sentence_a, sentence_b, is_next))
            
            # Negative example (random sentence)
            if i + 2 < len(sentences):
                sentence_a = sentences[i]
                sentence_b = sentences[i + 2]  # Skip one sentence
                is_next = 0
                nsp_data.append((sentence_a, sentence_b, is_next))
        
        return nsp_data
    
    def format_nsp_input(self, sentence_a, sentence_b):
        """Format input for NSP"""
        
        # Tokenize sentences
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b)
        
        # Create input sequence
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token] + tokens_b + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create segment IDs
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'token_type_ids': torch.tensor(segment_ids)
        }
    
    def nsp_loss(self, predictions, labels):
        """Calculate NSP loss"""
        return F.cross_entropy(predictions, labels)

class NSPTrainer:
    """NSP training utilities"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.nsp = NextSentencePrediction(tokenizer)
    
    def train_step(self, batch):
        """Single NSP training step"""
        
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.model(input_ids, token_type_ids=token_type_ids)
        loss = self.nsp.nsp_loss(outputs.logits, labels)
        
        return loss
    
    def evaluate_nsp(self, eval_data):
        """Evaluate NSP performance"""
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in eval_data:
                input_ids = batch['input_ids']
                token_type_ids = batch['token_type_ids']
                labels = batch['labels']
                
                outputs = self.model(input_ids, token_type_ids=token_type_ids)
                loss = self.nsp.nsp_loss(outputs.logits, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        return {
            'loss': total_loss / len(eval_data),
            'accuracy': correct_predictions / total_predictions
        }

# Usage
nsp_trainer = NSPTrainer(model, tokenizer)
```

---

## 🔤 Tokenization

### 1. Advanced Tokenization

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.processors import TemplateProcessing

class AdvancedTokenizer:
    """Advanced tokenization utilities"""
    
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.tokenizer = self._create_tokenizer()
    
    def _create_tokenizer(self):
        """Create BPE tokenizer"""
        
        tokenizer = Tokenizer(models.BPE())
        
        # Pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # Post-processor
        tokenizer.post_processor = TemplateProcessing(
            single="$A",
            pair="$A:0 $B:1",
            special_tokens=[
                ("<s>", 0),
                ("</s>", 1),
                ("<unk>", 2),
                ("<pad>", 3),
            ]
        )
        
        return tokenizer
    
    def train_tokenizer(self, text_files):
        """Train tokenizer on text files"""
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
            show_progress=True
        )
        
        self.tokenizer.train(text_files, trainer)
        
        return self.tokenizer
    
    def encode_text(self, text, add_special_tokens=True):
        """Encode text to tokens"""
        
        encoding = self.tokenizer.encode(text)
        
        if add_special_tokens:
            return encoding.ids
        else:
            return encoding.tokens
    
    def decode_tokens(self, token_ids):
        """Decode tokens to text"""
        
        return self.tokenizer.decode(token_ids)
    
    def analyze_tokenization(self, text):
        """Analyze tokenization patterns"""
        
        encoding = self.tokenizer.encode(text)
        
        analysis = {
            'original_text': text,
            'tokens': encoding.tokens,
            'token_ids': encoding.ids,
            'token_count': len(encoding.ids),
            'character_count': len(text),
            'compression_ratio': len(text) / len(encoding.ids),
            'average_token_length': sum(len(token) for token in encoding.tokens) / len(encoding.tokens)
        }
        
        return analysis
    
    def vocabulary_analysis(self):
        """Analyze vocabulary statistics"""
        
        vocab = self.tokenizer.get_vocab()
        
        analysis = {
            'vocab_size': len(vocab),
            'special_tokens': [token for token in vocab.keys() if token.startswith('<')],
            'most_common': sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10],
            'average_token_length': sum(len(token) for token in vocab.keys()) / len(vocab)
        }
        
        return analysis

class SubwordTokenization:
    """Subword tokenization strategies"""
    
    def __init__(self):
        self.strategies = {
            'bpe': self._bpe_tokenization,
            'wordpiece': self._wordpiece_tokenization,
            'unigram': self._unigram_tokenization,
            'sentencepiece': self._sentencepiece_tokenization
        }
    
    def _bpe_tokenization(self, text, vocab_size=30000):
        """Byte Pair Encoding tokenization"""
        
        # Simplified BPE implementation
        tokens = text.split()
        vocab = set(tokens)
        
        # BPE merge operations
        while len(vocab) < vocab_size:
            # Find most frequent pair
            pair_freq = {}
            for token in tokens:
                for i in range(len(token) - 1):
                    pair = token[i:i+2]
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1
            
            if not pair_freq:
                break
            
            # Merge most frequent pair
            most_frequent_pair = max(pair_freq, key=pair_freq.get)
            # Apply merge operation
            # (simplified implementation)
        
        return list(vocab)
    
    def _wordpiece_tokenization(self, text, vocab_size=30000):
        """WordPiece tokenization"""
        
        # Simplified WordPiece implementation
        tokens = text.split()
        vocab = set(tokens)
        
        # WordPiece uses likelihood instead of frequency
        # (simplified implementation)
        
        return list(vocab)
    
    def _unigram_tokenization(self, text, vocab_size=30000):
        """Unigram language model tokenization"""
        
        # Simplified Unigram implementation
        tokens = text.split()
        vocab = set(tokens)
        
        # Unigram uses language model probabilities
        # (simplified implementation)
        
        return list(vocab)
    
    def _sentencepiece_tokenization(self, text, vocab_size=30000):
        """SentencePiece tokenization"""
        
        # Simplified SentencePiece implementation
        tokens = text.split()
        vocab = set(tokens)
        
        # SentencePiece uses unigram language model
        # (simplified implementation)
        
        return list(vocab)

# Usage
advanced_tokenizer = AdvancedTokenizer()
subword_tokenizer = SubwordTokenization()
```

---

## 🧪 Exercises and Projects

### Exercise 1: Transformer Implementation
1. Implement complete transformer from scratch
2. Add different attention mechanisms
3. Implement positional encoding variants
4. Create attention visualization tools

### Exercise 2: Pre-training Tasks
1. Implement MLM training
2. Add NSP training
3. Create custom pre-training tasks
4. Evaluate pre-training performance

### Exercise 3: Tokenization Analysis
1. Compare different tokenization strategies
2. Analyze vocabulary efficiency
3. Create custom tokenizer
4. Optimize tokenization for specific domain

### Project: Custom LLM Training

**Objective**: Build and train a custom language model

**Requirements**:
- Implement transformer architecture
- Add pre-training tasks
- Create efficient tokenization
- Train on custom dataset

**Deliverables**:
- Complete transformer implementation
- Pre-training pipeline
- Tokenization system
- Trained model and evaluation

---

## 📖 Further Reading

### Essential Resources

1. **Transformer Architecture**
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [BERT Paper](https://arxiv.org/abs/1810.04805)
   - [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

2. **Pre-training Strategies**
   - [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
   - [ALBERT Paper](https://arxiv.org/abs/1909.11942)
   - [DeBERTa Paper](https://arxiv.org/abs/2006.03654)

3. **Tokenization**
   - [BPE Paper](https://arxiv.org/abs/1508.07909)
   - [WordPiece Paper](https://arxiv.org/abs/1609.08144)
   - [SentencePiece Paper](https://arxiv.org/abs/1808.06226)

### Advanced Topics

- **Efficient Attention**: Sparse attention, linear attention
- **Positional Encoding**: Rotary embeddings, relative positions
- **Pre-training Tasks**: Span corruption, denoising objectives
- **Tokenization**: Subword regularization, vocabulary optimization

### 2025 Trends

- **Efficiency**: Smaller models with similar performance
- **Multimodal**: Text + image + video integration
- **Reasoning**: Chain-of-thought and logical inference
- **Safety**: Alignment and robustness improvements

---

## 🎯 Key Takeaways

1. **Transformer Architecture**: Foundation of modern language models
2. **Attention Mechanisms**: Core innovation enabling long-range dependencies
3. **Pre-training Strategies**: Essential for learning language representations
4. **Tokenization**: Critical for efficient text processing
5. **Scalability**: Architecture enables training of very large models

---

*"Transformers have revolutionized NLP, but the journey is just beginning."*

**Next: [LLM Expert](llms_and_ai_models/39_llm_expert.md) → Advanced prompting and evaluation**