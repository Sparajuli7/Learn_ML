# NLP Advanced: Transformers, BERT, GPT, and Fine-tuning

*"The transformer architecture revolutionized NLP and created the foundation for modern AI"*

---

## 📚 Table of Contents

1. [Transformer Architecture](#transformer-architecture)
2. [BERT and Bidirectional Models](#bert-and-bidirectional-models)
3. [GPT and Generative Models](#gpt-and-generative-models)
4. [Fine-tuning Strategies](#fine-tuning-strategies)
5. [Real-World Applications](#real-world-applications)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Transformer Architecture

### The Attention Revolution

The Transformer architecture, introduced in "Attention Is All You Need" (2017), eliminated the need for recurrent connections and enabled parallel processing of sequences. This breakthrough made it possible to train much larger models and achieve unprecedented performance on NLP tasks.

### Key Components

#### 1. **Multi-Head Attention**

The core innovation that allows models to focus on different parts of the input:

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute attention scores"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(output)
        
        return output, attention_weights
```

#### 2. **Positional Encoding**

Since Transformers have no recurrence, positional information must be explicitly added:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

#### 3. **Complete Transformer Block**

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
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
```

---

## 🔄 BERT and Bidirectional Models

### BERT Architecture

BERT (Bidirectional Encoder Representations from Transformers) introduced bidirectional training, allowing the model to see the full context of each word.

#### 1. **BERT Tokenization**

```python
from transformers import BertTokenizer

class BERTProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = 512
    
    def tokenize_text(self, text):
        """Tokenize text for BERT"""
        # Add special tokens
        text = f"[CLS] {text} [SEP]"
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Convert to IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids += [0] * (self.max_length - len(token_ids))
        
        # Create attention mask
        attention_mask = [1 if token_id != 0 else 0 for token_id in token_ids]
        
        return {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
    
    def create_masked_tokens(self, text, mask_prob=0.15):
        """Create masked tokens for MLM training"""
        tokens = self.tokenizer.tokenize(text)
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 is ignored in loss
        
        # Randomly mask tokens
        for i in range(len(tokens)):
            if random.random() < mask_prob:
                labels[i] = self.tokenizer.convert_tokens_to_ids(tokens[i])
                masked_tokens[i] = '[MASK]'
        
        return masked_tokens, labels
```

#### 2. **BERT for Classification**

```python
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification head
        output = self.dropout(cls_output)
        logits = self.classifier(output)
        
        return logits

# Example usage
def train_bert_classifier(train_texts, train_labels, num_classes=3):
    """Train BERT classifier"""
    model = BERTClassifier(num_classes)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, 
                               return_tensors='pt')
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        
        outputs = model(train_encodings['input_ids'], 
                       train_encodings['attention_mask'])
        
        loss = criterion(outputs, torch.tensor(train_labels))
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model, tokenizer
```

---

## 🤖 GPT and Generative Models

### GPT Architecture

GPT (Generative Pre-trained Transformer) uses a decoder-only architecture for text generation.

#### 1. **GPT Tokenization and Generation**

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

class GPTGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50):
        """Generate text using GPT"""
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def calculate_perplexity(self, text):
        """Calculate perplexity of text"""
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
            loss = outputs.loss
        
        perplexity = torch.exp(loss)
        return perplexity.item()

# Example usage
gpt_generator = GPTGenerator()

# Generate text
prompt = "The future of artificial intelligence"
generated = gpt_generator.generate_text(prompt, max_length=50, temperature=0.8)
print(f"Generated: {generated}")

# Calculate perplexity
perplexity = gpt_generator.calculate_perplexity("This is a test sentence.")
print(f"Perplexity: {perplexity:.2f}")
```

#### 2. **Custom GPT Training**

```python
class CustomGPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
```

---

## 🎯 Fine-tuning Strategies

### 1. **Parameter-Efficient Fine-tuning**

#### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original weight + LoRA adaptation
        return x @ self.lora_A.T @ self.lora_B.T * self.scaling

class LoRABERT(nn.Module):
    def __init__(self, base_model, rank=16):
        super().__init__()
        self.base_model = base_model
        
        # Add LoRA to attention layers
        for layer in self.base_model.bert.encoder.layer:
            layer.attention.self.query = LoRALayer(
                layer.attention.self.query.in_features,
                layer.attention.self.query.out_features,
                rank
            )
            layer.attention.self.value = LoRALayer(
                layer.attention.self.value.in_features,
                layer.attention.self.value.out_features,
                rank
            )
    
    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)
```

#### 2. **Prompt Tuning**

```python
class PromptTuning(nn.Module):
    def __init__(self, base_model, prompt_length=20, prompt_dim=768):
        super().__init__()
        self.base_model = base_model
        self.prompt_length = prompt_length
        self.prompt_dim = prompt_dim
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, prompt_dim) * 0.02
        )
    
    def forward(self, input_ids, attention_mask):
        # Get base embeddings
        base_embeddings = self.base_model.embeddings.word_embeddings(input_ids)
        
        # Add prompt embeddings
        batch_size = base_embeddings.size(0)
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Concatenate
        combined_embeddings = torch.cat([prompt_embeddings, base_embeddings], dim=1)
        
        # Update attention mask
        prompt_mask = torch.ones(batch_size, self.prompt_length, device=input_ids.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Forward through base model
        outputs = self.base_model(inputs_embeds=combined_embeddings, 
                                attention_mask=combined_mask)
        
        return outputs
```

### 3. **Adapter Tuning**

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_projection = nn.Linear(hidden_size, adapter_size)
        self.up_projection = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states):
        # Down projection
        down = self.down_projection(hidden_states)
        down = self.activation(down)
        
        # Up projection
        up = self.up_projection(down)
        
        # Residual connection
        return hidden_states + up

class AdapterBERT(nn.Module):
    def __init__(self, base_model, adapter_size=64):
        super().__init__()
        self.base_model = base_model
        
        # Add adapters to each transformer layer
        for layer in self.base_model.bert.encoder.layer:
            layer.adapter = AdapterLayer(
                self.base_model.config.hidden_size, 
                adapter_size
            )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply adapters
        for layer in self.base_model.bert.encoder.layer:
            outputs.last_hidden_state = layer.adapter(outputs.last_hidden_state)
        
        return outputs
```

---

## 🎯 Real-World Applications

### 1. **Question Answering System**

```python
class QASystem:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    def answer_question(self, question, context):
        """Answer question given context"""
        # Tokenize
        inputs = self.tokenizer(
            question, context, return_tensors='pt', 
            max_length=512, truncation=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get start and end positions
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Extract answer
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        answer = self.tokenizer.convert_tokens_to_string(
            tokens[start_idx:end_idx+1]
        )
        
        return answer
```

### 2. **Text Summarization**

```python
class Summarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def summarize(self, text, max_length=130, min_length=30):
        """Generate summary of text"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', 
                              max_length=1024, truncation=True)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
```

---

## 🧪 Exercises and Projects

### Exercise 1: Implement Attention Visualization

```python
def visualize_attention(model, tokenizer, text, layer_idx=0, head_idx=0):
    """Visualize attention weights for a given layer and head"""
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention = outputs.attentions[layer_idx][0, head_idx]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention.numpy(), xticklabels=tokens, yticklabels=tokens)
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.show()
```

### Exercise 2: Build a Custom Fine-tuning Pipeline

```python
def fine_tune_bert_for_custom_task(model, tokenizer, train_data, 
                                  task_type='classification'):
    """Fine-tune BERT for custom task"""
    # Your implementation here
    pass
```

### Project 1: Multilingual Chatbot

Build a chatbot that can:
- Handle multiple languages
- Use BERT multilingual models
- Implement conversation memory
- Provide contextual responses

### Project 2: Document Q&A System

Create a system that can:
- Process large documents
- Answer questions about content
- Provide source citations
- Handle complex queries

### Project 3: Code Generation Assistant

Build an AI assistant that can:
- Generate code from natural language
- Complete partial code
- Debug and explain code
- Suggest improvements

---

## 📖 Further Reading

### Essential Papers
- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-training"

### Books
- "Transformers for Natural Language Processing" by Denis Rothman
- "Natural Language Processing with Transformers" by Lewis Tunstall

### Online Resources
- [Hugging Face Transformers Course](https://huggingface.co/course)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### Next Steps
- **[Computer Vision Basics](08_computer_vision_basics.md)**: Image processing and CNNs
- **[RL Basics](10_rl_basics.md)**: Reinforcement learning fundamentals
- **[ML Engineering](21_data_engineering.md)**: Production NLP systems

---

## 🎯 Key Takeaways

1. **Transformer Architecture**: Self-attention mechanism enables parallel processing and better long-range dependencies
2. **BERT**: Bidirectional training captures context from both directions
3. **GPT**: Unidirectional generation for text creation and completion
4. **Fine-tuning**: Efficient adaptation strategies for specific tasks
5. **Real-World Impact**: Powers modern chatbots, Q&A systems, and content generation
6. **2025 Relevance**: Foundation for all modern language models and multimodal AI

---

*"Transformers didn't just change NLP - they changed the entire field of AI."*

**Next: [Computer Vision Basics](08_computer_vision_basics.md) → Image processing, CNNs, and visual understanding** 