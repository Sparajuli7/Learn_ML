# Speech & Audio Processing

## Overview
Speech and audio processing is a specialized field that combines signal processing, machine learning, and linguistics to analyze, understand, and generate audio signals. It's essential for speech recognition, music analysis, audio synthesis, and many other applications.

## Audio Signal Fundamentals

### Signal Properties
```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
audio, sr = librosa.load('audio_file.wav', sr=16000)

# Basic properties
duration = len(audio) / sr  # Duration in seconds
amplitude = np.max(np.abs(audio))  # Peak amplitude
rms_energy = np.sqrt(np.mean(audio**2))  # Root mean square energy
```

### Digital Audio Representation
```python
# Audio signal properties
print(f"Sample rate: {sr} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Number of samples: {len(audio)}")
print(f"Bit depth: {audio.dtype}")

# Convert to different formats
audio_int16 = (audio * 32767).astype(np.int16)  # 16-bit PCM
audio_float32 = audio.astype(np.float32)  # 32-bit float
```

## Feature Extraction

### 1. Time-Domain Features
```python
def extract_time_features(audio):
    features = {}
    
    # Energy features
    features['rms_energy'] = np.sqrt(np.mean(audio**2))
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
    
    # Statistical features
    features['mean'] = np.mean(audio)
    features['std'] = np.std(audio)
    features['skewness'] = np.mean(((audio - np.mean(audio)) / np.std(audio))**3)
    features['kurtosis'] = np.mean(((audio - np.mean(audio)) / np.std(audio))**4)
    
    return features
```

### 2. Frequency-Domain Features
```python
def extract_frequency_features(audio, sr):
    # Compute FFT
    fft = np.fft.fft(audio)
    magnitude_spectrum = np.abs(fft)
    phase_spectrum = np.angle(fft)
    
    # Frequency bins
    freqs = np.fft.fftfreq(len(audio), 1/sr)
    
    # Spectral features
    features = {}
    features['spectral_centroid'] = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
    features['spectral_bandwidth'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * magnitude_spectrum) / np.sum(magnitude_spectrum))
    features['spectral_rolloff'] = np.percentile(freqs, 85)
    
    return features, magnitude_spectrum, phase_spectrum
```

### 3. Mel-Frequency Cepstral Coefficients (MFCC)
```python
def extract_mfcc(audio, sr, n_mfcc=13):
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Delta and delta-delta features
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Combine features
    mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    
    return mfcc_features
```

### 4. Mel-Spectrogram
```python
def extract_mel_spectrogram(audio, sr, n_mels=128, hop_length=512):
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels, 
        hop_length=hop_length
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec
```

## Speech Recognition

### 1. Hidden Markov Models (HMM)
```python
from hmmlearn import hmm

class HMMSpeechRecognizer:
    def __init__(self, n_components=5, n_iter=100):
        self.model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
        self.models = {}  # One model per word/class
    
    def train(self, features_dict):
        """Train HMM models for each word/class"""
        for word, features in features_dict.items():
            model = hmm.GaussianHMM(n_components=5, n_iter=100)
            model.fit(features)
            self.models[word] = model
    
    def predict(self, features):
        """Predict the most likely word/class"""
        scores = {}
        for word, model in self.models.items():
            scores[word] = model.score(features)
        
        return max(scores, key=scores.get)
```

### 2. Connectionist Temporal Classification (CTC)
```python
import torch
import torch.nn as nn

class CTCSpeechRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CTCSpeechRecognizer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def ctc_loss(predictions, targets, input_lengths, target_lengths):
    """Compute CTC loss"""
    return nn.CTCLoss()(predictions, targets, input_lengths, target_lengths)
```

### 3. Transformer-based ASR
```python
class TransformerASR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, vocab_size):
        super(TransformerASR, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, features)
        
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.transpose(0, 1)  # Back to (batch, seq_len, features)
        
        output = self.output_projection(transformer_out)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

## Audio Classification

### 1. CNN for Audio Classification
```python
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, 1, height, width)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
```

### 2. CRNN (Convolutional Recurrent Neural Network)
```python
class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm = nn.BatchNorm2d(128)
        
        # Recurrent layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.batch_norm(x)
        
        # Reshape for LSTM
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height, channels * width)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        x = torch.mean(lstm_out, dim=1)
        
        # Classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
```

## Audio Generation

### 1. WaveNet
```python
class WaveNet(nn.Module):
    def __init__(self, input_channels, residual_channels, skip_channels, out_channels, layers):
        super(WaveNet, self).__init__()
        
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.layers = layers
        
        # Input convolution
        self.input_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        # Dilated convolutions
        self.dilated_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for layer in range(layers):
            dilation = 2 ** layer
            self.dilated_convs.append(
                nn.Conv1d(residual_channels, residual_channels, 2, dilation=dilation, padding=dilation)
            )
            self.gate_convs.append(
                nn.Conv1d(residual_channels, residual_channels, 2, dilation=dilation, padding=dilation)
            )
            self.residual_convs.append(nn.Conv1d(residual_channels, residual_channels, 1))
            self.skip_convs.append(nn.Conv1d(residual_channels, skip_channels, 1))
        
        # Output layers
        self.output_conv = nn.Conv1d(skip_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.input_conv(x)
        skip = 0
        
        for layer in range(self.layers):
            residual = x
            
            # Dilated convolution
            x_dilated = self.dilated_convs[layer](x)
            x_gate = self.gate_convs[layer](x)
            
            # Gated activation
            x = torch.tanh(x_dilated) * torch.sigmoid(x_gate)
            
            # Residual and skip connections
            x = self.residual_convs[layer](x)
            x = x + residual
            skip = skip + self.skip_convs[layer](x)
        
        x = F.relu(skip)
        x = self.output_conv(x)
        return x
```

### 2. Tacotron 2 (Text-to-Speech)
```python
class Tacotron2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, decoder_dim):
        super(Tacotron2, self).__init__()
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(embedding_dim, encoder_dim, batch_first=True)
        
        # Attention mechanism
        self.attention = LocationSensitiveAttention(encoder_dim, decoder_dim)
        
        # Decoder
        self.decoder = nn.LSTM(encoder_dim + decoder_dim, decoder_dim, batch_first=True)
        
        # Mel-spectrogram prediction
        self.mel_predictor = nn.Linear(decoder_dim, 80)  # 80 mel bins
        
    def forward(self, text, mel_target=None):
        # Text embedding
        embedded = self.embedding(text)
        
        # Encoder
        encoder_outputs, _ = self.encoder(embedded)
        
        # Decoder with attention
        decoder_outputs = []
        attention_weights = []
        
        decoder_hidden = torch.zeros(1, text.size(0), self.decoder_dim)
        decoder_cell = torch.zeros(1, text.size(0), self.decoder_dim)
        
        for t in range(mel_target.size(1) if mel_target is not None else 1000):
            # Attention
            context, attention_weight = self.attention(decoder_hidden[-1], encoder_outputs)
            attention_weights.append(attention_weight)
            
            # Decoder step
            decoder_input = torch.cat([context, decoder_hidden[-1]], dim=1).unsqueeze(1)
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            # Mel prediction
            mel_output = self.mel_predictor(decoder_output.squeeze(1))
            decoder_outputs.append(mel_output)
            
            # Teacher forcing
            if mel_target is not None and t < mel_target.size(1):
                decoder_hidden = decoder_hidden + mel_target[:, t:t+1, :].transpose(0, 1)
        
        return torch.stack(decoder_outputs, dim=1), torch.stack(attention_weights, dim=1)

class LocationSensitiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(LocationSensitiveAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.attention_rnn = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        self.attention_layer = nn.Linear(decoder_dim, encoder_dim)
        self.location_layer = nn.Linear(1, encoder_dim)
        
    def forward(self, decoder_hidden, encoder_outputs, prev_attention=None):
        # Compute attention scores
        attention_hidden = self.attention_layer(decoder_hidden)
        
        if prev_attention is not None:
            location_features = self.location_layer(prev_attention.unsqueeze(-1))
            attention_hidden = attention_hidden + location_features
        
        attention_scores = torch.bmm(
            attention_hidden.unsqueeze(1), 
            encoder_outputs.transpose(1, 2)
        ).squeeze(1)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights
```

## Music Analysis

### 1. Beat Tracking
```python
def detect_beats(audio, sr):
    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    return tempo, beat_times, onset_times
```

### 2. Chord Recognition
```python
def detect_chords(audio, sr):
    # Chromagram
    chromagram = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Chord templates
    chord_templates = {
        'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'G': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        'Am': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        # Add more chord templates
    }
    
    # Chord recognition
    chords = []
    for frame in range(chromagram.shape[1]):
        frame_chroma = chromagram[:, frame]
        best_chord = None
        best_score = -1
        
        for chord_name, template in chord_templates.items():
            score = np.corrcoef(frame_chroma, template)[0, 1]
            if score > best_score:
                best_score = score
                best_chord = chord_name
        
        chords.append(best_chord)
    
    return chords
```

### 3. Music Genre Classification
```python
class MusicGenreClassifier(nn.Module):
    def __init__(self, num_genres):
        super(MusicGenreClassifier, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Classification layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_genres)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
```

## Audio Enhancement

### 1. Noise Reduction
```python
def spectral_subtraction(noisy_audio, noise_profile, alpha=1.0):
    """Spectral subtraction for noise reduction"""
    
    # Compute STFT
    stft = librosa.stft(noisy_audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise spectrum
    noise_spectrum = np.mean(np.abs(librosa.stft(noise_profile))**2, axis=1)
    
    # Spectral subtraction
    clean_magnitude = magnitude - alpha * np.sqrt(noise_spectrum.reshape(-1, 1))
    clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)  # Floor
    
    # Reconstruct signal
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft)
    
    return clean_audio
```

### 2. Audio Super-Resolution
```python
class AudioSuperResolution(nn.Module):
    def __init__(self, upscale_factor=4):
        super(AudioSuperResolution, self).__init__()
        self.upscale_factor = upscale_factor
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='linear')
        
        # Refinement layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Upsample
        x = self.upsample(x)
        
        # Refine
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x
```

## Evaluation Metrics

### 1. Speech Recognition Metrics
```python
def word_error_rate(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    from jiwer import wer
    return wer(reference, hypothesis)

def character_error_rate(reference, hypothesis):
    """Calculate Character Error Rate (CER)"""
    from jiwer import cer
    return cer(reference, hypothesis)
```

### 2. Audio Quality Metrics
```python
def signal_to_noise_ratio(clean_signal, noisy_signal):
    """Calculate Signal-to-Noise Ratio (SNR)"""
    noise = clean_signal - noisy_signal
    snr = 10 * np.log10(np.sum(clean_signal**2) / np.sum(noise**2))
    return snr

def perceptual_evaluation_of_speech_quality(clean_signal, processed_signal, sr):
    """Calculate PESQ score"""
    from pesq import pesq
    return pesq(sr, clean_signal, processed_signal, 'wb')
```

## Tools and Libraries

- **Librosa**: Audio and music analysis
- **PyAudio**: Audio I/O
- **SoundFile**: Audio file reading/writing
- **TorchAudio**: PyTorch audio processing
- **SpeechRecognition**: Speech recognition
- **Transformers**: Pre-trained speech models

## Best Practices

1. **Preprocessing**: Normalize audio and handle different sample rates
2. **Feature Engineering**: Choose appropriate features for your task
3. **Data Augmentation**: Use pitch shifting, time stretching, noise addition
4. **Model Architecture**: Select appropriate architecture for your task
5. **Evaluation**: Use task-specific metrics

## Next Steps

1. **Multilingual ASR**: Handle multiple languages
2. **Speaker Recognition**: Identify speakers from voice
3. **Emotion Recognition**: Detect emotions from speech
4. **Music Generation**: Generate music with AI
5. **Real-time Processing**: Process audio in real-time

---

*Speech and audio processing combines signal processing fundamentals with modern deep learning techniques to understand and generate audio content. From speech recognition to music analysis, these techniques are powering the next generation of audio applications.* 