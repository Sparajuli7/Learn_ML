# Speech & Audio Processing: From ASR to Music Generation

*"Transforming sound into intelligence and intelligence into sound"*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Audio Signal Processing](#audio-signal-processing)
3. [Speech Recognition (ASR)](#speech-recognition-asr)
4. [Text-to-Speech (TTS)](#text-to-speech-tts)
5. [Music Generation](#music-generation)
6. [Practical Implementation](#practical-implementation)
7. [Real-World Applications](#real-world-applications)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

---

## 🎯 Introduction

Speech and audio processing represents one of the most human-centric applications of AI, bridging the gap between human communication and machine understanding. From automatic speech recognition to music generation, from voice assistants to audio analysis, this field has seen remarkable advances in recent years.

### Historical Context

Audio processing began with basic signal processing techniques in the early 20th century. The 1950s saw the development of the first speech recognition systems, followed by Hidden Markov Models in the 1980s. The 2010s brought deep learning to audio, and the 2020s have seen transformer-based models revolutionize the field with models like Whisper, MusicLM, and AudioCraft.

### Current State (2025)

- **Foundation Models**: Large pre-trained models for audio understanding
- **Multimodal Audio**: Integrating audio with text, vision, and other modalities
- **Real-time Processing**: Low-latency systems for live applications
- **Personalization**: Speaker-adaptive and emotion-aware systems
- **Edge Computing**: On-device audio processing for privacy
- **Generative Audio**: High-quality music and speech synthesis

---

## 🎵 Audio Signal Processing

### Digital Audio Fundamentals

**Sampling Rate**: Number of samples per second (Hz)
```
Common rates: 8kHz (telephone), 16kHz (speech), 44.1kHz (CD), 48kHz (professional)
```

**Bit Depth**: Number of bits per sample
```
Common depths: 16-bit (CD), 24-bit (professional), 32-bit (floating point)
```

**Nyquist Theorem**: Sampling rate must be at least twice the highest frequency
```
f_sampling ≥ 2 × f_max
```

### Time-Domain Processing

**Amplitude Envelope**:
```
env(t) = |x(t)|
```

**Root Mean Square (RMS)**:
```
RMS = √(1/N Σᵢ x²(i))
```

**Zero Crossing Rate**:
```
ZCR = 1/N Σᵢ |sign(x(i)) - sign(x(i-1))|
```

### Frequency-Domain Processing

**Fourier Transform**:
```
X(f) = ∫ x(t) e^(-j2πft) dt
```

**Short-Time Fourier Transform (STFT)**:
```
X(t,f) = ∫ x(τ) w(τ-t) e^(-j2πfτ) dτ
```

**Mel-Frequency Cepstral Coefficients (MFCCs)**:
```
1. Apply STFT
2. Convert to mel scale
3. Apply log
4. Apply DCT
```

### Audio Features

**Spectral Features**:
- Spectral centroid
- Spectral rolloff
- Spectral bandwidth
- Spectral contrast

**Temporal Features**:
- Energy
- Zero crossing rate
- Linear predictive coding (LPC)
- Pitch (fundamental frequency)

---

## 🗣️ Speech Recognition (ASR)

### Traditional ASR Pipeline

**1. Preprocessing**:
```
- Noise reduction
- Normalization
- Voice activity detection (VAD)
```

**2. Feature Extraction**:
```
- MFCCs
- Filter banks
- Spectrograms
```

**3. Acoustic Model**:
```
- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- Deep Neural Networks (DNN)
```

**4. Language Model**:
```
- N-gram models
- Neural language models
- Transformer-based models
```

**5. Decoding**:
```
- Viterbi algorithm
- Beam search
- Connectionist Temporal Classification (CTC)
```

### Modern ASR with Deep Learning

**Connectionist Temporal Classification (CTC)**:
```
P(y|x) = Σₐ P(a|x) where a is alignment
```

**Sequence-to-Sequence Models**:
```
Encoder: h = Encoder(x)
Decoder: y = Decoder(h, y_prev)
```

**Transformer-based ASR**:
```
- Self-attention for acoustic modeling
- Cross-attention for alignment
- End-to-end training
```

### Whisper Architecture

**Encoder-Decoder Transformer**:
```
- 12-layer encoder
- 12-layer decoder
- 80 mel filter banks
- 50Hz to 15kHz frequency range
```

**Multi-task Learning**:
```
- Speech recognition
- Language identification
- Speech translation
- Timestamp prediction
```

---

## 🔊 Text-to-Speech (TTS)

### Traditional TTS Pipeline

**1. Text Analysis**:
```
- Text normalization
- Phonetic transcription
- Prosody prediction
```

**2. Acoustic Modeling**:
```
- Duration modeling
- Fundamental frequency (F0)
- Spectral envelope
```

**3. Speech Synthesis**:
```
- Concatenative synthesis
- Statistical parametric synthesis
- Neural synthesis
```

### Neural TTS

**Tacotron Architecture**:
```
- Encoder: Text → Phoneme embeddings
- Attention: Alignment mechanism
- Decoder: Mel spectrogram generation
- Vocoder: WaveNet for waveform generation
```

**FastSpeech**:
```
- Parallel generation
- Duration predictor
- Length regulator
- Non-autoregressive decoding
```

**YourTTS**:
```
- Zero-shot voice cloning
- Multilingual support
- Emotion control
- Style transfer
```

### Modern TTS Features

**Voice Cloning**:
```
- Few-shot adaptation
- Speaker embedding
- Style transfer
```

**Emotion Control**:
```
- Emotion embedding
- Prosody control
- Expressive synthesis
```

**Multilingual TTS**:
```
- Language-agnostic models
- Code-switching
- Accent adaptation
```

---

## 🎼 Music Generation

### Music Representation

**Piano Roll**:
```
- Time on x-axis
- Pitch on y-axis
- Velocity as intensity
```

**MIDI**:
```
- Note events (note_on, note_off)
- Control changes
- Tempo and time signature
```

**Symbolic Music**:
```
- Musical notation
- ABC notation
- MusicXML
```

### Neural Music Generation

**Music Transformer**:
```
- Relative positional encoding
- Attention for long sequences
- Autoregressive generation
```

**MuseNet**:
```
- Multi-instrument generation
- Style transfer
- Genre mixing
```

**MusicLM**:
```
- Text-to-music generation
- Audio continuation
- Style transfer
```

### Music Generation Techniques

**Autoregressive Models**:
```
P(x) = Πᵢ P(xᵢ | x₁, ..., xᵢ₋₁)
```

**Diffusion Models**:
```
- Forward process: Add noise
- Reverse process: Denoise
- Score-based generation
```

**GANs for Music**:
```
- Generator: Create music
- Discriminator: Evaluate quality
- Adversarial training
```

---

## 💻 Practical Implementation

### Setting Up the Environment

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
import transformers
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
```

### Audio Loading and Preprocessing

```python
def load_audio(file_path, sr=16000):
    """Load audio file with specified sample rate"""
    audio, sr_orig = librosa.load(file_path, sr=sr)
    return audio, sr

def preprocess_audio(audio, sr=16000):
    """Preprocess audio for ASR"""
    # Normalize
    audio = librosa.util.normalize(audio)
    
    # Apply pre-emphasis
    audio = librosa.effects.preemphasis(audio)
    
    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    return audio

def extract_features(audio, sr=16000, n_mfcc=13):
    """Extract MFCC features"""
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Extract delta and delta-delta
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Concatenate features
    features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2])
    
    return features.T  # Transpose for time as first dimension

def create_spectrogram(audio, sr=16000, n_fft=2048, hop_length=512):
    """Create mel spectrogram"""
    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec
```

### Speech Recognition Implementation

```python
class SimpleASR:
    def __init__(self, model_name="openai/whisper-base"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
    def transcribe(self, audio, sr=16000):
        """Transcribe audio to text"""
        # Ensure correct sample rate
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])
        
        # Decode transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription

def voice_activity_detection(audio, sr=16000, frame_length=2048, hop_length=512):
    """Detect voice activity in audio"""
    # Calculate energy
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    
    # Calculate threshold
    threshold = np.mean(energy) + 2 * np.std(energy)
    
    # Detect voice activity
    vad = energy > threshold
    
    return vad, energy

def speaker_diarization(audio, sr=16000):
    """Simple speaker diarization"""
    # Extract features
    features = extract_features(audio, sr)
    
    # Simple clustering (in practice, use more sophisticated methods)
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    return clusters
```

### Text-to-Speech Implementation

```python
class SimpleTTS:
    def __init__(self, model_name="facebook/fastspeech2-en-ljspeech"):
        # Note: This is a simplified example
        # In practice, use libraries like TTS or Coqui TTS
        self.model_name = model_name
        
    def synthesize(self, text, output_path="output.wav"):
        """Synthesize text to speech"""
        # This is a placeholder for actual TTS implementation
        # In practice, you would use:
        # - Coqui TTS
        # - Tacotron + WaveNet
        # - FastSpeech2
        # - YourTTS
        
        print(f"Synthesizing: {text}")
        print(f"Output saved to: {output_path}")
        
        # Placeholder: generate silence
        duration = 2.0  # seconds
        sample_rate = 22050
        audio = np.zeros(int(duration * sample_rate))
        
        # Save audio
        sf.write(output_path, audio, sample_rate)
        
        return audio

def text_normalization(text):
    """Normalize text for TTS"""
    import re
    
    # Convert numbers to words
    text = re.sub(r'\d+', lambda m: num2words(int(m.group())), text)
    
    # Expand abbreviations
    abbreviations = {
        'Mr.': 'Mister',
        'Dr.': 'Doctor',
        'vs.': 'versus',
        'etc.': 'et cetera'
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    
    return text

def prosody_prediction(text):
    """Predict prosody for TTS"""
    # Simple rule-based prosody
    # In practice, use neural models
    
    sentences = text.split('.')
    prosody = []
    
    for sentence in sentences:
        if sentence.strip():
            # Simple rules
            if '?' in sentence:
                prosody.append('question')
            elif '!' in sentence:
                prosody.append('exclamation')
            else:
                prosody.append('statement')
    
    return prosody
```

### Music Generation Implementation

```python
class SimpleMusicGenerator:
    def __init__(self):
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.scales = {
            'C_major': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'G_major': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
            'A_minor': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        }
    
    def generate_melody(self, scale='C_major', length=16):
        """Generate simple melody"""
        import random
        
        scale_notes = self.scales[scale]
        melody = []
        
        for _ in range(length):
            # Random note from scale
            note = random.choice(scale_notes)
            # Random duration (1, 2, or 4 beats)
            duration = random.choice([1, 2, 4])
            melody.append((note, duration))
        
        return melody
    
    def melody_to_midi(self, melody, output_path="melody.mid"):
        """Convert melody to MIDI"""
        from midiutil import MIDIFile
        
        # Create MIDI file
        midi = MIDIFile(1)
        track = 0
        time = 0
        channel = 0
        volume = 100
        
        # Set tempo
        midi.addTempo(track, time, 120)
        
        # Add notes
        for note, duration in melody:
            pitch = self.note_to_pitch(note)
            midi.addNote(track, channel, pitch, time, duration, volume)
            time += duration
        
        # Save MIDI file
        with open(output_path, "wb") as output_file:
            midi.writeFile(output_file)
        
        return output_path
    
    def note_to_pitch(self, note):
        """Convert note name to MIDI pitch"""
        note_names = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                     'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        # Assume middle octave (octave 4)
        return note_names[note] + 60

def create_audio_dataset(audio_files, labels, output_dir="dataset"):
    """Create audio dataset for training"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = []
    
    for audio_file, label in zip(audio_files, labels):
        # Load and preprocess audio
        audio, sr = load_audio(audio_file)
        audio = preprocess_audio(audio, sr)
        
        # Extract features
        features = extract_features(audio, sr)
        
        # Save features
        feature_file = os.path.join(output_dir, f"{label}_{len(dataset)}.npy")
        np.save(feature_file, features)
        
        dataset.append({
            'features': feature_file,
            'label': label,
            'audio_file': audio_file
        })
    
    return dataset

def audio_augmentation(audio, sr=16000):
    """Apply audio augmentation techniques"""
    augmented_audio = []
    
    # Original audio
    augmented_audio.append(audio)
    
    # Add noise
    noise = np.random.normal(0, 0.01, len(audio))
    augmented_audio.append(audio + noise)
    
    # Time stretching
    rate = np.random.uniform(0.8, 1.2)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    augmented_audio.append(stretched)
    
    # Pitch shifting
    steps = np.random.uniform(-4, 4)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    augmented_audio.append(shifted)
    
    return augmented_audio
```

### Complete Example: Speech Recognition System

```python
def speech_recognition_example():
    """Complete example of speech recognition system"""
    
    # Initialize ASR system
    asr = SimpleASR()
    
    # Generate synthetic audio (in practice, load real audio)
    duration = 3.0  # seconds
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple tone
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    noise = 0.01 * np.random.normal(0, 1, len(audio))
    audio = audio + noise
    
    # Preprocess audio
    audio = preprocess_audio(audio, sr)
    
    # Perform voice activity detection
    vad, energy = voice_activity_detection(audio, sr)
    
    # Transcribe audio
    try:
        transcription = asr.transcribe(audio, sr)
        print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Transcription failed: {e}")
        print("This is expected with synthetic audio")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0, duration, len(audio)), audio)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Plot energy
    plt.subplot(3, 1, 2)
    plt.plot(energy)
    plt.title('Energy')
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    
    # Plot VAD
    plt.subplot(3, 1, 3)
    plt.plot(vad)
    plt.title('Voice Activity Detection')
    plt.xlabel('Frame')
    plt.ylabel('VAD')
    
    plt.tight_layout()
    plt.show()
    
    return audio, transcription

def music_generation_example():
    """Complete example of music generation"""
    
    # Initialize music generator
    generator = SimpleMusicGenerator()
    
    # Generate melody
    melody = generator.generate_melody(scale='C_major', length=16)
    print("Generated melody:")
    for i, (note, duration) in enumerate(melody):
        print(f"{i+1:2d}. {note:2s} ({duration} beats)")
    
    # Convert to MIDI
    midi_file = generator.melody_to_midi(melody)
    print(f"MIDI file saved as: {midi_file}")
    
    return melody, midi_file

# Run examples
if __name__ == "__main__":
    print("=== Speech Recognition Example ===")
    audio, transcription = speech_recognition_example()
    
    print("\n=== Music Generation Example ===")
    melody, midi_file = music_generation_example()
```

---

## 🎯 Real-World Applications

### 1. Voice Assistants

**Smart Speakers**:
- Amazon Alexa, Google Assistant, Apple Siri
- Wake word detection
- Natural language understanding
- Multi-turn conversations

**Mobile Voice Assistants**:
- On-device processing for privacy
- Offline capabilities
- Personalized voice models

### 2. Accessibility

**Speech-to-Text for Hearing Impaired**:
- Real-time captioning
- Meeting transcription
- Video accessibility

**Text-to-Speech for Visually Impaired**:
- Screen readers
- Audiobook generation
- Navigation assistance

### 3. Healthcare

**Medical Transcription**:
- Clinical documentation
- Patient notes
- Medical report generation

**Voice Analysis**:
- Emotion detection
- Mental health monitoring
- Disease detection from voice

### 4. Entertainment

**Music Generation**:
- AI composers
- Style transfer
- Interactive music

**Voice Cloning**:
- Dubbing and localization
- Character voices
- Personalized content

### 5. Education

**Language Learning**:
- Pronunciation assessment
- Speech practice
- Interactive lessons

**Educational Content**:
- Audio book generation
- Lecture transcription
- Study material creation

### 6. Business Applications

**Customer Service**:
- Voice bots
- Call center automation
- Sentiment analysis

**Meeting Transcription**:
- Conference recording
- Note taking
- Action item extraction

---

## 🧪 Exercises and Projects

### Beginner Exercises

1. **Audio Feature Extraction**
   ```python
   # Extract MFCCs, spectral features, and temporal features
   # Compare different feature sets for classification
   # Visualize audio features
   ```

2. **Voice Activity Detection**
   ```python
   # Implement energy-based VAD
   # Compare with spectral-based methods
   # Evaluate on different audio types
   ```

3. **Simple TTS System**
   ```python
   # Build concatenative synthesis
   # Implement prosody prediction
   # Create voice cloning demo
   ```

### Intermediate Projects

1. **Music Genre Classification**
   - Extract audio features
   - Train CNN/LSTM models
   - Compare different architectures

2. **Speaker Recognition**
   - Implement speaker embedding
   - Build verification system
   - Handle multiple speakers

3. **Emotion Recognition**
   - Detect emotions from speech
   - Build multimodal system
   - Real-time emotion analysis

### Advanced Projects

1. **End-to-End ASR System**
   - Implement CTC-based model
   - Build attention mechanism
   - Handle multiple languages

2. **Neural TTS Pipeline**
   - Build Tacotron-like system
   - Implement vocoder
   - Add voice cloning

3. **Music Generation System**
   - Implement transformer for music
   - Build style transfer
   - Create interactive system

### Quiz Questions

1. **Conceptual Questions**
   - What is the Nyquist theorem and why is it important?
   - How do MFCCs differ from raw spectrograms?
   - What are the advantages of transformer-based ASR?

2. **Mathematical Questions**
   - Derive the STFT formula
   - Explain the mel scale conversion
   - Calculate MFCCs step by step

3. **Implementation Questions**
   - How would you handle different audio formats?
   - What are the trade-offs in choosing sample rate?
   - How do you evaluate ASR performance?

---

## 📖 Further Reading

### Essential Papers

1. **"Attention Is All You Need"** - Vaswani et al. (2017)
2. **"Robust Speech Recognition via Large-Scale Weak Supervision"** - Radford et al. (2022)
3. **"MusicLM: Generating Music From Text"** - Agostinelli et al. (2023)
4. **"Tacotron: Towards End-to-End Speech Synthesis"** - Wang et al. (2017)

### Books

1. **"Speech and Language Processing"** - Jurafsky & Martin
2. **"Digital Signal Processing"** - Proakis & Manolakis
3. **"Music and Machine Learning"** - Briot et al.

### Online Resources

1. **Libraries**: librosa, torchaudio, transformers
2. **Datasets**: LibriSpeech, Common Voice, GTZAN
3. **Competitions**: DCASE, AudioSet, MusicNet

### Next Steps

1. **Advanced Topics**: Multimodal audio, foundation models
2. **Production Systems**: Real-time processing, edge deployment
3. **Domain Specialization**: Music, healthcare, accessibility

---

## 🎯 Key Takeaways

1. **Signal Processing**: Foundation for all audio applications
2. **Feature Engineering**: Critical for model performance
3. **End-to-End Systems**: Modern approach with transformers
4. **Multimodal Integration**: Audio with text, vision, and other modalities
5. **Real-time Processing**: Essential for interactive applications

---

*"Sound is the bridge between human and machine intelligence."*

**Next: [Multimodal Learning](specialized_ml/15_multimodal_learning.md) → Integrating text, image, audio, and video**