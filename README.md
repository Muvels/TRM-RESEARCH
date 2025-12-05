# TRM-Research: Tiny Recursive Model for Audio & TTS

A Tiny Recursive Model (TRM) implementation for Text-to-Speech and audio token prediction using the Kyutai Mimi codec.

## Overview

This project implements the TRM architecture from ["Less is More: Recursive Reasoning with Tiny Networks"](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) and adapts it for:

1. **TTS (Text-to-Speech)**: `[speaker_id]text` → `mimi_tokens`
2. **Audio Token Prediction**: `mimi_tokens` → `mimi_tokens`

### Key Features

- **TTS Support**: Input text + speaker ID, output audio tokens
- **Recursive Reasoning**: TRM progressively refines predictions through multiple improvement cycles
- **Parameter Efficient**: Achieves strong results with only ~7-20M parameters
- **Mimi Integration**: Uses Kyutai's state-of-the-art neural audio codec for tokenization
- **Pre-tokenization**: Fast training with cached Mimi tokens

## Architecture

### Audio Token Prediction (TRM)

- **Offline**: Raw audio → `MimiEncoder` → Mimi tokens `[B, C, T]`
- **Model input**: Mimi tokens only (no waveform)

```
Mimi Tokens [B, C, T] (precomputed)
          ↓
  Flatten + Codebook Offsets
          ↓
   Token + Codebook Embeddings
          ↓
 (optional) Positional Encoding
          ↓
        x: Input Embeddings
          ↓
    Initialize y, z from learned vectors
          ↓
    ┌───────────────────────────────────────┐
    │ For K high-level steps (H_cycles):   │
    │   For n recursive steps (L_cycles):  │
    │     z = RecursiveBlock(x, y, z)      │
    │   y = AnswerUpdater(y, z)            │
    └───────────────────────────────────────┘
          ↓
     Output Projection
          ↓
  Predicted Mimi Tokens
```

### TTS (Text + Speaker → Mimi Tokens)

```
Text IDs [B, T_text]           Speaker ID [B]
          ↓                             ↓
    TextEncoder (Transformer)    Speaker Embedding
          ↓                             ↓
      Text Embeddings        Speaker Projection to audio dim
                 ↘          ↙
             Text Context [B, T_text, D_audio]
                          ↓
                 (Duration Predictor)
                          ↓
             T_audio frames (length for y, z)
                          ↓
  Initialize y, z ∈ R[B, T_audio, D_audio] + Audio Positional Encoding
                          ↓
    ┌───────────────────────────────────────┐
    │ For K high-level steps (H_cycles):   │
    │   For n recursive steps (L_cycles):  │
    │     z = RecursiveBlock(text_ctx,     │
    │                        y, z)         │
    │   y = AnswerUpdater(y, z)            │
    └───────────────────────────────────────┘
                          ↓
                   Output Projection
                          ↓
          Predicted Mimi Tokens (first codebook)
```

## Installation

```bash
# Clone the repository
cd TRM-RESEARCH

# Install dependencies
pip install -e .

# Optional: Install wandb for logging
pip install -e ".[wandb]"
```

## Quick Start

### TTS Training (Recommended)

```bash
# Step 1: Pre-tokenize audio with text/speaker info
python pretokenize.py --data-dir test_dataset --tts

# Step 2: Train TTS model (text + speaker → mimi tokens)
python train_tts.py \
    --data-dir test_dataset \
    --epochs 100 \
    --batch-size 8 \
    --max-audio-frames 256
```

**Input**: `[1]: Hey, hast du...` (speaker ID + text)  
**Output**: Mimi audio tokens

### Audio-only Training

```bash
# Pre-tokenize (audio only, no text)
python pretokenize.py --data-dir test_dataset

# Train with pre-tokenized data (faster)
python train.py --data-dir test_dataset --use-pretokenized
```

### Using the TTS Model

```python
from models import TTSTRM, TTSTRMConfig

# Create TTS model
config = TTSTRMConfig(
    text_vocab_size=256,
    num_speakers=2,
    H_cycles=3,
    L_cycles=6,
)
model = TTSTRM(config).cuda()

# Generate audio tokens from text
text = "Hallo, wie geht es dir?"
text_ids = encode_text(text)  # Character-level encoding
speaker_id = torch.tensor([1])

tokens = model.generate(
    text_ids=text_ids.unsqueeze(0),
    speaker_id=speaker_id,
    num_frames=100,
)
```

## Dataset Structure

For TTS training, organize your data as:

```
test_dataset/
└── de/
    └── <conversation_id>/
        ├── full_conversation.wav
        └── segments/
            ├── 001_speaker1.wav        # Line 1 audio
            ├── 002_speaker2.wav        # Line 2 audio
            ├── 003_speaker1.wav        # Line 3 audio
            └── vibevoice-podcast-script.txt
```

**Transcript format** (`vibevoice-podcast-script.txt`):
```
[1]: Hey, hast du letztens die neue Aufgaben-App ausprobiert?
[2]: Ja, total handy – sie synchronisiert sich automatisch mit meinem Kalender.
[1]: Ich nutze gerade „Todoist"...
```

The segment files map to transcript lines: `001_speaker1.wav` → line 1, etc.

## Model Configuration

Key TRM parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H_cycles` | 3 | High-level improvement steps (K) |
| `L_cycles` | 6 | Low-level recursive cycles (n) |
| `L_layers` | 2 | Depth of recursive blocks |
| `embed_dim` | 256 | Embedding dimension |
| `use_attention` | True | Use attention (False = MLP only) |

## Mimi Codec Details

- **Sample Rate**: 24 kHz
- **Frame Rate**: 12.5 Hz (80ms per frame)
- **Codebooks**: 32 RVQ codebooks (semantic + acoustic)
- **Vocabulary**: 2048 tokens per codebook
- **Bandwidth**: ~1.1 kbps

## Training Tips

- Start with short audio lengths (2-5s) to reduce memory usage
- Use batch size 1-2 on consumer GPUs / MPS
- The model has ~7-20M parameters depending on num_codebooks
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` on Apple Silicon for better memory handling

## References

- [TRM Paper](https://arxiv.org/abs/2510.04871) - "Less is More: Recursive Reasoning with Tiny Networks"
- [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) - Samsung AI Lab Montreal
- [Mimi Codec](https://huggingface.co/kyutai/mimi) - Kyutai Labs

## License

MIT License

