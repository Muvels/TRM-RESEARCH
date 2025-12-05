# TTS TRM: How It Works

A detailed technical guide to the Text-to-Speech Tiny Recursive Model (TTS TRM).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Input/Output Format](#inputoutput-format)
4. [The Mimi Audio Codec](#the-mimi-audio-codec)
5. [Multi-Codebook Prediction](#multi-codebook-prediction)
6. [Training Process](#training-process)
7. [Loss Computation](#loss-computation)
8. [Inference & Generation](#inference--generation)
9. [Sample Generation During Training](#sample-generation-during-training)
10. [CLI Reference](#cli-reference)

---

## Overview

The TTS TRM is a non-autoregressive text-to-speech model that converts text + speaker identity into discrete audio tokens (Mimi codec tokens). Unlike traditional autoregressive TTS models that generate one token at a time, TTS TRM uses **recursive refinement** to iteratively improve all output positions in parallel.

```
┌─────────────────────────────────────────────────────────────────┐
│                        TTS TRM Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input:                          Output:                        │
│   ┌──────────────┐               ┌────────────────────────────┐ │
│   │ "Hello world"│               │ Mimi Tokens [32 codebooks] │ │
│   │ Speaker ID: 0│  ──────────►  │ [32, T_frames]             │ │
│   └──────────────┘               └────────────────────────────┘ │
│                                           │                      │
│                                           ▼                      │
│                                  ┌────────────────┐              │
│                                  │  Mimi Decoder  │              │
│                                  └────────────────┘              │
│                                           │                      │
│                                           ▼                      │
│                                  ┌────────────────┐              │
│                                  │  Audio [24kHz] │              │
│                                  └────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### High-Level Structure

```
                    ┌─────────────────────────────────────────┐
                    │              TTS TRM Model               │
                    └─────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│  Text Encoder │              │ Speaker Embed │              │ Audio Decoder │
│  (Transformer)│              │  (Embedding)  │              │    (TRM)      │
└───────────────┘              └───────────────┘              └───────────────┘
```

### Component Details

#### 1. Text Encoder
- **Type**: Transformer Encoder
- **Input**: Character-level text tokens `[B, T_text]`
- **Output**: Contextual text embeddings `[B, T_text, D]`
- **Layers**: 4 (default)
- **Heads**: 8 (default)
- **Dimension**: 256 (default)

```python
# Text encoding flow
text_ids: [B, T_text]           # Character indices
    │
    ▼ Character Embedding
embeddings: [B, T_text, D]
    │
    ▼ Positional Encoding (Sinusoidal)
embeddings: [B, T_text, D]
    │
    ▼ Transformer Encoder (4 layers)
text_context: [B, T_text, D]    # Contextual representations
```

#### 2. Speaker Embedding
- **Type**: Learnable Embedding + Projection
- **Input**: Speaker ID `[B]`
- **Output**: Speaker vector `[B, D]`

The speaker embedding is added to all text positions to condition the generation on speaker identity.

#### 3. Audio Decoder (TRM Core)

The heart of the model uses **recursive refinement**:

```
┌────────────────────────────────────────────────────────────────────┐
│                      Recursive Refinement Loop                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Initialize:                                                        │
│    y = learnable_init_y  (answer)     [B, T_audio, D]              │
│    z = learnable_init_z  (latent)     [B, T_audio, D]              │
│                                                                     │
│  For h = 1 to H_cycles (3):           # High-level refinement      │
│    │                                                                │
│    │  For l = 1 to L_cycles (6):      # Low-level reasoning        │
│    │    │                                                          │
│    │    │  For each RecursiveBlock:                                │
│    │    │    z = SelfAttention(z)                                  │
│    │    │    z = CrossAttention(z, text_context)                   │
│    │    │    z = CrossAttention(z, y)                              │
│    │    │    z = FFN(z)                                            │
│    │                                                                │
│    │  y = AnswerUpdater(y, z)         # Update answer from latent  │
│                                                                     │
│  Output: y  [B, T_audio, D]                                        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Key insight**: The model maintains two representations:
- **y (answer)**: The current best guess for the audio representation
- **z (latent)**: A working memory that reasons about text, y, and itself

Through multiple cycles, z refines its understanding and updates y to be progressively better.

#### 4. Multi-Codebook Output Head

```
y: [B, T_audio, D]
    │
    ▼ For each codebook c = 0..31:
    │
    │   codebook_embed[c]: [1, D]         # Learned codebook embedding
    │   y_c = y + codebook_embed[c]       # Condition on codebook index
    │   logits[c] = output_proj[c](y_c)   # Per-codebook projection
    │
    ▼
logits: [B, 32, T_audio, 2048]            # All codebook predictions
```

---

## Input/Output Format

### Training Input

| Field | Shape | Description |
|-------|-------|-------------|
| `text_ids` | `[B, T_text]` | Character-level token indices |
| `text_mask` | `[B, T_text]` | Attention mask (1=valid, 0=padding) |
| `speaker_id` | `[B]` | Speaker index (0 to num_speakers-1) |
| `audio_tokens` | `[B, 32, T_audio]` | Target Mimi tokens (all 32 codebooks) |

### Training Output

| Field | Shape | Description |
|-------|-------|-------------|
| `logits` | `[B, 32, T_audio, 2048]` | Predicted logits for all codebooks |
| `loss` | scalar | Weighted cross-entropy loss |

### Inference Output

| Field | Shape | Description |
|-------|-------|-------------|
| `tokens` | `[B, 32, T_audio]` | Sampled audio tokens |

---

## The Mimi Audio Codec

[Mimi](https://huggingface.co/kyutai/mimi) is Kyutai's neural audio codec that compresses 24kHz audio to discrete tokens.

### Key Properties

| Property | Value |
|----------|-------|
| Sample Rate | 24,000 Hz |
| Frame Rate | 12.5 Hz (80ms per frame) |
| Num Codebooks | 32 |
| Codebook Size | 2,048 tokens each |
| Bitrate | ~1.1 kbps |

### Codebook Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mimi Codebook Structure                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Codebook 0   ████████████████████████████  Semantic Content    │
│               (WHAT is being said)                               │
│               - Phonetic information                             │
│               - Word identity                                    │
│               - Core linguistic content                          │
│                                                                  │
│  Codebooks    ████████████████              Acoustic Detail     │
│  1-7          (Early refinement)                                 │
│               - Prosody                                          │
│               - Basic pitch/energy                               │
│                                                                  │
│  Codebooks    ████████                      Fine Detail         │
│  8-31         (Acoustic texture)                                 │
│               - Timbre nuances                                   │
│               - High-frequency detail                            │
│               - Speaker characteristics                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Frame-to-Audio Relationship

```
1 second of audio at 24kHz = 24,000 samples
                           = 12.5 Mimi frames
                           = 12.5 × 32 = 400 tokens total

Example:
  Text: "Hello" (5 chars)
  Estimated frames: 5 × 3 = 15 frames
  Audio duration: 15 / 12.5 = 1.2 seconds
  Total tokens: 15 × 32 = 480 tokens
```

---

## Multi-Codebook Prediction

### Why Predict All 32 Codebooks?

The TTS TRM predicts **all 32 codebooks simultaneously** rather than just the semantic codebook. This allows:

1. **End-to-end training**: No need for a separate acoustic model
2. **Joint optimization**: All codebooks learn together
3. **Faster inference**: Single forward pass generates complete audio

### Codebook Embeddings

Each codebook has a learned embedding that tells the model which level of detail to predict:

```python
# Codebook conditioning
codebook_embed = Embedding(32, codebook_dim)  # [32, 64]

for c in range(32):
    cb_emb = codebook_embed[c]      # [64]
    cb_emb = project(cb_emb)        # [256]
    y_c = y + cb_emb                # Add codebook identity
    logits[c] = head[c](y_c)        # Predict this codebook
```

### Output Head Options

| Mode | Parameters | Description |
|------|------------|-------------|
| `share_output_heads=False` | 32 × (256 × 2048) = 16.8M | Separate projection per codebook |
| `share_output_heads=True` | 1 × (256 × 2048) = 0.5M | Shared projection, codebook-conditioned |

---

## Training Process

### Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                         Training Step                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Load Batch                                                      │
│     ├── text_ids:     [B, T_text]                                  │
│     ├── speaker_id:   [B]                                          │
│     ├── audio_tokens: [B, 32, T_audio]  (target)                   │
│     └── text_mask:    [B, T_text]                                  │
│                                                                     │
│  2. Forward Pass                                                    │
│     │                                                               │
│     ├── Encode text:    text_ctx = TextEncoder(text_ids)           │
│     ├── Add speaker:    text_ctx += speaker_proj(speaker_id)       │
│     ├── Init y, z:      [B, T_audio, D]                            │
│     │                                                               │
│     ├── Recursive refinement (H=3, L=6):                           │
│     │   └── z updates, y updates                                   │
│     │                                                               │
│     └── Output:         logits = MultiCodebookHead(y)              │
│                         [B, 32, T_audio, 2048]                     │
│                                                                     │
│  3. Compute Loss                                                    │
│     │                                                               │
│     ├── For each codebook c:                                       │
│     │   loss[c] = CrossEntropy(logits[:, c], audio_tokens[:, c])  │
│     │                                                               │
│     └── Weighted sum:                                               │
│         total_loss = Σ weight[c] × loss[c]                         │
│                                                                     │
│  4. Backward + Optimize                                             │
│     │                                                               │
│     ├── loss.backward()                                            │
│     ├── clip_grad_norm_(parameters, 1.0)                           │
│     └── optimizer.step()                                           │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Samples per batch |
| `lr` | 1e-4 | Learning rate |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `epochs` | 100 | Training epochs |
| `H_cycles` | 3 | High-level refinement cycles |
| `L_cycles` | 6 | Low-level reasoning cycles |

---

## Loss Computation

### Weighted Multi-Codebook Loss

Not all codebooks are equally important. The semantic codebook (0) is critical for intelligibility, while later codebooks add polish.

```
┌────────────────────────────────────────────────────────────────────┐
│                    Codebook Loss Weights                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Codebook    Weight    Importance                                  │
│  ────────    ──────    ──────────                                  │
│     0         2.0      ████████████████████  Semantic (critical)   │
│     1         1.43     ██████████████        Early acoustic        │
│     2         1.36     █████████████                               │
│     3         1.29     ████████████                                │
│     4         1.21     ███████████                                 │
│     5         1.14     ██████████                                  │
│     6         1.07     █████████                                   │
│     7         1.00     ████████              Transition            │
│   8-31        1.00     ████████              Fine acoustic         │
│                                                                     │
│  Formula:                                                           │
│    weight[0] = 2.0                                                 │
│    weight[1-7] = 1.5 - (0.5 × i / 7)                              │
│    weight[8-31] = 1.0                                              │
│    weights = normalize(weights)  # Mean = 1.0                      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Loss Calculation

```python
def compute_loss(logits, targets, weights):
    """
    logits:  [B, 32, T, 2048]
    targets: [B, 32, T]
    weights: [32]
    """
    losses = []
    for c in range(32):
        loss_c = cross_entropy(
            logits[:, c].reshape(-1, 2048),
            targets[:, c].reshape(-1),
            ignore_index=0  # Padding
        )
        losses.append(loss_c)
    
    losses = torch.stack(losses)  # [32]
    weighted_loss = (losses * weights).mean()
    
    return weighted_loss
```

---

## Inference & Generation

### Generation Process

```python
@torch.no_grad()
def generate(text, speaker_id, num_frames):
    # 1. Encode text
    text_ids = encode_text(text)
    
    # 2. Forward pass (no targets)
    logits, _ = model(text_ids, speaker_id, target_audio_frames=num_frames)
    # logits: [1, 32, T, 2048]
    
    # 3. Sample tokens for each codebook
    tokens = []
    for c in range(32):
        logits_c = logits[0, c] / temperature
        logits_c = top_k_filter(logits_c, k=50)
        probs = softmax(logits_c)
        tokens_c = multinomial(probs)
        tokens.append(tokens_c)
    
    tokens = stack(tokens)  # [32, T]
    
    # 4. Decode to audio
    audio = mimi.decode(tokens)  # [1, T_samples]
    
    return audio
```

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Randomness (lower = more deterministic) |
| `top_k` | 50 | Keep only top-k tokens |
| `top_p` | None | Nucleus sampling threshold |

### Duration Estimation

When `num_frames` is not specified, it's estimated from text length:

```python
num_frames = min(len(text) * 3, max_audio_frames)
# ~3 frames per character
# At 12.5 Hz, that's ~240ms per character
```

---

## Sample Generation During Training

### Purpose

Generate audio samples periodically during training to:
1. **Monitor quality**: Hear how the model improves
2. **Detect issues**: Catch problems early (gibberish, wrong speaker, etc.)
3. **Compare**: A/B test against ground truth

### Configuration

```bash
python train_tts.py \
    --sample-interval 500 \        # Generate every 500 steps
    --num-samples 3 \              # 3 samples per generation
    --sample-temperature 0.8 \     # Sampling temperature
    --sample-top-k 50              # Top-k filtering
```

### Custom Prompts

#### Option 1: CLI
```bash
--sample-prompts "Hello world" "Testing one two three" "The quick brown fox"
--sample-speaker 0
```

#### Option 2: File
```bash
--sample-prompts-file prompts.txt
```

File format (`prompts.txt`):
```
# Comments start with #
0|Hello, this is speaker zero.
1|And this is speaker one.
Just text uses default speaker.
```

### Output Structure

```
outputs/tts/
├── samples/
│   ├── sample_1_ground_truth.wav     # Reference (if from dataset)
│   ├── sample_2_ground_truth.wav
│   ├── sample_3_ground_truth.wav
│   │
│   ├── step_000500/
│   │   ├── sample_1_generated.wav
│   │   ├── sample_2_generated.wav
│   │   ├── sample_3_generated.wav
│   │   └── info.json
│   │
│   ├── step_001000/
│   │   └── ...
│   │
│   └── step_001500/
│       └── ...
│
└── checkpoints/
    ├── best.pt
    ├── step_1000.pt
    └── final.pt
```

---

## CLI Reference

### Full Command Reference

```bash
python train_tts.py [OPTIONS]
```

#### Data Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `test_dataset` | Data directory |
| `--token-dir` | `{data-dir}/mimi_tokens` | Pre-tokenized data |
| `--max-text-length` | `512` | Maximum text length (chars) |
| `--max-audio-frames` | `256` | Maximum audio frames |

#### Model Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--text-embed-dim` | `256` | Text embedding dimension |
| `--audio-embed-dim` | `256` | Audio embedding dimension |
| `--hidden-dim` | `512` | FFN hidden dimension |
| `--text-layers` | `4` | Text encoder layers |
| `--L-layers` | `2` | Recursive block layers |
| `--H-cycles` | `3` | High-level improvement cycles |
| `--L-cycles` | `6` | Low-level reasoning cycles |
| `--num-heads` | `8` | Attention heads |
| `--share-output-heads` | `False` | Share projection across codebooks |

#### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `0.1` | Weight decay |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--fp16` | `False` | Mixed precision training |

#### Logging Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-interval` | `10` | Log every N steps |
| `--eval-interval` | `500` | Evaluate every N steps |
| `--save-interval` | `1000` | Checkpoint every N steps |
| `--output-dir` | `outputs/tts` | Output directory |
| `--run-name` | auto | Run name for W&B |
| `--wandb` | `False` | Enable W&B logging |

#### Sample Generation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample-interval` | `0` | Generate every N steps (0=off) |
| `--num-samples` | `3` | Samples per generation |
| `--sample-prompts` | None | Custom text prompts |
| `--sample-prompts-file` | None | File with prompts |
| `--sample-speaker` | `0` | Speaker for custom prompts |
| `--sample-temperature` | `0.8` | Sampling temperature |
| `--sample-top-k` | `50` | Top-k filtering |

#### System Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-workers` | `4` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | Device (auto/cuda/mps/cpu) |

---

## Example Training Run

```bash
# Full training with sample generation
python train_tts.py \
    --data-dir test_dataset \
    --epochs 50 \
    --batch-size 16 \
    --lr 3e-4 \
    --H-cycles 3 \
    --L-cycles 6 \
    --sample-interval 500 \
    --sample-prompts "Hello, my name is Claude." "The weather is nice today." \
    --sample-speaker 0 \
    --wandb
```

Expected output:
```
Using device: cuda

📁 Setting up TTS data...
   Text vocab size: 87
   Speakers: 2
   Codebooks: 32
   Codebook size: 2048

📦 Creating TTS TRM model (full multi-codebook)...
   Parameters: 12,345,678 (12.35M)
   Output heads: separate per codebook

📊 Codebook loss weights:
   Semantic (CB0): 2.00
   Early acoustic (CB1-7): 1.21 avg
   Late acoustic (CB8-31): 1.00 avg

📝 Using custom prompts from CLI:
   1. [speaker 0] "Hello, my name is Claude."
   2. [speaker 0] "The weather is nice today."

🚀 Starting TTS training for 50 epochs...
   Total steps: 12500
   Input: text + speaker_id → Output: ALL 32 mimi codebooks
   Sample generation: every 500 steps (2 samples)

Epoch 1: 100%|████████| 250/250 [02:30<00:00, loss=4.2341, lr=3.00e-04]
...
```

---

## Appendix: Model Size Calculation

```
Component                          Parameters
─────────────────────────────────────────────
Text Encoder:
  - Embedding:                     256 × 256 = 65,536
  - 4 Transformer Layers:          4 × (4 × 256² + 2 × 256 × 512) ≈ 2.1M
  - Layer Norm:                    2 × 256 = 512

Speaker Embedding:
  - Embedding:                     11 × 64 = 704
  - Projection:                    64 × 256 = 16,384

Audio Decoder:
  - Init y, z:                     2 × 256 = 512
  - 2 RecursiveBlocks:             2 × (3 attention + FFN) ≈ 2.6M
  - AnswerUpdater:                 ≈ 0.8M

Multi-Codebook Head:
  - Codebook Embedding:            32 × 64 = 2,048
  - Codebook Projection:           64 × 256 = 16,384
  - 32 Output Heads:               32 × 256 × 2048 = 16.8M
  (or shared: 1 × 256 × 2048 = 0.5M)

─────────────────────────────────────────────
Total (separate heads):            ~22M parameters
Total (shared heads):              ~6M parameters
```

---

*Last updated: December 2024*


