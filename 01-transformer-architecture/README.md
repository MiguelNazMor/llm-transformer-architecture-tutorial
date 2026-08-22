# 1. The Transformer Architecture

## Why Transformers?

Before 2017, sequence-to-sequence tasks (translation, summarization, speech recognition) were dominated by recurrent neural networks (RNNs, LSTMs, GRUs) and convolutional models. These architectures had fundamental problems:

- **Sequential computation**: RNNs process tokens one at a time. You cannot compute token *t* until you've computed token *t-1*. This makes training painfully slow on long sequences.
- **Vanishing gradients**: Information from early tokens gets diluted as it propagates through many recurrent steps.
- **Long-range dependencies**: RNNs struggle to connect related words that are far apart in a sentence (e.g., "The cat that ate the fish that swam in the pond that was near the house **is** black" — the model must link "cat" to "is").

The Transformer, introduced by Vaswani et al. in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017), solved these problems by replacing recurrence entirely with a mechanism called **self-attention**. The key insight: instead of processing tokens sequentially, let every token look at every other token in parallel.

---

## The Big Picture

A Transformer is an **encoder-decoder** architecture built from stacked identical blocks. The original paper used 6 encoder blocks and 6 decoder blocks.

```
INPUT: "The cat sat on the mat"
  │
  ▼
┌─────────────────────────┐
│   Input Embedding        │  ← tokens → vectors (d_model = 512)
│ + Positional Encoding    │  ← inject order information
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Encoder Block × N      │  ← N = 6 in the original paper
│   ┌───────────────────┐  │
│   │ Multi-Head         │  │
│   │ Self-Attention     │  │
│   │     + Add & Norm   │  │
│   ├───────────────────┤  │
│   │ Feed-Forward       │  │
│   │ Network            │  │
│   │     + Add & Norm   │  │
│   └───────────────────┘  │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Decoder Block × N      │  ← also 6 blocks
│   ┌───────────────────┐  │
│   │ Masked Multi-Head  │  │  ← can only see past tokens
│   │ Self-Attention     │  │
│   │     + Add & Norm   │  │
│   ├───────────────────┤  │
│   │ Cross-Attention    │  │  ← attends to encoder output
│   │     + Add & Norm   │  │
│   ├───────────────────┤  │
│   │ Feed-Forward       │  │
│   │ Network            │  │
│   │     + Add & Norm   │  │
│   └───────────────────┘  │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Linear + Softmax       │  ← project to vocabulary size
└─────────────────────────┘
  │
  ▼
OUTPUT: "El gato se sentó en la alfombra"
```

---

## 1. Input Embedding + Positional Encoding

### Token Embeddings

Raw text is first tokenized (see tokenization section below) into a sequence of token IDs. Each token ID is mapped to a dense vector of size `d_model` (512 in the original paper) through a learned embedding matrix `E ∈ R^(vocab_size × d_model)`.

### Positional Encoding

Self-attention has no notion of token order — it treats the input as a set, not a sequence. To fix this, we add **positional encodings** to the input embeddings before feeding them into the first layer.

The original paper uses **sinusoidal** positional encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the token's position in the sequence
- `i` is the dimension index (0 to d_model/2 - 1)
- `10000^(2i/d_model)` creates a geometric progression of wavelengths from 2π to ~20000π

**Why sinusoids?**
- They produce unique encodings for every position.
- The model can easily learn to attend to relative positions because `PE(pos+k)` can be expressed as a linear function of `PE(pos)`.
- They generalize to sequence lengths not seen during training (unlike learned positional embeddings).
- The different frequencies allow the model to capture both short-range and long-range positional relationships.

Modern models (GPT, Llama) typically use **learned positional embeddings** or **RoPE (Rotary Position Embeddings)** instead, which we'll cover later.

> **Code:** [`src/embeddings.py`](src/embeddings.py) — `TokenEmbedding`, `sinusoidal_positional_encoding()`, `LearnedPositionalEmbedding`
> **Tests:** [`tests/test_embeddings.py`](tests/test_embeddings.py) — shape checks, unique positions, even/odd pattern, gradient flow for seen vs unseen tokens

---

## 2. Self-Attention — The Core Mechanism

Self-attention answers the question: *"For each word in the sequence, which other words should I pay attention to?"*

### Intuition

Consider the sentence: *"The animal didn't cross the street because it was too tired."*

What does "it" refer to? A human reader knows "it" refers to "the animal", not "the street". Self-attention lets the model learn these relationships automatically by computing attention scores between every pair of words.

### Scaled Dot-Product Attention

For each token, we compute three vectors:

- **Query (Q)**: "What am I looking for?" — the current token's representation of what it needs.
- **Key (K)**: "What do I contain?" — each token's representation of what information it offers.
- **Value (V)**: "What is my actual content?" — the information to aggregate.

All three are computed by multiplying the input embedding by learned weight matrices:

```
Q = X · W_Q    (d_model × d_k)
K = X · W_K    (d_model × d_k)
V = X · W_V    (d_model × d_v)
```

The attention formula:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Let's break this down step by step:

1. **`Q·K^T`** (compatibility scores): For each query, compute a dot product with every key. This gives a score for how much token *i* should attend to token *j*. Result shape: `(seq_len × seq_len)`.

2. **`/ √d_k`** (scaling): Without scaling, large `d_k` values push the softmax into regions with tiny gradients. Dividing by `√d_k` keeps the variance of the dot products at 1. This is why it's called *scaled* dot-product attention.

3. **`softmax(...)`** (normalization): Converts scores into a probability distribution over all tokens. Each row sums to 1. This means each token distributes its "attention budget" across all tokens.

4. **`· V`** (weighted sum): Each token's output is a weighted sum of all value vectors, where the weights are the attention probabilities.

### Why This Works

If token *i*'s query is similar to token *j*'s key, their dot product is large, and token *i* will incorporate a lot of token *j*'s value. The model learns to make queries and keys align when two tokens are semantically related.

> **Code:** [`src/attention.py`](src/attention.py) — `scaled_dot_product_attention()`, `create_causal_mask()`
> **Tests:** [`tests/test_attention.py`](tests/test_attention.py) — `TestScaledDotProductAttention` (output shape, sums to 1, causal mask, dropout)

---

## 3. Multi-Head Attention

A single attention function can only capture one type of relationship. **Multi-head attention** runs multiple attention operations in parallel, each with its own learned projections, allowing the model to attend to different aspects of the input simultaneously.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

The original paper uses `h = 8` heads with `d_k = d_v = d_model/h = 64`.

**What do different heads learn?** Research shows that different heads specialize in different linguistic phenomena:
- Some heads focus on **syntactic relations** (subject-verb agreement, dependency parsing).
- Others focus on **semantic relations** (coreference, synonymy).
- Some attend to **adjacent tokens** (local context), others to **far-away tokens** (global context).
- In practice, a few heads capture most of the useful patterns; many are redundant.

**Computational complexity**: O(n² · d_model) where n is the sequence length. This quadratic cost is the main bottleneck for long sequences and the reason for FlashAttention and other optimizations.

> **Code:** [`src/attention.py`](src/attention.py) — `MultiHeadAttention` (forward, backward, cross_attention_forward)
> **Tests:** [`tests/test_attention.py`](tests/test_attention.py) — `TestMultiHeadAttention` (shape, divisibility, cross-attn, determinism) + `TestAttentionTextLearning` (learns to focus, causal mask blocks future, gradient non-zero)

---

## 4. Feed-Forward Network (FFN)

After multi-head attention, each token's representation goes through a position-wise feed-forward network (the same FFN is applied to each position independently):

```
FFN(x) = ReLU(x·W_1 + b_1)·W_2 + b_2
```

In the original paper, the inner dimension is `d_ff = 2048` (4× d_model = 512).

**Purpose**: While attention mixes information *between* tokens, the FFN processes each token's representation *independently*. It adds non-linearity and increases the model's capacity to learn complex transformations. Together, attention + FFN allow the model to both gather context and transform representations.

Modern LLMs often use variants like **SwiGLU** activations instead of ReLU:

```
SwiGLU(x) = (x·W_1 ⊙ Swish(x·W_2))·W_3
```

Where `Swish(x) = x · σ(x)` and `⊙` is element-wise multiplication. SwiGLU consistently outperforms ReLU in large models (used in Llama, PaLM).

> **Code:** [`src/feed_forward.py`](src/feed_forward.py) — `FeedForward` (ReLU), `SwiGLUFeedForward` (SwiGLU), both with `forward()` and `backward()`
> **Tests:** [`tests/test_feed_forward.py`](tests/test_feed_forward.py) — shape, non-linearity, default d_ff + `TestFeedForwardTextLearning` (learns target, gradient non-zero, gradient check)

---

## 5. Add & Norm (Residual Connections + Layer Normalization)

Every sub-layer (attention and FFN) is wrapped with:

```
LayerNorm(x + Sublayer(x))
```

**Residual connection (`x + ...`)**: The input is added back to the sub-layer's output. This:
- Allows gradients to flow directly through the network during backpropagation.
- Makes it possible to train very deep networks (the original Transformer has up to ~100 layers if you count sub-layers).
- Lets the model easily "skip" a sub-layer if it's not useful (by learning near-zero outputs).

**Layer Normalization**: Normalizes the activations across the feature dimension (not the batch dimension, unlike BatchNorm). This stabilizes training and reduces sensitivity to weight initialization.

The original paper uses **post-layer norm** (add then normalize). Modern architectures (GPT-2, Llama) use **pre-layer norm** (normalize before the sub-layer):

```
x + Sublayer(LayerNorm(x))
```

Pre-norm is more stable during training, especially for very deep models.

> **Code:** [`src/transformer_block.py`](src/transformer_block.py) — `layer_norm()`, `backward_layer_norm()`, `RMSNorm`, `EncoderBlock`, `DecoderBlock` (all with `forward()` + `backward()`)
> **Tests:** [`tests/test_model.py`](tests/test_model.py) — `TestTransformer` (encode/decode shapes) + block-level gradient tests in `TestGPTTextLearning`

---

## 6. Encoder vs Decoder

### Encoder

- Processes the entire input sequence **bidirectionally** — each token can attend to all other tokens (both left and right).
- Produces contextualized representations of the input.
- Used in BERT and other understanding-focused models.

### Decoder

The decoder has two attention sub-layers:

1. **Masked self-attention**: Like encoder self-attention, but tokens can only attend to *previous* tokens (including themselves). This is achieved by setting attention scores for future positions to `-∞` before the softmax. This is essential for autoregressive generation — the model should not "cheat" by looking ahead.

2. **Cross-attention**: The queries come from the decoder, but the keys and values come from the encoder's output. This is how the decoder incorporates information from the input sequence.

### Decoder-Only Models (GPT, Llama)

Modern language models like GPT and Llama use only the decoder stack (no encoder, no cross-attention). They are trained to predict the next token given previous tokens (autoregressive language modeling). The masked self-attention naturally handles this.

**Why decoder-only?**
- Simpler architecture.
- Scales better for generative tasks.
- Can be trained on vast amounts of unlabeled text (next-token prediction is self-supervised).
- The same model can be used for many tasks by reformulating them as text generation (in-context learning).

> **Code:** [`src/model.py`](src/model.py) — `Transformer` (encoder-decoder), `GPT` (decoder-only with `forward()`, `backward()`, `generate()`)
> **Tests:** [`tests/test_model.py`](tests/test_model.py) — `TestGPT` (forward shape, determinism, generation, sinusoidal pos) + `TestGPTTextLearning` (loss decreases, predicts next token, generation contains corpus words, backward produces gradients)

---

## 7. Tokenization

Tokenization is the bridge between raw text and the model's vocabulary. It splits text into tokens and maps each token to an integer ID.

### Byte-Pair Encoding (BPE)

Used by GPT models. The algorithm:

1. Start with characters as the initial vocabulary.
2. Count all adjacent symbol pairs in the training corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the desired vocabulary size is reached.

**Example**: With BPE, "lower" might be tokenized as `["low", "er"]`, while "lowest" becomes `["low", "est"]`. Common words stay as single tokens; rare words get split into subword units.

### WordPiece

Used by BERT. Similar to BPE but merges based on likelihood improvement rather than raw frequency:

```
score = count(pair) / (count(first) · count(second))
```

### Unigram Language Model

Used by T5 and some Llama variants. Starts with a large vocabulary and iteratively removes tokens that increase the training loss the least.

### Special Tokens

- `[CLS]` (BERT): Classification token whose output represents the whole sequence.
- `[SEP]` (BERT): Separator between sentences.
- `<|endoftext|>` (GPT): End-of-text marker.
- `<s>`, `</s>` (Llama): Beginning and end of sequence.
- `[PAD]`: Padding token to make all sequences in a batch the same length.
- `[MASK]` (BERT): Masked token for masked language modeling.

> **Code:** [`src/tokenizer.py`](src/tokenizer.py) — `BPETokenizer` (train, encode, decode, encode_batch)
> **Tests:** [`tests/test_tokenizer.py`](tests/test_tokenizer.py) — initialization, training, roundtrip, unknown words, batch encoding + `TestBPETokenizerTextLearning` (frequent word becomes single token, fewer tokens than rare, vocab entries, encoding consistency)

---

## 8. Training the Transformer

### Original Training Objective (Translation)

The original Transformer was trained on machine translation (WMT 2014 English-German and English-French):

- **Loss**: Cross-entropy loss between predicted token probabilities and the ground-truth target tokens.
- **Label smoothing**: ε = 0.1, which prevents the model from becoming overconfident.
- **Optimizer**: Adam with β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹.
- **Learning rate schedule**: Warmup for 4000 steps, then decay proportionally to `1/√step`.
- **Dropout**: 0.1 on attention weights and FFN activations.
- **Batch**: ~25,000 source and target tokens per batch.

### Why This Training Recipe Matters

Many of these design choices (warmup + decay schedule, Adam β₂ = 0.98 instead of 0.999, label smoothing) were carefully tuned for the Transformer. Modern LLMs inherit many of these choices but with important modifications (AdamW, cosine schedules, larger batches).

> **Code:** [`src/trainer.py`](src/trainer.py) — `SGD` optimizer (with momentum + gradient clipping), `Trainer` (training loop), `prepare_text_batch()`
> **Code:** [`src/train.py`](src/train.py) — End-to-end training demo (tokenizer → model → train → generate)
> **Code:** [`src/model.py`](src/model.py) — `cross_entropy_loss()`, `softmax_cross_entropy_backward()`, `GPT.backward()`
> **Tests:** [`tests/test_text_prediction.py`](tests/test_text_prediction.py) — loss decreases, overfitting, next-token prediction, generation quality, gradient correctness (analytical vs numerical)

---

## 9. Beyond the Original: Modern Transformer Variants

### GPT (Decoder-Only)

- Only the decoder stack.
- Autoregressive left-to-right language modeling.
- Uses learned positional embeddings instead of sinusoidal.
- Pre-layer normalization (norm before attention/FFN).
- Scaled through increasing layers, width, and training data.

### BERT (Encoder-Only)

- Only the encoder stack.
- Trained with **Masked Language Modeling** (MLM): randomly mask 15% of tokens and predict them from context.
- Also trained with **Next Sentence Prediction** (NSP).
- Bidirectional context — ideal for understanding tasks (classification, QA, NER).

### T5 (Encoder-Decoder)

- Full encoder-decoder architecture.
- All NLP tasks cast as text-to-text: "translate English to German: The cat..." → "Die Katze..."
- Uses relative positional embeddings instead of absolute.

### Llama (Decoder-Only, Optimized)

- **RoPE** (Rotary Position Embeddings): Encodes relative position by rotating query and key vectors.
- **RMSNorm** instead of LayerNorm: Removes the mean-centering step for speed.
- **SwiGLU** activation in FFN: Better performance than ReLU/GELU.
- **Pre-norm** architecture for training stability.

> **Code (in this repo):**
> - SwiGLU FFN: [`src/feed_forward.py`](src/feed_forward.py) — `SwiGLUFeedForward`
> - RMSNorm: [`src/transformer_block.py`](src/transformer_block.py) — `RMSNorm`
> - Learned positional embeddings: [`src/embeddings.py`](src/embeddings.py) — `LearnedPositionalEmbedding`
> - GPT-style decoder-only model: [`src/model.py`](src/model.py) — `GPT`
> **Tests:** [`tests/test_feed_forward.py`](tests/test_feed_forward.py) (SwiGLU), [`tests/test_model.py`](tests/test_model.py) (`test_sinusoidal_pos`)

---

## Code Guide

The `src/` directory contains a full Transformer implementation in pure NumPy.
Every component is built from scratch — no deep learning framework required.

### Project Structure

```
src/
├── tokenizer.py         # BPE tokenizer (train, encode, decode)
├── attention.py         # Scaled dot-product + multi-head attention (+ backward)
├── embeddings.py        # Token embeddings + positional encodings (+ backward)
├── feed_forward.py      # ReLU FFN + SwiGLU FFN (+ backward)
├── transformer_block.py # Encoder & decoder blocks + RMSNorm (+ backward)
├── model.py             # Full Transformer + GPT-style model (+ backward)
├── trainer.py           # SGD optimizer + training loop
└── train.py             # End-to-end training demo

tests/
├── test_tokenizer.py          # Tokenizer + text learning tests
├── test_attention.py          # Attention + text learning tests
├── test_embeddings.py         # Embeddings + text learning tests
├── test_feed_forward.py       # FFN + text learning tests
├── test_model.py              # Model + text learning tests
└── test_text_prediction.py    # End-to-end training & prediction tests
```

### Setup

All commands are run from the repo root (`01-transformers-adapters-lora-qlora/`).

```bash
# Install dependencies (numpy, pytest, ruff)
uv sync

# The source code lives in a subfolder, so set PYTHONPATH:
export PYTHONPATH=01-transformer-architecture/src
```

### Run the Training Demo

The demo ties everything together: trains a BPE tokenizer on a tiny corpus,
creates a small GPT model, trains it with manual backpropagation, and shows
text generation before and after training.

```bash
PYTHONPATH=01-transformer-architecture/src uv run python 01-transformer-architecture/src/train.py
```

**Expected output:**

```
============================================================
Transformer Architecture — Training Demo
============================================================

[1/6] Training BPE tokenizer on a tiny corpus...
  Corpus:          10 sentences
  Vocabulary size: 300
  Learned merges:  40
  Time:            0.00s

  Sample:   "the cat sat on the mat"
  Token IDs: [261, 267, 272, 273, 261, 270]
  Tokens:    ['the', 'cat', 'sat', 'on', 'the', 'mat']

[2/6] Creating a small GPT model...
  Architecture:  d_model=64, heads=4, layers=2, d_ff=256
  Parameters:    120,704

[3/6] Text generation BEFORE training...
  "the cat" → "the cat Û ö  ö ! , ,"       ← gibberish (random weights)
  "the dog" → "the dog ö ì ö ¸ Ü  Ç"
  "the sun" → "the s u n   ½ ö  ¸ ½"

[4/6] Training the model with backpropagation...
  Initial loss: 5.7540
  Step  100/1000 | Epoch 10/100 | Loss: 1.2157
  Step  200/1000 | Epoch 20/100 | Loss: 0.3695
  ...
  Step 1000/1000 | Epoch 100/100 | Loss: 0.2084

  Final loss:   0.2084
  Loss reduction: 5.7540 → 0.2084 (96.4% decrease)
  Training time: 4.65s

[5/6] Text generation AFTER training...
  "the cat" → "the cat likes the warm mat the cat likes the"   ← coherent!
  "the dog" → "the dog likes the cold log the dog likes the"
  "the sun" → "the s u n is b ri g h t t o"

[6/6] Architecture details...
  Token embedding:   (300, 64)
  Position embedding: (32, 64)
  Block 0: attn heads=4, d_k=16, FFN=FeedForward
  Block 1: attn heads=4, d_k=16, FFN=FeedForward

============================================================
Demo complete!  All components verified:
  - BPE tokenizer (train, encode, decode)
  - Token + positional embeddings
  - Multi-head self-attention (with backward)
  - Feed-forward network (with backward)
  - Causal masking for autoregressive generation
  - Manual backpropagation + SGD optimizer
  - Real training loop with decreasing loss
============================================================
```

> **Note:** The model trains using **manual backpropagation** — analytical
> gradients are implemented by hand for every layer (attention, FFN,
> embeddings, layer norm).  No automatic differentiation framework is used.
> The before/after generation comparison shows the model going from gibberish
> to coherent corpus text as it learns.

### Run the Tests

```bash
PYTHONPATH=01-transformer-architecture/src uv run pytest 01-transformer-architecture/tests/ -v
```

**Expected output:** 71 passed in ~12s.

### Using Individual Modules

You can import and experiment with each module in a Python REPL or script.

#### Tokenizer (`tokenizer.py`)

Trains a BPE tokenizer and encodes/decodes text.

```python
from tokenizer import BPETokenizer

# Train on a corpus
corpus = ["the cat sat on the mat", "the dog sat on the log"]
tok = BPETokenizer(vocab_size=300)
tok.train(corpus)

# Encode text → token IDs
ids = tok.encode("the cat")
print(ids)  # e.g., [261, 267]

# Decode token IDs → text
text = tok.decode(ids)
print(text)  # "the cat"

# Batch encoding with padding
padded, masks = tok.encode_batch(["the cat", "the dog sat"], max_len=6)
print(padded)  # [[261, 267, 0, 0, 0, 0], [261, 269, 272, 0, 0, 0]]
print(masks)  # [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]]
```

#### Attention (`attention.py`)

Computes scaled dot-product attention and multi-head attention.

```python
import numpy as np
from attention import MultiHeadAttention, scaled_dot_product_attention, create_causal_mask

# Manual attention call
q = np.random.randn(2, 5, 64)  # (batch=2, seq_q=5, d_k=64)
k = np.random.randn(2, 7, 64)  # (batch=2, seq_k=7, d_k=64)
v = np.random.randn(2, 7, 64)  # (batch=2, seq_k=7, d_v=64)

output = scaled_dot_product_attention(q, k, v, training=False)
print(output.shape)  # (2, 5, 64)

# Multi-head attention layer
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = np.random.randn(2, 10, 512)

out = mha.forward(x, training=False)
print(out.shape)  # (2, 10, 512)

# Causal mask for autoregressive decoding
mask = create_causal_mask(4)
print(mask[0])
# [[1. 0. 0. 0.]
#  [1. 1. 0. 0.]
#  [1. 1. 1. 0.]
#  [1. 1. 1. 1.]]
```

#### Embeddings (`embeddings.py`)

Token embeddings and positional encodings.

```python
import numpy as np
from embeddings import (
    TokenEmbedding,
    sinusoidal_positional_encoding,
    LearnedPositionalEmbedding,
)

# Token embedding lookup
emb = TokenEmbedding(vocab_size=1000, d_model=512)
ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
x = emb.forward(ids)
print(x.shape)  # (1, 4, 512)

# Sinusoidal positional encoding (original Transformer)
pe = sinusoidal_positional_encoding(max_len=100, d_model=512)
print(pe.shape)  # (1, 100, 512)
print(pe[0, 0, :4])  # [0.0, 1.0, 0.0, 1.0] — sin(0)=0, cos(0)=1

# Learned positional embedding (GPT-style)
lpe = LearnedPositionalEmbedding(max_len=512, d_model=512)
pos = lpe.forward(seq_len=10)
print(pos.shape)  # (1, 10, 512)
```

#### Feed-Forward (`feed_forward.py`)

ReLU and SwiGLU feed-forward networks.

```python
import numpy as np
from feed_forward import FeedForward, SwiGLUFeedForward

x = np.random.randn(2, 10, 512)

# Original Transformer FFN (ReLU)
ffn_relu = FeedForward(d_model=512, d_ff=2048)
out = ffn_relu.forward(x)
print(out.shape)  # (2, 10, 512)
print(ffn_relu)  # FeedForward(d_model=512, d_ff=2048, activation=ReLU)

# Llama-style SwiGLU FFN
ffn_swiglu = SwiGLUFeedForward(d_model=512)
out = ffn_swiglu.forward(x)
print(out.shape)  # (2, 10, 512)
print(ffn_swiglu)  # SwiGLUFeedForward(d_model=512, d_ff=1536)
```

#### Transformer Blocks (`transformer_block.py`)

Encoder and decoder blocks with residual connections and layer norm.

```python
import numpy as np
from transformer_block import EncoderBlock, DecoderBlock, RMSNorm, layer_norm
from attention import create_causal_mask

x = np.random.randn(2, 10, 512)

# Encoder block (bidirectional self-attention)
enc = EncoderBlock(d_model=512, num_heads=8, d_ff=2048)
out = enc.forward(x, training=False)
print(out.shape)  # (2, 10, 512)

# Decoder block (masked self-attn + cross-attn + FFN)
dec = DecoderBlock(d_model=512, num_heads=8, d_ff=2048)
encoder_out = np.random.randn(2, 8, 512)  # from encoder
causal_mask = create_causal_mask(10)

out = dec.forward(x, encoder_out, causal_mask=causal_mask, training=False)
print(out.shape)  # (2, 10, 512)

# RMSNorm (used in Llama instead of LayerNorm)
rms = RMSNorm(d_model=512)
out = rms.forward(x)
print(out.shape)  # (2, 10, 512)
```

#### Full Models (`model.py`)

Complete Transformer (encoder-decoder) and GPT (decoder-only) models.

```python
import numpy as np
from model import GPT, Transformer, cross_entropy_loss

# --- GPT-style decoder-only model ---
gpt = GPT(
    vocab_size=1000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_len=256,
)

# Forward pass
input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
logits = gpt.forward(input_ids, training=False)
print(logits.shape)  # (1, 5, 1000) → (batch, seq_len, vocab_size)

# Compute loss (for training)
target_ids = np.array([[2, 3, 4, 5, 6]], dtype=np.int64)
loss = cross_entropy_loss(logits, target_ids)
print(f"Loss: {loss:.4f}")

# Autoregressive generation
prompt = np.array([[1, 2, 3]], dtype=np.int64)
generated = gpt.generate(prompt, max_new_tokens=10, temperature=0.8)
print(f"Generated {len(generated)} tokens")

# --- Encoder-decoder Transformer ---
transformer = Transformer(
    vocab_size=1000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
)

src_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
tgt_ids = np.array([[5, 6, 7]], dtype=np.int64)

logits = transformer.forward(src_ids, tgt_ids, training=False)
print(logits.shape)  # (1, 3, 1000)

# Encode only (get contextualized representations)
enc_out = transformer.encode(src_ids, training=False)
print(enc_out.shape)  # (1, 4, 512)
```

#### Training (`trainer.py`)

SGD optimizer and training loop for the GPT model.

```python
import numpy as np
from model import GPT
from tokenizer import BPETokenizer
from trainer import Trainer

# Train tokenizer
corpus = ["the cat sat on the mat", "the dog sat on the log"]
tok = BPETokenizer(vocab_size=300)
tok.train(corpus)

# Create model
model = GPT(vocab_size=len(tok), d_model=32, num_heads=4, num_layers=2, d_ff=64)

# Train
trainer = Trainer(model, tok, lr=0.1, momentum=0.9)
losses = trainer.train(corpus, epochs=100, seq_len=8)

print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")

# Generate text after training
prompt = np.array([tok.encode("the cat")], dtype=np.int64)
generated = model.generate(prompt, max_new_tokens=5, temperature=0.0)
print(tok.decode(generated))  # "the cat sat on the mat ..."
```

### Linting

```bash
uv run ruff check 01-transformer-architecture/src/ 01-transformer-architecture/tests/
uv run ruff format --check 01-transformer-architecture/src/ 01-transformer-architecture/tests/
```

---

## Key Takeaways

1. **Self-attention** is the core innovation — it lets every token directly interact with every other token in O(1) sequential operations (vs O(n) for RNNs).
2. **Multi-head attention** allows the model to focus on different types of relationships simultaneously.
3. **Residual connections + layer normalization** make deep Transformers trainable.
4. **Positional encodings** inject order information into an otherwise order-agnostic architecture.
5. The Transformer's **parallelizability** is what enables training on massive datasets and scaling to hundreds of billions of parameters.
6. Modern LLMs are almost exclusively **decoder-only** architectures, optimized with pre-norm, RoPE, SwiGLU, and RMSNorm.

---

## References

- Vaswani et al., "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Shazeer, "GLU Variants Improve Transformer" (2020) — SwiGLU
