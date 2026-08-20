# 4. LoRA — Low-Rank Adaptation

## The Key Insight

When you fine-tune a large pre-trained model, the **weight updates** (ΔW = W_finetuned - W_pretrained) during training have **low "intrinsic rank"**. This means the changes to the weight matrices can be represented compactly — you don't need to store or update a full `d × d` matrix; a low-rank decomposition suffices.

This insight, from Hu et al.'s ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021), is the foundation of LoRA.

---

## 1. How LoRA Works

Instead of updating a pre-trained weight matrix `W ∈ R^(d×k)` directly, LoRA represents the update as:

```
W' = W + ΔW
ΔW = A · B    (where A ∈ R^(d×r), B ∈ R^(r×k), and r << min(d, k))
```

- `W` is **frozen** (not updated during training).
- `A` and `B` are **trainable** low-rank matrices.
- `r` is the **rank** (typically 4-64, compared to d which is 4096+ in modern LLMs).

The forward pass becomes:

```
h = W·x + (α/r) · A·B·x
     ↑          ↑
  frozen    trainable (LoRA)
```

Where `α` is a scaling factor (more on this below).

### Visual Intuition

```
Full weight update (d × k, e.g., 4096 × 4096 = 16.8M params):
┌───────────────────────┐
│                       │
│         ΔW            │  ← 16.8M trainable parameters
│    (4096 × 4096)      │
│                       │
└───────────────────────┘

LoRA decomposition (r = 16):
┌──────┐   ┌───────────────────────┐
│  A   │   │                       │
│4096× │ × │          B            │  ← Only 4096×16 + 16×4096 = 131K params
│  16  │   │       (16 × 4096)     │
└──────┘   └───────────────────────┘
```

That's a **128× reduction** in trainable parameters for this one matrix.

---

## 2. Which Matrices to Apply LoRA To

In a Transformer, the main linear projections are:

| Matrix | Shape | Role |
|--------|-------|------|
| `W_q` | d × d | Query projection in attention |
| `W_k` | d × d | Key projection in attention |
| `W_v` | d × d | Value projection in attention |
| `W_o` | d × d | Output projection in attention |
| `W_up`, `W_down`, `W_gate` | d × d_ff (or d_ff × d) | FFN projections |

The original LoRA paper applied it only to **attention weights** (`W_q` and `W_v`), finding that this was sufficient for good performance. Subsequent research showed that applying LoRA to all linear layers (including FFN) can improve performance at the cost of more parameters.

**Common practice**: Apply LoRA to `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Optionally add `gate_proj`, `up_proj`, and `down_proj` for maximum performance.

---

## 3. Rank (r) and Alpha (α)

### Rank `r`

The rank determines the **capacity** of the LoRA adapter:
- **Low r (4-8)**: Very parameter-efficient. Works well for simple tasks or when you're close to the pre-training distribution.
- **Medium r (16-32)**: Good balance for most tasks. The original paper found r=16 works well across many benchmarks.
- **High r (64-128)**: More capacity. Useful for complex tasks that require significant adaptation.

**Important finding**: Performance saturates quickly with rank. r=1 already captures much of the benefit, and r=4-16 is often indistinguishable from full fine-tuning.

### Alpha `α`

Alpha is a scaling factor applied to the LoRA output:

```
h = W·x + (α/r) · A·B·x
```

- `α` controls the **magnitude** of the LoRA contribution.
- Higher α → stronger adaptation (the LoRA matrices have more influence).
- Common values: α = 2·r (e.g., r=16, α=32) or α = r (r=16, α=16).
- **α is NOT a hyperparameter that needs tuning** in the same way r does. Changing α is equivalent to changing the learning rate (it scales the effective step size). In practice, pick α = 2·r or α = r and tune the learning rate instead.

### Dropout

LoRA typically uses a small dropout (0.05-0.1) on the adapter path to prevent overfitting:

```
h = W·x + (α/r) · A · dropout(B·x)
```

---

## 4. Initialization

- **A**: Initialized with random Gaussian weights `N(0, σ²)`.
- **B**: Initialized to **zeros**.

This means `ΔW = A·B = 0` at the start of training, so the model begins with exactly its pre-trained behavior. As training progresses, B becomes non-zero and the adapter learns the necessary adaptations.

---

## 5. Training with LoRA

### Memory Savings

In full fine-tuning, you need to store:
- Model weights (FP16): 2 bytes × N parameters
- Gradients (FP16): 2 bytes × N parameters
- Optimizer states (Adam): 8 bytes × N parameters (FP32 momentum + FP32 variance)
- **Total**: ~12 bytes per parameter

For a 7B model: 7B × 12 = **84 GB** — impossible on most GPUs.

With LoRA (r=16 on attention only):
- Frozen weights: 2 bytes × 7B = 14 GB (no gradients, no optimizer states)
- Trainable LoRA params: ~30M → 30M × 12 = 0.36 GB
- **Total**: ~14.4 GB — fits on a single RTX 4090 (24 GB).

### Gradient Checkpointing

Even with LoRA, activations can consume significant memory. Gradient checkpointing trades compute for memory by recomputing activations during backpropagation instead of storing them.

### Mixed Precision Training

LoRA is typically used with:
- **FP16/BF16** for the frozen base model weights.
- **FP32** for the LoRA parameters (A and B matrices) to maintain training precision.

The Hugging Face `peft` + `bitsandbytes` libraries handle this automatically.

---

## 6. Merging LoRA Weights

After training, LoRA weights can be **merged** back into the base model:

```
W_merged = W + (α/r) · A · B
```

This produces a standard model checkpoint with **zero inference overhead** — no extra computation, no latency penalty. The merged model is identical in structure to the original pre-trained model.

```python
model = get_peft_model(base_model, lora_config)
model.train()  # train with LoRA
# ... training ...
model = model.merge_and_unload()  # merge LoRA into base weights
model.save_pretrained("my-finetuned-model")
```

### When to Merge vs Keep Separate

| Merge | Keep Separate |
|-------|---------------|
| Single task deployment | Multi-task serving |
| Zero inference overhead required | Need to switch between tasks quickly |
| Standard model format needed | Want to share base model across adapters |

---

## 7. LoRA vs Full Fine-Tuning vs Adapters

| Aspect | Full FT | Adapters | LoRA |
|--------|---------|----------|------|
| Trainable params | 100% | 0.5-8% | 0.1-1% |
| Memory for 7B model | ~84 GB | ~15 GB | ~14.4 GB |
| Inference latency | 0% overhead | +2-5% | 0% (when merged) |
| Task storage | 14 GB/task | 70 MB-1.1 GB | ~60 MB |
| Catastrophic forgetting | High risk | Low risk | Low risk |
| Performance | Best (baseline) | ~95-98% | ~95-99% |
| Training speed | Slowest | Fast | Fast |

---

## 8. Practical Tips

### Choosing Target Modules

- **Minimum**: `q_proj` and `v_proj` (original paper recommendation).
- **Good**: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- **Best**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

### Choosing Rank

- Start with `r=16`. It works well for most tasks.
- If underfitting, try `r=32` or `r=64`.
- If you need maximum parameter efficiency, `r=4` or `r=8` often works surprisingly well.

### Learning Rate

- LoRA typically uses higher learning rates than full fine-tuning: 1e-4 to 5e-4 (vs 5e-5 for full FT).
- If using a scheduler, cosine decay with warmup works well.

### Multi-GPU Training

- LoRA is compatible with DDP, FSDP, and DeepSpeed.
- When using FSDP/DeepSpeed ZeRO-3, the frozen base model weights are sharded across GPUs, and only LoRA parameters have full replicas.

---

## Key Takeaways

1. LoRA exploits the **low intrinsic rank** of weight updates during fine-tuning.
2. `W' = W + A·B` where only A and B are trained — W stays frozen.
3. Typically applied to attention projections (`q_proj`, `v_proj`, etc.).
4. **Rank r=16** and **α=32** are good defaults for most tasks.
5. Zero initialization of B means LoRA starts as identity.
6. Weights can be **merged** for zero inference overhead.
7. Memory savings are dramatic: 84 GB → 14.4 GB for a 7B model.

---

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021): https://arxiv.org/abs/2106.09685
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft/en/developer_guides/lora
- Biderman et al., "LoRA Learns Less and Forgets Less" (2024): https://arxiv.org/abs/2405.09673
