# 3. Adapters

## The Problem: Full Fine-Tuning Is Expensive

When you fully fine-tune a large language model for each new task, you need to:
- Store a complete copy of the model per task (e.g., ~14 GB per task for a 7B model).
- Run backpropagation through all parameters (huge memory and compute cost).
- Risk catastrophic forgetting of pre-trained knowledge.

**Adapters** solve this by inserting small, trainable bottleneck layers into the frozen pre-trained model. Only the adapter parameters are updated during fine-tuning — the original model weights stay frozen.

---

## 1. Core Idea

Instead of modifying the pre-trained weights, we **add new parameters** between the existing layers and only train those.

```
Pre-trained layer:    x → [FROZEN W] → h
Adapter layer:        x → [FROZEN W] → h → [ADAPTER] → h' → next layer
```

The adapter is a **bottleneck** architecture:

```
h → [Down-project: d → b] → ReLU → [Up-project: b → d] → h + residual → h'
```

Where:
- `d` is the model's hidden dimension (e.g., 768 for BERT-base, 4096 for Llama 7B).
- `b` is the **bottleneck size** (typically 8-256, much smaller than d).
- The residual connection ensures the adapter can learn the identity function (by initializing near zero).

### Why Bottlenecks?

The down-project + up-project structure forces the adapter to learn **compact representations** of the task-specific adaptation. This is parameter-efficient because `2·d·b` parameters (the two projection matrices) is much smaller than `d·d` (a full layer).

---

## 2. Adapter Variants

### Houlsby Adapters (Original, 2019)

Proposed by Houlsby et al. in ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751).

**Placement**: Two adapters per Transformer block — one after attention and one after the FFN.

```
Transformer Block with Houlsby Adapters:

x ──────────────────────────────────────┐
├→ Multi-Head Attention → [ADAPTER] → + ┤→ LayerNorm → [ADAPTER] → FFN → +
                                        └───────────────────────────────→ output
```

**Parameters per adapter**: 2 · d · b (down and up projections).
**Total trainable parameters**: 2 · L · 2 · d · b = 4 · L · d · b

For BERT-base (L=12, d=768, b=64): ~2.4M parameters (~2% of BERT's 110M).

### Pfeiffer Adapters (2021)

Proposed by Pfeiffer et al. in ["AdapterFusion: Non-Destructive Task Composition"](https://arxiv.org/abs/2005.00247).

**Placement**: Only one adapter per block — after the FFN (and after LayerNorm).

```
Transformer Block with Pfeiffer Adapter:

x → Multi-Head Attention → + → LayerNorm → FFN → + → LayerNorm → [ADAPTER] → output
```

**Parameters**: Half of Houlsby (~1.2M for BERT-base with b=64).

Pfeiffer adapters are more parameter-efficient and were shown to perform comparably to Houlsby adapters on most tasks.

### Parallel Adapters (He et al., 2022)

Instead of placing the adapter sequentially, place it **in parallel** with the existing sub-layer:

```
x → Multi-Head Attention → +
x → [ADAPTER] ──────────→ +
                            → output
```

**Advantage**: The adapter computation can run in parallel with the main layer, reducing inference latency compared to sequential adapters.

---

## 3. Inside an Adapter

Let's look at the exact computation inside a Houlsby-style adapter:

```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int):
        self.down_proj = nn.Linear(d_model, bottleneck)   # W_down ∈ R^(d×b)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)      # W_up ∈ R^(b×d)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.down_proj(x)    # (batch, seq, d) → (batch, seq, b)
        x = self.activation(x)   # non-linearity
        x = self.up_proj(x)      # (batch, seq, b) → (batch, seq, d)
        return residual + x      # skip connection
```

### Initialization

The up-projection (`W_up`) is initialized near zero (e.g., `N(0, 1e-5)`). This means the adapter initially behaves as an identity function (`f(x) ≈ x`), and the model starts with its pre-trained behavior. As training progresses, the adapter learns to deviate from the identity.

### Skip Connection

The residual connection (`residual + x`) is critical. Without it, the adapter would be forced to learn useful representations from scratch in a very low-rank space. With it, the adapter only needs to learn the **difference** between the pre-trained behavior and the desired task behavior.

---

## 4. Training with Adapters

### What Gets Trained

- **Frozen**: All pre-trained weights (attention projections, FFN weights, layer norms, embeddings).
- **Trainable**: Adapter parameters + task-specific head (classification layer, etc.) + optionally layer norms.

### Training Dynamics

- Learning rates are typically higher than full fine-tuning (e.g., 1e-3 to 1e-4 vs 5e-5).
- Because only a small fraction of parameters are updated, adapters converge faster.
- The frozen backbone prevents catastrophic forgetting — the model retains all its pre-trained knowledge.

### Batch Size and Optimizer

- Standard optimizer: AdamW.
- Batch size: Similar to full fine-tuning, but memory usage is much lower because gradients are only stored for adapter parameters.

---

## 5. AdapterFusion: Combining Multiple Adapters

What if you want a model that performs well on multiple tasks? With adapters, you can:

1. Train a separate adapter for each task.
2. At inference time, **compose** multiple adapters using **AdapterFusion**.

AdapterFusion learns attention weights over multiple adapters:

```
h_fused = Σᵢ αᵢ · Adapterᵢ(h)
```

Where `αᵢ` are learned attention weights that determine how much each task-specific adapter contributes. This allows the model to leverage knowledge from multiple tasks when solving a new one.

---

## 6. Advantages and Limitations

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Parameter efficiency** | Only 0.5-8% of parameters are trainable. |
| **No catastrophic forgetting** | Pre-trained weights stay frozen. |
| **Multi-task with one model** | Swap adapters to switch tasks; no need to reload models. |
| **Small storage footprint** | Adapters are typically 1-100 MB vs 14 GB for a full model. |
| **Fast task switching** | Load a new adapter in milliseconds. |
| **Modular** | Train adapters independently and combine later. |

### Limitations

| Limitation | Explanation |
|------------|-------------|
| **Inference latency** | Sequential adapters add computation to every forward pass (extra matrix multiplications). |
| **Limited capacity** | The bottleneck constrains how much task-specific knowledge can be stored. |
| **Less studied** | LoRA has become more popular, so the ecosystem (tools, best practices) is smaller. |
| **Architecture-specific** | Adapter placement depends on the model architecture; not as drop-in as some alternatives. |

---

## 7. When to Use Adapters

Adapters are ideal when:
- You have **many tasks** and want to serve them all from one base model.
- You need **fast task switching** (no model reloading).
- You care about **catastrophic forgetting** (e.g., the base model must retain its general capabilities).
- Storage is at a premium, but inference latency is not the primary concern.

---

## Key Takeaways

1. Adapters insert small **bottleneck layers** into a frozen pre-trained model.
2. The **down-project → activation → up-project** structure with a **residual connection** is the key design.
3. Near-zero initialization ensures the adapter starts as an **identity function**.
4. Adapters enable **multi-task serving** from a single base model by swapping adapter weights.
5. The main trade-off is **inference latency** vs **parameter efficiency**.

---

## References

- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019): https://arxiv.org/abs/1902.00751
- Pfeiffer et al., "AdapterFusion: Non-Destructive Task Composition for Transfer Learning" (2021): https://arxiv.org/abs/2005.00247
- He et al., "Towards a Unified View of Parameter-Efficient Transfer Learning" (2022): https://arxiv.org/abs/2110.04366
- Pfeiffer et al., "AdapterHub: A Framework for Adapting Transformers" (2020): https://arxiv.org/abs/2007.07779
