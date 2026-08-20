# 5. QLoRA — Quantized Low-Rank Adaptation

## The Problem: Even LoRA Has Limits

LoRA dramatically reduces the number of **trainable** parameters, but you still need to load the full pre-trained model into GPU memory in FP16/BF16:

| Model Size | FP16 Memory |
|------------|-------------|
| 7B | ~14 GB |
| 13B | ~26 GB |
| 34B | ~68 GB |
| 70B | ~140 GB |

A 70B model in FP16 won't fit on any single consumer GPU. Even a 34B model requires an A100 (40/80 GB) or multiple GPUs.

**QLoRA** (Dettmers et al., 2023) solves this by **quantizing the base model to 4 bits** before applying LoRA. This makes it possible to fine-tune a 70B model on a single 48 GB GPU.

---

## 1. How QLoRA Works

QLoRA = **Quantization** + **LoRA**

```
QLoRA pipeline:

1. Load base model in 4-bit (NF4 quantization)
2. Apply LoRA adapters (trainable in BF16/FP16)
3. Train only the LoRA parameters
4. (Optional) Merge and dequantize for deployment
```

The key components that make this possible:

### NF4 (NormalFloat4)

Standard 4-bit integer quantization (INT4) assumes values are uniformly distributed. But neural network weights are approximately **normally distributed** (bell curve). NF4 is a new data type designed for normally-distributed data, with quantization levels optimized for a Gaussian distribution:

- More quantization levels near zero (where most weights are).
- Fewer levels in the tails (where few weights are).
- Result: Better information preservation than INT4 for the same bit width.

### Double Quantization

Quantization introduces **quantization constants** (scaling factors) that also consume memory. For a large model, these constants can add up to ~0.5 GB. Double quantization quantizes these constants themselves to 8-bit, saving ~0.4 GB with negligible accuracy loss.

### Paged Optimizers

During training, optimizer states can cause out-of-memory (OOM) errors even if the model fits. QLoRA uses **paged optimizers** (similar to CPU virtual memory paging) that automatically offload optimizer states to CPU RAM when GPU memory is tight, then page them back when needed. This prevents OOM without manual tuning.

---

## 2. The Quantization Math

### Standard Quantization

Given a floating-point tensor `X`, quantization maps it to integer values:

```
X_int = round((X - offset) / scale)
X_dequant = X_int * scale + offset
```

Where `scale` and `offset` are computed to minimize quantization error.

### 4-bit Quantization

With 4 bits, you have only 2⁴ = 16 possible values. The challenge is mapping a continuous range (e.g., [-3, 3] for normalized weights) to just 16 discrete levels.

INT4 uses **uniform quantization**: the 16 levels are equally spaced.

NF4 uses **non-uniform quantization**: levels are denser near zero, matching the Gaussian distribution of weights.

### Block-wise Quantization

Instead of computing one scale for the entire tensor (which would be inaccurate), weights are quantized in **blocks** (e.g., 64 values per block), each with its own scale. This is a trade-off between accuracy and the memory overhead of storing per-block scales.

---

## 3. Memory Breakdown

Let's compare the memory usage for fine-tuning a **65B model** (similar to Llama 2 70B):

| Component | Full FT (FP16) | LoRA (FP16) | QLoRA (NF4) |
|-----------|----------------|-------------|-------------|
| Base model weights | 130 GB | 130 GB | 32.5 GB |
| LoRA parameters | — | ~0.2 GB | ~0.2 GB |
| Gradients | 130 GB | ~0.2 GB | ~0.2 GB |
| Optimizer states (Adam) | 260 GB | ~0.4 GB | ~0.4 GB |
| Quantization constants | — | — | ~2.5 GB |
| **Total** | **~520 GB** | **~131 GB** | **~35.8 GB** |

QLoRA reduces memory from 520 GB (impossible on any single machine) to ~36 GB (fits on a single A100 40 GB or even a high-end consumer setup with 48 GB).

---

## 4. Configuring QLoRA

### bitsandbytes 4-bit Config

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",           # Use NF4 data type
    bnb_4bit_use_double_quant=True,      # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for forward/backward
)
```

### LoRA Config for QLoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha
    target_modules=[         # Which modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,       # Dropout on LoRA layers
    bias="none",             # Don't train biases
    task_type="CAUSAL_LM",   # Task type
)
```

### Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch size = 4 × 4 = 16
    warmup_steps=100,
    max_steps=500,
    learning_rate=2e-4,
    fp16=True,                          # Or bf16=True
    logging_steps=10,
    output_dir="./qlora-output",
    optim="paged_adamw_8bit",           # Paged optimizer
    gradient_checkpointing=True,        # Save memory on activations
)
```

---

## 5. NF4 vs INT4 vs GPTQ

| Quantization | Type | Quality | Speed | Use Case |
|-------------|------|---------|-------|----------|
| **INT4** | Uniform 4-bit | Decent | Fast | Inference |
| **NF4** | NormalFloat4 | Better than INT4 | Fast | QLoRA training |
| **GPTQ** | Post-training quantization | Good | Fast | Inference-optimized |
| **AWQ** | Activation-aware | Better than GPTQ | Fast | Inference-optimized |

NF4 is specifically designed for **training** (QLoRA), while GPTQ and AWQ are designed for **inference**. For QLoRA, always use NF4.

---

## 6. Practical Considerations

### When QLoRA vs LoRA

| Scenario | Recommendation |
|----------|---------------|
| Model ≤ 7B, GPU ≥ 24 GB | LoRA (FP16/BF16) |
| Model 7-13B, GPU ≥ 24 GB | LoRA or QLoRA (QLoRA gives more headroom for batch size) |
| Model 13-34B, GPU 24 GB | QLoRA required |
| Model 34-70B, GPU 48 GB | QLoRA required |
| Model 70B+, GPU 24 GB | QLoRA + gradient checkpointing + small batch |

### Quality Impact

QLoRA achieves **~99% of full fine-tuning performance** on most benchmarks. The quality loss from 4-bit quantization is minimal because:
- NF4 preserves weight distribution better than INT4.
- LoRA adapters operate in BF16/FP16, compensating for any quantization error in the base weights.
- The base model was trained in higher precision, so the information is already encoded.

### Speed

QLoRA is slightly slower per step than FP16 LoRA because of the dequantization overhead on each forward pass. However, the reduced memory allows larger batch sizes, which can offset this.

---

## 7. The Full QLoRA Pipeline (Step by Step)

```
Step 1: Load tokenizer
Step 2: Configure 4-bit quantization (BitsAndBytesConfig)
Step 3: Load base model in 4-bit
Step 4: Prepare model for k-bit training (gradient checkpointing, etc.)
Step 5: Configure LoRA (LoraConfig)
Step 6: Wrap model with PEFT (get_peft_model)
Step 7: Load and tokenize dataset
Step 8: Configure training arguments (paged optimizer, etc.)
Step 9: Train with HuggingFace Trainer / SFTTrainer
Step 10: Save adapter weights (or merge and save full model)
```

---

## Key Takeaways

1. QLoRA = **4-bit quantization** (NF4) + **LoRA** = fine-tune 70B models on a single GPU.
2. **NF4** is a non-uniform quantization designed for Gaussian-distributed weights.
3. **Double quantization** saves memory by quantizing quantization constants.
4. **Paged optimizers** prevent OOM by offloading optimizer states to CPU.
5. Memory drops from ~520 GB (full FT) to ~36 GB (QLoRA) for a 65B model.
6. Quality is ~99% of full fine-tuning — the gap is negligible for most use cases.
7. Always use NF4 for QLoRA training, not INT4.

---

## References

- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023): https://arxiv.org/abs/2305.14314
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022): https://arxiv.org/abs/2208.07339
- bitsandbytes documentation: https://huggingface.co/docs/bitsandbytes
- Hugging Face QLoRA tutorial: https://huggingface.co/blog/4bit-transformers-bitsandbytes
