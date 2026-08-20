# 6. The PEFT Ecosystem

## What is PEFT?

**Parameter-Efficient Fine-Tuning (PEFT)** is a family of techniques that adapt large pre-trained models to downstream tasks by training only a small subset of parameters. The Hugging Face `peft` library provides a unified API for all major PEFT methods.

This section covers the ecosystem: the libraries, tools, and other PEFT methods beyond LoRA and Adapters.

---

## 1. The Core Libraries

### peft (Hugging Face)

The central library. It provides:

- Unified `get_peft_model()` API for all methods.
- Pre-built configurations for LoRA, Adapters, Prefix Tuning, Prompt Tuning, IA³.
- Seamless integration with `transformers` models.
- Weight merging and saving utilities.
- Multi-adapter support (load/save/switch between adapters).

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
# model is now trainable with only LoRA params
```

### bitsandbytes

The quantization backend. Provides:

- 4-bit and 8-bit quantization for model loading.
- NF4 and FP4 data types for QLoRA.
- 8-bit optimizers (AdamW8bit, etc.).
- Paged optimizers for memory-efficient training.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
)
```

### accelerate (Hugging Face)

Abstracts away hardware complexity:

- Automatic device placement (`device_map="auto"`).
- Mixed precision training (FP16, BF16).
- Multi-GPU training (DDP, FSDP, DeepSpeed integration).
- Gradient accumulation and checkpointing.

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

### transformers (Hugging Face)

The model backbone:

- Hundreds of pre-trained models (Llama, Mistral, Gemma, Qwen, etc.).
- `Trainer` and `SFTTrainer` classes with built-in PEFT support.
- Tokenizer implementations for all major tokenization algorithms.

---

## 2. Other PEFT Methods

Beyond LoRA and Adapters, the `peft` library supports several other techniques:

### Prefix Tuning

Adds trainable "prefix" vectors to the input of each Transformer layer (or just the first layer). These vectors are prepended to the key and value sequences in attention.

```
Normal:    K = W_k · [x₁, x₂, ..., xₙ]
           V = W_v · [x₁, x₂, ..., xₙ]

Prefix:    K = W_k · [p₁, ..., pₘ, x₁, x₂, ..., xₙ]
           V = W_v · [p₁, ..., pₘ, x₁, x₂, ..., xₙ]
                       ↑ prefix vectors (trainable)
```

- **Parameters**: 2 · L · m · d (L layers, m prefix length, d dimension).
- **Strength**: Simple, effective for generation tasks.
- **Weakness**: Prefix length reduces the effective context window.

### Prompt Tuning

A simplified version of prefix tuning: adds trainable tokens only to the **input embedding** layer (not every layer).

```
Normal:    Input = [e(x₁), e(x₂), ..., e(xₙ)]

Prompt:    Input = [p₁, ..., pₘ, e(x₁), e(x₂), ..., e(xₙ)]
                    ↑ prompt embeddings (trainable)
```

- **Parameters**: m · d (only m trainable vectors).
- **Strength**: Extremely parameter-efficient. Works well for large models (10B+).
- **Weakness**: Underperforms on smaller models (<1B).

### P-Tuning / P-Tuning v2

- **P-Tuning**: Adds trainable continuous prompts with an LSTM-based prompt encoder.
- **P-Tuning v2**: Applies continuous prompts at every layer (like prefix tuning but with different initialization).

### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Rescales activations using learned vectors:

```
h' = l_v ⊙ (W_v · x)    ← rescales value activations
h' = l_k ⊙ (W_k · x)    ← rescales key activations
h' = l_ff ⊙ FFN(x)      ← rescales FFN activations
```

Where `l_v`, `l_k`, `l_ff` are learned vectors (one scalar per dimension) and `⊙` is element-wise multiplication.

- **Parameters**: 3 · L · d (extremely small — often <0.01% of model params).
- **Strength**: The most parameter-efficient method. Comparable to full fine-tuning on some tasks.
- **Weakness**: Can underperform LoRA on complex tasks.

### VeRA (Vector-based Random Matrix Adaptation)

An even more parameter-efficient variant of LoRA that uses shared random matrices across layers:

```
ΔW = Λ_b · (A_shared · Λ_d · B_shared)
```

Where A_shared and B_shared are random frozen matrices shared across all layers, and only the diagonal scaling matrices Λ_b and Λ_d are trained.

- **Parameters**: Just 2 · d per layer (vs 2 · d · r for LoRA).
- **Strength**: Extremely parameter-efficient (10× fewer params than LoRA).
- **Weakness**: Slight quality degradation compared to LoRA.

---

## 3. Method Comparison

| Method | Trainable Params (7B) | Performance vs Full FT | Inference Overhead | Best For |
|--------|----------------------|------------------------|---------------------|----------|
| **Full FT** | 100% (7B) | Baseline | None | Maximum quality, large datasets |
| **LoRA** | ~0.1-1% (5-50M) | ~97-99% | None (merged) | General purpose, best balance |
| **QLoRA** | ~0.1-1% (5-50M) | ~96-99% | None (merged) | Large models on limited hardware |
| **Adapters** | ~0.5-8% (35-560M) | ~95-98% | +2-5% | Multi-task serving |
| **Prefix Tuning** | ~0.1-1% (5-50M) | ~90-97% | Reduces context | Generation tasks |
| **Prompt Tuning** | <0.01% (<1M) | ~85-95% | Reduces context | Very large models (10B+) |
| **IA³** | <0.01% (<1M) | ~90-96% | Minimal | Extreme parameter efficiency |

---

## 4. SFTTrainer (Supervised Fine-Tuning Trainer)

Hugging Face's `trl` library provides `SFTTrainer`, which is purpose-built for fine-tuning language models with PEFT:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(output_dir="./output", ...),
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,        # Direct PEFT integration
    max_seq_length=2048,            # Automatic truncation/padding
    dataset_text_field="text",      # Field containing the training text
    packing=True,                   # Pack multiple samples into one sequence
)

trainer.train()
```

Key features:
- **Packing**: Combines multiple short examples into a single sequence to minimize padding waste.
- **Auto-formatting**: Handles chat templates, instruction formatting.
- **Flash Attention 2** support.
- **NEFTune noise** for improved training (adds noise to embeddings).

---

## 5. The Training Loop (End to End)

A complete PEFT fine-tuning workflow:

```python
# 1. Quantization config (for QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2. Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.pad_token = tokenizer.eos_token

# 4. PEFT config
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    max_steps=500,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=2048,
)
trainer.train()

# 7. Save
trainer.save_model()  # saves adapter weights
# OR merge and save full model:
model = model.merge_and_unload()
model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

---

## 6. Tooling & Ecosystem

| Tool | Purpose |
|------|---------|
| **Weights & Biases** | Experiment tracking, loss curves, hyperparameter logging |
| **MLflow** | Model registry, experiment tracking, deployment |
| **TensorBoard** | Training visualization |
| **Axolotl** | YAML-based fine-tuning configs, multi-GPU training |
| **Unsloth** | Optimized LoRA/QLoRA kernels (2-5× faster) |
| **LLaMA-Factory** | Web UI + CLI for fine-tuning, supports 100+ models |
| **Hugging Face Hub** | Model sharing, versioning, inference endpoints |

---

## Key Takeaways

1. **PEFT** is the umbrella term for parameter-efficient fine-tuning methods.
2. The Hugging Face stack (`peft` + `bitsandbytes` + `accelerate` + `transformers` + `trl`) provides a complete pipeline.
3. **LoRA** is the most popular PEFT method, but **Adapters**, **Prefix Tuning**, **Prompt Tuning**, and **IA³** each have their niches.
4. **SFTTrainer** from `trl` simplifies supervised fine-tuning with built-in PEFT support.
5. Choose your method based on: model size, hardware constraints, task complexity, and whether you need multi-task serving.

---

## References

- Hugging Face PEFT: https://github.com/huggingface/peft
- Hugging Face TRL (SFTTrainer): https://github.com/huggingface/trl
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)
- Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)
- Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (IA³, 2022)
