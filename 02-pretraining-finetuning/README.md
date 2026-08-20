# 2. Pre-trained Models & Fine-Tuning

## The Pre-training Paradigm

The fundamental shift in NLP over the last 5-7 years is the move from training task-specific models from scratch to a two-stage approach:

1. **Pre-training**: Train a large model on a massive, general-purpose corpus using a self-supervised objective (no human labels needed).
2. **Fine-tuning**: Adapt the pre-trained model to a specific task using a smaller, labeled dataset.

This works because the pre-training phase teaches the model **general linguistic knowledge** — syntax, semantics, world knowledge, reasoning patterns — which transfers to downstream tasks.

---

## 1. Pre-training Objectives

### Autoregressive Language Modeling (GPT-style)

Predict the next token given all previous tokens:

```
P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)
```

- **Architecture**: Decoder-only (only masked self-attention, no encoder).
- **Direction**: Left-to-right (causal).
- **Loss**: Cross-entropy between predicted token distribution and actual next token.
- **Data**: Any text corpus — no labeling required.
- **Strength**: Natural for text generation tasks.
- **Limitation**: Can only use left context; misses right context.

**Example (GPT training)**:
```
Input:  "The cat sat on the"
Target: "cat sat on the mat"

The model predicts P("cat"|"The"), P("sat"|"The cat"), etc.
```

### Masked Language Modeling (BERT-style)

Randomly mask tokens in the input and train the model to predict them:

```
Input:  "The [MASK] sat on the [MASK]"
Target: "cat", "mat"
```

- **Architecture**: Encoder-only (bidirectional self-attention, no masking).
- **Direction**: Bidirectional — each token sees all other tokens (except masked ones).
- **Masking rate**: ~15% of tokens are masked (80% → [MASK], 10% → random token, 10% → unchanged).
- **Strength**: Excellent for understanding tasks (classification, NER, QA).
- **Limitation**: Not naturally suited for text generation.

**Why the 80/10/10 split?** During fine-tuning, the model never sees [MASK] tokens. The 10% random replacement and 10% unchanged tokens force the model to rely on context rather than just learning to fill [MASK] blanks.

### Span Corruption (T5-style)

Instead of masking individual tokens, mask entire spans of text and train the model to generate them:

```
Input:  "Thank you <X> me to your party <Y> week."
Target: "<X> for inviting <Y> last <Z>"
```

- **Architecture**: Encoder-decoder.
- **Strength**: Unified text-to-text framework; works for both understanding and generation.
- **Used by**: T5, UL2, Flan-T5.

### Next Sentence Prediction (BERT)

Given two sentences, predict whether the second sentence follows the first:

```
Input:  "[CLS] The man went to the store. [SEP] He bought milk. [SEP]"
Label:  IsNext (True)
```

- **Purpose**: Teaches the model about relationships between sentences.
- **Status**: Later research showed NSP is not very useful; models like RoBERTa dropped it and performed better.

---

## 2. Pre-training Data

### Scale

Modern LLMs are trained on **trillions of tokens**. For context:
- GPT-3: ~300 billion tokens
- Llama 2: 2 trillion tokens
- Llama 3: 15 trillion tokens

### Data Sources

| Source | Description | Example |
|--------|-------------|---------|
| Common Crawl | Web crawl (filtered) | C4, RefinedWeb |
| Books | Fiction and non-fiction | Books3, Gutenberg |
| Code | GitHub repositories | The Stack, BigCode |
| Wikipedia | Encyclopedia articles | All languages |
| Scientific papers | arXiv, PubMed | S2ORC |
| Social media | Reddit, Twitter | PushShift, OpenWebText |
| Curated datasets | High-quality instruction data | FLAN, OpenOrca |

### Data Quality Matters More Than Quantity

The **Chinchilla scaling laws** (Hoffmann et al., 2022) showed that for a given compute budget, it's better to train a smaller model on more data than a larger model on less data. But more recent research (Llama 3) shows that **data quality** matters even more — carefully filtered, deduplicated, and curated data produces better models than simply more data.

Key data preprocessing steps:
- **Deduplication**: Remove exact and near-duplicate documents (MinHash, SimHash).
- **Quality filtering**: Remove low-quality content using classifiers or heuristics.
- **PII removal**: Strip personally identifiable information.
- **Language filtering**: Keep only target-language content.
- **Toxicity filtering**: Remove hate speech and harmful content.

---

## 3. Full Fine-Tuning

Full fine-tuning updates **all parameters** of the pre-trained model on a task-specific dataset.

### How It Works

1. Load the pre-trained model weights.
2. Replace the output layer (language modeling head) with a task-specific head (e.g., a classification layer).
3. Train on the task dataset, typically with a much lower learning rate than pre-training (e.g., 5e-5 vs 1e-3).
4. All model parameters are updated during backpropagation.

### When to Use Full Fine-Tuning

- You have a **large, high-quality labeled dataset** (100K+ examples).
- The task is **very different** from the pre-training objective.
- You have the **compute budget** to train all parameters.
- You want to **maximize performance** and aren't constrained by memory or cost.

### Drawbacks

| Problem | Explanation |
|---------|-------------|
| **Memory cost** | Storing gradients and optimizer states for all parameters can require 4-5× the model size in memory. A 7B model may need 56-70 GB just for training state. |
| **Catastrophic forgetting** | The model may "forget" its pre-trained knowledge when aggressively fine-tuned on a narrow task. |
| **Storage cost** | Each fine-tuned task requires storing a full copy of the model (~14 GB for 7B in FP16). |
| **Slow iteration** | Training all parameters takes longer, making experimentation slower. |

### Mitigating Catastrophic Forgetting

- **Lower learning rates**: 1e-5 to 5e-5 instead of 1e-4.
- **Early stopping**: Stop when validation loss plateaus.
- **Mixed-task training**: Mix the target task data with general pre-training data.
- **Gradual unfreezing**: Start by training only the last layers, then gradually unfreeze earlier ones.
- **Elastic Weight Consolidation (EWC)**: Penalize large changes to parameters that are important for previous tasks.

---

## 4. Transfer Learning in NLP

The pre-train → fine-tune paradigm is a form of **transfer learning**: knowledge gained from one task (language modeling) is transferred to another (classification, QA, summarization).

### Why Transfer Learning Works for Language

Language has deep **shared structure** across tasks:
- **Syntax** (grammar, word order) is universal.
- **Semantics** (word meaning, relationships) transfers across domains.
- **World knowledge** (facts, commonsense) is useful for almost any task.
- **Reasoning patterns** (logic, inference) generalize.

A model that has learned that "Paris" is a city, that "is" indicates a property, and that questions often start with "What" can apply this knowledge to many different tasks.

### The Scale Hypothesis

As models get larger, they acquire **emergent abilities** — capabilities that are not present in smaller models and cannot be predicted by extrapolating from smaller-scale performance. Examples include:
- Few-shot learning (GPT-3)
- Chain-of-thought reasoning
- Instruction following
- Tool use

These abilities emerge because larger models learn more abstract and composable representations during pre-training.

---

## 5. Instruction Tuning

Instruction tuning is a special form of fine-tuning where the model is trained to follow natural language instructions.

### Format

```
Instruction: "Summarize the following article in one sentence."
Input: "<article text>"
Output: "<one-sentence summary>"
```

### Why Instruction Tuning Works

Pre-trained models are good at **completion** (continuing text) but not at **following instructions**. Instruction tuning bridges this gap by training on (instruction, response) pairs, teaching the model to:
- Understand what is being asked.
- Ignore irrelevant context.
- Produce the right output format.
- Refuse harmful or impossible requests.

### Datasets

- **FLAN**: ~1,800 tasks reformatted as instructions.
- **Dolly**: 15K human-generated instruction-response pairs.
- **OpenOrca**: ~1M GPT-4-generated instructions with responses.
- **Alpaca**: 52K GPT-3.5-generated instructions.
- **Self-Instruct**: Automated pipeline for generating instruction data.

---

## 6. RLHF (Reinforcement Learning from Human Feedback)

RLHF is a three-stage process for aligning models with human preferences:

### Stage 1: Supervised Fine-Tuning (SFT)
Fine-tune the pre-trained model on high-quality (prompt, response) pairs written by humans.

### Stage 2: Reward Model Training
- Generate multiple responses for each prompt using the SFT model.
- Humans rank the responses from best to worst.
- Train a **reward model** to predict human preference scores.

### Stage 3: PPO Fine-Tuning
- Use Proximal Policy Optimization (PPO) to fine-tune the SFT model.
- The reward model scores the model's outputs.
- A KL-divergence penalty prevents the model from drifting too far from the SFT model (which would cause it to produce nonsensical but high-reward outputs).

### DPO (Direct Preference Optimization)

A simpler alternative to RLHF that doesn't require training a separate reward model. DPO directly optimizes the policy from preference data using a binary cross-entropy loss, making the process more stable and easier to implement.

---

## 7. The Limits of Full Fine-Tuning → PEFT

The problems with full fine-tuning (memory, storage, catastrophic forgetting) motivate **Parameter-Efficient Fine-Tuning (PEFT)** methods, which update only a small fraction of parameters. This is the focus of the next sections (Adapters, LoRA, QLoRA).

### Why PEFT Is the Modern Standard

| Method | Trainable params (7B model) | Memory needed | Per-task storage |
|--------|-----------------------------|---------------|------------------|
| Full fine-tuning | 7B (100%) | ~56 GB | ~14 GB |
| Adapters | ~35-560M (0.5-8%) | ~12 GB | ~70 MB-1.1 GB |
| LoRA (r=16) | ~30M (0.4%) | ~12 GB | ~60 MB |
| QLoRA (4-bit) | ~30M (0.4%) | ~10 GB | ~60 MB |

---

## Key Takeaways

1. **Pre-training** teaches general language knowledge from massive unlabeled corpora.
2. **Fine-tuning** adapts this knowledge to specific tasks with small labeled datasets.
3. **Instruction tuning** makes models follow human instructions rather than just completing text.
4. **RLHF/DPO** aligns models with human preferences and values.
5. **Full fine-tuning** is increasingly replaced by **PEFT methods** due to memory, storage, and iteration speed advantages.
6. **Data quality** matters more than quantity for both pre-training and fine-tuning.

---

## References

- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2020)
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- Rafailov et al., "Direct Preference Optimization" (DPO, 2023)
