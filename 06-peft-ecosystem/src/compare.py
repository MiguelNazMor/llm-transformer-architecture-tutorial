"""PEFT method comparison — all methods applied to the same base model.

Loads the shared base model from chapter 1, then applies each PEFT method
to the same starting point.  Compares:
    - Trainable parameter count and memory savings
    - Training time (total and per step)
    - Loss before and after fine-tuning on the domain corpus (Harry Potter)
    - Loss on the general corpus (Alice in Wonderland) to measure knowledge
      retention vs catastrophic forgetting
    - Inference latency (time to generate text after training)
    - Generation quality on both general and domain-specific prompts

Run with:
    PYTHONPATH=06-peft-ecosystem/src:05-qlora/src:04-lora/src:03-adapters/src:02-pretraining-finetuning/src:01-transformer-architecture/src \\
        uv run python 06-peft-ecosystem/src/compare.py
"""

import sys
import time
from pathlib import Path

# Add all source directories.
for sub in [
    "01-transformer-architecture",
    "02-pretraining-finetuning",
    "03-adapters",
    "04-lora",
    "05-qlora",
]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / sub / "src"))

# Add the data/ directory for corpus loading.
_data_dir = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(_data_dir))

import numpy as np
from adapter_gpt import AdapterGPT
from loader import load_general_corpus, load_hp_corpus, split_corpus
from lora_gpt import LoRAGPT
from model import GPT, cross_entropy_loss, load_model, softmax_cross_entropy_backward
from qlora_gpt import QLoRAGPT
from tokenizer import BPETokenizer
from trainer import SGD, prepare_text_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "base_model.npz"
)
BASE_TOK_PATH = str(
    Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "base_tokenizer.json"
)

# ---------------------------------------------------------------------------
# Corpora and prompts
# ---------------------------------------------------------------------------

# Domain-specific corpus for fine-tuning (Harry Potter).
_HP_CORPUS = load_hp_corpus()
HP_TRAIN, HP_EVAL = split_corpus(_HP_CORPUS, eval_fraction=0.2)

# General corpus for knowledge-retention evaluation (Alice in Wonderland).
_GENERAL_CORPUS = load_general_corpus()
_, GENERAL_EVAL = split_corpus(_GENERAL_CORPUS, eval_fraction=0.2)

# Domain-specific prompts (Harry Potter).
HP_PROMPTS = ["Harry", "Hogwarts", "the wizard", "the spell"]

# General prompts (Alice in Wonderland style).
GENERAL_PROMPTS = ["the cat", "she said", "the queen", "Alice"]

# Sequence length for training and evaluation.
SEQ_LEN = 32


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_loss(model, tok, corpus, seq_len=SEQ_LEN):
    """Computes per-token average cross-entropy loss on a corpus."""
    total_nll = 0.0
    total_tokens = 0.0
    for text in corpus:
        inp, tgt, msk = prepare_text_batch([text], tok, seq_len)
        logits = model.forward(inp, mask=msk, training=False)
        token_count = float(np.sum(msk))
        total_nll += cross_entropy_loss(logits, tgt, msk) * token_count
        total_tokens += token_count
    return total_nll / max(total_tokens, 1.0)


def measure_inference_latency(model, tok, prompts, max_new_tokens=8, warmup=3, repeats=5):
    """Measures average time to generate text from a prompt.

    Args:
        model: The model to benchmark.
        tok: Tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Number of tokens to generate per prompt.
        warmup: Number of warmup iterations (not counted).
        repeats: Number of timed iterations.

    Returns:
        Average milliseconds per generation call.
    """
    # Warmup
    for _ in range(warmup):
        for prompt in prompts:
            p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
            model.generate(p_ids, max_new_tokens=max_new_tokens, temperature=0.0)

    # Timed
    t0 = time.perf_counter()
    for _ in range(repeats):
        for prompt in prompts:
            p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
            model.generate(p_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    elapsed = time.perf_counter() - t0

    total_calls = repeats * len(prompts)
    return (elapsed / total_calls) * 1000  # ms per call


def train_with_optimizer(model, tok, corpus, epochs=4, lr=0.1, seq_len=SEQ_LEN):
    """Trains a model using SGD optimizer (works for GPT, AdapterGPT, etc.).

    Returns:
        Tuple of (steps, elapsed_seconds).
    """
    opt = SGD(lr=lr, momentum=0.9, max_grad_norm=1.0)
    t0 = time.perf_counter()
    steps = 0
    for _ in range(epochs):
        for text in corpus:
            inp, tgt, msk = prepare_text_batch([text], tok, seq_len)
            model.zero_grad()
            logits = model.forward(inp, mask=msk, training=True)
            d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
            model.backward(d_logits)
            opt.step(model)
            steps += 1
    elapsed = time.perf_counter() - t0
    return steps, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs the PEFT method comparison."""
    # Check that shared base model exists.
    if not Path(BASE_MODEL_PATH).exists():
        print("ERROR: Shared base model not found.")
        print(f"  Expected at: {BASE_MODEL_PATH}")
        print("  Run: PYTHONPATH=01-transformer-architecture/src uv run python")
        print("       01-transformer-architecture/src/train.py")
        return

    print("=" * 100)
    print("PEFT Method Comparison — All methods on the same base model")
    print("=" * 100)

    # Load shared base model and tokenizer.
    print("\nLoading shared base model from chapter 1...")
    tok = BPETokenizer.load(BASE_TOK_PATH)
    base = load_model(BASE_MODEL_PATH)
    base_params = sum(p.size for p in base.get_params().values())
    print(f"  Vocab size:      {len(tok)}")
    print(f"  Base params:     {base_params:,}")
    print(f"  Domain corpus:   {len(HP_TRAIN)} HP sentences for training, {len(HP_EVAL)} held out")
    print(f"  General corpus:  {len(GENERAL_EVAL)} Alice sentences for retention check (held out)")

    # Show generation from base model (before any fine-tuning).
    print("\nBase model generation (no fine-tuning):")
    print("  General prompts:")
    for prompt in GENERAL_PROMPTS:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = base.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'    "{prompt}" → "{tok.decode(gen)}"')
    print("  HP prompts:")
    for prompt in HP_PROMPTS:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = base.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'    "{prompt}" → "{tok.decode(gen)}"')

    # Base model loss on held-out sets.
    base_domain_loss = compute_loss(base, tok, HP_EVAL)
    base_general_loss = compute_loss(base, tok, GENERAL_EVAL)
    print(f"\n  Base model loss on HP domain (held-out):      {base_domain_loss:.4f}")
    print(f"  Base model loss on general corpus (held-out): {base_general_loss:.4f}")

    # ------------------------------------------------------------------
    # Compare methods
    # ------------------------------------------------------------------
    results: list[dict] = []

    methods = [
        ("Full Fine-Tuning", lambda: _copy_model(base), "base"),
        ("Adapters (b=16)", lambda: AdapterGPT(_copy_model(base), bottleneck=16), "adapter"),
        ("LoRA (r=16)", lambda: LoRAGPT(_copy_model(base), rank=16), "lora"),
        ("QLoRA (r=16)", lambda: QLoRAGPT(_copy_model(base), rank=16), "qlora"),
    ]

    for name, factory, tag in methods:
        print(f"\n{'─' * 100}")
        print(f"  {name}")
        print(f"{'─' * 100}")

        model = factory()

        # Count trainable parameters.
        if tag == "base":
            trainable = base_params
        elif tag == "adapter":
            trainable = model.count_adapter_params()
        elif tag == "lora":
            trainable = model.count_lora_params()
        else:
            trainable = model.count_lora_params()

        pct = trainable / base_params * 100
        print(f"  Trainable params: {trainable:,} ({pct:.2f}% of base)")

        # Loss before fine-tuning (domain + general held-out sets).
        domain_loss_before = compute_loss(model, tok, HP_EVAL)
        general_loss_before = compute_loss(model, tok, GENERAL_EVAL)
        print(f"  Domain loss before:  {domain_loss_before:.4f}")
        print(f"  General loss before: {general_loss_before:.4f}")

        # Train on HP domain.
        steps, elapsed = train_with_optimizer(model, tok, HP_TRAIN, epochs=4, lr=0.1)
        ms_per_step = (elapsed / steps) * 1000
        print(
            f"  Training: {steps} steps in {elapsed:.1f}s "
            f"({ms_per_step:.1f} ms/step, {steps / elapsed:.0f} steps/s)"
        )

        # Loss after fine-tuning (on held-out sets, so no data leakage).
        domain_loss_after = compute_loss(model, tok, HP_EVAL)
        general_loss_after = compute_loss(model, tok, GENERAL_EVAL)
        domain_reduction = (1 - domain_loss_after / domain_loss_before) * 100
        general_change = (general_loss_after / general_loss_before - 1) * 100
        print(f"  Domain loss after:   {domain_loss_after:.4f} ({domain_reduction:.1f}% reduction)")
        print(
            f"  General loss after:  {general_loss_after:.4f} "
            f"({general_change:+.1f}% change — negative = retention, positive = forgetting)"
        )

        # Inference latency.
        latency_ms = measure_inference_latency(model, tok, HP_PROMPTS, max_new_tokens=8)
        print(f"  Inference latency: {latency_ms:.1f} ms per generation")

        # Generation on both domains.
        print("  Generation (HP domain):")
        for prompt in HP_PROMPTS:
            p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
            gen = model.generate(p_ids, max_new_tokens=8, temperature=0.0)
            print(f'    "{prompt}" → "{tok.decode(gen)}"')
        print("  Generation (general):")
        for prompt in GENERAL_PROMPTS:
            p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
            gen = model.generate(p_ids, max_new_tokens=8, temperature=0.0)
            print(f'    "{prompt}" → "{tok.decode(gen)}"')

        results.append(
            {
                "name": name,
                "trainable": trainable,
                "pct": pct,
                "domain_before": domain_loss_before,
                "domain_after": domain_loss_after,
                "domain_reduction": domain_reduction,
                "general_before": general_loss_before,
                "general_after": general_loss_after,
                "general_change": general_change,
                "train_ms": ms_per_step,
                "latency_ms": latency_ms,
            }
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    header = (
        f"{'Method':<25} {'Params':>9} {'%Base':>7} "
        f"{'DomBefore':>9} {'DomAfter':>9} {'DomRed':>7} "
        f"{'GenAfter':>9} {'GenChg':>7} "
        f"{'Train':>8} {'Latency':>9}"
    )
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")
    print(header)
    print("-" * 120)
    for r in results:
        print(
            f"{r['name']:<25} {r['trainable']:>9,} {r['pct']:>6.2f}% "
            f"{r['domain_before']:>9.4f} {r['domain_after']:>9.4f} "
            f"{r['domain_reduction']:>6.1f}% "
            f"{r['general_after']:>9.4f} {r['general_change']:>+6.1f}% "
            f"{r['train_ms']:>7.1f}ms {r['latency_ms']:>8.1f}ms"
        )
    print("-" * 120)

    # ------------------------------------------------------------------
    # How to read these results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 120}")
    print("HOW TO READ THESE RESULTS")
    print(f"{'=' * 120}")

    print("""
  COLUMN         WHAT IT MEANS
  ─────────────  ──────────────────────────────────────────────────────────
  Method         The fine-tuning technique being compared.

  Params         Number of trainable parameters.  Fewer = less memory
                 needed for gradients and optimizer states.  This is why
                 PEFT methods can run on consumer GPUs.

  %Base          Trainable params as a percentage of the full model size.
                 Full FT = 100%.  LoRA/Adapters typically < 10%.

  DomBefore      Cross-entropy loss on the HELD-OUT HP corpus BEFORE training.
                 High numbers mean the model is "surprised" by HP text.

  DomAfter       Cross-entropy loss on the HELD-OUT HP corpus AFTER training.
                 Should be lower than DomBefore — the model now expects
                 the domain-specific words.

  DomRed         Domain loss reduction: (Before - After) / Before × 100.
                 Higher = more effective domain learning.

  GenAfter       Cross-entropy loss on the general corpus (Alice in
                 Wonderland) AFTER training.  Compare to the base model's
                 general loss to assess knowledge retention.

  GenChg         General loss change: (After / Before - 1) × 100.
                 Negative = knowledge retained (good).
                 Positive = catastrophic forgetting (bad).
                 PEFT methods typically retain more general knowledge than
                 Full FT because most base weights are frozen.

  Train          Milliseconds per training step (forward + backward + update).
                 Lower = faster training.  PEFT methods are faster because
                 they compute gradients for fewer parameters.

  Latency        Milliseconds per text generation call (8 new tokens).
                 Measures inference speed AFTER training.  Adapters add
                 a small overhead at each layer; LoRA/QLoRA add extra
                 compute during generation unless weights are merged first.
""")

    # ------------------------------------------------------------------
    # Interpretation for this specific run
    # ------------------------------------------------------------------
    print("  INTERPRETATION FOR THIS RUN")
    print("  ──────────────────────────")

    # Find the best method by domain loss reduction
    best = max(results, key=lambda r: r["domain_reduction"])
    fastest = min(results, key=lambda r: r["train_ms"])
    fewest = min(results, key=lambda r: r["pct"])
    # Least catastrophic forgetting (most negative general_change = best retention)
    best_retention = min(results, key=lambda r: r["general_change"])

    print(
        f"  - Best domain learning: {best['name']} ({best['domain_reduction']:.1f}% loss reduction)"
    )
    print("    This method learned the HP domain patterns most effectively.")
    print()
    print(
        f"  - Best knowledge retention: {best_retention['name']} ({best_retention['general_change']:+.1f}% general loss change)"
    )
    print("    This method preserved the most general knowledge after fine-tuning.")
    print("    PEFT methods typically retain better because base weights are frozen.")
    print()
    print(f"  - Fastest training: {fastest['name']} ({fastest['train_ms']:.1f} ms/step)")
    print("    PEFT methods train faster because they update fewer parameters.")
    print()
    print(f"  - Fewest trainable params: {fewest['name']} ({fewest['pct']:.2f}% of base)")
    print("    This method needs the least GPU memory for training.")
    print()
    print("  Key insight: PEFT methods achieve most of the domain learning benefit")
    print("  while training only a tiny fraction of the parameters AND preserving")
    print("  more of the original general knowledge.  For large models (7B+),")
    print("  this is the difference between fitting on a consumer GPU vs needing")
    print("  a data center.")
    print("=" * 120)


def _copy_model(model: GPT) -> GPT:
    """Creates a deep copy of a GPT model via save/load."""
    import os
    import tempfile

    from model import save_model

    path = os.path.join(tempfile.gettempdir(), "_compare_model.npz")
    save_model(model, path)
    copy = load_model(path)
    os.remove(path)
    return copy


if __name__ == "__main__":
    main()
