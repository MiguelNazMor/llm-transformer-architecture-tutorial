"""Pre-training → Fine-Tuning demo.

Demonstrates the complete two-stage paradigm:
    1. Load a pre-trained GPT model (general corpus: Alice in Wonderland)
    2. Fine-tune it on a specific domain (Harry Potter)
    3. Compare generation before and after fine-tuning
    4. Show domain-specific perplexity improvement and knowledge retention

Run with:
    PYTHONPATH=02-pretraining-finetuning/src:01-transformer-architecture/src \\
        uv run python 02-pretraining-finetuning/src/demo.py
"""

import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
sys.path.insert(0, str(_transformer_src))

import numpy as np
from finetune import GENERAL_CORPUS, HP_CORPUS, evaluate_perplexity, finetune
from loader import split_corpus
from pretrain import load_shared_base_model, pretrain


def main() -> None:
    """Runs the pre-training → fine-tuning demo."""
    print("=" * 70)
    print("Pre-training → Fine-Tuning Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load shared base model (or train from scratch)
    # ------------------------------------------------------------------
    shared = load_shared_base_model()
    if shared is not None:
        model, tok = shared
        print("\n[1/5] Loaded shared base model from chapter 1.")
        pretrain_losses = None
    else:
        print("\n[1/5] No shared model found — pre-training from scratch...")
        model, tok, pretrain_losses = pretrain(
            vocab_size=2000,
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
            epochs=5,
            lr=0.1,
            seed=42,
            verbose=False,
        )
    if pretrain_losses is not None:
        print(f"  Initial loss: {pretrain_losses[0]:.4f}")
        print(f"  Final loss:   {pretrain_losses[-1]:.4f}")

    # Held-out sets for fair evaluation (no overlap with training).
    hp_train, hp_eval = split_corpus(HP_CORPUS, eval_fraction=0.2)
    general_train, general_eval = split_corpus(GENERAL_CORPUS, eval_fraction=0.2)

    # Show pre-trained generation on general and HP prompts.
    general_prompts = ["the cat", "she said", "the queen", "Alice"]
    hp_prompts = ["Harry", "Hogwarts", "the wizard", "the spell"]
    print("\n  Pre-trained generation (general prompts):")
    for prompt in general_prompts:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = model.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'    "{prompt}" → "{tok.decode(gen)}"')
    print("\n  Pre-trained generation (HP prompts — domain not yet seen):")
    for prompt in hp_prompts:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = model.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'    "{prompt}" → "{tok.decode(gen)}"')

    # ------------------------------------------------------------------
    # 2. Evaluate on both domains before fine-tuning
    # ------------------------------------------------------------------
    print("\n[2/5] Evaluating perplexity BEFORE fine-tuning (held-out sets)...")
    hp_ppl_before = evaluate_perplexity(model, tok, hp_eval)
    general_ppl_before = evaluate_perplexity(model, tok, general_eval)
    print(f"  HP domain perplexity:      {hp_ppl_before:.2f}")
    print(f"  General domain perplexity: {general_ppl_before:.2f}")

    # ------------------------------------------------------------------
    # 3. Fine-tune on HP domain
    # ------------------------------------------------------------------
    print("\n[3/5] Fine-tuning on Harry Potter domain...")
    finetune_losses = finetune(model, tok, hp_train, epochs=3, lr=0.05, verbose=False)
    print(f"  Fine-tuning initial loss: {finetune_losses[0]:.4f}")
    print(f"  Fine-tuning final loss:   {finetune_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluate after fine-tuning
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating perplexity AFTER fine-tuning (held-out sets)...")
    hp_ppl_after = evaluate_perplexity(model, tok, hp_eval)
    general_ppl_after = evaluate_perplexity(model, tok, general_eval)
    print(
        f"  HP domain perplexity:      {hp_ppl_after:.2f} "
        f"(was {hp_ppl_before:.2f}, "
        f"{(1 - hp_ppl_after / hp_ppl_before) * 100:.1f}% improvement)"
    )
    print(f"  General domain perplexity: {general_ppl_after:.2f} (was {general_ppl_before:.2f})")

    # ------------------------------------------------------------------
    # 5. Show generation after fine-tuning
    # ------------------------------------------------------------------
    print("\n[5/5] Generation AFTER fine-tuning...")
    print("\n  HP domain prompts (should show domain adaptation):")
    for prompt in hp_prompts:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = model.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'    "{prompt}" → "{tok.decode(gen)}"')

    print("\n  General prompts (knowledge retention check):")
    for prompt in general_prompts:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = model.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'    "{prompt}" → "{tok.decode(gen)}"')

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("  - Pre-trained model learned general language patterns")
    print("  - Fine-tuned model specializes in Harry Potter domain")
    print("  - Perplexity reported on held-out data (no train/eval overlap)")
    print("  - Some general knowledge retained (check perplexity change)")
    print("=" * 70)


if __name__ == "__main__":
    main()
