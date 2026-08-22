"""Adapter fine-tuning demo.

Demonstrates parameter-efficient fine-tuning using adapter layers:
    1. Load a pre-trained GPT model
    2. Wrap it with Houlsby-style adapters
    3. Fine-tune only the adapter parameters (base model frozen)
    4. Compare trainable parameter count vs full fine-tuning

Run with:
    PYTHONPATH=03-adapters/src:01-transformer-architecture/src \\
        uv run python 03-adapters/src/demo.py
"""

import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
sys.path.insert(0, str(_transformer_src))

# Add the data/ directory for corpus loading.
_data_dir = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(_data_dir))

import numpy as np
from adapter_gpt import AdapterGPT
from loader import load_combined_corpus, load_general_corpus, load_hp_corpus
from model import GPT, cross_entropy_loss
from tokenizer import BPETokenizer
from trainer import SGD, prepare_text_batch


def main() -> None:
    """Runs the adapter fine-tuning demo."""
    print("=" * 70)
    print("Adapter Fine-Tuning Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load shared base model (or train from scratch)
    # ------------------------------------------------------------------
    _base_path = Path(__file__).resolve().parents[2] / "01-transformer-architecture"
    _model_path = str(_base_path / "base_model.npz")
    _tok_path = str(_base_path / "base_tokenizer.json")

    if Path(_model_path).exists() and Path(_tok_path).exists():
        from model import load_model

        base = load_model(_model_path)
        tok = BPETokenizer.load(_tok_path)
        print("\n[1/4] Loaded shared base model from chapter 1.")
    else:
        print("\n[1/4] No shared model — training base model from scratch...")
        np.random.seed(42)
        general_corpus = load_general_corpus()
        combined_corpus = load_combined_corpus()
        tok = BPETokenizer(vocab_size=2000)
        tok.train(combined_corpus)
        base = GPT(
            vocab_size=len(tok),
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
            max_len=64,
            dropout_rate=0.0,
        )
        from trainer import Trainer

        Trainer(base, tok, lr=0.1, momentum=0.9).train(
            general_corpus, epochs=5, seq_len=32, verbose=False
        )

    print(
        f"  Base model: vocab={len(tok)}, params={sum(p.size for p in base.get_params().values()):,}"
    )

    # ------------------------------------------------------------------
    # 2. Wrap with adapters
    # ------------------------------------------------------------------
    print("\n[2/4] Wrapping with Houlsby adapters (bottleneck=16)...")
    adapter_model = AdapterGPT(base, bottleneck=16)
    adapter_params = adapter_model.count_adapter_params()
    base_params = sum(p.size for p in base.get_params().values())
    print(f"  Base model parameters:   {base_params:,} (frozen)")
    print(f"  Adapter parameters:      {adapter_params:,} (trainable)")
    print(f"  Trainable fraction:      {adapter_params / base_params * 100:.1f}%")

    # ------------------------------------------------------------------
    # 3. Fine-tune adapters on a new domain
    # ------------------------------------------------------------------
    print("\n[3/4] Fine-tuning adapters on Harry Potter domain...")
    domain_corpus = load_hp_corpus()

    inp, tgt, msk = prepare_text_batch(domain_corpus[:5], tok, seq_len=32)
    loss_before = cross_entropy_loss(adapter_model.forward(inp, mask=msk, training=False), tgt, msk)
    print(f"  Loss before fine-tuning: {loss_before:.4f}")

    opt = SGD(lr=0.1, momentum=0.9, max_grad_norm=1.0)
    for epoch in range(3):
        for text in domain_corpus:
            inp, tgt, msk = prepare_text_batch([text], tok, seq_len=32)
            adapter_model.zero_grad()
            logits = adapter_model.forward(inp, mask=msk, training=True)
            from model import softmax_cross_entropy_backward

            d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
            adapter_model.backward(d_logits)
            opt.step(adapter_model)

    loss_after = cross_entropy_loss(adapter_model.forward(inp, mask=msk, training=False), tgt, msk)
    print(f"  Loss after fine-tuning:  {loss_after:.4f}")
    print(f"  Reduction:               {(1 - loss_after / loss_before) * 100:.1f}%")

    # ------------------------------------------------------------------
    # 4. Show generation
    # ------------------------------------------------------------------
    print("\n[4/4] Generation after adapter fine-tuning...")
    prompts = ["Harry", "Hogwarts", "the wizard"]
    for prompt in prompts:
        p_ids = np.array([tok.encode(prompt)], dtype=np.int64)
        gen = adapter_model.generate(p_ids, max_new_tokens=8, temperature=0.0)
        print(f'  "{prompt}" → "{tok.decode(gen)}"')

    print("\n" + "=" * 70)
    print("Demo complete!")
    print(
        f"  - Only {adapter_params:,} adapter params trained "
        f"(vs {base_params:,} for full fine-tuning)"
    )
    print("  - Base model weights frozen (no catastrophic forgetting)")
    print("  - Adapters can be swapped for multi-task serving")
    print("=" * 70)


if __name__ == "__main__":
    main()
