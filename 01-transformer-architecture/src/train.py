"""Training demo for the Transformer architecture.

Demonstrates the full pipeline on a real corpus:
    1. Train a BPE tokenizer on a combined corpus (general + domain)
    2. Create a GPT model
    3. Pre-train the model on a general corpus (Alice in Wonderland)
    4. Show loss decreasing over training steps
    5. Show autoregressive text generation before and after training

This demo runs an ACTUAL training loop using the manual backpropagation
implementation.  The GPT model learns to predict the next token in the
general corpus, and we can observe the loss decreasing and the generated
text becoming more coherent.

The tokenizer is trained on the combined general + HP corpus so that
domain-specific vocabulary (Hogwarts, Gryffindor, Quidditch, etc.) gets
proper subword merges.  Pre-training itself uses only the general corpus.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add the data/ directory to the path for corpus loading.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(_DATA_DIR))

from loader import load_combined_corpus, load_general_corpus
from model import GPT, cross_entropy_loss, save_model
from tokenizer import BPETokenizer
from trainer import Trainer

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

# General pre-training corpus (Alice in Wonderland from Project Gutenberg).
GENERAL_CORPUS = load_general_corpus()

# Combined corpus for tokenizer training (general + HP so HP words get merges).
COMBINED_CORPUS = load_combined_corpus()


def count_parameters(model: GPT) -> int:
    """Counts the total number of trainable parameters in the model.

    Args:
        model: The GPT model.

    Returns:
        Total parameter count.
    """
    return sum(p.size for p in model.get_params().values())


def main() -> None:
    """Runs the training demo."""
    print("=" * 70)
    print("Transformer Architecture — Pre-training Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Train the tokenizer
    # ------------------------------------------------------------------
    print("\n[1/6] Training BPE tokenizer on combined corpus (general + HP)...")
    t0 = time.perf_counter()
    tokenizer = BPETokenizer(vocab_size=2000)
    tokenizer.train(COMBINED_CORPUS)
    t1 = time.perf_counter()

    print(f"  Combined corpus: {len(COMBINED_CORPUS)} sentences")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Learned merges:  {len(tokenizer.merges)}")
    print(f"  Time:            {t1 - t0:.2f}s")

    # Show how a sentence gets tokenized.
    sample = GENERAL_CORPUS[0]
    sample_ids = tokenizer.encode(sample)
    sample_tokens = [tokenizer.vocab.get(tid, "?") for tid in sample_ids]
    print(f'\n  Sample:   "{sample}"')
    print(f"  Token IDs: {sample_ids}")
    print(f"  Tokens:    {sample_tokens}")

    # ------------------------------------------------------------------
    # 2. Create the model
    # ------------------------------------------------------------------
    print("\n[2/6] Creating GPT model...")
    model = GPT(
        vocab_size=len(tokenizer),
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_len=64,
        dropout_rate=0.0,  # no dropout for deterministic training
        use_learned_pos=True,
    )
    n_params = count_parameters(model)
    print("  Architecture:  d_model=256, heads=8, layers=6, d_ff=1024")
    print(f"  Parameters:    {n_params:,}")

    # ------------------------------------------------------------------
    # 3. Show generation BEFORE training
    # ------------------------------------------------------------------
    print("\n[3/6] Text generation BEFORE training...")
    np.random.seed(42)
    prompts = ["the cat", "she said", "the queen", "Harry"]
    for prompt in prompts:
        prompt_ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        generated_ids = model.generate(prompt_ids, max_new_tokens=10, temperature=0.0)
        generated_text = tokenizer.decode(generated_ids)
        print(f'  "{prompt}" → "{generated_text}"')

    # ------------------------------------------------------------------
    # 4. Pre-train the model on the general corpus
    # ------------------------------------------------------------------
    print(f"\n[4/6] Pre-training on general corpus ({len(GENERAL_CORPUS)} sentences)...")
    trainer = Trainer(model, tokenizer, lr=0.1, momentum=0.9)

    # Compute initial loss
    from trainer import prepare_text_batch

    inp, tgt, msk = prepare_text_batch([GENERAL_CORPUS[0]], tokenizer, seq_len=32)
    initial_logits = model.forward(inp, mask=msk, training=False)
    initial_loss = cross_entropy_loss(initial_logits, tgt, msk)
    print(f"  Initial loss: {initial_loss:.4f}")

    t0 = time.perf_counter()
    trainer.train(GENERAL_CORPUS, epochs=5, seq_len=32, verbose=True)
    t1 = time.perf_counter()

    final_loss = trainer.loss_history[-1]
    print(f"\n  Final loss:   {final_loss:.4f}")
    print(
        f"  Loss reduction: {initial_loss:.4f} → {final_loss:.4f} "
        f"({(1 - final_loss / initial_loss) * 100:.1f}% decrease)"
    )
    print(f"  Training time: {t1 - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Show generation AFTER training
    # ------------------------------------------------------------------
    print("\n[5/6] Text generation AFTER pre-training...")
    for prompt in prompts:
        prompt_ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        generated_ids = model.generate(prompt_ids, max_new_tokens=10, temperature=0.0)
        generated_text = tokenizer.decode(generated_ids)
        print(f'  "{prompt}" → "{generated_text}"')

    # ------------------------------------------------------------------
    # 6. Show architecture details
    # ------------------------------------------------------------------
    print("\n[6/6] Architecture details...")
    print(f"  Token embedding:   {model.token_embedding.weight.shape}")
    if hasattr(model.pos_embedding, "weight"):
        print(f"  Position embedding: {model.pos_embedding.weight.shape}")  # type: ignore[union-attr]

    for i, block in enumerate(model.blocks):
        attn = block.self_attention
        ffn = block.ffn
        ffn_type = type(ffn).__name__
        print(f"  Block {i}: attn heads={attn.num_heads}, d_k={attn.d_k}, FFN={ffn_type}")

    print("\n" + "=" * 70)
    print("Demo complete!  All components verified:")
    print("  - BPE tokenizer (train, encode, decode)")
    print("  - Token + positional embeddings")
    print("  - Multi-head self-attention (with backward)")
    print("  - Feed-forward network (with backward)")
    print("  - Causal masking for autoregressive generation")
    print("  - Manual backpropagation + SGD optimizer")
    print("  - Real training loop with decreasing loss")
    print("=" * 70)

    # Save the trained model and tokenizer for use in other chapters.
    _repo_root = Path(__file__).resolve().parents[2]
    model_path = _repo_root / "01-transformer-architecture" / "base_model.npz"
    tok_path = _repo_root / "01-transformer-architecture" / "base_tokenizer.json"
    save_model(model, str(model_path))
    tokenizer.save(str(tok_path))
    print(f"\nModel saved to {model_path}")
    print(f"Tokenizer saved to {tok_path}")
    print("Chapters 2-6 can now load this shared base model.")


if __name__ == "__main__":
    main()
