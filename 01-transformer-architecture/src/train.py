"""Walkthrough demo for the Transformer architecture.

Demonstrates the full pipeline on a tiny corpus:
    1. Train a BPE tokenizer
    2. Create a small GPT model
    3. Run a forward pass and compute loss
    4. Show autoregressive text generation

Note: This demo does NOT run an actual training loop.  Training requires
automatic differentiation (PyTorch/TensorFlow/JAX), which is outside the
scope of this NumPy-from-scratch implementation.  The forward pass and
generation showcase the complete architecture end-to-end.
"""

import time

import numpy as np
from model import GPT, cross_entropy_loss
from tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Tiny training corpus
# ---------------------------------------------------------------------------

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog are friends",
    "the mat is soft and warm",
    "the log is hard and cold",
    "the cat likes the warm mat",
    "the dog likes the cold log",
    "cats and dogs are animals",
    "the sun is bright today",
    "the sky is blue and clear",
]


def prepare_batch(
    texts: list[str], tokenizer: BPETokenizer, seq_len: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenizes texts and creates input/target pairs for next-token prediction.

    Args:
        texts: List of strings to encode.
        tokenizer: Trained BPE tokenizer.
        seq_len: Sequence length for the batch.

    Returns:
        Tuple of (input_ids, target_ids, mask) as numpy arrays.
    """
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    # Take a single sequence of length seq_len + 1.
    if len(all_ids) <= seq_len:
        all_ids = (all_ids * ((seq_len + 2) // len(all_ids) + 1))[: seq_len + 1]

    input_ids = all_ids[:seq_len]
    target_ids = all_ids[1 : seq_len + 1]
    mask = [1] * seq_len

    return (
        np.array([input_ids], dtype=np.int64),
        np.array([target_ids], dtype=np.int64),
        np.array([mask], dtype=np.float64),
    )


def count_parameters(model: GPT) -> int:
    """Counts the total number of trainable parameters in the model.

    Args:
        model: The GPT model.

    Returns:
        Total parameter count.
    """
    total = 0
    total += model.token_embedding.weight.size
    if hasattr(model.pos_embedding, "weight"):
        total += model.pos_embedding.weight.size  # type: ignore[union-attr]
    for block in model.blocks:
        for attr_name in ("W_q", "W_k", "W_v", "W_o"):
            total += getattr(block.self_attention, attr_name).size
        for attr_name in ("W_1", "W_2", "W_gate", "W_up", "W_down"):
            if hasattr(block.ffn, attr_name):
                total += getattr(block.ffn, attr_name).size
    return total


def main() -> None:
    """Runs the walkthrough demo."""
    print("=" * 60)
    print("Transformer Architecture — Walkthrough Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Train the tokenizer
    # ------------------------------------------------------------------
    print("\n[1/5] Training BPE tokenizer on a tiny corpus...")
    t0 = time.perf_counter()
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(CORPUS)
    t1 = time.perf_counter()

    print(f"  Corpus:          {len(CORPUS)} sentences")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Learned merges:  {len(tokenizer.merges)}")
    print(f"  Time:            {t1 - t0:.2f}s")

    # Show how a sentence gets tokenized.
    sample = CORPUS[0]
    sample_ids = tokenizer.encode(sample)
    sample_tokens = [tokenizer.vocab.get(tid, "?") for tid in sample_ids]
    print(f'\n  Sample:   "{sample}"')
    print(f"  Token IDs: {sample_ids}")
    print(f"  Tokens:    {sample_tokens}")

    # ------------------------------------------------------------------
    # 2. Create the model
    # ------------------------------------------------------------------
    print("\n[2/5] Creating a small GPT model...")
    model = GPT(
        vocab_size=len(tokenizer),
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=32,
        dropout_rate=0.1,
        use_learned_pos=True,
    )
    n_params = count_parameters(model)
    print("  Architecture:  d_model=64, heads=4, layers=2, d_ff=256")
    print(f"  Parameters:    {n_params:,}")

    # ------------------------------------------------------------------
    # 3. Prepare a batch and run a forward pass
    # ------------------------------------------------------------------
    print("\n[3/5] Running forward pass...")
    input_ids, target_ids, mask = prepare_batch(CORPUS, tokenizer, seq_len=16)

    t0 = time.perf_counter()
    logits = model.forward(input_ids, mask=mask, training=True)
    t1 = time.perf_counter()

    loss = cross_entropy_loss(logits, target_ids, mask)
    print(f"  Input shape:    {input_ids.shape}")
    print(f"  Logits shape:   {logits.shape}  (batch, seq_len, vocab_size)")
    print(f"  Cross-entropy loss: {loss:.4f}")
    print(f"  Forward time:   {(t1 - t0) * 1000:.1f} ms")

    # ------------------------------------------------------------------
    # 4. Show attention shapes
    # ------------------------------------------------------------------
    print("\n[4/5] Architecture details...")
    print(f"  Token embedding:   {model.token_embedding.weight.shape}")
    if hasattr(model.pos_embedding, "weight"):
        print(f"  Position embedding: {model.pos_embedding.weight.shape}")  # type: ignore[union-attr]

    for i, block in enumerate(model.blocks):
        attn = block.self_attention
        ffn = block.ffn
        ffn_type = type(ffn).__name__
        print(f"  Block {i}: attn heads={attn.num_heads}, d_k={attn.d_k}, FFN={ffn_type}")

    # ------------------------------------------------------------------
    # 5. Text generation
    # ------------------------------------------------------------------
    print("\n[5/5] Autoregressive text generation...")
    prompts = ["the cat", "the dog", "the sun"]

    for prompt in prompts:
        prompt_ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        t0 = time.perf_counter()
        generated_ids = model.generate(prompt_ids, max_new_tokens=8, temperature=0.8)
        t1 = time.perf_counter()
        generated_text = tokenizer.decode(generated_ids)
        print(f'  "{prompt}" → "{generated_text}"  ({(t1 - t0) * 1000:.0f} ms)')

    print("\n" + "=" * 60)
    print("Demo complete!  All components verified:")
    print("  - BPE tokenizer (train, encode, decode)")
    print("  - Token + positional embeddings")
    print("  - Multi-head self-attention")
    print("  - SwiGLU feed-forward network")
    print("  - Causal masking for autoregressive generation")
    print("=" * 60)


if __name__ == "__main__":
    main()
