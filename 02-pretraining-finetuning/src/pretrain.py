"""Pre-training: next-token prediction on a general corpus.

Demonstrates the first stage of the pre-train → fine-tune paradigm:
training a GPT model on a diverse general corpus using next-token
prediction (autoregressive language modeling).

The pre-trained model learns general linguistic patterns — syntax,
semantics, and basic world knowledge from the training data.

Can either train from scratch or load the shared base model from
01-transformer-architecture/base_model.npz.
"""

import sys
from pathlib import Path

# Add the 01-transformer-architecture src to the path so we can import
# the GPT model, tokenizer, and trainer.
_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
sys.path.insert(0, str(_transformer_src))

# Add the data/ directory for corpus loading.
_data_dir = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(_data_dir))

import numpy as np
from loader import load_combined_corpus, load_general_corpus
from model import GPT, load_model
from tokenizer import BPETokenizer
from trainer import Trainer

# Path to the shared base model and tokenizer.
BASE_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "base_model.npz"
)
BASE_TOK_PATH = str(
    Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "base_tokenizer.json"
)


def load_shared_base_model() -> tuple[GPT, BPETokenizer] | None:
    """Loads the shared base model and tokenizer if they exist.

    Returns:
        Tuple of (model, tokenizer), or None if files don't exist.
    """
    if Path(BASE_MODEL_PATH).exists() and Path(BASE_TOK_PATH).exists():
        model = load_model(BASE_MODEL_PATH)
        tok = BPETokenizer.load(BASE_TOK_PATH)
        return model, tok
    return None


# ---------------------------------------------------------------------------
# General pre-training corpus (Alice in Wonderland from Project Gutenberg)
# ---------------------------------------------------------------------------

PRETRAIN_CORPUS = load_general_corpus()

# Combined corpus for tokenizer training (general + HP so HP words get merges).
COMBINED_CORPUS = load_combined_corpus()


def pretrain(
    vocab_size: int = 2000,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 1024,
    max_len: int = 64,
    epochs: int = 5,
    lr: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
    corpus: list[str] | None = None,
    tokenizer_corpus: list[str] | None = None,
    seq_len: int = 32,
) -> tuple[GPT, BPETokenizer, list[float]]:
    """Pre-trains a GPT model on a general corpus.

    Args:
        vocab_size: Target vocabulary size for the BPE tokenizer.
        d_model: Model hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of decoder blocks.
        d_ff: Feed-forward inner dimension.
        max_len: Maximum sequence length.
        epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed for reproducibility.
        verbose: If True, prints progress.
        corpus: Training corpus.  Defaults to the general Alice in Wonderland
            corpus.  Tests can pass a smaller corpus for speed.
        tokenizer_corpus: Corpus for tokenizer training.  Defaults to the
            combined general + HP corpus.  Tests can pass a smaller corpus.
        seq_len: Sequence length for training.

    Returns:
        Tuple of (pre-trained model, tokenizer, loss history).
    """
    np.random.seed(seed)

    train_corpus = corpus if corpus is not None else PRETRAIN_CORPUS
    tok_corpus = tokenizer_corpus if tokenizer_corpus is not None else COMBINED_CORPUS

    # Train tokenizer on the combined corpus (general + HP) so HP words
    # get proper subword merges.
    if verbose:
        print("Training BPE tokenizer on combined corpus...")
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(tok_corpus)

    # Create model.
    if verbose:
        print(
            f"Creating GPT model (vocab={len(tok)}, d_model={d_model}, "
            f"heads={num_heads}, layers={num_layers})..."
        )
    model = GPT(
        vocab_size=len(tok),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout_rate=0.0,
        use_learned_pos=True,
    )

    # Train on the general corpus only.
    if verbose:
        print(f"Pre-training for {epochs} epochs on {len(train_corpus)} sentences...")
    trainer = Trainer(model, tok, lr=lr, momentum=0.9)
    losses = trainer.train(train_corpus, epochs=epochs, seq_len=seq_len, verbose=verbose)

    if verbose:
        print(f"Pre-training complete.  Final loss: {losses[-1]:.4f}")

    return model, tok, losses
