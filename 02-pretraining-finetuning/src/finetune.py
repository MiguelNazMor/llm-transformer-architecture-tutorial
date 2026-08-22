"""Fine-tuning: adapting a pre-trained model to a specific domain.

Demonstrates the second stage of the pre-train → fine-tune paradigm:
taking a pre-trained GPT model and adapting it to a specific domain
or task using a smaller, focused dataset.

Key concepts demonstrated:
    - Domain adaptation: fine-tune on domain-specific text
    - Catastrophic forgetting: risk of losing pre-trained knowledge
    - Learning rate sensitivity: fine-tuning uses lower learning rates
"""

import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
sys.path.insert(0, str(_transformer_src))

# Add the data/ directory for corpus loading.
_data_dir = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(_data_dir))

import numpy as np
from loader import load_general_corpus, load_hp_corpus
from model import GPT, cross_entropy_loss
from tokenizer import BPETokenizer
from trainer import Trainer, prepare_text_batch

# ---------------------------------------------------------------------------
# Domain-specific fine-tuning corpora
# ---------------------------------------------------------------------------

# Harry Potter domain corpus for fine-tuning.
HP_CORPUS = load_hp_corpus()

# General corpus (Alice in Wonderland) — used to evaluate knowledge retention.
GENERAL_CORPUS = load_general_corpus()


def finetune(
    model: GPT,
    tokenizer: BPETokenizer,
    corpus: list[str],
    epochs: int = 50,
    lr: float = 0.05,
    seq_len: int = 32,
    verbose: bool = True,
) -> list[float]:
    """Fine-tunes a pre-trained model on a domain-specific corpus.

    Uses a lower learning rate than pre-training to avoid catastrophic
    forgetting of pre-trained knowledge.

    Args:
        model: Pre-trained GPT model (modified in-place).
        tokenizer: Trained BPE tokenizer.
        corpus: Domain-specific training texts.
        epochs: Number of fine-tuning epochs (typically fewer than pre-training).
        lr: Learning rate (typically lower than pre-training, e.g. 0.05 vs 0.1).
        seq_len: Sequence length for training.
        verbose: If True, prints progress.

    Returns:
        Loss history during fine-tuning.
    """
    if verbose:
        print(f"Fine-tuning on {len(corpus)} sentences for {epochs} epochs (lr={lr})...")

    trainer = Trainer(model, tokenizer, lr=lr, momentum=0.9)
    losses = trainer.train(corpus, epochs=epochs, seq_len=seq_len, verbose=verbose)

    if verbose:
        print(f"Fine-tuning complete.  Final loss: {losses[-1]:.4f}")

    return losses


def evaluate_perplexity(
    model: GPT,
    tokenizer: BPETokenizer,
    corpus: list[str],
    seq_len: int = 32,
) -> float:
    """Computes perplexity of the model on a given corpus.

    Perplexity = exp(loss).  Lower perplexity means the model is less
    "surprised" by the text — it predicts the corpus tokens with higher
    probability.

    Args:
        model: The GPT model to evaluate.
        tokenizer: Trained BPE tokenizer.
        corpus: Evaluation texts.
        seq_len: Sequence length for batching.

    Returns:
        Perplexity value (lower = better).
    """
    total_negative_log_likelihood = 0.0
    total_tokens = 0.0

    for text in corpus:
        inp, tgt, msk = prepare_text_batch([text], tokenizer, seq_len)
        logits = model.forward(inp, mask=msk, training=False)
        token_count = float(np.sum(msk))
        total_negative_log_likelihood += cross_entropy_loss(logits, tgt, msk) * token_count
        total_tokens += token_count

    avg_loss = total_negative_log_likelihood / max(total_tokens, 1.0)
    return float(np.exp(avg_loss))
