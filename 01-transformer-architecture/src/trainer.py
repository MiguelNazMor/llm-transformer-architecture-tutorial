"""SGD optimizer and training loop for the GPT model.

Provides:
    - SGD: Stochastic Gradient Descent optimizer with optional momentum.
    - Trainer: Runs a training loop on a GPT model with a tokenizer and corpus.

This module ties together the forward pass (model.py), the backward pass
(model.backward), and the optimizer to actually train the model on text.
"""

import numpy as np
from model import GPT, cross_entropy_loss, softmax_cross_entropy_backward
from numpy.typing import NDArray
from tokenizer import BPETokenizer


class SGD:
    """Stochastic Gradient Descent optimizer with optional momentum.

    Updates parameters in-place using:
        v = momentum * v + grad
        param -= lr * v

    Includes gradient clipping to prevent gradient explosion, which is
    common in Transformers trained with high learning rates.

    Attributes:
        lr: Learning rate.
        momentum: Momentum factor (0 = vanilla SGD).
        max_grad_norm: Maximum L2 norm for gradient clipping (0 = no clipping).
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, max_grad_norm: float = 1.0) -> None:
        """Initializes the SGD optimizer.

        Args:
            lr: Learning rate.
            momentum: Momentum factor in [0, 1).  0 means vanilla SGD.
            max_grad_norm: Maximum L2 norm for gradient clipping.  0 disables.
        """
        self.lr = lr
        self.momentum = momentum
        self.max_grad_norm = max_grad_norm
        self._velocity: dict[str, NDArray[np.float64]] = {}

    def step(self, model: GPT) -> None:
        """Performs a single optimization step.

        Args:
            model: The GPT model whose parameters should be updated.
        """
        params = model.get_params()
        grads = model.get_grads()

        # Global gradient clipping
        if self.max_grad_norm > 0:
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            if total_norm > self.max_grad_norm:
                scale = self.max_grad_norm / (total_norm + 1e-6)
                for name in grads:
                    grads[name] *= scale

        for name, param in params.items():
            grad = grads[name]
            if self.momentum > 0:
                if name not in self._velocity:
                    self._velocity[name] = np.zeros_like(param)
                self._velocity[name] = self.momentum * self._velocity[name] + grad
                param -= self.lr * self._velocity[name]
            else:
                param -= self.lr * grad

    def zero_grad(self, model: GPT) -> None:
        """Resets the model's gradient accumulators.

        Args:
            model: The GPT model whose gradients should be zeroed.
        """
        model.zero_grad()


def prepare_text_batch(
    texts: list[str], tokenizer: BPETokenizer, seq_len: int = 16
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Tokenizes texts and creates input/target pairs for next-token prediction.

    Concatenates all texts into one long token sequence, then slices it into
    input/target pairs shifted by one position (standard language modeling setup).

    Args:
        texts: List of strings to encode.
        tokenizer: Trained BPE tokenizer.
        seq_len: Sequence length for the batch.

    Returns:
        Tuple of (input_ids, target_ids, mask) as numpy arrays.
        - input_ids: shape (1, seq_len)
        - target_ids: shape (1, seq_len)
        - mask: shape (1, seq_len) — all 1s (no padding)
    """
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    # Ensure we have enough tokens for a full sequence.
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


def prepare_single_text(
    text: str, tokenizer: BPETokenizer, seq_len: int = 16
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Prepares a single text for training.

    Args:
        text: Input string.
        tokenizer: Trained BPE tokenizer.
        seq_len: Sequence length.

    Returns:
        Tuple of (input_ids, target_ids, mask) as numpy arrays.
    """
    return prepare_text_batch([text], tokenizer, seq_len)


class Trainer:
    """Training loop wrapper for the GPT model.

    Handles the forward pass, backward pass, and optimizer step for each
    training iteration.  Tracks loss history for monitoring.

    Attributes:
        model: The GPT model to train.
        optimizer: The SGD optimizer.
        loss_history: List of loss values after each step.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        lr: float = 0.05,
        momentum: float = 0.9,
        max_grad_norm: float = 1.0,
    ) -> None:
        """Initializes the trainer.

        Args:
            model: The GPT model to train.
            tokenizer: Trained BPE tokenizer.
            lr: Learning rate for the optimizer.
            momentum: Momentum factor for the optimizer.
            max_grad_norm: Maximum gradient norm for clipping (0 = no clipping).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = SGD(lr=lr, momentum=momentum, max_grad_norm=max_grad_norm)
        self.loss_history: list[float] = []

    def train_step(
        self,
        input_ids: NDArray[np.int64],
        target_ids: NDArray[np.int64],
        mask: NDArray[np.float64],
    ) -> float:
        """Performs a single training step (forward + backward + optimizer).

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            target_ids: Target token IDs of shape (batch, seq_len).
            mask: Padding mask of shape (batch, seq_len).

        Returns:
            The loss value for this step.
        """
        # Zero gradients
        self.optimizer.zero_grad(self.model)

        # Forward pass (with caching for backward)
        logits = self.model.forward(input_ids, mask=mask, training=True)

        # Compute loss
        loss = cross_entropy_loss(logits, target_ids, mask)

        # Backward pass
        d_logits = softmax_cross_entropy_backward(logits, target_ids, mask)
        self.model.backward(d_logits)

        # Optimizer step
        self.optimizer.step(self.model)

        self.loss_history.append(loss)
        return loss

    def train(
        self,
        corpus: list[str],
        epochs: int = 50,
        seq_len: int = 16,
        verbose: bool = True,
    ) -> list[float]:
        """Trains the model on a corpus for multiple epochs.

        Args:
            corpus: List of training strings.
            epochs: Number of passes through the corpus.
            seq_len: Sequence length for each training sample.
            verbose: If True, prints progress every 10% of training.

        Returns:
            List of loss values, one per step.
        """
        # Prepare training data: one batch per sentence
        batches = []
        for text in corpus:
            inp, tgt, msk = prepare_text_batch([text], self.tokenizer, seq_len)
            batches.append((inp, tgt, msk))

        total_steps = epochs * len(batches)
        step = 0

        for epoch in range(epochs):
            for inp, tgt, msk in batches:
                loss = self.train_step(inp, tgt, msk)
                step += 1

                if verbose and step % max(1, total_steps // 10) == 0:
                    print(
                        f"  Step {step}/{total_steps} | "
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Loss: {loss:.4f}"
                    )

        return self.loss_history
