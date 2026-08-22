"""Token embeddings and sinusoidal positional encodings.

Implements the embedding layer and the sinusoidal positional encoding scheme
from the original Transformer paper.
"""

import numpy as np
from numpy.typing import NDArray


class TokenEmbedding:
    """Learned token embedding lookup table.

    Maps token IDs to dense vectors of size d_model.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding dimension.
        weight: Embedding matrix of shape (vocab_size, d_model).
    """

    def __init__(self, vocab_size: int, d_model: int = 512) -> None:
        """Initializes token embeddings.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: Dimension of each embedding vector.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Xavier-like initialization scaled for embedding layers.
        self.weight = np.random.randn(vocab_size, d_model).astype(np.float64) * 0.02

    def forward(self, token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """Looks up embeddings for a batch of token IDs.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len).

        Returns:
            Embedding tensor of shape (batch, seq_len, d_model).
        """
        self._token_ids = token_ids  # cache for backward
        return self.weight[token_ids]

    def backward(self, grad_output: NDArray[np.float64]) -> None:
        """Accumulates gradients into self.grad_weight.

        Uses scatter-add: for each token ID in the input, the corresponding
        row of grad_weight is incremented by the gradient flowing back.

        Args:
            grad_output: Gradient w.r.t. the embedding output, shape
                (batch, seq_len, d_model).
        """
        if not hasattr(self, "grad_weight"):
            self.grad_weight = np.zeros_like(self.weight)
        np.add.at(
            self.grad_weight, self._token_ids.reshape(-1), grad_output.reshape(-1, self.d_model)
        )

    def __repr__(self) -> str:
        return f"TokenEmbedding(vocab_size={self.vocab_size}, d_model={self.d_model})"


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> NDArray[np.float64]:
    """Computes sinusoidal positional encodings.

    Uses the formula from Vaswani et al. (2017):

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_len: Maximum sequence length to pre-compute encodings for.
        d_model: Model hidden dimension.

    Returns:
        Positional encoding matrix of shape (1, max_len, d_model).
    """
    position = np.arange(max_len, dtype=np.float64)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float64) * (-np.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe = np.zeros((1, max_len, d_model), dtype=np.float64)
    pe[0, :, 0::2] = np.sin(position * div_term)
    pe[0, :, 1::2] = np.cos(position * div_term)
    return pe


class LearnedPositionalEmbedding:
    """Learned positional embeddings (used in GPT-style models).

    Unlike sinusoidal encodings, these are trainable parameters.

    Attributes:
        max_len: Maximum sequence length.
        d_model: Embedding dimension.
        weight: Embedding matrix of shape (max_len, d_model).
    """

    def __init__(self, max_len: int = 512, d_model: int = 512) -> None:
        """Initializes learned positional embeddings.

        Args:
            max_len: Maximum sequence length.
            d_model: Embedding dimension.
        """
        self.max_len = max_len
        self.d_model = d_model
        self.weight = np.random.randn(max_len, d_model).astype(np.float64) * 0.02

    def forward(self, seq_len: int) -> NDArray[np.float64]:
        """Returns positional embeddings for a sequence of length seq_len.

        Args:
            seq_len: Length of the current sequence (must be ≤ max_len).

        Returns:
            Embedding tensor of shape (1, seq_len, d_model).

        Raises:
            ValueError: If seq_len exceeds max_len.
        """
        if seq_len > self.max_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_len ({self.max_len}).")
        self._seq_len = seq_len  # cache for backward
        return self.weight[np.newaxis, :seq_len, :]

    def backward(self, grad_output: NDArray[np.float64]) -> None:
        """Accumulates gradients into self.grad_weight.

        Args:
            grad_output: Gradient w.r.t. the positional embedding output,
                shape (1, seq_len, d_model) or (batch, seq_len, d_model).
        """
        if not hasattr(self, "grad_weight"):
            self.grad_weight = np.zeros_like(self.weight)
        # grad_output may be (batch, seq_len, d_model); sum over batch dim.
        if grad_output.ndim == 3 and grad_output.shape[0] != 1:
            grad_output = grad_output.sum(axis=0, keepdims=True)
        self.grad_weight[: self._seq_len] += grad_output[0]

    def __repr__(self) -> str:
        return f"LearnedPositionalEmbedding(max_len={self.max_len}, d_model={self.d_model})"
