"""Prefix Tuning: prepend trainable vectors to attention keys and values.

Adds trainable "prefix" vectors to the key and value sequences in each
Transformer layer's self-attention.  The prefix vectors are learned
parameters; the base model weights stay frozen.

K' = W_k · [prefix_vectors, input_sequence]
V' = W_v · [prefix_vectors, input_sequence]
"""

import numpy as np
from numpy.typing import NDArray


class PrefixTuning:
    """Prefix tuning for a single Transformer layer.

    Adds m trainable prefix vectors to the key and value sequences.

    Attributes:
        num_prefix: Number of prefix tokens (m).
        d_model: Model hidden dimension.
        prefix_keys: Trainable prefix for keys, shape (num_prefix, d_model).
        prefix_values: Trainable prefix for values, shape (num_prefix, d_model).
    """

    def __init__(self, d_model: int, num_prefix: int = 10) -> None:
        """Initializes prefix tuning vectors.

        Args:
            d_model: Model hidden dimension.
            num_prefix: Number of prefix tokens to prepend.
        """
        self.num_prefix = num_prefix
        self.d_model = d_model

        # Initialize prefix vectors with small random values.
        self.prefix_keys = np.random.randn(num_prefix, d_model).astype(np.float64) * 0.02
        self.prefix_values = np.random.randn(num_prefix, d_model).astype(np.float64) * 0.02

    def forward(
        self, k: NDArray[np.float64], v: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Prepends prefix vectors to key and value sequences.

        Args:
            k: Key tensor of shape (batch, heads, seq_len, d_k).
            v: Value tensor of shape (batch, heads, seq_len, d_v).

        Returns:
            Tuple of (k_with_prefix, v_with_prefix), each with
            num_prefix extra positions prepended.
        """
        batch_size, num_heads, seq_len, d_k = k.shape
        d_v = v.shape[-1]

        # Expand prefix to (batch, heads, num_prefix, d_k/d_v).
        pk = np.tile(
            self.prefix_keys[np.newaxis, np.newaxis, :, :d_k],
            (batch_size, num_heads, 1, 1),
        )
        pv = np.tile(
            self.prefix_values[np.newaxis, np.newaxis, :, :d_v],
            (batch_size, num_heads, 1, 1),
        )

        k_with_prefix = np.concatenate([pk, k], axis=2)  # prepend
        v_with_prefix = np.concatenate([pv, v], axis=2)

        return k_with_prefix, v_with_prefix

    def backward(
        self,
        d_k: NDArray[np.float64],
        d_v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Backward pass — accumulates gradients into prefix vectors.

        Args:
            d_k: Gradient w.r.t. k_with_prefix, shape (batch, heads, seq+prefix, d_k).
            d_v: Gradient w.r.t. v_with_prefix, shape (batch, heads, seq+prefix, d_v).

        Returns:
            Tuple of (d_k_input, d_v_input) — gradients for the original
            input sequence (prefix portion stripped).
        """
        if not hasattr(self, "grad_prefix_keys"):
            self._init_grads()

        # Gradient for prefix: first num_prefix positions.
        self.grad_prefix_keys += d_k[:, :, : self.num_prefix, :].sum(axis=(0, 1))
        self.grad_prefix_values += d_v[:, :, : self.num_prefix, :].sum(axis=(0, 1))

        # Return gradient for the original sequence (strip prefix).
        return d_k[:, :, self.num_prefix :, :], d_v[:, :, self.num_prefix :, :]

    def _init_grads(self) -> None:
        """Initializes gradient accumulators."""
        self.grad_prefix_keys = np.zeros_like(self.prefix_keys)
        self.grad_prefix_values = np.zeros_like(self.prefix_values)

    def zero_grad(self) -> None:
        """Resets gradient accumulators."""
        self._init_grads()

    def __repr__(self) -> str:
        return (
            f"PrefixTuning(d_model={self.d_model}, num_prefix={self.num_prefix})"
        )
