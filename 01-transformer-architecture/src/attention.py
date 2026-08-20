"""Scaled dot-product attention and multi-head attention.

Implements the core attention mechanism from "Attention Is All You Need" using
only NumPy.  Supports both regular self-attention and causal (masked)
self-attention for autoregressive decoder models.
"""

import numpy as np
from numpy.typing import NDArray


def scaled_dot_product_attention(
    query: NDArray[np.float64],
    key: NDArray[np.float64],
    value: NDArray[np.float64],
    mask: NDArray[np.float64] | None = None,
    dropout_rate: float = 0.0,
    *,
    training: bool = True,
) -> NDArray[np.float64]:
    """Computes scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, seq_len_q, d_k).
        key: Key tensor of shape (batch, seq_len_k, d_k).
        value: Value tensor of shape (batch, seq_len_k, d_v).
        mask: Optional mask of shape (batch, seq_len_q, seq_len_k).
              Positions with mask == 0 are set to -inf before softmax.
        dropout_rate: Dropout probability applied to attention weights.
        training: If True, applies dropout during training.

    Returns:
        Attention output of shape (batch, seq_len_q, d_v).
    """
    d_k = query.shape[-1]

    # Compatibility scores: (batch, seq_len_q, seq_len_k)
    scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        # Expand mask to (batch, 1, seq_len_k) for broadcasting, then set masked
        # positions to a large negative value so they become 0 after softmax.
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax over the last dimension (the key sequence).
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    if training and dropout_rate > 0:
        keep_prob = 1.0 - dropout_rate
        dropout_mask = (np.random.rand(*attention_weights.shape) < keep_prob).astype(
            np.float64
        ) / keep_prob
        attention_weights *= dropout_mask

    return np.matmul(attention_weights, value)


class MultiHeadAttention:
    """Multi-head scaled dot-product attention.

    Splits the model dimension into h heads, runs attention in parallel, then
    concatenates and projects the results.

    Attributes:
        d_model: Total model dimension.
        num_heads: Number of attention heads.
        d_k: Dimension per head for queries and keys.
        d_v: Dimension per head for values.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initializes multi-head attention.

        Args:
            d_model: Model hidden dimension.
            num_heads: Number of parallel attention heads.
            dropout_rate: Dropout probability on attention weights.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout_rate = dropout_rate

        # Weight matrices for Q, K, V projections and output projection.
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float64) * scale

        # Bias terms (optional in the original paper; included for completeness).
        self.b_q = np.zeros(d_model, dtype=np.float64)
        self.b_k = np.zeros(d_model, dtype=np.float64)
        self.b_v = np.zeros(d_model, dtype=np.float64)
        self.b_o = np.zeros(d_model, dtype=np.float64)

    def forward(
        self,
        x: NDArray[np.float64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask of shape (batch, seq_len, seq_len).
            training: If True, applies dropout.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections.
        q = np.matmul(x, self.W_q) + self.b_q  # (batch, seq_len, d_model)
        k = np.matmul(x, self.W_k) + self.b_k
        v = np.matmul(x, self.W_v) + self.b_v

        # Reshape to (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k).
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_v).swapaxes(1, 2)

        # Expand mask to (batch, 1, 1, seq_len) for broadcasting across heads.
        if mask is not None:
            mask = mask[:, np.newaxis, :, :]

        # Attention per head.
        attn_out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_rate=self.dropout_rate, training=training
        )

        # Concatenate heads: (batch, seq_len, d_model).
        attn_out = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Output projection.
        return np.matmul(attn_out, self.W_o) + self.b_o

    def cross_attention_forward(
        self,
        x: NDArray[np.float64],
        encoder_output: NDArray[np.float64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass for cross-attention (decoder attending to encoder).

        Queries come from *x* (decoder), keys and values come from
        *encoder_output*.

        Args:
            x: Decoder input of shape (batch, seq_len_dec, d_model).
            encoder_output: Encoder output of shape (batch, seq_len_enc, d_model).
            mask: Optional mask of shape (batch, seq_len_dec, seq_len_enc).
            training: If True, applies dropout.

        Returns:
            Output tensor of shape (batch, seq_len_dec, d_model).
        """
        batch_size, seq_len_dec, _ = x.shape
        seq_len_enc = encoder_output.shape[1]

        q = np.matmul(x, self.W_q) + self.b_q
        k = np.matmul(encoder_output, self.W_k) + self.b_k
        v = np.matmul(encoder_output, self.W_v) + self.b_v

        q = q.reshape(batch_size, seq_len_dec, self.num_heads, self.d_k).swapaxes(1, 2)
        k = k.reshape(batch_size, seq_len_enc, self.num_heads, self.d_k).swapaxes(1, 2)
        v = v.reshape(batch_size, seq_len_enc, self.num_heads, self.d_v).swapaxes(1, 2)

        if mask is not None:
            mask = mask[:, np.newaxis, :, :]

        attn_out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_rate=self.dropout_rate, training=training
        )

        attn_out = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len_dec, self.d_model)
        return np.matmul(attn_out, self.W_o) + self.b_o


def create_causal_mask(seq_len: int) -> NDArray[np.float64]:
    """Creates a lower-triangular causal mask for autoregressive decoding.

    Tokens can only attend to themselves and previous positions.

    Args:
        seq_len: Sequence length.

    Returns:
        Boolean mask of shape (1, seq_len, seq_len) where True = allowed.
    """
    return np.tril(np.ones((1, seq_len, seq_len), dtype=np.float64))
