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
        # Mask arrives pre-expanded from MultiHeadAttention (includes head dim).
        # Set masked positions to a large negative value so they become 0
        # after softmax.
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
        expanded_mask = None
        if mask is not None:
            expanded_mask = mask[:, np.newaxis, :, :]

        # Attention per head.
        attn_out = scaled_dot_product_attention(
            q, k, v, mask=expanded_mask, dropout_rate=self.dropout_rate, training=training
        )

        # Concatenate heads: (batch, seq_len, d_model).
        attn_concat = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Output projection.
        out = np.matmul(attn_concat, self.W_o) + self.b_o

        # Cache for backward (only if training=True to save memory in eval mode).
        if training:
            self._cache = {
                "x": x,
                "q": q,
                "k": k,
                "v": v,
                "attn_concat": attn_concat,
                "expanded_mask": expanded_mask,
            }
            # Recompute attention weights for backward (needed for softmax gradient).
            d_k = q.shape[-1]
            scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
            if expanded_mask is not None:
                scores = np.where(expanded_mask == 0, -1e9, scores)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            self._cache["attn_weights"] = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        return out

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass for self-attention (and cross-attention).

        Supports both self-attention and cross-attention.  When called after
        cross_attention_forward(), the cache includes "x_enc" and only the Q
        gradient flows back to the decoder input (K and V gradients go to
        the encoder output, which is handled separately).

        Args:
            grad_output: Gradient w.r.t. the output, shape (batch, seq_len, d_model).

        Returns:
            Gradient w.r.t. the input, shape (batch, seq_len, d_model).
        """
        if not hasattr(self, "grad_W_q"):
            self._init_grads()
        c = self._cache
        x = c["x"]
        batch_size, seq_len, _ = x.shape

        # For cross-attention, K and V come from encoder output, not from x.
        x_kv = c.get("x_enc", x)  # use encoder output if available (cross-attn)

        # Backprop through output projection: out = attn_concat @ W_o + b_o
        self.grad_W_o += np.matmul(
            c["attn_concat"].reshape(-1, self.d_model).T, grad_output.reshape(-1, self.d_model)
        )
        self.grad_b_o += grad_output.reshape(-1, self.d_model).sum(axis=0)
        d_attn_concat = np.matmul(grad_output, self.W_o.T)  # (batch, seq_len, d_model)

        # Reshape back to (batch, num_heads, seq_len, d_k)
        d_attn_out = d_attn_concat.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(
            1, 2
        )

        # Backprop through attention: attn_out = attn_weights @ v
        d_attn_weights = np.matmul(
            d_attn_out, c["v"].swapaxes(-2, -1)
        )  # (batch, heads, seq_q, seq_k)
        d_v = np.matmul(
            c["attn_weights"].swapaxes(-2, -1), d_attn_out
        )  # (batch, heads, seq_k, d_v)

        # Backprop through softmax
        # d_softmax: for each row, ds_i = s_i * (ds_i - sum_j(s_j * ds_j))
        attn_weights = c["attn_weights"]
        d_scores = attn_weights * (
            d_attn_weights - np.sum(attn_weights * d_attn_weights, axis=-1, keepdims=True)
        )

        # Backprop through scaling: scores = QK^T / sqrt(d_k)
        d_k = c["q"].shape[-1]
        d_scores /= np.sqrt(d_k)

        # Apply mask gradient: masked positions had scores = -1e9, so their
        # softmax output was ~0, and their gradient is ~0. No special handling needed.

        # Backprop through QK^T: scores = Q @ K^T
        d_q = np.matmul(d_scores, c["k"])  # (batch, heads, seq_q, d_k)
        d_k_mat = np.matmul(d_scores.swapaxes(-2, -1), c["q"])  # (batch, heads, seq_k, d_k)

        # Reshape Q, K, V gradients back to (batch, seq_len, d_model)
        d_q_flat = d_q.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)
        d_k_flat = d_k_mat.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)
        d_v_flat = d_v.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)

        # Backprop through linear projections.
        # Q comes from x (decoder input).  K and V come from x_kv (which is
        # x for self-attn, or encoder_output for cross-attn).
        self.grad_W_q += np.matmul(
            x.reshape(-1, self.d_model).T, d_q_flat.reshape(-1, self.d_model)
        )
        self.grad_W_k += np.matmul(
            x_kv.reshape(-1, self.d_model).T, d_k_flat.reshape(-1, self.d_model)
        )
        self.grad_W_v += np.matmul(
            x_kv.reshape(-1, self.d_model).T, d_v_flat.reshape(-1, self.d_model)
        )
        self.grad_b_q += d_q_flat.reshape(-1, self.d_model).sum(axis=0)
        self.grad_b_k += d_k_flat.reshape(-1, self.d_model).sum(axis=0)
        self.grad_b_v += d_v_flat.reshape(-1, self.d_model).sum(axis=0)

        # Gradient w.r.t. input.
        # For self-attention: all three paths (Q, K, V) flow back to x.
        # For cross-attention: only the Q path flows back to the decoder input.
        if "x_enc" in c:
            d_x = np.matmul(d_q_flat, self.W_q.T)  # only Q path for cross-attn
        else:
            d_x = (
                np.matmul(d_q_flat, self.W_q.T)
                + np.matmul(d_k_flat, self.W_k.T)
                + np.matmul(d_v_flat, self.W_v.T)
            )
        return d_x

    def _init_grads(self) -> None:
        """Initializes gradient accumulators to zero."""
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_b_o = np.zeros_like(self.b_o)

    def zero_grad(self) -> None:
        """Resets all gradient accumulators to zero."""
        self._init_grads()

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

        expanded_mask = None
        if mask is not None:
            expanded_mask = mask[:, np.newaxis, :, :]

        attn_out = scaled_dot_product_attention(
            q, k, v, mask=expanded_mask, dropout_rate=self.dropout_rate, training=training
        )

        attn_concat = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len_dec, self.d_model)
        out = np.matmul(attn_concat, self.W_o) + self.b_o

        # Cache for backward.  "x_enc" signals to backward() that this was
        # cross-attention, so K/V gradients use encoder_output instead of x.
        if training:
            self._cache = {
                "x": x,
                "x_enc": encoder_output,  # K and V source for cross-attention
                "q": q,
                "k": k,
                "v": v,
                "attn_concat": attn_concat,
                "expanded_mask": expanded_mask,
            }
            d_k = q.shape[-1]
            scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
            if expanded_mask is not None:
                scores = np.where(expanded_mask == 0, -1e9, scores)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            self._cache["attn_weights"] = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        return out


def create_causal_mask(seq_len: int) -> NDArray[np.float64]:
    """Creates a lower-triangular causal mask for autoregressive decoding.

    Tokens can only attend to themselves and previous positions.

    Args:
        seq_len: Sequence length.

    Returns:
        Boolean mask of shape (1, seq_len, seq_len) where True = allowed.
    """
    return np.tril(np.ones((1, seq_len, seq_len), dtype=np.float64))
