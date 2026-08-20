"""Transformer encoder and decoder blocks.

Each block stacks multi-head attention and feed-forward sub-layers with
residual connections and layer normalization.
"""

import numpy as np
from attention import MultiHeadAttention
from feed_forward import FeedForward, SwiGLUFeedForward
from numpy.typing import NDArray


def layer_norm(x: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    """Applies Layer Normalization over the last dimension.

    Args:
        x: Input tensor of shape (..., d_model).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of the same shape.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class RMSNorm:
    """Root Mean Square Layer Normalization.

    Used in Llama instead of standard LayerNorm.  RMSNorm removes the
    mean-centering step, making it faster with comparable performance.

        RMSNorm(x) = x / RMS(x) * γ

    where RMS(x) = sqrt(mean(x²)).

    Attributes:
        d_model: Normalization dimension.
        gamma: Learnable scale parameter.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initializes RMSNorm.

        Args:
            d_model: Dimension to normalize over.
            eps: Small constant for numerical stability.
        """
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float64)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applies RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of the same shape.
        """
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.gamma


class EncoderBlock:
    """A single Transformer encoder block.

    Consists of:
        1. Multi-head self-attention → Add & Norm
        2. Feed-forward network → Add & Norm

    Uses pre-layer normalization (norm before sub-layer), which is more stable
    for training deep models.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        *,
        use_swiglu: bool = False,
    ) -> None:
        """Initializes an encoder block.

        Args:
            d_model: Model hidden dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward inner dimension.
            dropout_rate: Dropout probability.
            use_swiglu: If True, uses SwiGLU FFN instead of ReLU.
        """
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn: FeedForward | SwiGLUFeedForward
        if use_swiglu:
            self.ffn = SwiGLUFeedForward(d_model, d_ff)
        else:
            self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: NDArray[np.float64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass through the encoder block.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional attention mask.
            training: If True, applies dropout.

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        # Self-attention with pre-norm and residual.
        attn_out = self.attention.forward(layer_norm(x), mask=mask, training=training)
        x = x + attn_out

        # FFN with pre-norm and residual.
        ffn_out = self.ffn.forward(layer_norm(x))
        return x + ffn_out


class DecoderBlock:
    """A single Transformer decoder block.

    Consists of:
        1. Masked multi-head self-attention → Add & Norm
        2. Cross-attention (to encoder output) → Add & Norm
        3. Feed-forward network → Add & Norm

    Uses pre-layer normalization for training stability.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        *,
        use_swiglu: bool = False,
    ) -> None:
        """Initializes a decoder block.

        Args:
            d_model: Model hidden dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward inner dimension.
            dropout_rate: Dropout probability.
            use_swiglu: If True, uses SwiGLU FFN instead of ReLU.
        """
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn: FeedForward | SwiGLUFeedForward
        if use_swiglu:
            self.ffn = SwiGLUFeedForward(d_model, d_ff)
        else:
            self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: NDArray[np.float64],
        encoder_output: NDArray[np.float64],
        causal_mask: NDArray[np.float64] | None = None,
        cross_mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass through the decoder block.

        Args:
            x: Decoder input of shape (batch, seq_len_dec, d_model).
            encoder_output: Encoder output of shape (batch, seq_len_enc, d_model).
            causal_mask: Mask for self-attention (prevents attending to future).
            cross_mask: Mask for cross-attention (e.g., padding mask).
            training: If True, applies dropout.

        Returns:
            Output of shape (batch, seq_len_dec, d_model).
        """
        # Masked self-attention.
        normed = layer_norm(x)
        attn_out = self.self_attention.forward(normed, mask=causal_mask, training=training)
        x = x + attn_out

        # Cross-attention to encoder.
        normed = layer_norm(x)
        cross_out = self.cross_attention.cross_attention_forward(
            normed, encoder_output, mask=cross_mask, training=training
        )
        x = x + cross_out

        # FFN.
        ffn_out = self.ffn.forward(layer_norm(x))
        return x + ffn_out
