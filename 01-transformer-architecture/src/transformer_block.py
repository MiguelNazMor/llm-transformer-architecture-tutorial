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


def backward_layer_norm(
    x: NDArray[np.float64], grad_out: NDArray[np.float64], eps: float = 1e-6
) -> NDArray[np.float64]:
    """Backward pass through layer normalization (no learnable gamma/beta).

    Args:
        x: Original input to layer_norm, shape (..., d_model).
        grad_out: Gradient w.r.t. the normalized output, shape (..., d_model).
        eps: Same eps used in forward pass.

    Returns:
        Gradient w.r.t. the input x, shape (..., d_model).
    """
    d_model = x.shape[-1]
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)

    x_normed = (x - mean) * std_inv
    n = d_model

    # Standard layer norm gradient (without gamma/beta parameters)
    d_x = (
        (1.0 / n)
        * std_inv
        * (
            n * grad_out
            - np.sum(grad_out, axis=-1, keepdims=True)
            - x_normed * np.sum(grad_out * x_normed, axis=-1, keepdims=True)
        )
    )
    return d_x


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
        x_in = x  # save for backward
        normed_1 = layer_norm(x)
        attn_out = self.attention.forward(normed_1, mask=mask, training=training)
        x = x + attn_out

        # FFN with pre-norm and residual.
        x_after_attn = x  # save for backward
        normed_2 = layer_norm(x)
        ffn_out = self.ffn.forward(normed_2)
        out = x + ffn_out

        # Cache for backward
        if training:
            self._cache = {
                "x_in": x_in,
                "x_after_attn": x_after_attn,
            }

        return out

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass through the encoder block.

        Pre-norm architecture: each sub-layer is norm(input) with a residual
        connection.  Gradient flows through residual + through norm backward.

        Args:
            grad_output: Gradient w.r.t. the block output, shape (batch, seq, d_model).

        Returns:
            Gradient w.r.t. the block input, shape (batch, seq, d_model).
        """
        c = self._cache

        # out = x_after_attn + ffn_out
        # ffn_out = ffn(norm(x_after_attn))
        d_ffn_out = grad_output
        d_normed_2 = self.ffn.backward(d_ffn_out)
        d_x_after_attn = grad_output + backward_layer_norm(c["x_after_attn"], d_normed_2)

        # x_after_attn = x_in + attn_out
        # attn_out = self_attn(norm(x_in))
        d_attn_out = d_x_after_attn
        d_normed_1 = self.attention.backward(d_attn_out)
        d_x_in = d_x_after_attn + backward_layer_norm(c["x_in"], d_normed_1)

        return d_x_in

    def zero_grad(self) -> None:
        """Resets gradient accumulators for all sub-layers."""
        self.attention.zero_grad()
        self.ffn.zero_grad()


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
        x_in = x  # save original input for backward
        normed_1 = layer_norm(x)
        attn_out = self.self_attention.forward(normed_1, mask=causal_mask, training=training)
        x = x + attn_out

        # Cross-attention to encoder.
        x_after_attn = x  # save for backward
        normed_2 = layer_norm(x)
        cross_out = self.cross_attention.cross_attention_forward(
            normed_2, encoder_output, mask=cross_mask, training=training
        )
        x = x + cross_out

        # FFN.
        x_after_cross = x  # save for backward
        normed_3 = layer_norm(x)
        ffn_out = self.ffn.forward(normed_3)
        out = x + ffn_out

        # Cache for backward — store INPUTS to layer_norm (not outputs)
        if training:
            self._cache = {
                "x_in": x_in,  # input to first layer_norm
                "x_after_attn": x_after_attn,  # input to second layer_norm
                "x_after_cross": x_after_cross,  # input to third layer_norm
            }

        return out

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass through the decoder block.

        Uses pre-norm architecture: each sub-layer is norm(sublayer_input)
        with a residual connection around the whole norm+sublayer.

        Args:
            grad_output: Gradient w.r.t. the block output, shape (batch, seq, d_model).

        Returns:
            Gradient w.r.t. the block input, shape (batch, seq, d_model).
        """
        c = self._cache

        # out = x_after_cross + ffn_out
        # ffn_out = ffn(norm(x_after_cross))
        # d_x_after_cross = grad_output (residual) + backward_norm(x_after_cross, d_normed)
        d_ffn_out = grad_output  # residual contribution passes through
        d_normed_3 = self.ffn.backward(d_ffn_out)
        d_x_after_cross = grad_output + backward_layer_norm(c["x_after_cross"], d_normed_3)

        # x_after_cross = x_after_attn + cross_out
        # cross_out = cross_attn(norm(x_after_attn), encoder_output)
        # Cross-attention backward is not implemented for GPT (dummy encoder).
        # The gradient flows through the residual only.
        d_x_after_attn = d_x_after_cross  # cross-attention gradient is ~0 with dummy encoder

        # x_after_attn = x_in + attn_out
        # attn_out = self_attn(norm(x_in))
        d_attn_out = d_x_after_attn  # residual contribution
        d_normed_1 = self.self_attention.backward(d_attn_out)
        d_x_in = d_x_after_attn + backward_layer_norm(c["x_in"], d_normed_1)

        return d_x_in

    def zero_grad(self) -> None:
        """Resets gradient accumulators for all sub-layers."""
        self.self_attention.zero_grad()
        self.ffn.zero_grad()
