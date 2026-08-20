"""Position-wise feed-forward networks.

Implements both the original ReLU FFN from Vaswani et al. (2017) and the
SwiGLU variant used in modern LLMs (Llama, PaLM).
"""

import numpy as np
from numpy.typing import NDArray


def _swish(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Swish (SiLU) activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


class FeedForward:
    """Original Transformer feed-forward network with ReLU.

    FFN(x) = ReLU(x · W_1 + b_1) · W_2 + b_2

    Attributes:
        d_model: Input/output dimension.
        d_ff: Inner (hidden) dimension.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048) -> None:
        """Initializes a ReLU feed-forward layer.

        Args:
            d_model: Model hidden dimension.
            d_ff: Inner feed-forward dimension (typically 4× d_model).
        """
        self.d_model = d_model
        self.d_ff = d_ff

        scale = np.sqrt(2.0 / d_model)
        self.W_1 = np.random.randn(d_model, d_ff).astype(np.float64) * scale
        self.W_2 = np.random.randn(d_ff, d_model).astype(np.float64) * scale
        self.b_1 = np.zeros(d_ff, dtype=np.float64)
        self.b_2 = np.zeros(d_model, dtype=np.float64)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applies the ReLU feed-forward transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        hidden = np.maximum(0, np.matmul(x, self.W_1) + self.b_1)
        return np.matmul(hidden, self.W_2) + self.b_2

    def __repr__(self) -> str:
        return f"FeedForward(d_model={self.d_model}, d_ff={self.d_ff}, activation=ReLU)"


class SwiGLUFeedForward:
    """SwiGLU feed-forward network used in Llama, PaLM, etc.

    SwiGLU(x) = (Swish(x · W_gate) ⊙ (x · W_up)) · W_down

    The gate projection and up projection each have dimension d_ff.
    Because SwiGLU uses an element-wise gate, we use 2/3 of the typical d_ff
    to keep the parameter count comparable to a standard FFN.

    Attributes:
        d_model: Input/output dimension.
        d_ff: Inner dimension for gate and up projections.
    """

    def __init__(self, d_model: int = 512, d_ff: int | None = None) -> None:
        """Initializes a SwiGLU feed-forward layer.

        Args:
            d_model: Model hidden dimension.
            d_ff: Inner dimension.  If None, uses (8/3) * d_model rounded to
                  a multiple of 256, which keeps parameter count similar to
                  a standard FFN with d_ff = 4 * d_model.
        """
        self.d_model = d_model
        if d_ff is None:
            # Standard Llama convention: 8/3 * d_model → same params as 4× FFN.
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256  # round to multiple of 256
        self.d_ff = d_ff

        scale = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, d_ff).astype(np.float64) * scale
        self.W_up = np.random.randn(d_model, d_ff).astype(np.float64) * scale
        self.W_down = np.random.randn(d_ff, d_model).astype(np.float64) * scale

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applies the SwiGLU transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        gate = _swish(np.matmul(x, self.W_gate))
        up = np.matmul(x, self.W_up)
        return np.matmul(gate * up, self.W_down)

    def __repr__(self) -> str:
        return f"SwiGLUFeedForward(d_model={self.d_model}, d_ff={self.d_ff})"
