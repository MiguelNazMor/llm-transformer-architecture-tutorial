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
        self._x = x  # cache for backward
        self._pre_activated = np.matmul(x, self.W_1) + self.b_1
        self._hidden = np.maximum(0, self._pre_activated)
        return np.matmul(self._hidden, self.W_2) + self.b_2

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass through the ReLU feed-forward layer.

        Args:
            grad_output: Gradient w.r.t. the output, shape (batch, seq_len, d_model).

        Returns:
            Gradient w.r.t. the input, shape (batch, seq_len, d_model).
        """
        if not hasattr(self, "grad_W_1"):
            self._init_grads()
        # Backprop through W_2 + b_2
        self.grad_W_2 += np.matmul(
            self._hidden.reshape(-1, self.d_ff).T, grad_output.reshape(-1, self.d_model)
        )
        self.grad_b_2 += grad_output.reshape(-1, self.d_model).sum(axis=0)
        d_hidden = np.matmul(grad_output, self.W_2.T)  # (batch, seq, d_ff)

        # Backprop through ReLU
        d_pre = d_hidden * (self._pre_activated > 0).astype(np.float64)

        # Backprop through W_1 + b_1
        self.grad_W_1 += np.matmul(
            self._x.reshape(-1, self.d_model).T, d_pre.reshape(-1, self.d_ff)
        )
        self.grad_b_1 += d_pre.reshape(-1, self.d_ff).sum(axis=0)
        d_x = np.matmul(d_pre, self.W_1.T)  # (batch, seq, d_model)
        return d_x

    def _init_grads(self) -> None:
        """Initializes gradient accumulators to zero."""
        self.grad_W_1 = np.zeros_like(self.W_1)
        self.grad_W_2 = np.zeros_like(self.W_2)
        self.grad_b_1 = np.zeros_like(self.b_1)
        self.grad_b_2 = np.zeros_like(self.b_2)

    def zero_grad(self) -> None:
        """Resets all gradient accumulators to zero."""
        self._init_grads()

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
        self._x = x  # cache for backward
        self._gate_pre = np.matmul(x, self.W_gate)
        self._gate = _swish(self._gate_pre)
        self._up = np.matmul(x, self.W_up)
        self._gate_times_up = self._gate * self._up
        return np.matmul(self._gate_times_up, self.W_down)

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass through the SwiGLU feed-forward layer.

        Args:
            grad_output: Gradient w.r.t. the output, shape (batch, seq_len, d_model).

        Returns:
            Gradient w.r.t. the input, shape (batch, seq_len, d_model).
        """
        if not hasattr(self, "grad_W_gate"):
            self._init_grads()
        # Backprop through W_down
        self.grad_W_down += np.matmul(
            self._gate_times_up.reshape(-1, self.d_ff).T, grad_output.reshape(-1, self.d_model)
        )
        d_gate_times_up = np.matmul(grad_output, self.W_down.T)  # (batch, seq, d_ff)

        # gate * up → d_gate, d_up
        d_gate = d_gate_times_up * self._up
        d_up = d_gate_times_up * self._gate

        # Backprop through swish(gate_pre)
        # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sigmoid_gate = 1.0 / (1.0 + np.exp(-self._gate_pre))
        d_gate_pre = d_gate * (sigmoid_gate + self._gate_pre * sigmoid_gate * (1.0 - sigmoid_gate))

        # Backprop through W_gate and W_up
        self.grad_W_gate += np.matmul(
            self._x.reshape(-1, self.d_model).T, d_gate_pre.reshape(-1, self.d_ff)
        )
        self.grad_W_up += np.matmul(
            self._x.reshape(-1, self.d_model).T, d_up.reshape(-1, self.d_ff)
        )

        # Gradient w.r.t. input
        d_x = np.matmul(d_gate_pre, self.W_gate.T) + np.matmul(d_up, self.W_up.T)
        return d_x

    def _init_grads(self) -> None:
        """Initializes gradient accumulators to zero."""
        self.grad_W_gate = np.zeros_like(self.W_gate)
        self.grad_W_up = np.zeros_like(self.W_up)
        self.grad_W_down = np.zeros_like(self.W_down)

    def zero_grad(self) -> None:
        """Resets all gradient accumulators to zero."""
        self._init_grads()

    def __repr__(self) -> str:
        return f"SwiGLUFeedForward(d_model={self.d_model}, d_ff={self.d_ff})"
