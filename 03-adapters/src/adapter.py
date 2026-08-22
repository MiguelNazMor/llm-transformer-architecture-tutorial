"""Adapter layer for parameter-efficient fine-tuning.

Implements the bottleneck adapter architecture from Houlsby et al. (2019):
    h' = h + UpProject(ReLU(DownProject(h)))

where:
    - DownProject: d_model → bottleneck (learned)
    - UpProject:   bottleneck → d_model (learned, initialized near zero)
    - The residual connection ensures the adapter starts as identity.

Only the adapter parameters are trained — the base model weights stay frozen.
"""

import numpy as np
from numpy.typing import NDArray


class AdapterLayer:
    """A bottleneck adapter layer for Transformer blocks.

    Architecture:
        h → DownProject(d→b) → ReLU → UpProject(b→d) → + h → h'

    The up-projection is initialized near zero so the adapter starts as
    an identity function: f(h) ≈ h.

    Attributes:
        d_model: Model hidden dimension.
        bottleneck: Bottleneck (inner) dimension.
        W_down: Down-projection matrix of shape (d_model, bottleneck).
        W_up: Up-projection matrix of shape (bottleneck, d_model).
    """

    def __init__(self, d_model: int, bottleneck: int = 64) -> None:
        """Initializes an adapter layer.

        Args:
            d_model: Model hidden dimension.
            bottleneck: Bottleneck size (much smaller than d_model).
        """
        self.d_model = d_model
        self.bottleneck = bottleneck

        # Xavier-like init for down-projection.
        scale = np.sqrt(2.0 / d_model)
        self.W_down = np.random.randn(d_model, bottleneck).astype(np.float64) * scale
        self.b_down = np.zeros(bottleneck, dtype=np.float64)

        # Near-zero init for up-projection → adapter starts as identity.
        self.W_up = np.random.randn(bottleneck, d_model).astype(np.float64) * 1e-5
        self.b_up = np.zeros(d_model, dtype=np.float64)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through the adapter.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        self._x = x  # cache for backward
        self._hidden = np.maximum(0, np.matmul(x, self.W_down) + self.b_down)
        adapter_out = np.matmul(self._hidden, self.W_up) + self.b_up
        return x + adapter_out  # residual connection

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass through the adapter.

        Args:
            grad_output: Gradient w.r.t. the adapter output, shape
                (batch, seq_len, d_model).

        Returns:
            Gradient w.r.t. the input, shape (batch, seq_len, d_model).
        """
        if not hasattr(self, "grad_W_down"):
            self._init_grads()

        # Adapter path: grad flows through residual + adapter
        # grad_output flows to both the residual path and the adapter path
        d_adapter_out = grad_output  # gradient through adapter path
        d_x_residual = grad_output  # gradient through residual path

        # Backprop through up-projection
        self.grad_W_up += np.matmul(
            self._hidden.reshape(-1, self.bottleneck).T,
            d_adapter_out.reshape(-1, self.d_model),
        )
        self.grad_b_up += d_adapter_out.reshape(-1, self.d_model).sum(axis=0)
        d_hidden = np.matmul(d_adapter_out, self.W_up.T)

        # Backprop through ReLU
        d_hidden[self._hidden <= 0] = 0

        # Backprop through down-projection
        self.grad_W_down += np.matmul(
            self._x.reshape(-1, self.d_model).T,
            d_hidden.reshape(-1, self.bottleneck),
        )
        self.grad_b_down += d_hidden.reshape(-1, self.bottleneck).sum(axis=0)
        d_x_adapter = np.matmul(d_hidden, self.W_down.T)

        return d_x_residual + d_x_adapter

    def _init_grads(self) -> None:
        """Initializes gradient accumulators to zero."""
        self.grad_W_down = np.zeros_like(self.W_down)
        self.grad_W_up = np.zeros_like(self.W_up)
        self.grad_b_down = np.zeros_like(self.b_down)
        self.grad_b_up = np.zeros_like(self.b_up)

    def zero_grad(self) -> None:
        """Resets all gradient accumulators."""
        self._init_grads()

    def __repr__(self) -> str:
        return (
            f"AdapterLayer(d_model={self.d_model}, bottleneck={self.bottleneck})"
        )
