"""LoRA (Low-Rank Adaptation) layer.

Implements the LoRA technique from Hu et al. (2021):
    W' = W + (α/r) · A · B

where:
    - W is the frozen pre-trained weight matrix
    - A ∈ R^(d×r) and B ∈ R^(r×k) are trainable low-rank matrices
    - r is the rank (typically 4-64)
    - α is a scaling factor (typically 2·r)

Key properties:
    - Zero initialization of B means ΔW = 0 at start (identity behavior)
    - Weights can be merged: W_merged = W + (α/r) · A · B
    - Merging eliminates inference overhead
"""

import numpy as np
from numpy.typing import NDArray


class LoRALinear:
    """A linear layer with a LoRA adapter.

    Forward:  h = x @ W + (α/r) · x @ A @ B

    W is frozen (no gradients).  Only A and B are trained.

    Attributes:
        W: Frozen weight matrix of shape (in_features, out_features).
        A: Trainable low-rank matrix of shape (in_features, rank).
        B: Trainable low-rank matrix of shape (rank, out_features).
        rank: LoRA rank (r).
        alpha: Scaling factor (α).
    """

    def __init__(
        self,
        W: NDArray[np.float64],
        rank: int = 16,
        alpha: float | None = None,
    ) -> None:
        """Wraps a weight matrix with a LoRA adapter.

        Args:
            W: Pre-trained weight matrix (copied and frozen).
            rank: LoRA rank (r).
            alpha: Scaling factor.  If None, defaults to 2 * rank.
        """
        self.W = W.copy()  # frozen copy
        self.in_features, self.out_features = W.shape
        self.rank = min(rank, min(self.in_features, self.out_features))
        self.alpha = alpha if alpha is not None else 2.0 * self.rank
        self.scaling = self.alpha / self.rank

        # A: random Gaussian init
        self.A = np.random.randn(self.in_features, self.rank).astype(np.float64) * 0.02
        # B: zero init → ΔW = A·B = 0 at start
        self.B = np.zeros((self.rank, self.out_features), dtype=np.float64)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass with LoRA.

        Args:
            x: Input of shape (..., in_features).

        Returns:
            Output of shape (..., out_features).
        """
        self._x = x  # cache for backward

        # Frozen path
        base_out = np.matmul(x, self.W)

        # LoRA path
        lora_out = self.scaling * np.matmul(np.matmul(x, self.A), self.B)

        # Cache intermediate for backward
        self._lora_hidden = np.matmul(x, self.A)  # (..., rank)

        return base_out + lora_out

    def backward(self, grad_output: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward pass through the LoRA layer.

        Only computes gradients for A and B — W gradients are not computed.

        Args:
            grad_output: Gradient w.r.t. the output, shape (..., out_features).

        Returns:
            Gradient w.r.t. the input, shape (..., in_features).
        """
        if not hasattr(self, "grad_A"):
            self._init_grads()

        # Gradient through LoRA path: grad_out @ B^T @ A^T
        # lora_out = scaling * x @ A @ B
        # d(lora_out)/dB = scaling * (x @ A)^T @ grad_out
        # d(lora_out)/dA = scaling * x^T @ (grad_out @ B^T)
        flat_x = self._x.reshape(-1, self.in_features)
        flat_grad = grad_output.reshape(-1, self.out_features)
        flat_hidden = self._lora_hidden.reshape(-1, self.rank)

        # Gradient for B: dL/dB = scaling * (x @ A)^T @ grad_out
        self.grad_B += self.scaling * np.matmul(flat_hidden.T, flat_grad)

        # Gradient for A: dL/dA = scaling * x^T @ (grad_out @ B^T)
        d_hidden = self.scaling * np.matmul(flat_grad, self.B.T)  # (N, rank)
        self.grad_A += np.matmul(flat_x.T, d_hidden)

        # Gradient w.r.t. input: from both frozen and LoRA paths
        d_x_frozen = np.matmul(grad_output, self.W.T)
        d_x_lora = np.matmul(d_hidden, self.A.T).reshape(grad_output.shape[:-1] + (self.in_features,))

        return d_x_frozen + d_x_lora

    def _init_grads(self) -> None:
        """Initializes gradient accumulators."""
        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)

    def zero_grad(self) -> None:
        """Resets gradient accumulators."""
        self._init_grads()

    def merge(self) -> NDArray[np.float64]:
        """Merges LoRA weights into the base weight matrix.

        Returns:
            W_merged = W + (α/r) · A · B
        """
        return self.W + self.scaling * np.matmul(self.A, self.B)

    def __repr__(self) -> str:
        return (
            f"LoRALinear(in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha})"
        )
