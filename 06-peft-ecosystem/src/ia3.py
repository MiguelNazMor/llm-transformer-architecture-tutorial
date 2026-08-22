"""IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).

Rescales key, value, and FFN activations using learned vectors:
    h' = l_v ⊙ (W_v · x)    ← rescales value activations
    h' = l_k ⊙ (W_k · x)    ← rescales key activations
    h' = l_ff ⊙ FFN(x)      ← rescales FFN activations

Where l_v, l_k, l_ff are learned vectors of shape (d_model,)
and ⊙ is element-wise multiplication.

IA³ is extremely parameter-efficient: only 3 * L * d_model scalars
per layer (vs 2 * d_model * bottleneck for adapters).
"""

import numpy as np
from numpy.typing import NDArray


class IA3Layer:
    """IA³ rescaling vectors for a single Transformer layer.

    Attributes:
        d_model: Model hidden dimension.
        l_k: Learned key rescaling vector, shape (d_model,).
        l_v: Learned value rescaling vector, shape (d_model,).
        l_ff: Learned FFN rescaling vector, shape (d_model,).
    """

    def __init__(self, d_model: int) -> None:
        """Initializes IA³ rescaling vectors.

        All vectors start at 1.0 (identity — no rescaling).

        Args:
            d_model: Model hidden dimension.
        """
        self.d_model = d_model
        self.l_k = np.ones(d_model, dtype=np.float64)
        self.l_v = np.ones(d_model, dtype=np.float64)
        self.l_ff = np.ones(d_model, dtype=np.float64)

    def rescale_key(self, k: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rescales key activations: k' = l_k ⊙ k.

        Args:
            k: Key tensor of shape (..., d_model).

        Returns:
            Rescaled key tensor.
        """
        self._k = k
        return k * self.l_k

    def rescale_value(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rescales value activations: v' = l_v ⊙ v.

        Args:
            v: Value tensor of shape (..., d_model).

        Returns:
            Rescaled value tensor.
        """
        self._v = v
        return v * self.l_v

    def rescale_ffn(self, ffn_out: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rescales FFN output: ffn' = l_ff ⊙ ffn_out.

        Args:
            ffn_out: FFN output of shape (..., d_model).

        Returns:
            Rescaled FFN output.
        """
        self._ffn = ffn_out
        return ffn_out * self.l_ff

    def backward(
        self,
        d_k_out: NDArray[np.float64] | None = None,
        d_v_out: NDArray[np.float64] | None = None,
        d_ffn_out: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Backward pass — accumulates gradients into rescaling vectors.

        Args:
            d_k_out: Gradient w.r.t. rescaled key output.
            d_v_out: Gradient w.r.t. rescaled value output.
            d_ffn_out: Gradient w.r.t. rescaled FFN output.

        Returns:
            Tuple of (d_k, d_v, d_ffn) — gradients w.r.t. the original
            (unscaled) inputs.
        """
        if not hasattr(self, "grad_l_k"):
            self._init_grads()

        d_k, d_v, d_ffn = None, None, None

        if d_k_out is not None:
            # d(l_k ⊙ k)/dl_k = k * d_k_out  (element-wise)
            self.grad_l_k += (self._k * d_k_out).sum(axis=tuple(range(d_k_out.ndim - 1)))
            d_k = d_k_out * self.l_k

        if d_v_out is not None:
            self.grad_l_v += (self._v * d_v_out).sum(axis=tuple(range(d_v_out.ndim - 1)))
            d_v = d_v_out * self.l_v

        if d_ffn_out is not None:
            self.grad_l_ff += (self._ffn * d_ffn_out).sum(axis=tuple(range(d_ffn_out.ndim - 1)))
            d_ffn = d_ffn_out * self.l_ff

        return d_k, d_v, d_ffn

    def _init_grads(self) -> None:
        """Initializes gradient accumulators."""
        self.grad_l_k = np.zeros(self.d_model, dtype=np.float64)
        self.grad_l_v = np.zeros(self.d_model, dtype=np.float64)
        self.grad_l_ff = np.zeros(self.d_model, dtype=np.float64)

    def zero_grad(self) -> None:
        """Resets gradient accumulators."""
        self._init_grads()

    def count_params(self) -> int:
        """Returns the number of trainable parameters."""
        return self.l_k.size + self.l_v.size + self.l_ff.size

    def __repr__(self) -> str:
        return f"IA3Layer(d_model={self.d_model})"
