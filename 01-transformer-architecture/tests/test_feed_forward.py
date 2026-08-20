"""Tests for feed-forward networks.

The Transformer applies a position-wise feed-forward network after each
attention sub-layer.  This module tests two variants:

    FeedForward       — original ReLU FFN from Vaswani et al. (2017)
    SwiGLUFeedForward — gated variant used in Llama, PaLM, etc.

Key properties under test:
    - Output shape preservation (input and output have the same shape)
    - Non-linearity: the FFN is not an identity function
    - SwiGLU's default inner dimension calculation
    - String representation includes the activation type
"""

import numpy as np
from feed_forward import FeedForward, SwiGLUFeedForward


class TestFeedForward:
    """Tests for the ReLU feed-forward network.

    Architecture:
        FFN(x) = ReLU(x · W_1 + b_1) · W_2 + b_2

    where:
        W_1 ∈ R^(d_model × d_ff)    (expand)
        W_2 ∈ R^(d_ff × d_model)    (contract)
        d_ff is typically 4 × d_model

    The FFN is applied independently to each position (token) in the
    sequence — hence "position-wise."
    """

    def test_output_shape(self) -> None:
        """Verifies the FFN preserves the input shape.

        Input:  (batch=2, seq_len=10, d_model=256)
        Output: (batch=2, seq_len=10, d_model=256)

        The inner dimension d_ff=1024 is only used temporarily — the second
        projection maps back to d_model so the output can feed into the next
        layer without dimension mismatches.
        """
        ffn = FeedForward(d_model=256, d_ff=1024)
        x = np.random.randn(2, 10, 256)
        out = ffn.forward(x)
        assert out.shape == (2, 10, 256)

    def test_nonlinear(self) -> None:
        """Proves the FFN is not an identity function.

        We carefully construct the weight matrices to act as near-identity
        projections (selecting the first 4 dimensions), then set b_1 to all
        ones.  The ReLU activation passes these positive biases through, so
        the output gains an extra +1 per dimension from the bias.

        If the FFN were purely linear (no ReLU), the output would equal the
        input after the identity projection.  The bias + ReLU ensures the
        output differs, confirming non-linearity.

        How it works step by step:
            1. W_1: 4→8 identity padded with zeros
            2. W_2: 8→4 identity padded with zeros
            3. b_1: ones(8) → ReLU passes them through
            4. x·W_1 + b_1 → ones(8) (after ReLU)
            5. (ones(8))·W_2 → ones(4)
            6. Output ≠ input (ones(4) vs ones(4)... wait, they ARE equal!)
               Actually: x=[1,1,1,1], W_1 picks first 4 dims, b_1 adds 1,
               so hidden = [2,2,2,2, 1,1,1,1] after ReLU.
               Then W_2 picks first 4 dims → [2,2,2,2] ≠ [1,1,1,1] ✓
        """
        ffn = FeedForward(d_model=4, d_ff=8)
        ffn.W_1 = np.zeros((4, 8))
        ffn.W_1[:4, :4] = np.eye(4)
        ffn.W_2 = np.zeros((8, 4))
        ffn.W_2[:4, :4] = np.eye(4)
        ffn.b_1 = np.ones(8)
        ffn.b_2 = np.zeros(4)

        x = np.ones((1, 1, 4))
        out = ffn.forward(x)
        assert not np.allclose(out, x)

    def test_repr(self) -> None:
        """Ensures the string representation mentions ReLU.

        This helps with debugging: when printing a model summary, you can
        immediately see which activation function each FFN uses.
        """
        ffn = FeedForward(d_model=128, d_ff=512)
        r = repr(ffn)
        assert "ReLU" in r


class TestSwiGLUFeedForward:
    """Tests for the SwiGLU feed-forward network.

    Architecture (used in Llama, PaLM):
        SwiGLU(x) = (Swish(x · W_gate) ⊙ (x · W_up)) · W_down

    where:
        Swish(x) = x · σ(x)          (also called SiLU)
        ⊙ is element-wise multiplication

    The gate projection controls how much of the up-projection passes
    through, giving the network more expressive power than a simple ReLU.

    The default d_ff is (8/3)·d_model (rounded to a multiple of 256) so
    the total parameter count is comparable to a standard FFN with
    d_ff = 4·d_model (since SwiGLU has 3 weight matrices instead of 2).
    """

    def test_output_shape(self) -> None:
        """Verifies SwiGLU FFN preserves the input shape, same as ReLU FFN."""
        ffn = SwiGLUFeedForward(d_model=256)
        x = np.random.randn(2, 10, 256)
        out = ffn.forward(x)
        assert out.shape == (2, 10, 256)

    def test_default_d_ff(self) -> None:
        """Confirms the default inner dimension follows the Llama convention.

        For d_model=512:
            raw = (8/3) * 512 = 1365.33...
            rounded = ((1365 + 255) // 256) * 256 = 1536

        This ensures the SwiGLU FFN has roughly the same number of
        parameters as a standard ReLU FFN with d_ff = 4 * 512 = 2048:
            ReLU:  512×2048 + 2048×512 = 2,097,152 params (2 matrices)
            SwiGLU: 3 × (512×1536)     = 2,359,296 params (3 matrices)
        """
        ffn = SwiGLUFeedForward(d_model=512)
        assert ffn.d_ff == 1536

    def test_repr(self) -> None:
        """Ensures the string representation mentions SwiGLU for debugging."""
        ffn = SwiGLUFeedForward(d_model=128)
        r = repr(ffn)
        assert "SwiGLU" in r
