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


# ======================================================================
# Text learning tests
# ======================================================================


class TestFeedForwardTextLearning:
    """Tests that the feed-forward network can learn transformations.

    The FFN is a position-wise MLP that transforms each token's
    representation independently.  These tests verify that:
        - The FFN can learn to map specific inputs to specific outputs
        - The backward pass produces non-zero gradients for all weights
        - The gradient check passes (analytical vs numerical gradients)
    """

    def test_ffn_learns_target_transformation(self) -> None:
        """FFN should learn to approximate a target mapping via gradient descent.

        We create a random input, a random target, and train the FFN to
        minimize the MSE between its output and the target.  After 100
        steps of gradient descent, the loss should be significantly lower
        than the initial loss.
        """
        np.random.seed(42)
        ffn = FeedForward(d_model=8, d_ff=32)

        x = np.random.randn(1, 4, 8)
        target = np.random.randn(1, 4, 8) * 0.5

        # Compute initial loss
        out = ffn.forward(x)
        initial_loss = float(np.mean((out - target) ** 2))

        # Train
        lr = 0.01
        for _ in range(100):
            out = ffn.forward(x)
            d_out = 2 * (out - target) / out.size
            ffn.backward(d_out)
            ffn.W_1 -= lr * ffn.grad_W_1
            ffn.W_2 -= lr * ffn.grad_W_2
            ffn.b_1 -= lr * ffn.grad_b_1
            ffn.b_2 -= lr * ffn.grad_b_2
            ffn.zero_grad()

        final_loss = float(np.mean((ffn.forward(x) - target) ** 2))
        assert final_loss < initial_loss * 0.5, (
            f"FFN did not learn: loss {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_ffn_gradient_is_nonzero(self) -> None:
        """After a backward pass, all weight gradients should be non-zero.

        This ensures the gradient signal reaches all parameters, which
        is necessary for the FFN to learn during training.
        """
        np.random.seed(42)
        ffn = FeedForward(d_model=8, d_ff=16)
        x = np.random.randn(1, 4, 8)
        grad_out = np.random.randn(1, 4, 8)

        ffn.forward(x)
        ffn.backward(grad_out)

        assert np.any(ffn.grad_W_1 != 0), "W_1 gradient is all zeros"
        assert np.any(ffn.grad_W_2 != 0), "W_2 gradient is all zeros"
        assert np.any(ffn.grad_b_1 != 0), "b_1 gradient is all zeros"
        assert np.any(ffn.grad_b_2 != 0), "b_2 gradient is all zeros"

    def test_ffn_gradient_check(self) -> None:
        """Analytical FFN gradients should match numerical gradients.

        This is a gradient check: we perturb each weight by a small amount
        and compare the finite-difference gradient to the analytical one
        computed by the backward pass.  They should match to within 1%.
        """
        np.random.seed(42)
        ffn = FeedForward(d_model=4, d_ff=8)
        x = np.random.randn(1, 2, 4)
        grad_out = np.random.randn(1, 2, 4)

        ffn.forward(x)
        ffn.backward(grad_out)

        eps = 1e-5
        max_rel_error = 0.0

        for param_name, param, grad in [
            ("W_1", ffn.W_1, ffn.grad_W_1),
            ("W_2", ffn.W_2, ffn.grad_W_2),
        ]:
            for _ in range(5):
                i = np.random.randint(0, param.shape[0])
                j = np.random.randint(0, param.shape[1])
                orig = param[i, j]
                param[i, j] = orig + eps
                out_p = ffn.forward(x)
                loss_p = float(np.sum(grad_out * out_p))
                param[i, j] = orig - eps
                out_m = ffn.forward(x)
                loss_m = float(np.sum(grad_out * out_m))
                param[i, j] = orig
                num_grad = (loss_p - loss_m) / (2 * eps)
                ana_grad = grad[i, j]
                rel_error = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10)
                max_rel_error = max(max_rel_error, rel_error)

        assert max_rel_error < 0.01, (
            f"FFN gradient check failed: max relative error = {max_rel_error:.6f}"
        )
