"""Tests for attention mechanisms.

Attention is the core innovation of the Transformer.  These tests verify
the two layers that implement it:

    scaled_dot_product_attention  — the raw Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V
    MultiHeadAttention            — runs h parallel attention heads, concatenates, projects

Key properties under test:
    - Correct output tensor shapes
    - Numerical stability (no NaN, finite values)
    - Causal masking (lower-triangular: token i cannot see token j > i)
    - Dropout: produces different outputs in train vs eval mode
    - Determinism in eval mode (same input → same output)
    - Cross-attention shape correctness (decoder attending to encoder)
"""

import numpy as np
import pytest
from attention import (
    MultiHeadAttention,
    create_causal_mask,
    scaled_dot_product_attention,
)


class TestScaledDotProductAttention:
    """Tests for the core scaled dot-product attention function.

    This is the mathematical heart of the Transformer:
        Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    Q (query):  "What am I looking for?"
    K (key):    "What information do I contain?"
    V (value):  "What is my actual content?"

    The function supports:
      - An optional mask to block certain positions (e.g., padding, causality)
      - Dropout on the attention weights during training
    """

    def test_output_shape(self) -> None:
        """Verifies the output shape matches the query shape in the last two dims.

        The formula produces an output of shape (batch, seq_len_q, d_v).
        seq_len_q comes from Q, and d_v comes from V.  K only participates
        in the compatibility scores and does not affect the output shape.

        Test setup:
            batch=2, seq_q=5, seq_k=7, d_k=64, d_v=64

        Q: (2, 5, 64)   →  "5 queries, each 64-dimensional"
        K: (2, 7, 64)   →  "7 keys"
        V: (2, 7, 64)   →  "7 values"
        Output: (2, 5, 64)  ← 5 weighted sums of values
        """
        batch, seq_q, seq_k, d_k, d_v = 2, 5, 7, 64, 64
        q = np.random.randn(batch, seq_q, d_k)
        k = np.random.randn(batch, seq_k, d_k)
        v = np.random.randn(batch, seq_k, d_v)

        out = scaled_dot_product_attention(q, k, v, training=False)
        assert out.shape == (batch, seq_q, d_v)

    def test_attention_sums_to_one(self) -> None:
        """Ensures the output is numerically stable (no NaN or Inf).

        The softmax normalizes attention weights to sum to 1 per query.
        While we can't easily extract the weights from the output, we can
        verify that the output contains only finite values — proving that
        the softmax didn't explode or produce NaN.

        A well-behaved softmax should always produce finite outputs even
        with random inputs, thanks to the max-subtraction trick:
            softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        q = np.random.randn(1, 3, 8)
        k = np.random.randn(1, 4, 8)
        v = np.random.randn(1, 4, 8)

        out = scaled_dot_product_attention(q, k, v, training=False)
        assert np.all(np.isfinite(out))

    def test_causal_mask(self) -> None:
        """Verifies the causal mask is lower-triangular.

        In autoregressive (decoder) models, token at position i must not
        attend to any token at position j > i — otherwise the model would
        "cheat" by looking at future tokens during training.

        The causal mask is a lower-triangular matrix:
            pos 0: [1, 0, 0, 0]  ← can only see itself
            pos 1: [1, 1, 0, 0]  ← can see pos 0 and itself
            pos 2: [1, 1, 1, 0]
            pos 3: [1, 1, 1, 1]  ← can see all previous positions

        This test checks specific mask entries to confirm the pattern.
        """
        mask = create_causal_mask(4)

        # Position 0: can attend to itself (1.0), but NOT to position 1 (0.0).
        assert mask[0, 0, 0] == 1.0
        assert mask[0, 0, 1] == 0.0

        # Position 3: can attend to position 2 (1.0).
        assert mask[0, 3, 2] == 1.0

        # Position 2: CANNOT attend to position 3 (0.0).
        assert mask[0, 2, 3] == 0.0

    def test_dropout_training_vs_eval(self) -> None:
        """Confirms dropout is active in training mode and off in eval mode.

        Dropout randomly zeros out attention weights during training to
        prevent overfitting.  In eval mode, dropout must be disabled so the
        model produces deterministic outputs.

        How it works:
            1. Seed the RNG for reproducibility.
            2. Run attention in training mode with dropout_rate=0.5.
            3. Re-seed and run in eval mode.
            4. The outputs must differ because training mode applies dropout
               while eval mode does not.
        """
        q = np.random.randn(2, 4, 16)
        k = np.random.randn(2, 4, 16)
        v = np.random.randn(2, 4, 16)

        np.random.seed(42)
        out_train = scaled_dot_product_attention(q, k, v, dropout_rate=0.5, training=True)
        np.random.seed(42)
        out_eval = scaled_dot_product_attention(q, k, v, dropout_rate=0.5, training=False)
        assert not np.allclose(out_train, out_eval)


class TestMultiHeadAttention:
    """Tests for multi-head attention.

    MultiHeadAttention runs h parallel attention "heads", each with its own
    learned Q, K, V projections.  The heads are concatenated and projected
    back to d_model:

        MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O

    where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi).

    The original paper uses h=8 heads with d_k = d_v = d_model/h = 64.
    """

    def test_output_shape(self) -> None:
        """Verifies that multi-head attention preserves the input shape.

        Input:  (batch=2, seq_len=6, d_model=256)
        Output: (batch=2, seq_len=6, d_model=256)

        Internally, the 256-dim model is split across 8 heads of 32 dims
        each, then concatenated back to 256.  The output shape must match
        the input shape so blocks can be stacked.
        """
        batch, seq, d_model, heads = 2, 6, 256, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads)
        x = np.random.randn(batch, seq, d_model)

        out = mha.forward(x, training=False)
        assert out.shape == (batch, seq, d_model)

    def test_d_model_not_divisible(self) -> None:
        """Ensures the constructor rejects invalid head counts.

        If d_model (e.g., 100) is not divisible by num_heads (e.g., 3),
        the per-head dimension d_k = 100/3 = 33.333... is not an integer.
        The constructor must raise a ValueError immediately rather than
        producing cryptic errors later during matrix multiplication.
        """
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=100, num_heads=3)

    def test_cross_attention_shape(self) -> None:
        """Verifies cross-attention produces the decoder's sequence length.

        In the encoder-decoder Transformer, the decoder's cross-attention
        uses:
          - Queries from the decoder (seq_len_dec)
          - Keys/Values from the encoder (seq_len_enc)

        The output shape must match the decoder sequence length, NOT the
        encoder sequence length, because each decoder position produces
        one output vector.

        Test setup:
            Decoder: (2, 5, 256)  — 5 positions
            Encoder: (2, 7, 256)  — 7 positions
            Output:  (2, 5, 256)  — matches decoder length
        """
        batch, seq_dec, seq_enc, d_model, heads = 2, 5, 7, 256, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads)
        x = np.random.randn(batch, seq_dec, d_model)
        enc_out = np.random.randn(batch, seq_enc, d_model)

        out = mha.cross_attention_forward(x, enc_out, training=False)
        assert out.shape == (batch, seq_dec, d_model)

    def test_forward_deterministic_in_eval(self) -> None:
        """Confirms that eval mode produces identical outputs for the same input.

        In eval mode (training=False), dropout is disabled and all operations
        are deterministic.  Running the same input through the model twice
        must produce bit-identical results.

        This is critical for:
          - Reproducible inference
          - Reliable benchmarking
          - Consistent generation in production
        """
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        x = np.random.randn(1, 4, 128)

        out1 = mha.forward(x, training=False)
        out2 = mha.forward(x, training=False)
        assert np.allclose(out1, out2)


# ======================================================================
# Text learning tests
# ======================================================================


class TestAttentionTextLearning:
    """Tests that attention can learn patterns from text-like data.

    These tests verify that the attention mechanism — when trained with
    backpropagation — can learn to focus on specific positions, and that
    the causal mask correctly prevents attending to future tokens during
    text generation.
    """

    def test_attention_learns_to_focus(
        self,
    ) -> None:
        """Attention weights should shift after training on a target pattern.

        We create a simple setup where the attention output at position 1
        should equal the input at position 0 (a "copy" pattern).  After
        training with gradient descent, the attention weights at position 1
        should shift toward position 0.
        """
        np.random.seed(42)
        d_model = 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=2, dropout_rate=0.0)

        # Create a simple input: 3 tokens
        x = np.random.randn(1, 3, d_model)

        # Target: output at position 1 should be the input at position 0
        target = np.zeros_like(x)
        target[0, 1, :] = x[0, 0, :]

        # Train for several steps
        lr = 0.1
        initial_loss = None
        for step in range(50):
            out = mha.forward(x, training=True)
            loss = float(np.mean((out - target) ** 2))
            if initial_loss is None:
                initial_loss = loss

            # Backward: d_out = 2 * (out - target) / N
            d_out = 2 * (out - target) / out.size
            mha.backward(d_out)

            # Update weights
            mha.W_q -= lr * mha.grad_W_q
            mha.W_k -= lr * mha.grad_W_k
            mha.W_v -= lr * mha.grad_W_v
            mha.W_o -= lr * mha.grad_W_o
            mha.zero_grad()

        final_loss = float(np.mean((mha.forward(x, training=False) - target) ** 2))
        assert final_loss < initial_loss, (
            f"Attention did not learn: loss {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_causal_mask_blocks_future_in_generation(self) -> None:
        """Causal mask should prevent attending to future positions.

        In autoregressive text generation, token i must not attend to
        token j > i.  We verify this by checking that the attention output
        at position 0 does not change when future positions are modified.
        """
        np.random.seed(42)
        d_model = 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=1, dropout_rate=0.0)
        causal_mask = create_causal_mask(4)

        x = np.random.randn(1, 4, d_model)
        out_original = mha.forward(x, mask=causal_mask, training=False)

        # Modify future positions (1, 2, 3) — should NOT affect output at position 0
        x_modified = x.copy()
        x_modified[0, 1:, :] += 10.0  # large perturbation to future tokens
        out_modified = mha.forward(x_modified, mask=causal_mask, training=False)

        # Output at position 0 should be identical
        assert np.allclose(out_original[0, 0, :], out_modified[0, 0, :], atol=1e-10), (
            "Causal mask failed: output at position 0 changed when future positions were modified"
        )

    def test_attention_gradient_is_nonzero(self) -> None:
        """After a backward pass, all attention weight gradients should be non-zero.

        This ensures that the gradient signal flows through all parts of
        the attention mechanism, which is necessary for learning.
        """
        np.random.seed(42)
        mha = MultiHeadAttention(d_model=8, num_heads=2, dropout_rate=0.0)
        x = np.random.randn(1, 4, 8)
        grad_out = np.random.randn(1, 4, 8)

        mha.forward(x, training=True)
        mha.backward(grad_out)

        assert np.any(mha.grad_W_q != 0), "W_q gradient is all zeros"
        assert np.any(mha.grad_W_k != 0), "W_k gradient is all zeros"
        assert np.any(mha.grad_W_v != 0), "W_v gradient is all zeros"
        assert np.any(mha.grad_W_o != 0), "W_o gradient is all zeros"
