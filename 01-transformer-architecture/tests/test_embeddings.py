"""Tests for token and positional embeddings.

Embeddings convert discrete token IDs into continuous vector representations
that the Transformer can process.  This module tests three components:

    TokenEmbedding               — learned lookup table: token ID → dense vector
    sinusoidal_positional_encoding — fixed sine/cosine functions of position
    LearnedPositionalEmbedding   — trainable position vectors (GPT-style)

Key properties under test:
    - Correct output shapes (batch, seq_len, d_model)
    - Token embeddings: same ID → same vector, different IDs → different vectors
    - Sinusoidal PE: every position has a unique encoding
    - Sinusoidal PE: pos=0 has sin(0)=0 on even dims, cos(0)=1 on odd dims
    - Learned PE: rejects sequences longer than max_len
"""

import numpy as np
import pytest
from embeddings import (
    LearnedPositionalEmbedding,
    TokenEmbedding,
    sinusoidal_positional_encoding,
)


class TestTokenEmbedding:
    """Tests for the TokenEmbedding lookup table.

    TokenEmbedding is a simple weight matrix W ∈ R^(vocab_size × d_model).
    Given a tensor of token IDs, it returns W[ids] — the rows of W
    corresponding to each token.

    This is the first layer of every Transformer: raw token IDs enter,
    dense vectors come out.
    """

    def test_output_shape(self) -> None:
        """Verifies that a batch of token IDs produces the expected shape.

        Input:  (batch=2, seq_len=3) integer token IDs
        Output: (2, 3, d_model=64) float vectors

        Each of the 6 token IDs (2×3) is replaced by its 64-dimensional
        embedding vector.
        """
        emb = TokenEmbedding(vocab_size=100, d_model=64)
        ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        out = emb.forward(ids)
        assert out.shape == (2, 3, 64)

    def test_lookup(self) -> None:
        """Confirms the fundamental property of embeddings: same ID → same vector.

        This is a two-part test:

        Part 1 — Different IDs produce different vectors:
            Input [0, 1]: the vectors at positions 0 and 1 should NOT be
            equal because they come from different rows of the weight matrix.

        Part 2 — Same ID produces the same vector:
            Input [0, 0]: both positions should be identical because they
            both read row 0 of the weight matrix.

        This property is what makes embeddings a lookup table rather than
        a transformation.
        """
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        ids = np.array([[0, 1]], dtype=np.int64)
        out = emb.forward(ids)
        # Different tokens: positions 0 and 1 must differ.
        assert np.allclose(out[0, 0], out[0, 1]) is False

        ids2 = np.array([[0, 0]], dtype=np.int64)
        out2 = emb.forward(ids2)
        # Same token: both positions must be identical.
        assert np.allclose(out2[0, 0], out2[0, 1])


class TestSinusoidalPositionalEncoding:
    """Tests for the sinusoidal positional encoding.

    From "Attention Is All You Need" (Vaswani et al., 2017):

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    These fixed (non-learned) encodings inject position information into
    the otherwise order-agnostic self-attention mechanism.  Different
    frequencies (controlled by the 10000^(2i/d_model) term) allow the model
    to capture both short-range and long-range positional patterns.
    """

    def test_output_shape(self) -> None:
        """Verifies the encoding matrix has the expected dimensions.

        The output is (1, max_len, d_model) — a single "batch" of encodings
        that gets broadcast across the actual batch during the forward pass.
        The leading dimension of 1 avoids redundant copies.
        """
        pe = sinusoidal_positional_encoding(max_len=100, d_model=64)
        assert pe.shape == (1, 100, 64)

    def test_unique_positions(self) -> None:
        """Ensures every position has a distinct encoding vector.

        If two positions had identical encodings, the model couldn't
        distinguish them — defeating the purpose of positional information.

        How it works:
            1. Generate encodings for 50 positions with 128 dimensions.
            2. Flatten each position's 128-dim vector.
            3. Compare every pair (i, j) — none should be equal.
               (np.allclose with default tolerances is sufficient.)

        The sinusoids at different frequencies guarantee uniqueness because
        the wavelengths form a geometric progression from 2π to ~20000π,
        and no two integer positions produce identical values across all
        frequencies simultaneously.
        """
        pe = sinusoidal_positional_encoding(max_len=50, d_model=128)
        flat = pe[0].reshape(50, -1)
        for i in range(50):
            for j in range(i + 1, 50):
                assert not np.allclose(flat[i], flat[j])

    def test_even_odd_pattern(self) -> None:
        """Verifies the sin/cos alternation at position 0.

        At position 0:
          - Even dimensions (0, 2, 4, ...): sin(0) = 0
          - Odd dimensions  (1, 3, 5, ...): cos(0) = 1

        This pattern is a direct consequence of the formula:
            PE(0, 2i)   = sin(0) = 0
            PE(0, 2i+1) = cos(0) = 1

        It confirms the encoding is computed correctly at the boundary.
        """
        pe = sinusoidal_positional_encoding(max_len=10, d_model=64)
        assert np.allclose(pe[0, 0, 0::2], 0.0, atol=1e-10)
        assert np.allclose(pe[0, 0, 1::2], 1.0, atol=1e-10)


class TestLearnedPositionalEmbedding:
    """Tests for learned positional embeddings (GPT-style).

    Unlike sinusoidal encodings, learned positional embeddings are trainable
    parameters.  GPT models use these instead of fixed sinusoids.

    The embedding matrix has shape (max_len, d_model) — one learnable vector
    per position up to max_len.
    """

    def test_output_shape(self) -> None:
        """Verifies that requesting a sequence length returns the right shape.

        Input:  seq_len=10
        Output: (1, 10, d_model=64)

        Only the first 10 rows of the embedding matrix are returned.
        """
        pe = LearnedPositionalEmbedding(max_len=100, d_model=64)
        out = pe.forward(seq_len=10)
        assert out.shape == (1, 10, 64)

    def test_exceeds_max_len(self) -> None:
        """Ensures the embedding rejects sequences longer than max_len.

        If seq_len > max_len, the embedding matrix doesn't have enough rows
        to assign a vector to every position.  The method must raise a
        ValueError rather than silently producing incorrect results (e.g.,
        wrapping around or indexing out of bounds).

        This is why models specify a max_len (context window) — they can
        only handle sequences up to that length.
        """
        pe = LearnedPositionalEmbedding(max_len=10, d_model=32)
        with pytest.raises(ValueError, match="exceeds"):
            pe.forward(seq_len=11)


# ======================================================================
# Text learning tests
# ======================================================================


class TestEmbeddingTextLearning:
    """Tests that embeddings learn meaningful representations from text.

    These tests verify that:
        - Token embeddings produce consistent vectors for the same token
        - The backward pass correctly accumulates gradients for seen tokens
        - Gradients for unseen tokens remain zero
        - Positional embeddings receive gradients during training
    """

    def test_same_token_same_vector(self) -> None:
        """The same token ID should always map to the same embedding vector.

        This is the fundamental property of an embedding lookup: it's a
        deterministic function from token ID to vector.  If token 5 appears
        at positions 0, 2, and 4, all three should produce the same vector.
        """
        np.random.seed(42)
        emb = TokenEmbedding(vocab_size=100, d_model=32)
        ids = np.array([[5, 10, 5, 20, 5]], dtype=np.int64)
        out = emb.forward(ids)

        # Token 5 appears at positions 0, 2, 4 — all should have the same vector
        assert np.allclose(out[0, 0, :], out[0, 2, :])
        assert np.allclose(out[0, 0, :], out[0, 4, :])
        # Token 10 at position 1 should be different from token 5
        assert not np.allclose(out[0, 0, :], out[0, 1, :])

    def test_embedding_gradient_for_seen_tokens(self) -> None:
        """Backward pass should produce non-zero gradients for tokens in the input.

        When the embedding output is used and gradients flow back, only the
        tokens that appeared in the input should receive gradient updates.
        Tokens not in the input should have zero gradient.
        """
        np.random.seed(42)
        emb = TokenEmbedding(vocab_size=100, d_model=16)
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        emb.forward(ids)

        grad = np.random.randn(1, 3, 16)
        emb.backward(grad)

        # Tokens 1, 2, 3 were in the input — should have non-zero gradient
        assert np.any(emb.grad_weight[1, :] != 0), "Token 1 has zero gradient"
        assert np.any(emb.grad_weight[2, :] != 0), "Token 2 has zero gradient"
        assert np.any(emb.grad_weight[3, :] != 0), "Token 3 has zero gradient"

    def test_embedding_gradient_for_unseen_tokens_is_zero(self) -> None:
        """Backward pass should produce zero gradients for tokens NOT in the input.

        Only tokens that appeared in the forward pass should receive gradient.
        This ensures that the embedding for unseen tokens doesn't change
        during training, which is important for stability.
        """
        np.random.seed(42)
        emb = TokenEmbedding(vocab_size=100, d_model=16)
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        emb.forward(ids)

        grad = np.random.randn(1, 3, 16)
        emb.backward(grad)

        # Tokens 4, 5, 99 were NOT in the input — should have zero gradient
        assert np.all(emb.grad_weight[4, :] == 0), "Token 4 has non-zero gradient"
        assert np.all(emb.grad_weight[5, :] == 0), "Token 5 has non-zero gradient"
        assert np.all(emb.grad_weight[99, :] == 0), "Token 99 has non-zero gradient"

    def test_positional_embedding_receives_gradient(self) -> None:
        """Learned positional embeddings should receive gradients during backward.

        When the model is trained, the positional embeddings should be
        updated just like the token embeddings.  This test verifies that
        the backward pass produces non-zero gradients for the position
        embeddings.
        """
        np.random.seed(42)
        pe = LearnedPositionalEmbedding(max_len=10, d_model=16)
        pe.forward(seq_len=5)

        # Simulate gradient flowing back (batch=2 for generality)
        grad = np.random.randn(2, 5, 16)
        pe.backward(grad)

        # Positions 0-4 were used — should have non-zero gradient
        for pos in range(5):
            assert np.any(pe.grad_weight[pos, :] != 0), f"Position {pos} has zero gradient"

        # Positions 5-9 were NOT used — should have zero gradient
        for pos in range(5, 10):
            assert np.all(pe.grad_weight[pos, :] == 0), (
                f"Position {pos} has non-zero gradient (should be zero)"
            )
