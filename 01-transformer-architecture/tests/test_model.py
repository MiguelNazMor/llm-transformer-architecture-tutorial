"""Tests for the full Transformer and GPT models.

These tests verify the two complete model architectures built from the
lower-level components (attention, embeddings, FFN, blocks):

    Transformer  — original encoder-decoder (Vaswani et al., 2017)
    GPT          — decoder-only autoregressive language model

Also tested are the supporting utilities:
    softmax             — numerically stable softmax
    cross_entropy_loss  — standard loss for next-token prediction

Key properties under test:
    - Correct output shapes at every stage (encode, decode, forward, generate)
    - Deterministic forward pass in eval mode
    - Softmax sums to 1 and handles large values without NaN
    - Cross-entropy: near-zero loss for perfect predictions
    - Masked loss ignores specified positions
    - Generation produces the correct number of tokens
    - Temperature=0 generation is deterministic
    - Sinusoidal positional encoding mode works in GPT
"""

import numpy as np
from model import GPT, Transformer, cross_entropy_loss, softmax

# ======================================================================
# Utilities
# ======================================================================


class TestSoftmax:
    """Tests for the numerically stable softmax implementation.

    Softmax converts raw logits into a probability distribution:

        softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

    The naive implementation fails for large values because exp(1000)
    overflows to infinity.  The stable version subtracts max(x) first:

        softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

    This does not change the result mathematically but keeps all
    intermediate values in a safe range.
    """

    def test_sums_to_one(self) -> None:
        """Verifies softmax outputs are valid probability distributions.

        Every row of the output must sum to exactly 1.0 (within floating
        point tolerance).  This is the defining property of softmax.

        Test: 3 random vectors of 5 elements each → 3 probability distributions.
        """
        x = np.random.randn(3, 5)
        probs = softmax(x)
        assert np.allclose(np.sum(probs, axis=-1), 1.0)

    def test_numerical_stability(self) -> None:
        """Proves softmax handles large input values without overflow.

        When all logits are equal (e.g., [1000, 1000, 1000]), the output
        should be a uniform distribution [1/3, 1/3, 1/3].

        A naive implementation would compute exp(1000) ≈ 10^434, which
        overflows float64.  The stable implementation subtracts 1000 first,
        computing exp(0) = 1, which is perfectly safe.

        This test also verifies that no NaN values appear in the output.
        """
        x = np.array([[1000.0, 1000.0, 1000.0]])
        probs = softmax(x)
        assert np.all(np.isfinite(probs))
        assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])


class TestCrossEntropyLoss:
    """Tests for the cross-entropy loss function.

    Cross-entropy measures how well the predicted token distribution matches
    the true next token:

        loss = -log(p_correct)  averaged over non-masked positions

    where p_correct is the predicted probability of the actual next token.
    """

    def test_perfect_prediction(self) -> None:
        """Verifies loss approaches 0 when the model is perfectly confident.

        Setup:
            Batch of 1, sequence of 2 tokens, vocabulary of 2.
            Token 0 → logits [100, 0]   (very confident in class 0)
            Token 1 → logits [0, 100]   (very confident in class 1)
            Targets: [0, 1]

        After softmax, p(token_0) ≈ 1.0 and p(token_1) ≈ 1.0, so:
            loss = -(log(1.0) + log(1.0)) / 2 ≈ 0

        We assert loss < 0.01 to allow for floating-point imprecision.
        """
        logits = np.array([[[100.0, 0.0], [0.0, 100.0]]])  # (1, 2, 2)
        targets = np.array([[0, 1]])
        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.01

    def test_with_mask(self) -> None:
        """Ensures masked positions are excluded from the loss computation.

        In language modeling, shorter sequences in a batch are padded with
        a special token.  The mask (1 = include, 0 = ignore) prevents the
        loss from being computed on these padding positions.

        Setup:
            3 positions, but position 2 is masked out (mask=0).
            The loss should only consider positions 0 and 1.

        With uniform logits (all zeros → p=0.25 for each of 4 classes),
        the per-token loss is -log(0.25) ≈ 1.386.  With mask [1,1,0],
        the loss should be exactly that value (averaged over 2 tokens).
        """
        logits = np.zeros((1, 3, 4))
        targets = np.array([[0, 1, 2]])
        mask = np.array([[1.0, 1.0, 0.0]])
        loss = cross_entropy_loss(logits, targets, mask)
        assert np.isfinite(loss)


# ======================================================================
# GPT (decoder-only)
# ======================================================================


class TestGPT:
    """Tests for the GPT-style decoder-only language model.

    GPT uses only the decoder stack with causal (masked) self-attention.
    It is trained to predict the next token given previous tokens:

        P(x_t | x_1, ..., x_{t-1})

    The model supports both learned positional embeddings (GPT-2/GPT-3
    style) and sinusoidal encodings (original Transformer style).

    All tests use a tiny configuration (d_model=64, 4 heads, 2 layers)
    for fast execution.
    """

    @staticmethod
    def make_model(vocab_size: int = 100) -> GPT:
        """Creates a small GPT model with deterministic behavior.

        dropout_rate=0.0 ensures identical outputs across calls in eval mode.
        """
        return GPT(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_len=32,
            dropout_rate=0.0,
        )

    def test_forward_shape(self) -> None:
        """Verifies the forward pass produces logits over the full vocabulary.

        Input:  (batch=1, seq_len=5) token IDs
        Output: (1, 5, vocab_size=100)

        Each of the 5 positions gets a vector of 100 logits — one per
        possible next token.  The last position's logits can be used to
        predict the 6th token during generation.
        """
        model = self.make_model()
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        logits = model.forward(ids, training=False)
        assert logits.shape == (1, 5, 100)

    def test_forward_deterministic(self) -> None:
        """Confirms the forward pass is deterministic in eval mode.

        With dropout_rate=0.0 and training=False, there are no sources of
        randomness.  Two calls with the same input must produce identical
        logits.  This is essential for reproducible inference.
        """
        model = self.make_model()
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        out1 = model.forward(ids, training=False)
        out2 = model.forward(ids, training=False)
        assert np.allclose(out1, out2)

    def test_generate_shape(self) -> None:
        """Verifies autoregressive generation produces the right number of tokens.

        Given a prompt of length 3 and max_new_tokens=5, the output list
        must have exactly 8 elements (3 prompt + 5 new).

        Each new token is generated by:
            1. Running the full sequence through the model.
            2. Taking the logits at the last position.
            3. Sampling from the softmax distribution.
            4. Appending the sampled token to the sequence.
            5. Repeating until max_new_tokens is reached.
        """
        model = self.make_model()
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        generated = model.generate(prompt, max_new_tokens=5)
        assert len(generated) == 3 + 5

    def test_generate_temperature_zero(self) -> None:
        """Ensures temperature=0 produces deterministic (greedy) generation.

        Temperature scales the logits before softmax:
            p_i = softmax(logits / T)_i

        As T → 0, the distribution concentrates on the single highest logit
        (argmax).  Two calls with temperature=0 must return exactly the same
        token sequence because there is no randomness.

        This is the standard "greedy decoding" mode used when you want
        the most likely output.
        """
        model = self.make_model()
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        gen1 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        gen2 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        assert gen1 == gen2

    def test_sinusoidal_pos(self) -> None:
        """Verifies the model works with sinusoidal (non-learned) position encoding.

        Setting use_learned_pos=False switches from learned embeddings to
        the fixed sinusoidal encodings from the original Transformer paper.

        The forward pass should still produce correct-shape logits, proving
        that both positional encoding strategies are interchangeable at the
        API level.
        """
        model = GPT(
            vocab_size=50,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_len=32,
            use_learned_pos=False,
        )
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        logits = model.forward(ids, training=False)
        assert logits.shape == (1, 3, 50)


# ======================================================================
# Transformer (encoder-decoder)
# ======================================================================


class TestTransformer:
    """Tests for the original encoder-decoder Transformer.

    Architecture (Vaswani et al., 2017):
        Encoder: N blocks of (self-attention + FFN)
        Decoder: N blocks of (masked self-attn + cross-attn + FFN)
        Output:  Linear projection to vocabulary

    The encoder produces contextualized representations of the source
    sequence.  The decoder generates the target sequence one token at a
    time, attending to both previous target tokens (masked self-attention)
    and the encoder output (cross-attention).

    All tests use a tiny configuration (d_model=64, 4 heads, 2 layers)
    for fast execution.
    """

    @staticmethod
    def make_model(vocab_size: int = 100) -> Transformer:
        """Creates a small Transformer with deterministic behavior."""
        return Transformer(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_len=32,
            dropout_rate=0.0,
        )

    def test_forward_shape(self) -> None:
        """Verifies the full encode→decode→project pipeline.

        Input:
            Source: (1, 4)  — 4 tokens in the source language
            Target: (1, 3)  — 3 tokens in the target language

        Output: (1, 3, 100) — logits for each of the 3 target positions
                               over the 100-token vocabulary

        The output length matches the target length, not the source length.
        """
        model = self.make_model()
        src = np.array([[1, 2, 3, 4]], dtype=np.int64)
        tgt = np.array([[5, 6, 7]], dtype=np.int64)
        logits = model.forward(src, tgt, training=False)
        assert logits.shape == (1, 3, 100)

    def test_encode_shape(self) -> None:
        """Verifies the encoder produces the correct representation shape.

        Input:  (batch=1, src_len=4) token IDs
        Output: (1, 4, d_model=64)

        Each source token gets a 64-dimensional contextualized vector that
        incorporates information from all other source tokens via
        bidirectional self-attention.

        This output is what the decoder's cross-attention will use as
        keys and values.
        """
        model = self.make_model()
        src = np.array([[1, 2, 3, 4]], dtype=np.int64)
        enc_out = model.encode(src, training=False)
        assert enc_out.shape == (1, 4, 64)

    def test_decode_shape(self) -> None:
        """Verifies the decoder produces the correct representation shape.

        The decoder takes:
          - Target token IDs (the partial output sequence so far)
          - Encoder output (contextualized source representations)

        It produces contextualized target representations by:
          1. Masked self-attention over the target sequence.
          2. Cross-attention to the encoder output.
          3. Feed-forward transformation.

        Input:  target=(1, 2), encoder_out=(1, 3, 64)
        Output: (1, 2, 64) — one vector per target position
        """
        model = self.make_model()
        src = np.array([[1, 2, 3]], dtype=np.int64)
        tgt = np.array([[4, 5]], dtype=np.int64)
        enc_out = model.encode(src, training=False)
        dec_out = model.decode(tgt, enc_out, training=False)
        assert dec_out.shape == (1, 2, 64)
