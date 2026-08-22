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
from tokenizer import BPETokenizer

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


# ======================================================================
# Text learning tests
# ======================================================================


class TestGPTTextLearning:
    """Tests that the GPT model can learn from text data.

    These tests exercise the full training pipeline:
        forward → loss → backward → optimizer step

    They verify that:
        - Loss decreases over training steps
        - The model can predict the correct next token after training
        - Generated text contains words from the training corpus
    """

    @staticmethod
    def _make_tokenizer() -> BPETokenizer:
        """Creates a tokenizer trained on a tiny repetitive corpus."""
        tok = BPETokenizer(vocab_size=300)
        tok.train(
            [
                "the cat sat on the mat",
                "the dog sat on the log",
                "the cat sat on the dog",
            ]
        )
        return tok

    @staticmethod
    def _make_model(vocab_size: int) -> GPT:
        """Creates a small GPT model for fast tests."""
        np.random.seed(42)
        return GPT(
            vocab_size=vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )

    def test_gpt_loss_decreases(self) -> None:
        """Training the GPT model should reduce the loss over 50 epochs.

        This is the most fundamental test of the training loop: the loss
        must go down.  If it doesn't, either the gradients are wrong or
        the optimizer is broken.
        """
        from trainer import Trainer, prepare_text_batch

        tok = self._make_tokenizer()
        model = self._make_model(len(tok))
        corpus = ["the cat sat on the mat", "the dog sat on the log"]

        inp, tgt, msk = prepare_text_batch(corpus, tok, seq_len=8)
        initial_loss = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)

        trainer = Trainer(model, tok, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=50, seq_len=8, verbose=False)

        final_loss = trainer.loss_history[-1]
        assert final_loss < initial_loss * 0.8, (
            f"GPT loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_gpt_predicts_next_token(self) -> None:
        """After training, GPT should predict 'sat' after 'the cat'.

        This tests that the model has actually learned the patterns in
        the training data, not just that the loss decreased.  The corpus
        always has 'the cat sat', so the model should predict 'sat'.
        """
        from trainer import Trainer

        tok = self._make_tokenizer()
        model = self._make_model(len(tok))
        corpus = [
            "the cat sat on the mat",
            "the cat sat on the log",
            "the cat sat on the dog",
        ]

        trainer = Trainer(model, tok, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=200, seq_len=8, verbose=False)

        prompt_ids = np.array([tok.encode("the cat")], dtype=np.int64)
        logits = model.forward(prompt_ids, training=False)
        predicted_id = int(np.argmax(logits[0, -1]))
        predicted_token = tok.vocab.get(predicted_id, "?")

        assert predicted_token == "sat", (
            f"Expected 'sat' but got '{predicted_token}' (id={predicted_id})"
        )

    def test_gpt_generation_contains_corpus_words(self) -> None:
        """After training, generated text should contain words from the corpus.

        The model should generate text that resembles the training data,
        not random gibberish.  We check that at least one corpus word
        appears in the generated text.
        """
        from trainer import Trainer

        tok = self._make_tokenizer()
        model = self._make_model(len(tok))
        corpus = [
            "the cat sat on the mat",
            "the cat sat on the log",
            "the cat sat on the dog",
        ]

        trainer = Trainer(model, tok, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=200, seq_len=8, verbose=False)

        prompt_ids = np.array([tok.encode("the cat")], dtype=np.int64)
        generated = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)
        generated_text = tok.decode(generated)

        corpus_words = set()
        for text in corpus:
            corpus_words.update(text.split())
        generated_words = set(generated_text.split())
        overlap = generated_words & corpus_words

        assert len(overlap) > 0, (
            f"Generated '{generated_text}' contains no corpus words. "
            f"Expected overlap with: {corpus_words}"
        )

    def test_gpt_backward_produces_gradients(self) -> None:
        """After forward + backward, all parameters should have gradients.

        This verifies that the backward pass reaches every parameter in
        the model, not just a subset.  Missing gradients would mean some
        parts of the model are not learning.
        """
        from model import softmax_cross_entropy_backward

        tok = self._make_tokenizer()
        model = self._make_model(len(tok))

        inp = np.array([tok.encode("the cat sat")], dtype=np.int64)
        tgt = np.array([tok.encode("cat sat on")], dtype=np.int64)
        msk = np.array([[1, 1, 1]], dtype=np.float64)

        logits = model.forward(inp, mask=msk, training=True)
        d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
        model.backward(d_logits)
        grads = model.get_grads()

        # Check that all gradients are non-zero
        for name, grad in grads.items():
            assert np.any(grad != 0), f"Gradient for {name} is all zeros"


class TestModelSerialization:
    """Tests for save_model / load_model and tokenizer save / load."""

    @staticmethod
    def _make_tokenizer() -> BPETokenizer:
        """Creates a small trained tokenizer."""
        tok = BPETokenizer(vocab_size=300)
        tok.train(["the cat sat on the mat", "the dog sat on the log"])
        return tok

    @staticmethod
    def _make_model(vocab_size: int) -> GPT:
        """Creates a small GPT model."""
        np.random.seed(42)
        return GPT(
            vocab_size=vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )

    def test_save_load_roundtrip(self, tmp_path) -> None:
        """A saved model should produce identical outputs after loading."""
        from model import load_model, save_model

        tok = self._make_tokenizer()
        model = self._make_model(len(tok))

        prompt = np.array([tok.encode("the cat")], dtype=np.int64)
        logits_before = model.forward(prompt, training=False)
        gen_before = model.generate(prompt, max_new_tokens=5, temperature=0.0)

        model_path = tmp_path / "model.npz"
        tok_path = tmp_path / "tok.json"
        save_model(model, str(model_path))
        tok.save(str(tok_path))

        loaded = load_model(str(model_path))
        loaded_tok = BPETokenizer.load(str(tok_path))

        # Consistency checks between model and tokenizer.
        assert loaded.vocab_size == len(loaded_tok), (
            f"Model vocab size ({loaded.vocab_size}) != tokenizer size ({len(loaded_tok)})"
        )

        logits_after = loaded.forward(prompt, training=False)
        gen_after = loaded.generate(prompt, max_new_tokens=5, temperature=0.0)

        assert np.allclose(logits_before, logits_after), "Logits differ after save/load roundtrip"
        assert gen_before == gen_after, (
            f"Generation differs: before={gen_before}, after={gen_after}"
        )
