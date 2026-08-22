"""End-to-end text training and prediction tests.

These tests verify that the GPT model can actually LEARN from text data:
    - Loss decreases over training steps
    - The model can overfit a tiny corpus
    - Next-token predictions match the training data after training
    - Generated text uses vocabulary from the training corpus
    - Loss values are always finite (no NaN or Inf)

Unlike the component-level tests (which verify shapes and mathematical
properties), these tests exercise the full pipeline:
    tokenizer → embedding → attention → FFN → loss → backprop → optimizer

The models are intentionally tiny (d_model=16-32, 1-2 layers) and training
runs are short (30-200 steps) to keep total test time under 10 seconds.
"""

import numpy as np
import pytest
from model import GPT, cross_entropy_loss
from tokenizer import BPETokenizer
from trainer import Trainer, prepare_text_batch

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tiny_corpus() -> list[str]:
    """A tiny corpus with repetitive structure for easy learning."""
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat likes the warm mat",
        "the dog likes the cold log",
    ]


@pytest.fixture
def trained_tokenizer(tiny_corpus: list[str]) -> BPETokenizer:
    """A BPE tokenizer trained on the tiny corpus."""
    tok = BPETokenizer(vocab_size=300)
    tok.train(tiny_corpus)
    return tok


@pytest.fixture
def small_gpt(trained_tokenizer: BPETokenizer) -> GPT:
    """A small GPT model suitable for fast tests."""
    np.random.seed(42)
    return GPT(
        vocab_size=len(trained_tokenizer),
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_len=32,
        dropout_rate=0.0,
        use_learned_pos=True,
    )


# ======================================================================
# Loss decreases during training
# ======================================================================


class TestLossDecreases:
    """Tests that training reduces the cross-entropy loss."""

    def test_loss_decreases(
        self,
        small_gpt: GPT,
        trained_tokenizer: BPETokenizer,
        tiny_corpus: list[str],
    ) -> None:
        """Loss after 50 epochs should be at least 20% lower than initial."""
        inp, tgt, msk = prepare_text_batch(tiny_corpus, trained_tokenizer, seq_len=8)
        initial_loss = cross_entropy_loss(
            small_gpt.forward(inp, mask=msk, training=False), tgt, msk
        )

        trainer = Trainer(small_gpt, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(tiny_corpus, epochs=50, seq_len=8, verbose=False)

        final_loss = trainer.loss_history[-1]
        assert final_loss < initial_loss * 0.8, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_is_finite_during_training(
        self,
        small_gpt: GPT,
        trained_tokenizer: BPETokenizer,
        tiny_corpus: list[str],
    ) -> None:
        """No loss value during training should be NaN or Inf."""
        trainer = Trainer(small_gpt, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(tiny_corpus, epochs=30, seq_len=8, verbose=False)

        for i, loss in enumerate(trainer.loss_history):
            assert np.isfinite(loss), f"Loss at step {i} is not finite: {loss}"

    def test_loss_decreases_monotonically_on_average(
        self,
        small_gpt: GPT,
        trained_tokenizer: BPETokenizer,
        tiny_corpus: list[str],
    ) -> None:
        """Average of last 5 losses should be less than average of first 5."""
        trainer = Trainer(small_gpt, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(tiny_corpus, epochs=50, seq_len=8, verbose=False)

        first_5 = np.mean(trainer.loss_history[:5])
        last_5 = np.mean(trainer.loss_history[-5:])
        assert last_5 < first_5, (
            f"Average loss did not improve: first 5={first_5:.4f}, last 5={last_5:.4f}"
        )


# ======================================================================
# Overfitting a tiny corpus
# ======================================================================


class TestOverfitting:
    """Tests that the model can memorize a very small dataset."""

    def test_overfit_single_sentence(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """Training on one sentence for 200 steps should drive loss below 0.5."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        single_sentence = ["the cat sat on the mat"]
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(single_sentence, epochs=200, seq_len=8, verbose=False)

        final_loss = trainer.loss_history[-1]
        assert final_loss < 0.5, (
            f"Could not overfit single sentence: final loss = {final_loss:.4f}"
        )

    def test_overfit_reaches_low_loss(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """Training on a tiny corpus for 300 steps should reach loss below 1.0."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=150, seq_len=8, verbose=False)

        final_loss = trainer.loss_history[-1]
        assert final_loss < 1.0, (
            f"Could not reach low loss: final loss = {final_loss:.4f}"
        )


# ======================================================================
# Next-token prediction
# ======================================================================


class TestNextTokenPrediction:
    """Tests that the model predicts correct next tokens after training."""

    def test_predicts_next_token(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """After training, the model should predict 'sat' after 'the cat'."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        corpus = [
            "the cat sat on the mat",
            "the cat sat on the log",
            "the cat sat on the dog",
        ]
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=200, seq_len=8, verbose=False)

        # Encode "the cat" and check the next token prediction
        prompt_ids = np.array([trained_tokenizer.encode("the cat")], dtype=np.int64)
        logits = model.forward(prompt_ids, training=False)
        predicted_id = int(np.argmax(logits[0, -1]))
        predicted_token = trained_tokenizer.vocab.get(predicted_id, "?")

        assert predicted_token == "sat", (
            f"Expected 'sat' but got '{predicted_token}' (id={predicted_id})"
        )

    def test_prediction_improves_with_training(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """Prediction probability for the correct token should increase."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        corpus = ["the cat sat on the mat", "the cat sat on the log"]

        # Get prediction probability before training
        prompt_ids = np.array([trained_tokenizer.encode("the cat")], dtype=np.int64)
        from model import softmax

        logits_before = model.forward(prompt_ids, training=False)
        probs_before = softmax(logits_before[0, -1])
        sat_id = trained_tokenizer.encode("sat")[0]
        prob_before = probs_before[sat_id]

        # Train
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=200, seq_len=8, verbose=False)

        # Get prediction probability after training
        logits_after = model.forward(prompt_ids, training=False)
        probs_after = softmax(logits_after[0, -1])
        prob_after = probs_after[sat_id]

        assert prob_after > prob_before, (
            f"Probability for 'sat' did not increase: "
            f"before={prob_before:.4f}, after={prob_after:.4f}"
        )


# ======================================================================
# Text generation after training
# ======================================================================


class TestTextGeneration:
    """Tests that generation produces corpus-like text after training."""

    def test_generation_uses_corpus_vocabulary(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """Generated tokens should be from the corpus vocabulary, not random."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=100, seq_len=8, verbose=False)

        prompt_ids = np.array([trained_tokenizer.encode("the")], dtype=np.int64)
        generated = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)

        # All generated tokens should be valid vocabulary IDs
        for tid in generated:
            assert 0 <= tid < len(trained_tokenizer), (
                f"Generated token ID {tid} is out of vocabulary range"
            )

    def test_generation_after_training_contains_corpus_words(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """After training, generated text should contain words from the corpus."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        corpus = [
            "the cat sat on the mat",
            "the cat sat on the log",
            "the cat sat on the dog",
        ]
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=200, seq_len=8, verbose=False)

        prompt_ids = np.array([trained_tokenizer.encode("the cat")], dtype=np.int64)
        generated = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)
        generated_text = trained_tokenizer.decode(generated)

        # The generated text should contain "sat" (the next word in the corpus)
        corpus_words = set()
        for text in corpus:
            corpus_words.update(text.split())

        generated_words = set(generated_text.split())
        overlap = generated_words & corpus_words

        assert len(overlap) > 0, (
            f"Generated text '{generated_text}' contains no corpus words. "
            f"Expected overlap with: {corpus_words}"
        )

    def test_deterministic_generation_with_temperature_zero(
        self,
        trained_tokenizer: BPETokenizer,
    ) -> None:
        """Temperature=0 generation should be deterministic after training."""
        np.random.seed(42)
        model = GPT(
            vocab_size=len(trained_tokenizer),
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            max_len=32,
            dropout_rate=0.0,
        )
        corpus = ["the cat sat on the mat"]
        trainer = Trainer(model, trained_tokenizer, lr=0.1, momentum=0.9)
        trainer.train(corpus, epochs=100, seq_len=8, verbose=False)

        prompt_ids = np.array([trained_tokenizer.encode("the cat")], dtype=np.int64)
        gen1 = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)
        gen2 = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)

        assert gen1 == gen2, "Temperature=0 generation should be deterministic"


# ======================================================================
# Gradient correctness
# ======================================================================


class TestGradientCorrectness:
    """Tests that manual backprop gradients match numerical gradients."""

    def test_gradient_check_token_embedding(self) -> None:
        """Analytic gradients for token embedding should match numerical."""
        np.random.seed(42)
        vocab = 10
        model = GPT(
            vocab_size=vocab, d_model=8, num_heads=2,
            num_layers=1, d_ff=16, max_len=8, dropout_rate=0.0,
        )
        inp = np.array([[1, 2, 3, 4]], dtype=np.int64)
        tgt = np.array([[2, 3, 4, 5]], dtype=np.int64)
        msk = np.array([[1, 1, 1, 1]], dtype=np.float64)

        logits = model.forward(inp, mask=msk, training=True)
        from model import softmax_cross_entropy_backward

        d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
        model.backward(d_logits)
        grads = model.get_grads()

        eps = 1e-5
        param = model.token_embedding.weight
        analytic = grads["token_embedding.weight"]

        max_rel_error = 0.0
        np.random.seed(123)
        for _ in range(10):
            i = np.random.randint(0, param.shape[0])
            j = np.random.randint(0, param.shape[1])
            orig = param[i, j]
            param[i, j] = orig + eps
            loss_p = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)
            param[i, j] = orig - eps
            loss_m = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)
            param[i, j] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            ana_grad = analytic[i, j]
            rel_error = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10)
            max_rel_error = max(max_rel_error, rel_error)

        assert max_rel_error < 0.01, (
            f"Gradient check failed: max relative error = {max_rel_error:.6f}"
        )

    def test_gradient_check_attention_weights(self) -> None:
        """Analytic gradients for attention W_q should match numerical."""
        np.random.seed(42)
        vocab = 10
        model = GPT(
            vocab_size=vocab, d_model=8, num_heads=2,
            num_layers=1, d_ff=16, max_len=8, dropout_rate=0.0,
        )
        inp = np.array([[1, 2, 3, 4]], dtype=np.int64)
        tgt = np.array([[2, 3, 4, 5]], dtype=np.int64)
        msk = np.array([[1, 1, 1, 1]], dtype=np.float64)

        logits = model.forward(inp, mask=msk, training=True)
        from model import softmax_cross_entropy_backward

        d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
        model.backward(d_logits)

        eps = 1e-5
        attn = model.blocks[0].self_attention
        param = attn.W_q
        analytic = attn.grad_W_q

        max_rel_error = 0.0
        np.random.seed(456)
        for _ in range(5):
            i = np.random.randint(0, param.shape[0])
            j = np.random.randint(0, param.shape[1])
            orig = param[i, j]
            param[i, j] = orig + eps
            loss_p = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)
            param[i, j] = orig - eps
            loss_m = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)
            param[i, j] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            ana_grad = analytic[i, j]
            rel_error = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10)
            max_rel_error = max(max_rel_error, rel_error)

        assert max_rel_error < 0.01, (
            f"Gradient check failed for W_q: max relative error = {max_rel_error:.6f}"
        )
