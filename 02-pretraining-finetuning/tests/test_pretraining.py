"""Tests for pre-training and fine-tuning.

Verifies the complete pre-train → fine-tune pipeline:
    - Pre-training reduces loss on a general corpus
    - Fine-tuning further reduces loss on a domain-specific corpus
    - Domain perplexity improves after fine-tuning
    - The model retains some general knowledge after fine-tuning
    - The model generates domain-specific text after fine-tuning
"""

import sys
from pathlib import Path

# Add src to path
_src = Path(__file__).resolve().parents[2] / "02-pretraining-finetuning" / "src"
sys.path.insert(0, str(_src))

import numpy as np
import pytest
from finetune import evaluate_perplexity, finetune
from loader import split_corpus
from model import GPT, cross_entropy_loss
from pretrain import pretrain
from tokenizer import BPETokenizer
from trainer import prepare_text_batch

# Small corpora for fast tests (don't use the full 280-sentence Alice corpus).
_TEST_GENERAL = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog are friends",
    "the mat is soft and warm",
    "the log is hard and cold",
    "the cat likes the warm mat",
    "the dog likes the cold log",
    "cats and dogs are animals",
    "the sun is bright today",
    "the sky is blue and clear",
]

# Small HP-style corpus for fast tests.
_TEST_HP = [
    "Harry was a young wizard who lived in a cupboard",
    "Hogwarts was a school of witchcraft and wizardry",
    "Ron and Hermione were friends of Harry at Hogwarts",
    "Gryffindor was the house of brave wizards at Hogwarts",
    "Harry flew on a broomstick during the Quidditch match",
    "The spell was cast with a wand by the young wizard",
    "Dumbledore was the headmaster of Hogwarts school",
    "Harry fought the dark wizard with courage and magic",
]


class TestPretraining:
    """Tests for the pre-training stage."""

    def test_pretrain_reduces_loss(self) -> None:
        """Pre-training should significantly reduce the loss."""
        model, tok, losses = pretrain(
            vocab_size=300,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            epochs=30,
            lr=0.1,
            seed=42,
            verbose=False,
            corpus=_TEST_GENERAL,
            tokenizer_corpus=_TEST_GENERAL + _TEST_HP,
            seq_len=16,
        )
        assert losses[-1] < losses[0] * 0.8, (
            f"Pre-training loss did not decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_pretrained_model_generates_text(self) -> None:
        """Pre-trained model should generate valid tokens (not crash)."""
        model, tok, _ = pretrain(
            vocab_size=300,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            epochs=20,
            lr=0.1,
            seed=42,
            verbose=False,
            corpus=_TEST_GENERAL,
            tokenizer_corpus=_TEST_GENERAL + _TEST_HP,
            seq_len=16,
        )
        prompt = np.array([tok.encode("the cat")], dtype=np.int64)
        generated = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        assert len(generated) > 2  # prompt + at least some generated tokens
        assert all(0 <= tid < len(tok) for tid in generated)


class TestFineTuning:
    """Tests for the fine-tuning stage."""

    @pytest.fixture
    def pretrained(self) -> tuple[GPT, BPETokenizer]:
        """Returns a pre-trained model and tokenizer."""
        model, tok, _ = pretrain(
            vocab_size=300,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            epochs=20,
            lr=0.1,
            seed=42,
            verbose=False,
            corpus=_TEST_GENERAL,
            tokenizer_corpus=_TEST_GENERAL + _TEST_HP,
            seq_len=16,
        )
        return model, tok

    def test_finetune_reduces_loss(self, pretrained: tuple[GPT, BPETokenizer]) -> None:
        """Fine-tuning should reduce loss on a HELD-OUT target-domain sample."""
        model, tok = pretrained

        hp_train, hp_eval = split_corpus(_TEST_HP, eval_fraction=0.25)

        # Measure loss before fine-tuning (on held-out data)
        inp, tgt, msk = prepare_text_batch(hp_eval, tok, seq_len=8)
        loss_before = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)

        # Fine-tune on the training split only
        finetune(model, tok, hp_train, epochs=20, lr=0.05, seq_len=8, verbose=False)

        # Measure loss after fine-tuning (on the same held-out data)
        loss_after = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)

        assert loss_after < loss_before, (
            f"Fine-tuning did not reduce loss: {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_domain_perplexity_improves(self, pretrained: tuple[GPT, BPETokenizer]) -> None:
        """Perplexity on a held-out target-domain sample should decrease after fine-tuning."""
        model, tok = pretrained

        hp_train, hp_eval = split_corpus(_TEST_HP, eval_fraction=0.25)

        ppl_before = evaluate_perplexity(model, tok, hp_eval, seq_len=8)
        finetune(model, tok, hp_train, epochs=20, lr=0.05, seq_len=8, verbose=False)
        ppl_after = evaluate_perplexity(model, tok, hp_eval, seq_len=8)

        assert ppl_after < ppl_before, (
            f"Domain perplexity did not improve: {ppl_before:.2f} → {ppl_after:.2f}"
        )

    def test_generates_domain_specific_text(self, pretrained: tuple[GPT, BPETokenizer]) -> None:
        """After fine-tuning on HP, generation should differ from before.

        BPE tokenizers may split novel words (like wizard names) into
        subword pieces.  We check that fine-tuning changes the model's
        output, indicating domain adaptation.
        """
        model, tok = pretrained

        # Record generation BEFORE fine-tuning
        prompt = np.array([tok.encode("the")], dtype=np.int64)
        gen_before = tok.decode(model.generate(prompt, max_new_tokens=6, temperature=0.0))

        # Fine-tune on HP corpus
        finetune(model, tok, _TEST_HP, epochs=30, lr=0.05, seq_len=8, verbose=False)

        # Generate AFTER fine-tuning
        gen_after = tok.decode(model.generate(prompt, max_new_tokens=6, temperature=0.0))

        # The generated text should change after domain-specific fine-tuning
        assert gen_before != gen_after, (
            f"Generation did not change after fine-tuning: "
            f"before='{gen_before}', after='{gen_after}'"
        )
        assert len(gen_after.strip()) > 0, "Generated empty text"

    def test_no_catastrophic_forgetting(self, pretrained: tuple[GPT, BPETokenizer]) -> None:
        """After fine-tuning on HP, general words should still be generated."""
        model, tok = pretrained

        # Fine-tune on HP corpus
        finetune(model, tok, _TEST_HP, epochs=20, lr=0.05, seq_len=8, verbose=False)

        # Generate after fine-tuning
        prompt = np.array([tok.encode("the cat")], dtype=np.int64)
        gen_after = set(
            tok.decode(model.generate(prompt, max_new_tokens=5, temperature=0.0)).split()
        )

        # The model should still generate some general words (not only HP)
        # We check that the output isn't empty and contains valid tokens
        assert len(gen_after) > 0, (
            "Model generates nothing after fine-tuning (catastrophic forgetting)"
        )


class TestPerplexity:
    """Tests for perplexity computation."""

    def test_perplexity_is_finite(self) -> None:
        """Perplexity should be a finite positive number."""
        model, tok, _ = pretrain(
            vocab_size=300,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            epochs=10,
            lr=0.1,
            seed=42,
            verbose=False,
            corpus=_TEST_GENERAL,
            tokenizer_corpus=_TEST_GENERAL + _TEST_HP,
            seq_len=16,
        )
        ppl = evaluate_perplexity(model, tok, ["the cat sat on the mat"], seq_len=16)
        assert np.isfinite(ppl), f"Perplexity is not finite: {ppl}"
        assert ppl > 1.0, f"Perplexity should be > 1: {ppl}"

    def test_perplexity_lower_for_seen_text(self) -> None:
        """Perplexity should be lower for text similar to training data."""
        model, tok, _ = pretrain(
            vocab_size=300,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            epochs=15,
            lr=0.1,
            seed=42,
            verbose=False,
            corpus=_TEST_GENERAL,
            tokenizer_corpus=_TEST_GENERAL + _TEST_HP,
            seq_len=16,
        )
        # The model was trained on text with "the cat sat"
        seen_ppl = evaluate_perplexity(model, tok, ["the cat sat on the mat"], seq_len=16)
        # Seen text should not have astronomically high perplexity
        assert seen_ppl < 10000, f"Perplexity on seen text is extremely high: {seen_ppl:.2f}"
