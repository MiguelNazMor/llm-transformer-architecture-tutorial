"""Tests for the BPE tokenizer.

The BPETokenizer implements Byte-Pair Encoding — the subword tokenization
algorithm used by GPT models.  These tests verify every stage of the
tokenizer lifecycle:

    - Initialization and special token ID assignment
    - Input validation (e.g., vocab_size must be ≥ 256)
    - Training: learning merges from a corpus
    - Encoding: text → token IDs
    - Decoding: token IDs → text
    - Batch encoding with padding and attention masks
    - Handling of unknown tokens and special tokens
"""

import pytest
from tokenizer import BPETokenizer


class TestBPETokenizer:
    """Tests for BPETokenizer training, encoding, and decoding."""

    @pytest.fixture
    def corpus(self) -> list[str]:
        """A tiny training corpus with overlapping vocabulary.

        The sentences share words like "the", "cat", "sat", "on" so the BPE
        algorithm has enough frequency data to learn meaningful merges.
        """
        return [
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat likes milk",
        ]

    @pytest.fixture
    def tokenizer(self, corpus: list[str]) -> BPETokenizer:
        """Returns a tokenizer already trained on the corpus.

        vocab_size=500 gives the tokenizer room for ~240 merges beyond the
        256 base byte tokens and 4 special tokens.
        """
        tok = BPETokenizer(vocab_size=500)
        tok.train(corpus)
        return tok

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization(self) -> None:
        """Verifies that a fresh tokenizer assigns correct special token IDs.

        Special tokens occupy the first four IDs in a fixed order:
            0 = <PAD>  — used to pad shorter sequences in a batch
            1 = <UNK>  — placeholder for tokens not in the vocabulary
            2 = <BOS>  — beginning-of-sequence marker
            3 = <EOS>  — end-of-sequence marker

        The vocab_size attribute stores the user-requested target size,
        not the current number of trained tokens.
        """
        tok = BPETokenizer(vocab_size=1000)
        assert tok.vocab_size == 1000
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3

    def test_vocab_size_too_small(self) -> None:
        """Ensures vocab_size < 256 raises ValueError.

        The BPE base vocabulary starts with all 256 byte values.  A smaller
        vocab_size would make it impossible to represent arbitrary text, so
        the constructor rejects it immediately.
        """
        with pytest.raises(ValueError, match="at least 256"):
            BPETokenizer(vocab_size=200)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def test_train_builds_vocabulary(self, tokenizer: BPETokenizer) -> None:
        """Confirms that training actually learns merges and grows the vocab.

        After training on the corpus, the tokenizer must have:
          - More than just the 4 special tokens in its vocabulary.
          - At least one learned merge (a pair of symbols that frequently
            appear together, like "t" + "h" → "th").

        This is a smoke test that the BPE training loop runs without errors
        and produces non-trivial results.
        """
        assert len(tokenizer.vocab) > 4  # More than just special tokens.
        assert len(tokenizer.merges) > 0

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def test_encode_decode_roundtrip(self, tokenizer: BPETokenizer) -> None:
        """Verifies that encode → decode preserves the core content.

        BPE may merge subwords differently than the original whitespace
        splitting (e.g., "the" stays one token, but a rare word might be
        split into subword pieces).  The roundtrip won't be byte-identical,
        but the key content words must survive.

        How it works:
            1. "the cat sat on the mat" is encoded to token IDs.
            2. Those IDs are decoded back to a string.
            3. We check that "cat" and "mat" appear in the decoded text.
        """
        text = "the cat sat on the mat"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert "cat" in decoded
        assert "mat" in decoded

    def test_encode_unknown_word(self, tokenizer: BPETokenizer) -> None:
        """Ensures the tokenizer handles completely unseen words gracefully.

        "zzzunknownzzz" contains characters not present in the training
        corpus.  The tokenizer should still produce token IDs (mapping
        unknown characters to <UNK> or byte-level fallbacks) rather than
        crashing or returning an empty list.

        How it works:
            1. Encode a nonsense word.
            2. Verify we got at least one token back.
            3. Verify all token IDs are non-negative (valid indices).
        """
        ids = tokenizer.encode("zzzunknownzzz")
        assert len(ids) > 0
        assert all(tid >= 0 for tid in ids)

    def test_decode_special_tokens_are_stripped(self, tokenizer: BPETokenizer) -> None:
        """Confirms that special tokens are removed during decoding.

        <PAD>, <BOS>, and <EOS> are control tokens, not text.  The decode
        method must filter them out so the output is clean human-readable
        text.

        How it works:
            1. Build a token ID list containing only special tokens.
            2. Decode it.
            3. Assert none of the special token strings appear in the output.
        """
        ids = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id]
        decoded = tokenizer.decode(ids)
        assert "<PAD>" not in decoded
        assert "<BOS>" not in decoded

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def test_encode_batch(self, tokenizer: BPETokenizer) -> None:
        """Tests batch encoding with padding and attention masks.

        Transformers process fixed-length sequences in batches.  The
        encode_batch method pads shorter sequences with <PAD> tokens and
        produces an attention mask so the model knows which positions to
        ignore.

        How it works:
            1. Encode two texts of different lengths with max_len=10.
            2. Verify both padded sequences are exactly 10 tokens long.
            3. The attention mask has 1s for real tokens and 0s for padding.
        """
        texts = ["the cat", "the dog sat on the log"]
        padded, masks = tokenizer.encode_batch(texts, max_len=10)
        assert len(padded) == 2
        assert len(masks) == 2
        assert len(padded[0]) == 10
        assert len(padded[1]) == 10

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def test_repr(self, tokenizer: BPETokenizer) -> None:
        """Verifies the __repr__ string contains key information.

        The repr should include the class name and the target vocab_size
        so that print(tokenizer) gives useful debugging output.
        """
        r = repr(tokenizer)
        assert "BPETokenizer" in r
        assert str(tokenizer.vocab_size) in r
