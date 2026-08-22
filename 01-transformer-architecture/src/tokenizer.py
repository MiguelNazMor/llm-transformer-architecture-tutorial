"""Byte-Pair Encoding (BPE) tokenizer.

Implements a minimal BPE tokenizer from scratch, following the algorithm used
by GPT models.  Includes training (learning merges from a corpus), encoding
(text → token IDs), and decoding (token IDs → text).
"""

from __future__ import annotations

from collections import Counter
from typing import Self


class BPETokenizer:
    """A Byte-Pair Encoding tokenizer.

    Learns subword merges from a training corpus and uses them to split text
    into tokens.  Supports special tokens for padding, unknown words, and
    sequence boundaries.

    Attributes:
        vocab_size: Target vocabulary size (including base characters).
        merges: Ordered list of (a, b) token pairs to merge.
        vocab: Mapping from token ID to token string.
        special_tokens: Dict of special token names to their IDs.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab_size: int = 1000) -> None:
        """Initializes the tokenizer with a target vocabulary size.

        Args:
            vocab_size: Maximum number of tokens in the vocabulary.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (to cover all bytes).")

        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[int, str] = {}

        # Reserve IDs for special tokens.
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self._next_id = 4

        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Assigns IDs to special tokens in the vocabulary."""
        self.vocab[self.pad_id] = self.PAD_TOKEN
        self.vocab[self.unk_id] = self.UNK_TOKEN
        self.vocab[self.bos_id] = self.BOS_TOKEN
        self.vocab[self.eos_id] = self.EOS_TOKEN

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: list[str]) -> Self:
        """Learns BPE merges from a text corpus.

        Args:
            corpus: List of strings to learn merges from.

        Returns:
            Self for method chaining.
        """
        # Start with character-level tokenization for every word.
        word_freqs = self._count_word_frequencies(corpus)
        splits: dict[str, list[str]] = {word: list(word) for word in word_freqs}

        num_merges = self.vocab_size - 256 - 4  # 4 special tokens
        for _ in range(num_merges):
            pair_freqs = self._count_pair_frequencies(splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)  # type: ignore[arg-type]
            self.merges.append(best_pair)
            self._apply_merge(splits, best_pair)

        self._build_vocab()
        return self

    @staticmethod
    def _count_word_frequencies(corpus: list[str]) -> dict[str, int]:
        """Counts how many times each whitespace-delimited word appears."""
        freqs: Counter[str] = Counter()
        for text in corpus:
            freqs.update(text.split())
        return dict(freqs)

    @staticmethod
    def _count_pair_frequencies(
        splits: dict[str, list[str]],
        word_freqs: dict[str, int],
    ) -> dict[tuple[str, str], int]:
        """Counts adjacent symbol-pair frequencies across all words."""
        pair_freqs: Counter[tuple[str, str]] = Counter()
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    @staticmethod
    def _apply_merge(splits: dict[str, list[str]], pair: tuple[str, str]) -> None:
        """Merges all occurrences of *pair* inside every word's symbol list."""
        a, b = pair
        for word in splits:
            symbols = splits[word]
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i : i + 2] = [a + b]
                else:
                    i += 1

    def _build_vocab(self) -> None:
        """Builds the id→token vocabulary from learned merges."""
        # Base vocabulary: individual bytes + special tokens.
        for i in range(256):
            token = bytes([i]).decode("latin-1", errors="replace")
            self.vocab[self._next_id] = token
            self._next_id += 1

        for a, b in self.merges:
            self.vocab[self._next_id] = a + b
            self._next_id += 1

        # Build reverse mapping for encoding.
        self._token_to_id = {v: k for k, v in self.vocab.items()}

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Converts text into a list of token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        token_ids: list[int] = []
        for word in text.split():
            token_ids.extend(self._encode_word(word))
        return token_ids

    def _encode_word(self, word: str) -> list[int]:
        """Encodes a single word using the learned BPE merges."""
        if not word:
            return []

        symbols = list(word)
        while len(symbols) > 1:
            # Find the earliest merge that applies.
            merged = False
            for a, b in self.merges:
                for i in range(len(symbols) - 1):
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i : i + 2] = [a + b]
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

        ids: list[int] = []
        for sym in symbols:
            ids.append(self._token_to_id.get(sym, self.unk_id))
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Converts token IDs back into text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        tokens: list[str] = []
        for tid in token_ids:
            token = self.vocab.get(tid, self.UNK_TOKEN)
            if token not in (
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ):
                tokens.append(token)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def encode_batch(
        self, texts: list[str], max_len: int | None = None
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encodes a batch of texts with padding and attention masks.

        Args:
            texts: List of strings to encode.
            max_len: Maximum sequence length.  If None, uses the longest text.

        Returns:
            Tuple of (padded_input_ids, attention_mask).
        """
        encoded = [self.encode(t) for t in texts]
        if max_len is None:
            max_len = max(len(ids) for ids in encoded)

        padded: list[list[int]] = []
        masks: list[int] = []
        for ids in encoded:
            mask = [1] * min(len(ids), max_len) + [0] * max(0, max_len - len(ids))
            masks.append(mask)
            if len(ids) >= max_len:
                padded.append(ids[:max_len])
            else:
                padded.append(ids + [self.pad_id] * (max_len - len(ids)))

        return padded, masks

    def __len__(self) -> int:
        """Returns the current vocabulary size."""
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"trained_tokens={len(self.vocab)}, "
            f"merges={len(self.merges)})"
        )

    def vocab_size_trained(self) -> int:
        """Returns the actual number of tokens learned (not the target)."""
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves the tokenizer to a JSON file.

        Args:
            path: File path (e.g., "tokenizer.json").
        """
        import json

        data = {
            "vocab_size": self.vocab_size,
            "merges": [(a, b) for a, b in self.merges],
            "vocab": {str(k): v for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Loads a tokenizer from a JSON file saved by save().

        Args:
            path: File path to the saved tokenizer.

        Returns:
            A BPETokenizer with restored vocabulary and merges.
        """
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = [(a, b) for a, b in data["merges"]]
        tok.vocab = {int(k): v for k, v in data["vocab"].items()}
        tok._next_id = max(tok.vocab.keys()) + 1 if tok.vocab else 4
        tok._token_to_id = {v: k for k, v in tok.vocab.items()}
        return tok

    # ------------------------------------------------------------------
    # Serialization helpers for demo purposes
    # ------------------------------------------------------------------

    def _encode_char_level(self, text: str) -> list[int]:
        """Encodes text at character level (no BPE merges applied).

        Useful as a baseline comparison and for untrained tokenizers.
        """
        ids: list[int] = []
        for ch in text:
            tid = self._token_to_id.get(ch, self.unk_id)
            ids.append(tid)
        return ids
