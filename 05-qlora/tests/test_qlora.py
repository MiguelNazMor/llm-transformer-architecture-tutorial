"""Tests for NF4 quantization and QLoRA fine-tuning."""

import numpy as np
import pytest
from quantize import (
    _nf4_quantization_levels,
    count_quantized_memory,
    dequantize_nf4,
    quantize_nf4,
)


class TestNF4Quantization:
    """Tests for NF4 quantization functions."""

    def test_levels_are_sorted(self) -> None:
        """NF4 levels should be in ascending order."""
        levels = _nf4_quantization_levels()
        assert np.all(np.diff(levels) > 0)
        assert len(levels) == 16

    def test_quantize_dequantize_roundtrip(self) -> None:
        """Quantization + dequantization should approximately preserve values."""
        np.random.seed(42)
        W = np.random.randn(128, 64).astype(np.float64) * 0.1
        q, s = quantize_nf4(W)
        W_recovered = dequantize_nf4(q, s, W.shape)

        # Should be reasonably close (4-bit has inherent error)
        rel_error = np.mean(np.abs(W - W_recovered)) / (np.mean(np.abs(W)) + 1e-8)
        assert rel_error < 0.5, f"Quantization error too large: {rel_error:.4f}"

    def test_zero_weights_quantize_correctly(self) -> None:
        """All-zero weights should quantize and dequantize to zeros."""
        W = np.zeros((64, 32), dtype=np.float64)
        q, s = quantize_nf4(W)
        W_recovered = dequantize_nf4(q, s, W.shape)
        assert np.allclose(W_recovered, 0, atol=1e-10)

    def test_quantize_preserves_shape(self) -> None:
        """Dequantized weights should have the original shape."""
        W = np.random.randn(100, 50).astype(np.float64)
        q, s = quantize_nf4(W)
        W_recovered = dequantize_nf4(q, s, W.shape)
        assert W_recovered.shape == W.shape

    def test_memory_reduction(self) -> None:
        """NF4 should significantly reduce memory vs FP64."""
        fp64_mb, nf4_mb, reduction = count_quantized_memory(1_000_000)
        assert reduction > 70, f"Memory reduction only {reduction:.1f}%"
        assert nf4_mb < fp64_mb * 0.3


class TestQLoRAGPT:
    """Tests for the QLoRA GPT model."""

    @pytest.fixture
    def setup(self):
        """Creates a pre-trained model wrapped with QLoRA."""
        import sys
        from pathlib import Path

        _tsrc = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
        sys.path.insert(0, str(_tsrc))
        from model import GPT
        from qlora_gpt import QLoRAGPT
        from tokenizer import BPETokenizer
        from trainer import Trainer

        np.random.seed(42)
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        tok = BPETokenizer(vocab_size=300)
        tok.train(corpus)
        base = GPT(
            vocab_size=len(tok),
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64,
            max_len=16,
            dropout_rate=0.0,
        )
        Trainer(base, tok, lr=0.1, momentum=0.9).train(corpus, epochs=20, seq_len=8, verbose=False)
        return QLoRAGPT(base, rank=8), tok

    def test_forward_shape(self, setup) -> None:
        """QLoRA GPT produces correct output shapes."""
        model, tok = setup
        ids = np.array([tok.encode("the cat")], dtype=np.int64)
        logits = model.forward(ids, training=False)
        assert logits.shape == (1, len(ids[0]), len(tok))

    def test_quantization_reduces_model_size(self, setup) -> None:
        """Quantized weights should use much less memory than FP64."""
        model, _ = setup
        base_params = sum(int(np.prod(shape)) for _, _, shape in model.quantized_weights.values())
        fp64_mb, nf4_mb, reduction = count_quantized_memory(base_params)
        assert reduction > 70, f"Memory reduction: {reduction:.1f}%"

    def test_lora_params_are_trainable(self, setup) -> None:
        """LoRA parameters should exist and be non-zero in size."""
        model, _ = setup
        params = model.get_params()
        assert len(params) > 0
        lora_count = model.count_lora_params()
        assert lora_count > 0

    def test_generation_works(self, setup) -> None:
        """Generation should work without crashing."""
        model, tok = setup
        prompt = np.array([tok.encode("the cat")], dtype=np.int64)
        gen = model.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert len(gen) > 2
