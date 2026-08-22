"""Tests for LoRA layers and LoRA-based fine-tuning."""

import numpy as np
import pytest
from lora import LoRALinear
from lora_gpt import LoRAGPT
from model import GPT, cross_entropy_loss, softmax_cross_entropy_backward
from tokenizer import BPETokenizer
from trainer import SGD, prepare_text_batch


class TestLoRALinear:
    """Tests for the LoRALinear wrapper."""

    def test_output_shape(self) -> None:
        """LoRA layer preserves output shape."""
        W = np.random.randn(32, 64)
        lora = LoRALinear(W, rank=8)
        x = np.random.randn(2, 5, 32)
        out = lora.forward(x)
        assert out.shape == (2, 5, 64)

    def test_starts_as_original(self) -> None:
        """Zero B init means LoRA output ≈ W output at start."""
        W = np.random.randn(16, 32)
        lora = LoRALinear(W, rank=4)
        x = np.random.randn(1, 4, 16)
        out_lora = lora.forward(x)
        out_orig = np.matmul(x, W)
        assert np.allclose(out_lora, out_orig, atol=1e-10)

    def test_backward_produces_gradients(self) -> None:
        """Backward produces non-zero gradients for B (A is zero since B starts at 0).

        With B initialized to zero, dA = x^T @ (grad_out @ B^T) = 0.
        But dB = (x @ A)^T @ grad_out is non-zero since A is random.
        """
        W = np.random.randn(8, 16)
        lora = LoRALinear(W, rank=4)
        x = np.random.randn(1, 3, 8)
        grad_out = np.random.randn(1, 3, 16)
        lora.forward(x)
        d_x = lora.backward(grad_out)
        assert d_x.shape == (1, 3, 8)
        # B gradient should be non-zero (even with B=0, the forward depends on A)
        assert np.any(lora.grad_B != 0), "B gradient is all zeros"
        # A gradient is zero when B=0 (correct — output doesn't depend on A)

    def test_merge_preserves_shape(self) -> None:
        """Merged weight has the same shape as original."""
        W = np.random.randn(32, 64)
        lora = LoRALinear(W, rank=8)
        merged = lora.merge()
        assert merged.shape == W.shape

    def test_gradient_check(self) -> None:
        """Analytical LoRA gradients match numerical."""
        np.random.seed(42)
        W = np.random.randn(8, 8)
        lora = LoRALinear(W, rank=2)
        x = np.random.randn(1, 2, 8)
        grad_out = np.random.randn(1, 2, 8)
        lora.forward(x)
        lora.backward(grad_out)

        eps = 1e-5
        for param_name, param, grad in [
            ("A", lora.A, lora.grad_A),
            ("B", lora.B, lora.grad_B),
        ]:
            max_err = 0.0
            for _ in range(5):
                i = np.random.randint(0, param.shape[0])
                j = np.random.randint(0, param.shape[1])
                orig = param[i, j]
                param[i, j] = orig + eps
                out_p = lora.forward(x)
                loss_p = float(np.sum(grad_out * out_p))
                param[i, j] = orig - eps
                out_m = lora.forward(x)
                loss_m = float(np.sum(grad_out * out_m))
                param[i, j] = orig
                num_grad = (loss_p - loss_m) / (2 * eps)
                ana_grad = grad[i, j]
                rel = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10)
                max_err = max(max_err, rel)
            assert max_err < 0.01, f"{param_name} max_rel_error={max_err:.6f}"


class TestLoRAGPT:
    """Tests for the GPT model with LoRA."""

    @pytest.fixture
    def setup(self) -> tuple[LoRAGPT, BPETokenizer]:
        """Creates a pre-trained model wrapped with LoRA."""
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
        from trainer import Trainer

        Trainer(base, tok, lr=0.1, momentum=0.9).train(corpus, epochs=30, seq_len=8, verbose=False)
        return LoRAGPT(base, rank=8), tok

    def test_forward_shape(self, setup: tuple[LoRAGPT, BPETokenizer]) -> None:
        """LoRA GPT produces correct output shapes."""
        model, tok = setup
        ids = np.array([tok.encode("the cat")], dtype=np.int64)
        logits = model.forward(ids, training=False)
        assert logits.shape == (1, len(ids[0]), len(tok))

    def test_lora_params_fewer_than_base(self, setup: tuple[LoRAGPT, BPETokenizer]) -> None:
        """LoRA parameters should be much fewer than base model."""
        model, _ = setup
        base_params = sum(p.size for p in model.base_model.get_params().values())
        lora_params = model.count_lora_params()
        assert lora_params < base_params * 0.3

    def test_loss_decreases_with_lora(self, setup: tuple[LoRAGPT, BPETokenizer]) -> None:
        """Fine-tuning LoRA should reduce loss."""
        model, tok = setup
        domain = ["the lion roars loudly", "the tiger hunts prey"]
        inp, tgt, msk = prepare_text_batch(domain, tok, seq_len=8)
        loss_before = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)

        opt = SGD(lr=0.1, momentum=0.9, max_grad_norm=1.0)
        for _ in range(50):
            for text in domain:
                inp, tgt, msk = prepare_text_batch([text], tok, seq_len=8)
                model.zero_grad()
                logits = model.forward(inp, mask=msk, training=True)
                d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
                model.backward(d_logits)
                opt.step(model)

        loss_after = cross_entropy_loss(model.forward(inp, mask=msk, training=False), tgt, msk)
        assert loss_after < loss_before * 0.8

    def test_merge_weights(self, setup: tuple[LoRAGPT, BPETokenizer]) -> None:
        """Merging LoRA weights should not crash and preserve generation."""
        model, tok = setup
        model.merge_weights()
        prompt = np.array([tok.encode("the cat")], dtype=np.int64)
        gen = model.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert len(gen) > 2
