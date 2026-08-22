"""Tests for adapter layers and adapter-based fine-tuning."""

import numpy as np
import pytest
from adapter import AdapterLayer
from adapter_gpt import AdapterGPT
from model import GPT, cross_entropy_loss, softmax_cross_entropy_backward
from tokenizer import BPETokenizer
from trainer import SGD, prepare_text_batch


class TestAdapterLayer:
    """Tests for the AdapterLayer bottleneck module."""

    def test_output_shape(self) -> None:
        """Adapter preserves input shape."""
        adapter = AdapterLayer(d_model=32, bottleneck=8)
        x = np.random.randn(2, 5, 32)
        out = adapter.forward(x)
        assert out.shape == (2, 5, 32)

    def test_starts_near_identity(self) -> None:
        """Near-zero init means adapter output ≈ input at start."""
        adapter = AdapterLayer(d_model=32, bottleneck=8)
        x = np.random.randn(1, 4, 32)
        out = adapter.forward(x)
        # Output should be very close to input (up-proj initialized near zero)
        diff = np.mean(np.abs(out - x))
        assert diff < 0.1, f"Adapter output differs from input by {diff:.6f}"

    def test_backward_produces_gradients(self) -> None:
        """Backward pass produces non-zero gradients."""
        adapter = AdapterLayer(d_model=16, bottleneck=4)
        x = np.random.randn(1, 3, 16)
        adapter.forward(x)
        grad_out = np.random.randn(1, 3, 16)
        d_x = adapter.backward(grad_out)

        assert d_x.shape == (1, 3, 16)
        assert np.any(adapter.grad_W_down != 0)
        assert np.any(adapter.grad_W_up != 0)

    def test_gradient_check(self) -> None:
        """Analytical gradients match numerical gradients."""
        np.random.seed(42)
        adapter = AdapterLayer(d_model=8, bottleneck=4)
        x = np.random.randn(1, 2, 8)
        grad_out = np.random.randn(1, 2, 8)

        adapter.forward(x)
        adapter.backward(grad_out)

        eps = 1e-5
        for param_name, param, grad in [
            ("W_down", adapter.W_down, adapter.grad_W_down),
            ("W_up", adapter.W_up, adapter.grad_W_up),
        ]:
            for _ in range(5):
                i = np.random.randint(0, param.shape[0])
                j = np.random.randint(0, param.shape[1])
                orig = param[i, j]
                param[i, j] = orig + eps
                out_p = adapter.forward(x)
                loss_p = float(np.sum(grad_out * out_p))
                param[i, j] = orig - eps
                out_m = adapter.forward(x)
                loss_m = float(np.sum(grad_out * out_m))
                param[i, j] = orig
                num_grad = (loss_p - loss_m) / (2 * eps)
                ana_grad = grad[i, j]
                rel = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10)
                assert rel < 0.01, f"{param_name}[{i},{j}]: rel_error={rel:.6f}"


class TestAdapterGPT:
    """Tests for the GPT model wrapped with adapters."""

    @pytest.fixture
    def setup(self) -> tuple[AdapterGPT, BPETokenizer]:
        """Creates a pre-trained base model wrapped with adapters."""
        np.random.seed(42)
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        tok = BPETokenizer(vocab_size=300)
        tok.train(corpus)

        base = GPT(
            vocab_size=len(tok), d_model=32, num_heads=2,
            num_layers=1, d_ff=64, max_len=16, dropout_rate=0.0,
        )
        from trainer import Trainer
        Trainer(base, tok, lr=0.1, momentum=0.9).train(
            corpus, epochs=30, seq_len=8, verbose=False
        )
        adapter_model = AdapterGPT(base, bottleneck=8)
        return adapter_model, tok

    def test_forward_shape(self, setup: tuple[AdapterGPT, BPETokenizer]) -> None:
        """Adapter GPT produces correct output shapes."""
        model, tok = setup
        ids = np.array([tok.encode("the cat")], dtype=np.int64)
        logits = model.forward(ids, training=False)
        assert logits.shape == (1, len(ids[0]), len(tok))

    def test_adapter_params_fewer_than_base(self, setup: tuple[AdapterGPT, BPETokenizer]) -> None:
        """Adapter parameters should be much fewer than base model parameters."""
        model, _ = setup
        base_params = sum(p.size for p in model.base_model.get_params().values())
        adapter_params = model.count_adapter_params()
        assert adapter_params < base_params * 0.3, (
            f"Adapter params ({adapter_params}) should be < 30% of "
            f"base params ({base_params})"
        )

    def test_loss_decreases_with_adapter_training(
        self, setup: tuple[AdapterGPT, BPETokenizer]
    ) -> None:
        """Fine-tuning adapters should reduce loss on new domain."""
        model, tok = setup
        domain = ["the lion roars in the savanna", "the tiger hunts in the forest"]

        inp, tgt, msk = prepare_text_batch(domain, tok, seq_len=8)
        loss_before = cross_entropy_loss(
            model.forward(inp, mask=msk, training=False), tgt, msk
        )

        opt = SGD(lr=0.1, momentum=0.9, max_grad_norm=1.0)
        for _ in range(40):
            for text in domain:
                inp, tgt, msk = prepare_text_batch([text], tok, seq_len=8)
                model.zero_grad()
                logits = model.forward(inp, mask=msk, training=True)
                d_logits = softmax_cross_entropy_backward(logits, tgt, msk)
                model.backward(d_logits)
                opt.step(model)

        loss_after = cross_entropy_loss(
            model.forward(inp, mask=msk, training=False), tgt, msk
        )
        assert loss_after < loss_before * 0.8, (
            f"Adapter training did not reduce loss: {loss_before:.4f} → {loss_after:.4f}"
        )
