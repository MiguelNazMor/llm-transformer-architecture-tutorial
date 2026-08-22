"""Tests for Prefix Tuning and IA3 implementations."""

import numpy as np
from ia3 import IA3Layer
from prefix_tuning import PrefixTuning


class TestPrefixTuning:
    """Tests for prefix tuning vectors."""

    def test_forward_adds_prefix(self) -> None:
        """Prefix tuning should prepend vectors to K and V."""
        pt = PrefixTuning(d_model=32, num_prefix=5)
        k = np.random.randn(2, 4, 10, 32)  # (batch, heads, seq, d_k)
        v = np.random.randn(2, 4, 10, 32)
        k_out, v_out = pt.forward(k, v)

        # Should have 5 extra positions
        assert k_out.shape == (2, 4, 15, 32)
        assert v_out.shape == (2, 4, 15, 32)

    def test_backward_strips_prefix(self) -> None:
        """Backward should strip prefix and return original-length gradients."""
        pt = PrefixTuning(d_model=16, num_prefix=3)
        k = np.random.randn(1, 2, 5, 16)
        v = np.random.randn(1, 2, 5, 16)
        pt.forward(k, v)

        d_k_in = np.random.randn(1, 2, 8, 16)  # 5 + 3 prefix
        d_v_in = np.random.randn(1, 2, 8, 16)
        d_k_out, d_v_out = pt.backward(d_k_in, d_v_in)

        assert d_k_out.shape == (1, 2, 5, 16)
        assert d_v_out.shape == (1, 2, 5, 16)

    def test_backward_produces_gradients(self) -> None:
        """Backward should produce non-zero prefix gradients."""
        pt = PrefixTuning(d_model=8, num_prefix=2)
        k = np.random.randn(1, 1, 3, 8)
        v = np.random.randn(1, 1, 3, 8)
        pt.forward(k, v)

        d_k = np.random.randn(1, 1, 5, 8)
        d_v = np.random.randn(1, 1, 5, 8)
        pt.backward(d_k, d_v)

        assert np.any(pt.grad_prefix_keys != 0)
        assert np.any(pt.grad_prefix_values != 0)


class TestIA3:
    """Tests for IA3 rescaling vectors."""

    def test_initial_state_is_identity(self) -> None:
        """IA3 vectors start at 1.0 (identity — no change)."""
        ia3 = IA3Layer(d_model=16)
        k = np.random.randn(1, 4, 16)
        assert np.allclose(ia3.rescale_key(k), k)

    def test_rescale_shapes(self) -> None:
        """Rescaling preserves tensor shapes."""
        ia3 = IA3Layer(d_model=32)
        k = np.random.randn(2, 4, 10, 32)
        v = np.random.randn(2, 4, 10, 32)
        ffn = np.random.randn(2, 10, 32)

        assert ia3.rescale_key(k).shape == k.shape
        assert ia3.rescale_value(v).shape == v.shape
        assert ia3.rescale_ffn(ffn).shape == ffn.shape

    def test_backward_produces_gradients(self) -> None:
        """Backward produces non-zero gradients for rescaling vectors."""
        ia3 = IA3Layer(d_model=8)
        k = np.random.randn(1, 3, 8)
        v = np.random.randn(1, 3, 8)
        ffn = np.random.randn(1, 3, 8)

        ia3.rescale_key(k)
        ia3.rescale_value(v)
        ia3.rescale_ffn(ffn)

        d_k = np.random.randn(1, 3, 8)
        d_v = np.random.randn(1, 3, 8)
        d_ffn = np.random.randn(1, 3, 8)

        ia3.backward(d_k_out=d_k, d_v_out=d_v, d_ffn_out=d_ffn)

        assert np.any(ia3.grad_l_k != 0)
        assert np.any(ia3.grad_l_v != 0)
        assert np.any(ia3.grad_l_ff != 0)

    def test_param_count(self) -> None:
        """IA3 has exactly 3 * d_model parameters per layer."""
        d_model = 128
        ia3 = IA3Layer(d_model)
        assert ia3.count_params() == 3 * d_model

    def test_gradient_check(self) -> None:
        """Analytical IA3 gradients match numerical."""
        np.random.seed(42)
        ia3 = IA3Layer(d_model=4)
        k = np.random.randn(1, 2, 4)
        v = np.random.randn(1, 2, 4)

        ia3.rescale_key(k)
        ia3.rescale_value(v)

        d_k_out = np.random.randn(1, 2, 4)
        d_v_out = np.random.randn(1, 2, 4)
        ia3.backward(d_k_out=d_k_out, d_v_out=d_v_out)

        eps = 1e-5
        # Check l_k gradient
        orig = ia3.l_k[0]
        ia3.l_k[0] = orig + eps
        k_out_p = ia3.rescale_key(k)
        loss_p = float(np.sum(d_k_out * k_out_p))
        ia3.l_k[0] = orig - eps
        k_out_m = ia3.rescale_key(k)
        loss_m = float(np.sum(d_k_out * k_out_m))
        ia3.l_k[0] = orig
        num_grad = (loss_p - loss_m) / (2 * eps)
        ana_grad = ia3.grad_l_k[0]
        rel = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10)
        assert rel < 0.01, f"l_k gradient: rel_error={rel:.6f}"
