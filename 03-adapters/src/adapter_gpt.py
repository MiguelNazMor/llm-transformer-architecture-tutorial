"""GPT model with adapter layers for parameter-efficient fine-tuning.

Wraps the base GPT model from 01-transformer-architecture and inserts
adapter layers after each attention and FFN sub-layer.  Only the adapter
parameters are trained — the base model weights stay frozen.

This demonstrates the Houlsby adapter pattern: two adapters per block
(one after attention, one after FFN).
"""

import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
sys.path.insert(0, str(_transformer_src))

import numpy as np
from adapter import AdapterLayer
from model import GPT
from numpy.typing import NDArray
from transformer_block import backward_layer_norm, layer_norm


class AdapterGPT:
    """GPT model with adapter layers for parameter-efficient fine-tuning.

    Inserts Houlsby-style adapters after each attention sub-layer and
    each FFN sub-layer in every decoder block.  The base GPT weights
    are frozen; only adapter parameters are trained.

    Attributes:
        base_model: The frozen base GPT model.
        adapters: List of (attn_adapter, ffn_adapter) per block.
        tokenizer: The BPE tokenizer (shared with base model).
    """

    def __init__(self, base_model: GPT, bottleneck: int = 16) -> None:
        """Wraps a GPT model with adapter layers.

        Args:
            base_model: A pre-trained GPT model (weights will be frozen).
            bottleneck: Bottleneck dimension for each adapter.
        """
        self.base_model = base_model
        self.d_model = base_model.d_model
        self.vocab_size = base_model.vocab_size
        self.max_len = base_model.max_len

        # Create two adapters per block (Houlsby pattern).
        self.adapters: list[tuple[AdapterLayer, AdapterLayer]] = []
        for _ in base_model.blocks:
            self.adapters.append((
                AdapterLayer(base_model.d_model, bottleneck),
                AdapterLayer(base_model.d_model, bottleneck),
            ))

        # Weight tying: output head shares with token embedding (frozen).
        self.lm_head = base_model.lm_head

    def forward(
        self,
        token_ids: NDArray[np.int64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass through the GPT model with adapters.

        The base model runs with training=False (no dropout, no caching)
        so its weights are not updated.  Adapters are applied on top.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len).
            mask: Optional padding mask.
            training: If True, caches for backward pass.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        from attention import create_causal_mask
        from embeddings import LearnedPositionalEmbedding

        batch_size, seq_len = token_ids.shape

        # Embeddings (frozen — no caching needed).
        x = self.base_model.token_embedding.forward(token_ids)
        if isinstance(self.base_model.pos_embedding, LearnedPositionalEmbedding):
            x = x + self.base_model.pos_embedding.forward(seq_len)
        else:
            x = x + self.base_model.pos_embedding[:, :seq_len, :]

        # Causal mask.
        causal_mask = create_causal_mask(seq_len)
        if mask is not None:
            causal_mask = causal_mask * mask[:, np.newaxis, :]

        dummy_encoder = np.zeros((batch_size, 1, self.d_model), dtype=np.float64)

        # Run through each decoder block, inserting adapters.
        self._block_outputs: list[NDArray[np.float64]] = []
        for i, block in enumerate(self.base_model.blocks):
            # Self-attention sub-layer (frozen forward).
            normed = layer_norm(x)
            attn_out = block.self_attention.forward(
                normed, mask=causal_mask, training=False
            )
            x = x + attn_out

            # Adapter after attention (trainable).
            if training:
                self._block_outputs.append(x.copy())
            x = self.adapters[i][0].forward(x)

            # Cross-attention sub-layer (frozen, dummy encoder).
            normed = layer_norm(x)
            cross_out = block.cross_attention.cross_attention_forward(
                normed, dummy_encoder, mask=None, training=False
            )
            x = x + cross_out

            # FFN sub-layer (frozen forward).
            normed = layer_norm(x)
            ffn_out = block.ffn.forward(normed)
            x = x + ffn_out

            # Adapter after FFN (trainable).
            if training:
                self._block_outputs.append(x.copy())
            x = self.adapters[i][1].forward(x)

        x_normed = layer_norm(x)
        logits = np.matmul(x_normed, self.lm_head.T)

        # Cache for backward.
        if training:
            self._cache = {
                "token_ids": token_ids,
                "x_pre_final_norm": x,
                "x_normed": x_normed,
            }

        return logits

    def backward(self, d_logits: NDArray[np.float64]) -> None:
        """Backward pass through the adapter GPT model.

        Only adapter parameters receive gradients — base model weights
        are frozen (their backward is not called).

        Args:
            d_logits: Gradient w.r.t. logits, shape (batch, seq_len, vocab_size).
        """
        c = self._cache

        # Output projection (weight-tied, frozen — gradient not accumulated).
        d_x_normed = np.matmul(d_logits, self.lm_head)
        d_x = backward_layer_norm(c["x_pre_final_norm"], d_x_normed)

        # Backward through blocks in reverse, only through adapters.
        # Base model sub-layers are skipped (frozen).
        for i in reversed(range(len(self.base_model.blocks))):
            # Adapter after FFN (trainable).
            d_x = self.adapters[i][1].backward(d_x)
            # Adapter after attention (trainable).
            d_x = self.adapters[i][0].backward(d_x)

    def zero_grad(self) -> None:
        """Resets all adapter gradient accumulators."""
        for attn_adapter, ffn_adapter in self.adapters:
            attn_adapter.zero_grad()
            ffn_adapter.zero_grad()

    def get_params(self) -> dict[str, NDArray[np.float64]]:
        """Returns all trainable adapter parameters."""
        params: dict[str, NDArray[np.float64]] = {}
        for i, (attn_adapter, ffn_adapter) in enumerate(self.adapters):
            params[f"block.{i}.attn_adapter.W_down"] = attn_adapter.W_down
            params[f"block.{i}.attn_adapter.W_up"] = attn_adapter.W_up
            params[f"block.{i}.attn_adapter.b_down"] = attn_adapter.b_down
            params[f"block.{i}.attn_adapter.b_up"] = attn_adapter.b_up
            params[f"block.{i}.ffn_adapter.W_down"] = ffn_adapter.W_down
            params[f"block.{i}.ffn_adapter.W_up"] = ffn_adapter.W_up
            params[f"block.{i}.ffn_adapter.b_down"] = ffn_adapter.b_down
            params[f"block.{i}.ffn_adapter.b_up"] = ffn_adapter.b_up
        return params

    def get_grads(self) -> dict[str, NDArray[np.float64]]:
        """Returns all adapter parameter gradients."""
        grads: dict[str, NDArray[np.float64]] = {}
        for i, (attn_adapter, ffn_adapter) in enumerate(self.adapters):
            grads[f"block.{i}.attn_adapter.W_down"] = attn_adapter.grad_W_down
            grads[f"block.{i}.attn_adapter.W_up"] = attn_adapter.grad_W_up
            grads[f"block.{i}.attn_adapter.b_down"] = attn_adapter.grad_b_down
            grads[f"block.{i}.attn_adapter.b_up"] = attn_adapter.grad_b_up
            grads[f"block.{i}.ffn_adapter.W_down"] = ffn_adapter.grad_W_down
            grads[f"block.{i}.ffn_adapter.W_up"] = ffn_adapter.grad_W_up
            grads[f"block.{i}.ffn_adapter.b_down"] = ffn_adapter.grad_b_down
            grads[f"block.{i}.ffn_adapter.b_up"] = ffn_adapter.grad_b_up
        return grads

    def generate(
        self,
        prompt_ids: NDArray[np.int64],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generates tokens autoregressively (same interface as GPT)."""
        generated: list[int] = list(prompt_ids[0])
        for _ in range(max_new_tokens):
            ctx = generated[-self.max_len :]
            inp = np.array([ctx], dtype=np.int64)
            logits = self.forward(inp, training=False)
            next_logits = logits[0, -1, :]
            if temperature > 0:
                from model import softmax
                next_logits = next_logits / temperature
                probs = softmax(next_logits)
                next_token = int(np.random.choice(self.vocab_size, p=probs))
            else:
                next_token = int(np.argmax(next_logits))
            generated.append(next_token)
        return generated

    def count_adapter_params(self) -> int:
        """Returns the total number of trainable adapter parameters."""
        total = 0
        for attn_a, ffn_a in self.adapters:
            total += attn_a.W_down.size + attn_a.W_up.size
            total += attn_a.b_down.size + attn_a.b_up.size
            total += ffn_a.W_down.size + ffn_a.W_up.size
            total += ffn_a.b_down.size + ffn_a.b_up.size
        return total
