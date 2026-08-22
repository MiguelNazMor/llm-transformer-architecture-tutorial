"""GPT model with LoRA applied to attention weight matrices.

Wraps the base GPT model and replaces attention projection matrices
(W_q, W_k, W_v, W_o) with LoRA-wrapped versions.  Only the LoRA A and B
matrices are trained — the base weights stay frozen.

This demonstrates the standard LoRA setup: apply to attention projections
with rank r=16 (or similar) and alpha=2*r.
"""

import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
sys.path.insert(0, str(_transformer_src))

import numpy as np
from lora import LoRALinear
from model import GPT
from numpy.typing import NDArray


class LoRAGPT:
    """GPT model with LoRA applied to attention projections.

    Each attention layer's W_q, W_k, W_v, W_o are replaced with
    LoRALinear wrappers.  The base weights are frozen; only the
    low-rank A and B matrices are trained.

    Attributes:
        base_model: The frozen base GPT model.
        lora_layers: List of LoRA-wrapped attention layers per block.
        tokenizer: Shared BPE tokenizer.
    """

    def __init__(self, base_model: GPT, rank: int = 16, alpha: float | None = None) -> None:
        """Wraps a GPT model with LoRA adapters on attention weights.

        Args:
            base_model: A pre-trained GPT model.
            rank: LoRA rank (r).
            alpha: LoRA alpha scaling factor.  Defaults to 2 * rank.
        """
        self.base_model = base_model
        self.d_model = base_model.d_model
        self.vocab_size = base_model.vocab_size
        self.max_len = base_model.max_len

        # Wrap each block's attention weights with LoRA.
        self.lora_layers: list[dict[str, LoRALinear]] = []
        for block in base_model.blocks:
            attn = block.self_attention
            lora_dict = {
                "W_q": LoRALinear(attn.W_q, rank=rank, alpha=alpha),
                "W_k": LoRALinear(attn.W_k, rank=rank, alpha=alpha),
                "W_v": LoRALinear(attn.W_v, rank=rank, alpha=alpha),
                "W_o": LoRALinear(attn.W_o, rank=rank, alpha=alpha),
            }
            self.lora_layers.append(lora_dict)

        # Output head (weight-tied, frozen).
        self.lm_head = base_model.lm_head

    def forward(
        self,
        token_ids: NDArray[np.int64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass with LoRA-enhanced attention.

        Runs the base model forward pass but replaces attention projections
        with LoRA-wrapped versions.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len).
            mask: Optional padding mask.
            training: If True, caches for backward.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        from attention import create_causal_mask, scaled_dot_product_attention
        from embeddings import LearnedPositionalEmbedding
        from transformer_block import layer_norm

        batch_size, seq_len = token_ids.shape

        # Embeddings (frozen).
        x = self.base_model.token_embedding.forward(token_ids)
        if isinstance(self.base_model.pos_embedding, LearnedPositionalEmbedding):
            x = x + self.base_model.pos_embedding.forward(seq_len)
        else:
            x = x + self.base_model.pos_embedding[:, :seq_len, :]

        causal_mask = create_causal_mask(seq_len)
        if mask is not None:
            causal_mask = causal_mask * mask[:, np.newaxis, :]

        dummy_encoder = np.zeros((batch_size, 1, self.d_model), dtype=np.float64)

        self._cache_blocks: list[dict] = []
        for i, block in enumerate(self.base_model.blocks):
            lora = self.lora_layers[i]
            attn = block.self_attention
            block_cache: dict = {"x": x.copy()}

            # ---- Self-attention with LoRA ----
            normed = layer_norm(x)

            # Q, K, V projections with LoRA
            q = lora["W_q"].forward(normed) + attn.b_q
            k = lora["W_k"].forward(normed) + attn.b_k
            v = lora["W_v"].forward(normed) + attn.b_v

            # Reshape for multi-head
            q = q.reshape(batch_size, seq_len, attn.num_heads, attn.d_k).swapaxes(1, 2)
            k = k.reshape(batch_size, seq_len, attn.num_heads, attn.d_k).swapaxes(1, 2)
            v = v.reshape(batch_size, seq_len, attn.num_heads, attn.d_v).swapaxes(1, 2)

            expanded_mask = None
            if causal_mask is not None:
                expanded_mask = causal_mask[:, np.newaxis, :, :]

            attn_out = scaled_dot_product_attention(q, k, v, mask=expanded_mask)
            attn_concat = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

            # Output projection with LoRA
            attn_output = lora["W_o"].forward(attn_concat) + attn.b_o
            x = x + attn_output

            # ---- Cross-attention (frozen, dummy encoder) ----
            normed = layer_norm(x)
            cross_out = block.cross_attention.cross_attention_forward(
                normed, dummy_encoder, mask=None, training=False
            )
            x = x + cross_out

            # ---- FFN (frozen) ----
            normed = layer_norm(x)
            ffn_out = block.ffn.forward(normed)
            x = x + ffn_out

            block_cache["attn_concat"] = attn_concat
            block_cache["attn_output"] = attn_output
            self._cache_blocks.append(block_cache)

        x_normed = layer_norm(x)
        logits = np.matmul(x_normed, self.lm_head.T)

        if training:
            self._cache = {"x_normed": x_normed, "x_pre_final_norm": x}

        return logits

    def backward(self, d_logits: NDArray[np.float64]) -> None:
        """Backward pass — only LoRA parameters receive gradients."""
        from transformer_block import backward_layer_norm

        c = self._cache
        d_x_normed = np.matmul(d_logits, self.lm_head)
        d_x = backward_layer_norm(c["x_pre_final_norm"], d_x_normed)

        # Backward through blocks in reverse — only LoRA layers get gradients.
        for i in reversed(range(len(self.base_model.blocks))):
            lora = self.lora_layers[i]
            # FFN backward (frozen — no grad accumulation)
            # Cross-attn backward (frozen — no grad accumulation)
            # Self-attn backward through LoRA layers only
            for name in ["W_o", "W_v", "W_k", "W_q"]:
                d_x = lora[name].backward(d_x)

    def zero_grad(self) -> None:
        """Resets all LoRA gradient accumulators."""
        for lora_dict in self.lora_layers:
            for layer in lora_dict.values():
                layer.zero_grad()

    def get_params(self) -> dict[str, NDArray[np.float64]]:
        """Returns all trainable LoRA parameters."""
        params: dict[str, NDArray[np.float64]] = {}
        for i, lora_dict in enumerate(self.lora_layers):
            for name, layer in lora_dict.items():
                params[f"block.{i}.{name}.A"] = layer.A
                params[f"block.{i}.{name}.B"] = layer.B
        return params

    def get_grads(self) -> dict[str, NDArray[np.float64]]:
        """Returns all LoRA parameter gradients."""
        grads: dict[str, NDArray[np.float64]] = {}
        for i, lora_dict in enumerate(self.lora_layers):
            for name, layer in lora_dict.items():
                grads[f"block.{i}.{name}.A"] = layer.grad_A
                grads[f"block.{i}.{name}.B"] = layer.grad_B
        return grads

    def generate(
        self, prompt_ids: NDArray[np.int64],
        max_new_tokens: int = 20, temperature: float = 1.0,
    ) -> list[int]:
        """Autoregressive generation (same interface as GPT)."""
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

    def count_lora_params(self) -> int:
        """Returns the total number of trainable LoRA parameters."""
        total = 0
        for lora_dict in self.lora_layers:
            for layer in lora_dict.values():
                total += layer.A.size + layer.B.size
        return total

    def merge_weights(self) -> None:
        """Merges LoRA weights into the base attention weight matrices."""
        for i, block in enumerate(self.base_model.blocks):
            lora = self.lora_layers[i]
            attn = block.self_attention
            attn.W_q = lora["W_q"].merge()
            attn.W_k = lora["W_k"].merge()
            attn.W_v = lora["W_v"].merge()
            attn.W_o = lora["W_o"].merge()
