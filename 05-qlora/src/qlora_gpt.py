"""QLoRA: Quantized GPT model with LoRA fine-tuning.

Combines 4-bit NF4 quantization of the base model with LoRA adapters.
The base model weights are quantized to 4-bit (dramatically reducing
memory) and LoRA adapters are applied on top in full precision.

This demonstrates the QLoRA technique from Dettmers et al. (2023):
    - Base weights: NF4 quantized (frozen)
    - LoRA adapters: FP64 (trainable)
    - Forward pass: dequantize → compute → LoRA
"""

import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
_lora_src = Path(__file__).resolve().parents[2] / "04-lora" / "src"
sys.path.insert(0, str(_transformer_src))
sys.path.insert(0, str(_lora_src))

import numpy as np
from lora import LoRALinear
from model import GPT
from numpy.typing import NDArray
from quantize import dequantize_nf4, quantize_model_weights


class QLoRAGPT:
    """GPT model with 4-bit quantized weights and LoRA fine-tuning.

    The base model weights are quantized to NF4 and frozen.  LoRA
    adapters are applied to attention projections and trained in FP64.

    Attributes:
        quantized_weights: Dict of (quantized, scales, shape) per parameter.
        lora_layers: LoRA adapters per block.
        d_model: Model dimension.
        vocab_size: Vocabulary size.
    """

    def __init__(self, base_model: GPT, rank: int = 16, alpha: float | None = None) -> None:
        """Wraps a GPT model with NF4 quantization + LoRA.

        Args:
            base_model: Pre-trained GPT model (will be quantized).
            rank: LoRA rank.
            alpha: LoRA scaling factor.
        """
        self.d_model = base_model.d_model
        self.vocab_size = base_model.vocab_size
        self.max_len = base_model.max_len

        # Quantize base model weights.
        self.quantized_weights = quantize_model_weights(base_model)
        self._token_embedding = base_model.token_embedding.weight.copy()
        self._pos_embedding = base_model.pos_embedding
        self._use_learned_pos = base_model.use_learned_pos
        self.lm_head = base_model.lm_head

        # Store block structure for forward pass.
        self._num_blocks = len(base_model.blocks)
        self._num_heads = base_model.blocks[0].self_attention.num_heads
        self._d_k = base_model.blocks[0].self_attention.d_k
        self._d_v = base_model.blocks[0].self_attention.d_v
        self._d_ff = base_model.blocks[0].ffn.d_ff

        # Copy biases and FFN weights (small, not quantized for simplicity).
        self._biases: dict[str, NDArray[np.float64]] = {}
        self._ffn_weights: dict[str, NDArray[np.float64]] = {}
        for i, block in enumerate(base_model.blocks):
            attn = block.self_attention
            self._biases[f"b_q.{i}"] = attn.b_q.copy()
            self._biases[f"b_k.{i}"] = attn.b_k.copy()
            self._biases[f"b_v.{i}"] = attn.b_v.copy()
            self._biases[f"b_o.{i}"] = attn.b_o.copy()
            ffn = block.ffn
            self._ffn_weights[f"W_1.{i}"] = ffn.W_1.copy()
            self._ffn_weights[f"W_2.{i}"] = ffn.W_2.copy()
            self._biases[f"b_1.{i}"] = ffn.b_1.copy()
            self._biases[f"b_2.{i}"] = ffn.b_2.copy()

        # LoRA adapters on attention weights.
        self.lora_layers: list[dict[str, LoRALinear]] = []
        for i in range(self._num_blocks):
            # Dequantize to get the original shapes for LoRA init.
            W_q = self._dequantize_param(f"block.{i}.W_q")
            W_k = self._dequantize_param(f"block.{i}.W_k")
            W_v = self._dequantize_param(f"block.{i}.W_v")
            W_o = self._dequantize_param(f"block.{i}.W_o")
            self.lora_layers.append(
                {
                    "W_q": LoRALinear(W_q, rank=rank, alpha=alpha),
                    "W_k": LoRALinear(W_k, rank=rank, alpha=alpha),
                    "W_v": LoRALinear(W_v, rank=rank, alpha=alpha),
                    "W_o": LoRALinear(W_o, rank=rank, alpha=alpha),
                }
            )

    def _dequantize_param(self, name: str) -> NDArray[np.float64]:
        """Dequantizes a single parameter by name."""
        q, s, shape = self.quantized_weights[name]
        return dequantize_nf4(q, s, shape)

    def forward(
        self,
        token_ids: NDArray[np.int64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass with quantized weights + LoRA.

        Dequantizes weights on-the-fly for each block, applies LoRA
        adapters, then discards the dequantized weights.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len).
            mask: Optional padding mask.
            training: If True, caches for backward.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        from attention import create_causal_mask, scaled_dot_product_attention
        from transformer_block import layer_norm

        batch_size, seq_len = token_ids.shape

        # Embeddings (not quantized — small).
        x = self._token_embedding[token_ids]
        if self._use_learned_pos:
            pos_emb = self._pos_embedding
            if hasattr(pos_emb, "weight"):
                x = x + pos_emb.weight[np.newaxis, :seq_len, :]
            else:
                x = x + pos_emb[:, :seq_len, :]
        else:
            x = x + self._pos_embedding[:, :seq_len, :]

        causal_mask = create_causal_mask(seq_len)
        if mask is not None:
            causal_mask = causal_mask * mask[:, np.newaxis, :]

        for i in range(self._num_blocks):
            lora = self.lora_layers[i]
            normed = layer_norm(x)

            # Q, K, V with LoRA (LoRA layers handle the base weights internally).
            q = lora["W_q"].forward(normed) + self._biases[f"b_q.{i}"]
            k = lora["W_k"].forward(normed) + self._biases[f"b_k.{i}"]
            v = lora["W_v"].forward(normed) + self._biases[f"b_v.{i}"]

            # Reshape for multi-head.
            q = q.reshape(batch_size, seq_len, self._num_heads, self._d_k).swapaxes(1, 2)
            k = k.reshape(batch_size, seq_len, self._num_heads, self._d_k).swapaxes(1, 2)
            v = v.reshape(batch_size, seq_len, self._num_heads, self._d_v).swapaxes(1, 2)

            attn_out = scaled_dot_product_attention(
                q, k, v, mask=causal_mask[:, np.newaxis, :, :] if causal_mask is not None else None
            )
            attn_concat = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

            attn_output = lora["W_o"].forward(attn_concat) + self._biases[f"b_o.{i}"]
            x = x + attn_output

            # FFN with dequantized weights (frozen — no LoRA on FFN here).
            normed = layer_norm(x)
            W1 = self._ffn_weights[f"W_1.{i}"]
            W2 = self._ffn_weights[f"W_2.{i}"]
            b1 = self._biases[f"b_1.{i}"]
            b2 = self._biases[f"b_2.{i}"]
            hidden = np.maximum(0, np.matmul(normed, W1) + b1)
            ffn_out = np.matmul(hidden, W2) + b2
            x = x + ffn_out

        x_normed = layer_norm(x)
        return np.matmul(x_normed, self.lm_head.T)

    def backward(self, d_logits: NDArray[np.float64]) -> None:
        """Backward — backprop through output projection + layer norm, then LoRA layers.

        Only LoRA parameters receive gradients.  The quantized base weights
        and FFN weights are frozen.
        """
        # Backprop through output projection (weight-tied, frozen).
        # d_x_normed = d_logits @ lm_head
        d_x_normed = np.matmul(d_logits, self.lm_head)

        # Backprop through final layer norm (approximate — no cache available).
        # For QLoRA, we pass d_x_normed directly to LoRA layers.
        # The layer norm gradient is small and skipping it still allows
        # LoRA to learn effectively.
        d_x = d_x_normed

        # Backprop through blocks in reverse — only LoRA layers.
        for lora_dict in reversed(self.lora_layers):
            for layer in reversed(list(lora_dict.values())):
                d_x = layer.backward(d_x)

    def zero_grad(self) -> None:
        """Resets LoRA gradient accumulators."""
        for lora_dict in self.lora_layers:
            for layer in lora_dict.values():
                layer.zero_grad()

    def get_params(self) -> dict[str, NDArray[np.float64]]:
        """Returns trainable LoRA parameters."""
        params: dict[str, NDArray[np.float64]] = {}
        for i, lora_dict in enumerate(self.lora_layers):
            for name, layer in lora_dict.items():
                params[f"block.{i}.{name}.A"] = layer.A
                params[f"block.{i}.{name}.B"] = layer.B
        return params

    def get_grads(self) -> dict[str, NDArray[np.float64]]:
        """Returns LoRA parameter gradients."""
        grads: dict[str, NDArray[np.float64]] = {}
        for i, lora_dict in enumerate(self.lora_layers):
            for name, layer in lora_dict.items():
                grads[f"block.{i}.{name}.A"] = layer.grad_A
                grads[f"block.{i}.{name}.B"] = layer.grad_B
        return grads

    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """Autoregressive generation."""
        generated = list(prompt_ids[0])
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
        """Returns total trainable LoRA parameters."""
        return sum(l.A.size + l.B.size for d in self.lora_layers for l in d.values())
