"""Full Transformer models: encoder-decoder and GPT-style decoder-only.

Provides two model classes:
    - Transformer: Original encoder-decoder architecture (Vaswani et al., 2017).
    - GPT: Decoder-only autoregressive language model (GPT-2 style).
"""

import numpy as np
from attention import create_causal_mask
from embeddings import (
    LearnedPositionalEmbedding,
    TokenEmbedding,
    sinusoidal_positional_encoding,
)
from numpy.typing import NDArray
from transformer_block import DecoderBlock, EncoderBlock, backward_layer_norm, layer_norm


def softmax(x: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    """Numerically stable softmax along the given axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def cross_entropy_loss(
    logits: NDArray[np.float64],
    targets: NDArray[np.int64],
    mask: NDArray[np.float64] | None = None,
) -> float:
    """Computes mean cross-entropy loss over non-padded tokens.

    Args:
        logits: Predicted logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        mask: Optional mask of shape (batch, seq_len).  1 = include, 0 = ignore.

    Returns:
        Scalar loss value.
    """
    batch_size, seq_len, vocab_size = logits.shape

    probs = softmax(logits.reshape(-1, vocab_size), axis=-1)
    targets_flat = targets.reshape(-1)

    # Gather probabilities of the correct tokens.
    correct_probs = probs[np.arange(batch_size * seq_len), targets_flat]
    log_probs = -np.log(np.clip(correct_probs, 1e-12, 1.0))

    if mask is not None:
        mask_flat = mask.reshape(-1)
        loss = np.sum(log_probs * mask_flat) / np.maximum(np.sum(mask_flat), 1)
    else:
        loss = np.mean(log_probs)

    return float(loss)


def softmax_cross_entropy_backward(
    logits: NDArray[np.float64],
    targets: NDArray[np.int64],
    mask: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Gradient of cross-entropy loss w.r.t. logits.

    For softmax + cross-entropy, the gradient simplifies to:
        d_logits = (probs - one_hot) / N

    where N is the number of non-masked tokens.

    Args:
        logits: Logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        mask: Optional mask of shape (batch, seq_len).  1 = include, 0 = ignore.

    Returns:
        Gradient w.r.t. logits, same shape as logits.
    """
    batch_size, seq_len, vocab_size = logits.shape
    probs = softmax(logits.reshape(-1, vocab_size), axis=-1)
    targets_flat = targets.reshape(-1)

    # one-hot encoding of targets
    one_hot = np.zeros((batch_size * seq_len, vocab_size), dtype=np.float64)
    one_hot[np.arange(batch_size * seq_len), targets_flat] = 1.0

    d_logits = probs - one_hot  # (batch*seq, vocab)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        d_logits *= mask_flat[:, np.newaxis]
        n_tokens = np.maximum(np.sum(mask_flat), 1)
    else:
        n_tokens = batch_size * seq_len

    d_logits /= n_tokens
    return d_logits.reshape(batch_size, seq_len, vocab_size)


class Transformer:
    """Original encoder-decoder Transformer (Vaswani et al., 2017).

    Architecture:
        - Token embeddings + sinusoidal positional encodings
        - N encoder blocks (self-attention + FFN)
        - N decoder blocks (masked self-attn + cross-attn + FFN)
        - Final linear projection to vocabulary

    Designed for sequence-to-sequence tasks like machine translation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initializes the Transformer model.

        Args:
            vocab_size: Size of the shared vocabulary.
            d_model: Model hidden dimension.
            num_heads: Number of attention heads per block.
            num_layers: Number of encoder and decoder blocks (each).
            d_ff: Feed-forward inner dimension.
            max_len: Maximum sequence length for positional encodings.
            dropout_rate: Dropout probability.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Shared token embeddings for encoder and decoder.
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Positional encodings are fixed (sinusoidal), not learned.
        self.pos_encoding = sinusoidal_positional_encoding(max_len, d_model)

        # Encoder and decoder stacks.
        self.encoder_blocks = [
            EncoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ]
        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ]

        # Output projection to vocabulary.
        self.output_proj = np.random.randn(d_model, vocab_size).astype(np.float64) * 0.02

    def encode(
        self,
        src_ids: NDArray[np.int64],
        src_mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Encodes a source sequence.

        Args:
            src_ids: Source token IDs of shape (batch, src_len).
            src_mask: Optional padding mask of shape (batch, src_len).
            training: If True, applies dropout.

        Returns:
            Encoder output of shape (batch, src_len, d_model).
        """
        _, src_len = src_ids.shape

        x = self.token_embedding.forward(src_ids)
        x = x + self.pos_encoding[:, :src_len, :]

        # Expand padding mask to (batch, 1, src_len) — broadcasts to
        # (batch, src_len, src_len) for self-attention, then attention
        # expands to (batch, 1, src_len, src_len) internally.
        attn_mask = None
        if src_mask is not None:
            attn_mask = src_mask[:, np.newaxis, :]

        for block in self.encoder_blocks:
            x = block.forward(x, mask=attn_mask, training=training)

        return layer_norm(x)

    def decode(
        self,
        tgt_ids: NDArray[np.int64],
        encoder_output: NDArray[np.float64],
        tgt_mask: NDArray[np.float64] | None = None,
        src_mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Decodes a target sequence given encoder output.

        Args:
            tgt_ids: Target token IDs of shape (batch, tgt_len).
            encoder_output: Encoder output of shape (batch, src_len, d_model).
            tgt_mask: Optional padding mask for target of shape (batch, tgt_len).
            src_mask: Optional padding mask for source of shape (batch, src_len).
            training: If True, applies dropout.

        Returns:
            Decoder output of shape (batch, tgt_len, d_model).
        """
        batch_size, tgt_len = tgt_ids.shape

        x = self.token_embedding.forward(tgt_ids)
        x = x + self.pos_encoding[:, :tgt_len, :]

        causal_mask = create_causal_mask(tgt_len)
        if tgt_mask is not None:
            # Combine causal and padding masks.
            # tgt_mask is (batch, tgt_len); expand to (batch, 1, tgt_len)
            # so it broadcasts to (batch, tgt_len, tgt_len).
            causal_mask = causal_mask * tgt_mask[:, np.newaxis, :]

        cross_mask = None
        if src_mask is not None:
            # src_mask is (batch, src_len); expand to (batch, 1, src_len)
            # so it broadcasts to (batch, tgt_len, src_len) in attention.
            cross_mask = src_mask[:, np.newaxis, :]

        for block in self.decoder_blocks:
            x = block.forward(
                x,
                encoder_output,
                causal_mask=causal_mask,
                cross_mask=cross_mask,
                training=training,
            )

        return layer_norm(x)

    def forward(
        self,
        src_ids: NDArray[np.int64],
        tgt_ids: NDArray[np.int64],
        src_mask: NDArray[np.float64] | None = None,
        tgt_mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Full forward pass: encode source, decode target, project to vocab.

        Args:
            src_ids: Source token IDs of shape (batch, src_len).
            tgt_ids: Target token IDs of shape (batch, tgt_len).
            src_mask: Optional source padding mask.
            tgt_mask: Optional target padding mask.
            training: If True, applies dropout.

        Returns:
            Logits of shape (batch, tgt_len, vocab_size).
        """
        encoder_output = self.encode(src_ids, src_mask, training=training)
        decoder_output = self.decode(tgt_ids, encoder_output, tgt_mask, src_mask, training=training)
        return np.matmul(decoder_output, self.output_proj)


# ---------------------------------------------------------------------------
# GPT-style decoder-only model
# ---------------------------------------------------------------------------


class GPT:
    """A GPT-style decoder-only Transformer for language modeling.

    Uses only the decoder stack with causal (masked) self-attention.
    Supports both learned and sinusoidal positional embeddings.

    Attributes:
        vocab_size: Vocabulary size.
        d_model: Model hidden dimension.
        num_layers: Number of decoder blocks.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout_rate: float = 0.1,
        *,
        use_learned_pos: bool = True,
    ) -> None:
        """Initializes a GPT-style language model.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: Model hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of decoder (transformer) blocks.
            d_ff: Feed-forward inner dimension.
            max_len: Maximum sequence length.
            dropout_rate: Dropout probability.
            use_learned_pos: If True, uses learned positional embeddings
                             (GPT style).  Otherwise, sinusoidal.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.use_learned_pos = use_learned_pos
        if use_learned_pos:
            self.pos_embedding: LearnedPositionalEmbedding | NDArray[np.float64] = (
                LearnedPositionalEmbedding(max_len, d_model)
            )
        else:
            self.pos_embedding = sinusoidal_positional_encoding(max_len, d_model)

        self.blocks = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ]

        # Output projection shares weights with token embedding (weight tying).
        self.lm_head = self.token_embedding.weight

    def forward(
        self,
        token_ids: NDArray[np.int64],
        mask: NDArray[np.float64] | None = None,
        *,
        training: bool = True,
    ) -> NDArray[np.float64]:
        """Forward pass through the GPT model.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len).
            mask: Optional padding mask of shape (batch, seq_len).
            training: If True, applies dropout.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.shape

        x = self.token_embedding.forward(token_ids)

        # Add positional information.
        if isinstance(self.pos_embedding, LearnedPositionalEmbedding):
            pos = self.pos_embedding.forward(seq_len)
        else:
            pos = self.pos_embedding[:, :seq_len, :]
        x = x + pos

        # Causal mask: can only attend to previous positions.
        # Shape: (1, seq_len, seq_len) — lower triangular.
        causal_mask = create_causal_mask(seq_len)
        if mask is not None:
            # Combine causal mask with padding mask.
            # mask is (batch, seq_len) — 1 for real tokens, 0 for padding.
            # We mask out key positions that are padded.
            # Result: (batch, seq_len, seq_len) — 3D for attention.
            causal_mask = causal_mask * mask[:, np.newaxis, :]

        # Decoder blocks use a dummy encoder output for cross-attention.
        # In a decoder-only model, cross-attention is skipped by passing None.
        # We pass zeros so the block structure is preserved.
        dummy_encoder = np.zeros((batch_size, 1, self.d_model), dtype=np.float64)

        block_inputs: list[NDArray[np.float64]] = []
        for block in self.blocks:
            if training:
                block_inputs.append(x)
            x = block.forward(
                x,
                dummy_encoder,
                causal_mask=causal_mask,
                cross_mask=None,
                training=training,
            )

        x_normed = layer_norm(x)
        logits = np.matmul(x_normed, self.lm_head.T)

        # Cache for backward
        if training:
            self._cache = {
                "token_ids": token_ids,
                "seq_len": seq_len,
                "block_inputs": block_inputs,
                "x_pre_final_norm": x,  # before final layer norm
                "x_normed": x_normed,
            }

        return logits

    def backward(self, d_logits: NDArray[np.float64]) -> None:
        """Backward pass through the GPT model.

        Accumulates gradients into all parameters (token embedding, pos
        embedding, attention/FFN weights in each block).  After calling
        backward, use get_grads() to retrieve the gradient dictionary.

        Args:
            d_logits: Gradient w.r.t. the logits, shape (batch, seq_len, vocab_size).
        """
        c = self._cache

        # Backprop through output projection (weight-tied with token embedding)
        # logits = x_normed @ lm_head.T
        # lm_head = token_embedding.weight (shape: vocab_size, d_model)
        # d_x_normed = d_logits @ lm_head  (since lm_head.T.T = lm_head)
        d_x_normed = np.matmul(d_logits, self.lm_head)  # (batch, seq, d_model)
        # Gradient w.r.t. lm_head (= token_embedding.weight):
        # d_lm_head = d_logits.T @ x_normed → shape (vocab, d_model)
        d_lm_head = np.matmul(
            d_logits.reshape(-1, self.vocab_size).T, c["x_normed"].reshape(-1, self.d_model)
        )

        # Backprop through final layer norm
        d_x = backward_layer_norm(c["x_pre_final_norm"], d_x_normed)

        # Backprop through decoder blocks in reverse
        for i in reversed(range(len(self.blocks))):
            d_x = self.blocks[i].backward(d_x)

        # Backprop through positional embedding
        if isinstance(self.pos_embedding, LearnedPositionalEmbedding):
            self.pos_embedding.backward(d_x)  # accumulates into grad_weight

        # Backprop through token embedding
        # The token embedding receives gradients from TWO paths:
        #   1. The embedding lookup (forward path)
        #   2. The output projection (weight tying)
        # We need to accumulate both into the same weight matrix.
        self.token_embedding.backward(d_x)  # accumulates from lookup path
        # Add the output projection gradient (weight tying)
        if not hasattr(self.token_embedding, "grad_weight"):
            self.token_embedding.grad_weight = np.zeros_like(self.token_embedding.weight)
        self.token_embedding.grad_weight += d_lm_head

    def zero_grad(self) -> None:
        """Resets all gradient accumulators in the model."""
        # Token embedding
        self.token_embedding.grad_weight = np.zeros_like(self.token_embedding.weight)
        # Position embedding (only if learned)
        if isinstance(self.pos_embedding, LearnedPositionalEmbedding):
            self.pos_embedding.grad_weight = np.zeros_like(self.pos_embedding.weight)
        # Blocks
        for block in self.blocks:
            block.zero_grad()

    def get_params(self) -> dict[str, NDArray[np.float64]]:
        """Returns a dictionary of all trainable parameters.

        Returns:
            Dict mapping parameter names to their arrays.
        """
        params: dict[str, NDArray[np.float64]] = {}
        params["token_embedding.weight"] = self.token_embedding.weight
        if isinstance(self.pos_embedding, LearnedPositionalEmbedding):
            params["pos_embedding.weight"] = self.pos_embedding.weight
        for i, block in enumerate(self.blocks):
            attn = block.self_attention
            params[f"block.{i}.W_q"] = attn.W_q
            params[f"block.{i}.W_k"] = attn.W_k
            params[f"block.{i}.W_v"] = attn.W_v
            params[f"block.{i}.W_o"] = attn.W_o
            params[f"block.{i}.b_q"] = attn.b_q
            params[f"block.{i}.b_k"] = attn.b_k
            params[f"block.{i}.b_v"] = attn.b_v
            params[f"block.{i}.b_o"] = attn.b_o
            ffn = block.ffn
            if hasattr(ffn, "W_1"):
                params[f"block.{i}.W_1"] = ffn.W_1
                params[f"block.{i}.W_2"] = ffn.W_2
                params[f"block.{i}.b_1"] = ffn.b_1
                params[f"block.{i}.b_2"] = ffn.b_2
            elif hasattr(ffn, "W_gate"):
                params[f"block.{i}.W_gate"] = ffn.W_gate
                params[f"block.{i}.W_up"] = ffn.W_up
                params[f"block.{i}.W_down"] = ffn.W_down
        return params

    def get_grads(self) -> dict[str, NDArray[np.float64]]:
        """Returns a dictionary of all parameter gradients.

        Must be called after forward() and backward().

        Returns:
            Dict mapping gradient names to their arrays (same keys as get_params).
        """
        grads: dict[str, NDArray[np.float64]] = {}
        grads["token_embedding.weight"] = self.token_embedding.grad_weight
        if isinstance(self.pos_embedding, LearnedPositionalEmbedding):
            grads["pos_embedding.weight"] = self.pos_embedding.grad_weight
        for i, block in enumerate(self.blocks):
            attn = block.self_attention
            grads[f"block.{i}.W_q"] = attn.grad_W_q
            grads[f"block.{i}.W_k"] = attn.grad_W_k
            grads[f"block.{i}.W_v"] = attn.grad_W_v
            grads[f"block.{i}.W_o"] = attn.grad_W_o
            grads[f"block.{i}.b_q"] = attn.grad_b_q
            grads[f"block.{i}.b_k"] = attn.grad_b_k
            grads[f"block.{i}.b_v"] = attn.grad_b_v
            grads[f"block.{i}.b_o"] = attn.grad_b_o
            ffn = block.ffn
            if hasattr(ffn, "grad_W_1"):
                grads[f"block.{i}.W_1"] = ffn.grad_W_1
                grads[f"block.{i}.W_2"] = ffn.grad_W_2
                grads[f"block.{i}.b_1"] = ffn.grad_b_1
                grads[f"block.{i}.b_2"] = ffn.grad_b_2
            elif hasattr(ffn, "grad_W_gate"):
                grads[f"block.{i}.W_gate"] = ffn.grad_W_gate
                grads[f"block.{i}.W_up"] = ffn.grad_W_up
                grads[f"block.{i}.W_down"] = ffn.grad_W_down
        return grads

    def generate(
        self,
        prompt_ids: NDArray[np.int64],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generates tokens autoregressively from a prompt.

        Args:
            prompt_ids: Prompt token IDs of shape (1, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.  Lower = more deterministic.

        Returns:
            List of generated token IDs (prompt + new tokens).
        """
        generated: list[int] = list(prompt_ids[0])

        for _ in range(max_new_tokens):
            # Truncate to max_len if needed.
            ctx = generated[-self.max_len :]
            inp = np.array([ctx], dtype=np.int64)

            logits = self.forward(inp, training=False)  # (1, ctx_len, vocab_size)
            next_logits = logits[0, -1, :]  # logits for last position

            if temperature > 0:
                next_logits = next_logits / temperature
                probs = softmax(next_logits)
                next_token = int(np.random.choice(self.vocab_size, p=probs))
            else:
                next_token = int(np.argmax(next_logits))

            generated.append(next_token)

        return generated


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------


def save_model(model: GPT, path: str) -> None:
    """Saves a GPT model to a .npz file.

    All weights, biases, and configuration are saved so the model can be
    reconstructed exactly with load_model().

    Args:
        model: The GPT model to save.
        path: File path (e.g., "base_model.npz").
    """
    data: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {}

    # Configuration
    data["_vocab_size"] = np.array(model.vocab_size, dtype=np.int64)
    data["_d_model"] = np.array(model.d_model, dtype=np.int64)
    data["_num_heads"] = np.array(model.blocks[0].self_attention.num_heads, dtype=np.int64)
    data["_num_layers"] = np.array(len(model.blocks), dtype=np.int64)
    data["_d_ff"] = np.array(model.blocks[0].ffn.d_ff, dtype=np.int64)
    data["_max_len"] = np.array(model.max_len, dtype=np.int64)
    data["_use_learned_pos"] = np.array(1 if model.use_learned_pos else 0, dtype=np.int64)

    # Token embedding
    data["token_embedding.weight"] = model.token_embedding.weight

    # Position embedding (if learned)
    if model.use_learned_pos and hasattr(model.pos_embedding, "weight"):
        data["pos_embedding.weight"] = model.pos_embedding.weight  # type: ignore[union-attr]

    # Blocks
    for i, block in enumerate(model.blocks):
        attn = block.self_attention
        data[f"block.{i}.self_attn.W_q"] = attn.W_q
        data[f"block.{i}.self_attn.W_k"] = attn.W_k
        data[f"block.{i}.self_attn.W_v"] = attn.W_v
        data[f"block.{i}.self_attn.W_o"] = attn.W_o
        data[f"block.{i}.self_attn.b_q"] = attn.b_q
        data[f"block.{i}.self_attn.b_k"] = attn.b_k
        data[f"block.{i}.self_attn.b_v"] = attn.b_v
        data[f"block.{i}.self_attn.b_o"] = attn.b_o

        cross = block.cross_attention
        data[f"block.{i}.cross_attn.W_q"] = cross.W_q
        data[f"block.{i}.cross_attn.W_k"] = cross.W_k
        data[f"block.{i}.cross_attn.W_v"] = cross.W_v
        data[f"block.{i}.cross_attn.W_o"] = cross.W_o
        data[f"block.{i}.cross_attn.b_q"] = cross.b_q
        data[f"block.{i}.cross_attn.b_k"] = cross.b_k
        data[f"block.{i}.cross_attn.b_v"] = cross.b_v
        data[f"block.{i}.cross_attn.b_o"] = cross.b_o

        ffn = block.ffn
        if hasattr(ffn, "W_1"):
            data[f"block.{i}.ffn.W_1"] = ffn.W_1
            data[f"block.{i}.ffn.W_2"] = ffn.W_2
            data[f"block.{i}.ffn.b_1"] = ffn.b_1
            data[f"block.{i}.ffn.b_2"] = ffn.b_2
        elif hasattr(ffn, "W_gate"):
            data[f"block.{i}.ffn.W_gate"] = ffn.W_gate
            data[f"block.{i}.ffn.W_up"] = ffn.W_up
            data[f"block.{i}.ffn.W_down"] = ffn.W_down

    np.savez_compressed(path, **data)


def load_model(path: str) -> GPT:
    """Loads a GPT model from a .npz file saved by save_model().

    Args:
        path: File path to the saved model.

    Returns:
        A GPT model with restored weights and configuration.
    """
    data = np.load(path)

    # Configuration
    vocab_size = int(data["_vocab_size"])
    d_model = int(data["_d_model"])
    num_heads = int(data["_num_heads"])
    num_layers = int(data["_num_layers"])
    d_ff = int(data["_d_ff"])
    max_len = int(data["_max_len"])
    use_learned_pos = bool(int(data["_use_learned_pos"]))

    # Create model with the same config
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout_rate=0.0,
        use_learned_pos=use_learned_pos,
    )

    # Restore token embedding
    model.token_embedding.weight = data["token_embedding.weight"].copy()

    # Restore position embedding
    if use_learned_pos and "pos_embedding.weight" in data:
        model.pos_embedding.weight = data["pos_embedding.weight"].copy()  # type: ignore[union-attr]

    # Restore blocks
    for i in range(num_layers):
        attn = model.blocks[i].self_attention
        attn.W_q = data[f"block.{i}.self_attn.W_q"].copy()
        attn.W_k = data[f"block.{i}.self_attn.W_k"].copy()
        attn.W_v = data[f"block.{i}.self_attn.W_v"].copy()
        attn.W_o = data[f"block.{i}.self_attn.W_o"].copy()
        attn.b_q = data[f"block.{i}.self_attn.b_q"].copy()
        attn.b_k = data[f"block.{i}.self_attn.b_k"].copy()
        attn.b_v = data[f"block.{i}.self_attn.b_v"].copy()
        attn.b_o = data[f"block.{i}.self_attn.b_o"].copy()

        cross = model.blocks[i].cross_attention
        cross.W_q = data[f"block.{i}.cross_attn.W_q"].copy()
        cross.W_k = data[f"block.{i}.cross_attn.W_k"].copy()
        cross.W_v = data[f"block.{i}.cross_attn.W_v"].copy()
        cross.W_o = data[f"block.{i}.cross_attn.W_o"].copy()
        cross.b_q = data[f"block.{i}.cross_attn.b_q"].copy()
        cross.b_k = data[f"block.{i}.cross_attn.b_k"].copy()
        cross.b_v = data[f"block.{i}.cross_attn.b_v"].copy()
        cross.b_o = data[f"block.{i}.cross_attn.b_o"].copy()

        ffn = model.blocks[i].ffn
        if hasattr(ffn, "W_1"):
            ffn.W_1 = data[f"block.{i}.ffn.W_1"].copy()
            ffn.W_2 = data[f"block.{i}.ffn.W_2"].copy()
            ffn.b_1 = data[f"block.{i}.ffn.b_1"].copy()
            ffn.b_2 = data[f"block.{i}.ffn.b_2"].copy()
        elif hasattr(ffn, "W_gate"):
            ffn.W_gate = data[f"block.{i}.ffn.W_gate"].copy()
            ffn.W_up = data[f"block.{i}.ffn.W_up"].copy()
            ffn.W_down = data[f"block.{i}.ffn.W_down"].copy()

    # Re-establish weight tying
    model.lm_head = model.token_embedding.weight

    return model
