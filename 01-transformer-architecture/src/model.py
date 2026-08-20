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
from transformer_block import DecoderBlock, EncoderBlock, layer_norm


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

        # Expand mask to (batch, 1, 1, src_len) for attention.
        attn_mask = None
        if src_mask is not None:
            attn_mask = src_mask[:, np.newaxis, np.newaxis, :]

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
            tgt_mask_expanded = tgt_mask[:, np.newaxis, np.newaxis, :]
            causal_mask = causal_mask * tgt_mask_expanded

        cross_mask = None
        if src_mask is not None:
            cross_mask = src_mask[:, np.newaxis, np.newaxis, :]

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
            x = x + self.pos_embedding.forward(seq_len)
        else:
            x = x + self.pos_embedding[:, :seq_len, :]

        # Causal mask: can only attend to previous positions.
        causal_mask = create_causal_mask(seq_len)
        if mask is not None:
            mask_expanded = mask[:, np.newaxis, np.newaxis, :]
            causal_mask = causal_mask * mask_expanded

        # Decoder blocks use a dummy encoder output for cross-attention.
        # In a decoder-only model, cross-attention is skipped by passing None.
        # We pass zeros so the block structure is preserved.
        dummy_encoder = np.zeros((batch_size, 1, self.d_model), dtype=np.float64)

        for block in self.blocks:
            x = block.forward(
                x,
                dummy_encoder,
                causal_mask=causal_mask,
                cross_mask=None,
                training=training,
            )

        x = layer_norm(x)
        return np.matmul(x, self.lm_head.T)

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
