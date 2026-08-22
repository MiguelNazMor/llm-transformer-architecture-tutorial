"""NF4-style 4-bit quantization.

Implements a simplified NormalFloat4 quantization for demonstration:
    - Computes 16 quantization levels optimized for Gaussian-distributed data
    - Quantizes weights block-by-block (64 values per block)
    - Dequantizes on-the-fly during forward pass

This is a pedagogical implementation showing the core concepts.
Production systems use bitsandbytes which has optimized CUDA kernels.

Key concepts:
    - NF4: non-uniform quantization levels (denser near zero)
    - Block-wise: each block of 64 weights gets its own scale
    - Dequantization: X_fp = X_int * scale (linear dequant)
"""

import numpy as np
from numpy.typing import NDArray

BLOCK_SIZE = 64  # standard block size for 4-bit quantization


def _nf4_quantization_levels() -> NDArray[np.float64]:
    """Returns the 16 NF4 quantization levels.

    NF4 uses levels optimized for a standard normal distribution.
    The levels are the quantiles of N(0,1) at positions that minimize
    quantization error.

    Returns:
        Array of 16 quantization levels, sorted ascending.
    """
    # Normalize the levels so they have unit variance
    # (standard practice for NF4).
    levels = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
        -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ], dtype=np.float64)
    return levels


def quantize_nf4(weight: NDArray[np.float64]) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Quantizes a weight matrix to NF4.

    Weights are quantized in blocks of BLOCK_SIZE.  Each block gets its
    own scale factor computed as the maximum absolute value in the block.

    Args:
        weight: Float64 weight matrix of any shape (flattened internally).

    Returns:
        Tuple of (quantized_weights, scales):
        - quantized_weights: uint8 array of 4-bit indices (2 per byte)
        - scales: float64 array of per-block scale factors
    """
    flat = weight.ravel()
    n = flat.size
    levels = _nf4_quantization_levels()

    # Pad to multiple of BLOCK_SIZE.
    padded_size = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = np.zeros(padded_size, dtype=np.float64)
    padded[:n] = flat

    n_blocks = padded_size // BLOCK_SIZE
    scales = np.zeros(n_blocks, dtype=np.float64)
    # uint8 array: each byte stores two 4-bit indices
    quantized = np.zeros((padded_size + 1) // 2, dtype=np.uint8)

    for b in range(n_blocks):
        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = padded[start:end]

        # Scale: max absolute value in block
        scale = np.max(np.abs(block))
        if scale < 1e-8:
            scale = 1.0  # avoid division by zero for all-zero blocks
        scales[b] = scale

        # Normalize to [-1, 1] range.
        normalized = np.clip(block / scale, -1.0, 1.0)

        # Find nearest NF4 level for each weight.
        indices = np.zeros(BLOCK_SIZE, dtype=np.uint8)
        for i in range(BLOCK_SIZE):
            val = normalized[i]
            # Find nearest level
            idx = int(np.argmin(np.abs(levels - val)))
            indices[i] = idx

        # Pack two 4-bit indices per byte.
        for i in range(0, BLOCK_SIZE, 2):
            byte_idx = start // 2 + i // 2
            quantized[byte_idx] = (indices[i] & 0x0F) | ((indices[i + 1] & 0x0F) << 4)

    return quantized, scales


def dequantize_nf4(
    quantized: NDArray[np.uint8],
    scales: NDArray[np.float64],
    original_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Dequantizes NF4 weights back to float64.

    Args:
        quantized: uint8 array of packed 4-bit indices.
        scales: Per-block scale factors.
        original_shape: Shape of the original weight matrix.

    Returns:
        Dequantized float64 array with the original shape.
    """
    levels = _nf4_quantization_levels()
    n_original = int(np.prod(original_shape))
    padded_size = ((n_original + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    result = np.zeros(padded_size, dtype=np.float64)

    for b in range(len(scales)):
        start = b * BLOCK_SIZE
        scale = scales[b]

        for i in range(0, BLOCK_SIZE, 2):
            byte_idx = start // 2 + i // 2
            byte_val = quantized[byte_idx]
            idx_lo = byte_val & 0x0F
            idx_hi = (byte_val >> 4) & 0x0F

            result[start + i] = levels[idx_lo] * scale
            if start + i + 1 < padded_size:
                result[start + i + 1] = levels[idx_hi] * scale

    return result[:n_original].reshape(original_shape)


def quantize_model_weights(model) -> dict:
    """Quantizes all weight matrices in a GPT model to NF4.

    Args:
        model: A GPT model instance.

    Returns:
        Dict mapping parameter names to (quantized, scales, shape) tuples.
    """
    quantized_weights: dict = {}
    params = model.get_params()
    for name, param in params.items():
        q, s = quantize_nf4(param)
        quantized_weights[name] = (q, s, param.shape)
    return quantized_weights


def count_quantized_memory(original_params: int) -> tuple[float, float, float]:
    """Estimates memory usage of original vs quantized weights.

    Args:
        original_params: Total number of float64 parameters.

    Returns:
        Tuple of (fp64_mb, nf4_mb, reduction_pct).
    """
    fp64_bytes = original_params * 8
    # 4-bit: 0.5 bytes per weight + scales (float64 per 64 weights)
    nf4_weight_bytes = original_params * 0.5
    n_blocks = (original_params + BLOCK_SIZE - 1) // BLOCK_SIZE
    nf4_scale_bytes = n_blocks * 8  # float64 per block
    nf4_total = nf4_weight_bytes + nf4_scale_bytes

    fp64_mb = fp64_bytes / (1024 * 1024)
    nf4_mb = nf4_total / (1024 * 1024)
    reduction = (1 - nf4_total / fp64_bytes) * 100
    return fp64_mb, nf4_mb, reduction
