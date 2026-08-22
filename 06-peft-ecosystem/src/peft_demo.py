"""PEFT method comparison demo.

Compares the parameter efficiency of different PEFT methods:
    - Full fine-tuning (baseline)
    - Adapters
    - LoRA
    - Prefix Tuning
    - IA3

Run with:
    PYTHONPATH=06-peft-ecosystem/src:05-qlora/src:04-lora/src:03-adapters/src:01-transformer-architecture/src \\
        uv run python 06-peft-ecosystem/src/peft_demo.py
"""

import sys
from pathlib import Path

for sub in ["01-transformer-architecture", "03-adapters", "04-lora", "05-qlora"]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / sub / "src"))

from model import GPT
from tokenizer import BPETokenizer


def count_base_params(model: GPT) -> int:
    """Counts total parameters in a GPT model."""
    return sum(p.size for p in model.get_params().values())


def count_adapter_params(d_model: int, bottleneck: int, num_layers: int) -> int:
    """Counts parameters for Houlsby adapters (2 per block)."""
    # Per adapter: W_down (d×b) + W_up (b×d) + biases
    per_adapter = 2 * d_model * bottleneck + d_model + bottleneck
    return 2 * num_layers * per_adapter  # 2 adapters per block


def count_lora_params(d_model: int, rank: int, num_layers: int) -> int:
    """Counts LoRA parameters (applied to W_q, W_k, W_v, W_o)."""
    # Per matrix: A (d×r) + B (r×d)
    per_matrix = 2 * d_model * rank
    return 4 * num_layers * per_matrix  # 4 matrices per block


def count_prefix_params(d_model: int, num_prefix: int, num_layers: int) -> int:
    """Counts prefix tuning parameters."""
    return 2 * num_layers * num_prefix * d_model


def count_ia3_params(d_model: int, num_layers: int) -> int:
    """Counts IA3 parameters (l_k, l_v, l_ff per layer)."""
    return 3 * num_layers * d_model


def main() -> None:
    """Prints a comparison of PEFT method parameter counts."""
    # Example: Llama 7B-like model
    d_model = 4096
    num_layers = 32
    bottleneck = 64  # typical adapter bottleneck
    rank = 16  # typical LoRA rank
    num_prefix = 10  # typical prefix length

    # Simulate a full model for baseline
    tok = BPETokenizer(vocab_size=32000)
    tok.train(["placeholder text for tokenizer"])
    model = GPT(
        vocab_size=len(tok), d_model=d_model, num_heads=32,
        num_layers=num_layers, d_ff=11008, max_len=2048, dropout_rate=0.0,
    )
    full_params = count_base_params(model)

    adapter_params = count_adapter_params(d_model, bottleneck, num_layers)
    lora_params = count_lora_params(d_model, rank, num_layers)
    prefix_params = count_prefix_params(d_model, num_prefix, num_layers)
    ia3_params = count_ia3_params(d_model, num_layers)

    methods = [
        ("Full Fine-Tuning", full_params, 0.0),
        ("Adapters (b=64)", adapter_params, 0.0),
        ("LoRA (r=16)", lora_params, 0.0),
        ("Prefix Tuning (m=10)", prefix_params, 0.0),
        ("IA³", ia3_params, 0.0),
    ]

    print("=" * 70)
    print("PEFT Method Comparison (d_model=4096, layers=32)")
    print("=" * 70)
    print(f"{'Method':<30} {'Trainable':>12} {'% of Full':>10} {'Storage':>10}")
    print("-" * 70)

    for name, params, _ in methods:
        pct = params / full_params * 100
        storage_mb = params * 8 / (1024 * 1024)  # FP64 bytes → MB
        print(f"{name:<30} {params:>12,} {pct:>9.2f}% {storage_mb:>9.1f} MB")

    print("-" * 70)
    print("\nKey insight: PEFT methods train 0.01-8% of parameters")
    print("vs full fine-tuning, enabling fine-tuning on consumer GPUs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
