"""
Evaluation script: compute perplexity on WikiText-2 test set.
Supports: base model (no LoRA), QLoRA with residual branches.

Usage:
  python evaluate.py                         # Evaluate with LoRA (best checkpoint)
  python evaluate.py --baseline              # Evaluate base quantized model only
  python evaluate.py --lora_path outputs/final  # Specify checkpoint path
"""

import argparse
import math
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from config import TrainConfig
from model_utils import (
    load_model_and_tokenizer, inject_lora, freeze_base_model,
    load_lora_weights, print_trainable_params,
)
from train import tokenize_and_chunk, collate_fn


@torch.no_grad()
def compute_perplexity(model, dataloader, desc="Evaluating"):
    """Compute perplexity over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    start_time = time.time()
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)

        # Count actual tokens (exclude padding if any)
        num_tokens = batch["labels"].numel()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    elapsed = time.time() - start_time
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "time_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate QLoRA model")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate base quantized model without LoRA")
    parser.add_argument("--lora_path", type=str, default="outputs/best",
                        help="Path to saved LoRA weights")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg = TrainConfig()

    # Load dataset
    print("Loading WikiText-2 test set...")
    raw_dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)

    # Load quantized model
    print("Loading quantized base model...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    test_dataset = tokenize_and_chunk(raw_dataset["test"], tokenizer, cfg.block_size)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    results = {}

    if args.baseline:
        # Evaluate base model only
        print("\n" + "=" * 60)
        print("Evaluating BASE quantized model (no LoRA)")
        print("=" * 60)
        results["baseline"] = compute_perplexity(model, test_loader)
    else:
        # --- Baseline first ---
        print("\n" + "=" * 60)
        print("Evaluating BASE quantized model (no LoRA)")
        print("=" * 60)
        results["baseline"] = compute_perplexity(model, test_loader)

        # --- Inject LoRA and load weights ---
        print("\n" + "=" * 60)
        print(f"Evaluating QLoRA model (loading from {args.lora_path})")
        print("=" * 60)
        inject_lora(model, cfg.lora)
        freeze_base_model(model)
        load_lora_weights(model, args.lora_path)
        print_trainable_params(model)

        results["qlora"] = compute_perplexity(model, test_loader)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n[{name.upper()}]")
        print(f"  Loss:             {r['loss']:.4f}")
        print(f"  Perplexity:       {r['perplexity']:.2f}")
        print(f"  Inference time:   {r['time_seconds']:.2f}s")
        print(f"  Tokens/sec:       {r['tokens_per_second']:.0f}")
        print(f"  Total tokens:     {r['total_tokens']:,}")

    if "baseline" in results and "qlora" in results:
        base_ppl = results["baseline"]["perplexity"]
        qlora_ppl = results["qlora"]["perplexity"]
        print(f"\nPerplexity change: {base_ppl:.2f} -> {qlora_ppl:.2f} "
              f"({'improved' if qlora_ppl < base_ppl else 'degraded'})")
        speedup = results["baseline"]["tokens_per_second"] / results["qlora"]["tokens_per_second"]
        print(f"Throughput ratio (base/qlora): {speedup:.2f}x")


if __name__ == "__main__":
    main()
