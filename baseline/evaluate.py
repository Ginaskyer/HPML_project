"""
Evaluation script: compute perplexity on WikiText-2 test set.
Supports: full fine-tuned model.

Usage:
  python evaluate.py
  python evaluate.py --ckpt_path outputs/best --precision bf16
"""

import argparse
import math
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from config import TrainConfig
from model_utils import load_model_and_tokenizer, load_finetuned_weights
from train import tokenize_and_chunk, collate_fn


@torch.no_grad()
def compute_perplexity(model, dataloader, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    start_time = time.time()
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)

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
    parser = argparse.ArgumentParser(description="Evaluate full fine-tuning model")
    parser.add_argument("--ckpt_path", type=str, default="outputs/best")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp32"])
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.precision = args.precision

    # Load dataset
    print("Loading WikiText-2 test set...")
    raw_dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)

    # tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Optional: load fine-tuned weights
    if args.ckpt_path is not None:
        model = load_finetuned_weights(model, args.ckpt_path)

    test_dataset = tokenize_and_chunk(raw_dataset["test"], tokenizer, cfg.block_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load fine-tuned model
    print("\n" + "=" * 60)
    print(f"Evaluating FULL FINE-TUNED model ({args.precision})")
    print("=" * 60)

    results = compute_perplexity(model, test_loader)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Loss:             {results['loss']:.4f}")
    print(f"Perplexity:       {results['perplexity']:.2f}")
    print(f"Inference time:   {results['time_seconds']:.2f}s")
    print(f"Tokens/sec:       {results['tokens_per_second']:.0f}")
    print(f"Total tokens:     {results['total_tokens']:,}")


if __name__ == "__main__":
    main()