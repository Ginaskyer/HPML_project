"""
Evaluation script: compute perplexity on WikiText-2 test set.

Usage:
  python evaluate.py --merged_path outputs/merged   # Load HF merged checkpoint
  python evaluate.py                                # QLoRA eval (best LoRA + baseline)
  python evaluate.py --baseline                     # Base quantized model only
  python evaluate.py --lora_path outputs/final      # Specify LoRA checkpoint path
"""

import argparse
import math
import time
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TrainConfig
from model_utils import (
    load_model_and_tokenizer, inject_lora, freeze_base_model,
    load_lora_weights, print_trainable_params,
)
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


def load_merged_checkpoint(path):
    """Load a HuggingFace merged checkpoint saved by save_merged_model()."""
    print(f"Loading merged HF checkpoint from {path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def print_results(results):
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

    if "baseline" in results and len(results) > 1:
        base_ppl = results["baseline"]["perplexity"]
        other_key = next(k for k in results if k != "baseline")
        other_ppl = results[other_key]["perplexity"]
        print(f"\nPerplexity change: {base_ppl:.2f} -> {other_ppl:.2f} "
              f"({'improved' if other_ppl < base_ppl else 'degraded'})")
        speedup = results["baseline"]["tokens_per_second"] / results[other_key]["tokens_per_second"]
        print(f"Throughput ratio (base/{other_key}): {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Evaluate QLoRA / merged model")
    parser.add_argument("--merged_path", type=str, default=None,
                        help="Path to a merged HF checkpoint (saved by save_merged_model). "
                             "When set, all other loading flags are ignored.")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate base quantized model without LoRA")
    parser.add_argument("--lora_path", type=str, default="outputs/best",
                        help="Path to saved LoRA weights (.pt)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg = TrainConfig()

    print("Loading WikiText-2 test set...")
    raw_dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)

    results = {}

    if args.merged_path:
        # ------------------------------------------------------------------
        # Mode: load a fully-merged HF checkpoint
        # ------------------------------------------------------------------
        model, tokenizer = load_merged_checkpoint(args.merged_path)

        test_dataset = tokenize_and_chunk(raw_dataset["test"], tokenizer, cfg.block_size)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, pin_memory=True,
        )
        print(f"Test samples: {len(test_dataset)}")

        print("\n" + "=" * 60)
        print(f"Evaluating MERGED model ({args.merged_path})")
        print("=" * 60)
        results["merged"] = compute_perplexity(model, test_loader)

    else:
        # ------------------------------------------------------------------
        # Mode: quantized base model (+ optional LoRA)
        # ------------------------------------------------------------------
        print("Loading quantized base model...")
        model, tokenizer = load_model_and_tokenizer(cfg)

        if not args.baseline:
            print("Loading quantized base model...")
            model, tokenizer = prepare_model(cfg)
            load_lora_weights(model, args.lora_path)

        test_dataset = tokenize_and_chunk(raw_dataset["test"], tokenizer, cfg.block_size)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, pin_memory=True,
        )
        print(f"Test samples: {len(test_dataset)}")

        print("\n" + "=" * 60)
        print("Evaluating model")
        print("=" * 60)
        results["baseline"] = compute_perplexity(model, test_loader)

    print_results(results)


if __name__ == "__main__":
    main()
