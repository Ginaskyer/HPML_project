"""
Full fine-tuning training script on WikiText-2.
Usage: python train.py  --precision bf16
"""

import argparse
import math
import os
import time
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler

from config import TrainConfig
from model_utils import prepare_model, save_full_checkpoint, cuda_sync, get_gpu_memory_gb, reset_gpu_memory


def tokenize_and_chunk(dataset, tokenizer, block_size):
    """Tokenize text and create fixed-length chunks for causal LM training."""

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    chunked = tokenized.map(
        group_texts,
        batched=True,
        desc="Chunking",
    )
    return chunked


def collate_fn(batch):
    """Collate batch into tensors."""
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


def train(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Load model
    print("=" * 60)
    print(f"Loading full-parameter model for fine-tuning ({cfg.precision})...")
    print("=" * 60)
    model, tokenizer = prepare_model(cfg)

    # Load dataset
    print("\nLoading WikiText-2 dataset...")
    raw_dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
    train_dataset = tokenize_and_chunk(raw_dataset["train"], tokenizer, cfg.block_size)
    val_dataset = tokenize_and_chunk(raw_dataset["validation"], tokenizer, cfg.block_size)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Full fine-tuning optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler
    num_training_steps = cfg.num_epochs * len(train_loader) // cfg.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    use_amp = (cfg.precision.lower() == "bf16" and model.device.type == "cuda")
    scaler = None  # bf16 does not need GradScaler

    # -----------------------------
    # GPU Warmup (not counted)
    # -----------------------------
    print("\nWarming up GPU...")
    model.train()
    warmup_steps = min(5, len(train_loader))
    warmup_iter = iter(train_loader)

    for _ in range(warmup_steps):
        batch = next(warmup_iter)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        loss.backward()
        optimizer.zero_grad()

    cuda_sync()
    print(f"Warmup finished: {warmup_steps} steps")

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting full fine-tuning for {cfg.num_epochs} epochs")
    print(f"Precision: {cfg.precision}")
    print(f"Total optimization steps: {num_training_steps}")
    print(f"{'=' * 60}\n")

    global_step = 0
    best_val_loss = float("inf")

    reset_gpu_memory()
    cuda_sync()
    training_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        cuda_sync()
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / cfg.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch + 1}/{cfg.num_epochs} | "
                        f"Step {global_step}/{num_training_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

        # Epoch validation
        val_loss = evaluate_loss(model, val_loader, cfg.precision)
        cuda_sync()
        epoch_time = time.time() - epoch_start
        peak_allocated, peak_reserved = get_gpu_memory_gb()
        ppl = math.exp(val_loss) if val_loss < 100 else float("inf")
        print(
            f"\n--- Epoch {epoch + 1} done in {epoch_time:.1f}s | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {ppl:.2f} | "
            f"Peak Allocated: {peak_allocated:.2f} GB | "
            f"Peak Reserved: {peak_reserved:.2f} GB ---\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_full_checkpoint(model, tokenizer, os.path.join(cfg.output_dir, "best"))

    cuda_sync()
    total_training_time = time.time() - training_start
    peak_allocated, peak_reserved = get_gpu_memory_gb()

    print("\n" + "=" * 60)
    print("FULL FINETUNING TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total training time: {total_training_time:.1f}s")
    print(f"Total training time: {total_training_time / 60:.2f} min")
    print(f"Peak GPU allocated memory: {peak_allocated:.2f} GB")
    print(f"Peak GPU reserved memory:  {peak_reserved:.2f} GB")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    # Save final checkpoint
    save_full_checkpoint(model, tokenizer, os.path.join(cfg.output_dir, "final"))


@torch.no_grad()
def evaluate_loss(model, dataloader, precision):
    """Compute average loss on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    use_amp = (precision.lower() == "bf16" and model.device.type == "cuda")

    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
        else:
            outputs = model(**batch)

        total_loss += outputs.loss.item()
        total_steps += 1

    return total_loss / total_steps


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Training precision",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.precision = args.precision
    cfg.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()
