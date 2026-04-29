"""
Model loading, and weight save/load utilities.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TrainConfig

def print_trainable_params(model):
    """Print the number of trainable vs total parameters."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable:,} | Total params: {total:,} | "
          f"Trainable%: {100 * trainable / total:.4f}%")


def get_torch_dtype(precision: str):
    """Map precision string to torch dtype."""
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'bf16' or 'fp32'.")


def prepare_model(cfg: TrainConfig):
    """Full pipeline: load full model for full-parameter training."""
    model, tokenizer = load_model_and_tokenizer(cfg)

    print("Full fine-tuning mode: all model parameters are trainable.")
    print_trainable_params(model)

    return model, tokenizer


def load_model_and_tokenizer(cfg: TrainConfig):
    """Load the full model and tokenizer."""
    torch_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def save_full_checkpoint(model, tokenizer, save_dir):
    """Save full fine-tuned model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

import os
import torch


def load_finetuned_weights(model, ckpt_path):
    model_state = dict(model.named_parameters())

    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
    safe_path = os.path.join(ckpt_path, "model.safetensors")

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        weight_path = bin_path
    else:
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
        weight_path = safe_path

    for name, weight in state_dict.items():
        if name in model_state:
            model_state[name].data.copy_(weight.to(model_state[name].device))

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    print(f"Loaded {len(state_dict)} tensors from {weight_path}")
    return model