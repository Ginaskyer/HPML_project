"""
Model loading, LoRA injection, and weight save/load utilities.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora import LoRALinear
from config import TrainConfig


def get_bnb_config(quant_cfg):
    """Create BitsAndBytesConfig from our QuantConfig."""
    compute_dtype = getattr(torch, quant_cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(cfg: TrainConfig):
    """Load the quantized model and tokenizer."""
    bnb_config = get_bnb_config(cfg.quant)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def inject_lora(model, lora_cfg):
    """
    Replace target linear layers with LoRALinear wrappers.
    Returns the list of injected LoRA module names for reference.
    """
    injected = []

    for name, module in model.named_modules():
        # Check if any target module name matches the current module name
        for target in lora_cfg.target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear) or \
               (hasattr(module, 'in_features') and hasattr(module, 'out_features') and name.endswith(target)):
                # Get the parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name

                # Detect device of the base layer
                if hasattr(module, 'weight'):
                    device = module.weight.device
                else:
                    # For quantized layers, weight may be stored differently
                    device = next(module.parameters()).device

                # Replace with LoRA-wrapped version, placed on same device
                lora_layer = LoRALinear(
                    base_layer=module,
                    rank=lora_cfg.rank,
                    alpha=lora_cfg.alpha,
                    dropout=lora_cfg.dropout,
                )
                lora_layer.lora_A = nn.Parameter(lora_layer.lora_A.to(device))
                lora_layer.lora_B = nn.Parameter(lora_layer.lora_B.to(device))
                setattr(parent, attr_name, lora_layer)
                injected.append(name)

    return injected


def freeze_base_model(model):
    """Freeze all parameters, then unfreeze only LoRA parameters."""
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True


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


def save_lora_weights(model, save_path):
    """Save only the LoRA adapter weights."""
    os.makedirs(save_path, exist_ok=True)
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.cpu()

    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pt"))
    print(f"Saved {len(lora_state_dict)} LoRA tensors to {save_path}/lora_weights.pt")


def load_lora_weights(model, load_path):
    """Load LoRA adapter weights into the model."""
    weight_path = os.path.join(load_path, "lora_weights.pt")
    lora_state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    model_state = dict(model.named_parameters())
    loaded = 0
    for name, weight in lora_state_dict.items():
        if name in model_state:
            model_state[name].data.copy_(weight.to(model_state[name].device))
            loaded += 1
        else:
            print(f"Warning: {name} not found in model")

    print(f"Loaded {loaded}/{len(lora_state_dict)} LoRA tensors from {weight_path}")


def prepare_model(cfg: TrainConfig):
    """Full pipeline: load quantized model, inject LoRA, freeze base."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    injected = inject_lora(model, cfg.lora)
    freeze_base_model(model)

    print(f"Injected LoRA into {len(injected)} layers: {injected[:3]}...")
    print_trainable_params(model)

    return model, tokenizer
