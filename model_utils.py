"""
Model loading, LoRA injection, and weight save/load utilities.
"""

import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora import LoRALinear
from config import TrainConfig
from rotation import fuse_rotation, fuse_weight, load_or_create_R1

def get_bnb_config(quant_cfg):
    """Create BitsAndBytesConfig from our QuantConfig."""
    compute_dtype = getattr(torch, quant_cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(cfg: TrainConfig, quantize: bool = True):
    """Load the model (optionally 4-bit quantized) and tokenizer."""
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if quantize:
        kwargs["quantization_config"] = get_bnb_config(cfg.quant)
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def quantize_linears_to_4bit(model, quant_cfg, skip_names=("lm_head",)):
    """Replace every nn.Linear (outside skip_names) with bnb.nn.Linear4bit.

    Must be called after any offline weight transformation (e.g. rotation fusion)
    since 4-bit Params4bit stores weights in a packed layout that forbids
    direct matmul on .data.
    """
    compute_dtype = getattr(torch, quant_cfg.bnb_4bit_compute_dtype)

    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(s in name for s in skip_names):
            to_replace.append((name, module))

    for name, module in to_replace:
        parent_name, _, attr_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        device = module.weight.device

        new_linear = bnb.nn.Linear4bit(
            input_features=module.in_features,
            output_features=module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=quant_cfg.bnb_4bit_use_double_quant,
            quant_type=quant_cfg.bnb_4bit_quant_type,
        )
        new_linear.weight = bnb.nn.Params4bit(
            module.weight.data,
            requires_grad=False,
            compress_statistics=quant_cfg.bnb_4bit_use_double_quant,
            quant_type=quant_cfg.bnb_4bit_quant_type,
        )
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        new_linear = new_linear.to(device)
        setattr(parent, attr_name, new_linear)
        del module
    torch.cuda.empty_cache()


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
    """Full pipeline: load bf16 model -> fuse norms -> rotate -> quantize 4-bit
    -> inject LoRA -> freeze base."""
    # Load in bf16 first: rotation/norm-fusion must see full-precision weights,
    # not the packed Params4bit layout produced by BitsAndBytesConfig.
    model, tokenizer = load_model_and_tokenizer(cfg, quantize=False)
    print(model)

    # Untie lm_head from embed_tokens before any offline weight transform.
    # Tied weights would cause fuse_weight to leak the final RMSNorm scale into
    # the embedding table and rotate_embeddings+rotate_head to double-rotate the
    # shared matrix.
    if getattr(model.config, "tie_word_embeddings", False):
        embed_weight = model.model.embed_tokens.weight
        if model.lm_head.weight is embed_weight:
            model.lm_head.weight = nn.Parameter(embed_weight.data.clone())
        model.config.tie_word_embeddings = False

    fuse_weight(model)

    R1 = load_or_create_R1(mode="online",
                           device="cuda",
                           dim=model.config.hidden_size)
    R1 = R1.weight.detach()
    gpu_count = torch.cuda.device_count()
    R1_per_gpu = {}
    if gpu_count > 1:
        print(f"[INFO] Detected {gpu_count} GPUs, creating R1 copies...")
        for i in range(gpu_count):
            R1_per_gpu[f"cuda:{i}"] = R1.to(f"cuda:{i}")
            print(f"[INFO] R1 copied to cuda:{i}")
    else:
        R1_per_gpu["cuda:0"] = R1.to(device="cuda")
    fuse_rotation(model, R1_per_gpu, None)

    if cfg.quant.load_in_4bit:
        quantize_linears_to_4bit(model, cfg.quant)

    injected = inject_lora(model, cfg.lora)
    freeze_base_model(model)

    print(f"Injected LoRA into {len(injected)} layers: {injected[:3]}...")
    print_trainable_params(model)

    return model, tokenizer
