"""
optimized_model.py — INT4 inference matching evaluate.py pipeline.

int4_base : Qwen/Qwen3-0.6B → BnB NF4 int4            (no LoRA)
int4      : Qwen/Qwen3-0.6B → BnB NF4 int4 → inject lora_weights.pt
"""

import os
import sys

import torch
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# HPML LoRA utilities
# ---------------------------------------------------------------------------
_HPML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HPML_project-main"))
if _HPML_DIR not in sys.path:
    sys.path.insert(0, _HPML_DIR)

from config import TrainConfig
from model_utils import inject_lora, freeze_base_model, load_lora_weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_CKPT_DIR = os.path.join(_ROOT, "lora")   # contains lora_weights.pt

QuantMode = Literal["int4_base", "int4"]

_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_model(quant_mode: QuantMode = "int4", device: str = "cuda"):
    """
    Mirrors evaluate.py:
      1. Load Qwen/Qwen3-0.6B with BnB NF4 int4 (same as load_model_and_tokenizer)
      2. For int4: inject LoRA + load lora_weights.pt  (same as inject_lora / load_lora_weights)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Optimized model requires CUDA (int4 quantization is GPU-only)")

    cfg = TrainConfig()   # cfg.model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    if quant_mode == "int4":
        inject_lora(model, cfg.lora)
        freeze_base_model(model)
        load_lora_weights(model, LORA_CKPT_DIR)

    model.eval()
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128,
) -> dict:
    """
    Run a single generation pass.

    Returns:
        {
            "output_text":    str,
            "input_ids_len":  int,
            "output_ids_len": int,
            "logits":         torch.Tensor,  # [T, vocab_size], float32, CPU
            "output_ids":     torch.Tensor,  # [T], int64, CPU
        }
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            output_scores=True,
            return_dict_in_generate=True,
        )

    output_ids = outputs.sequences[0, input_ids_len:].cpu()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    logits = torch.stack(outputs.scores, dim=0).squeeze(1).cpu().float()

    return {
        "output_text":    output_text,
        "input_ids_len":  input_ids_len,
        "output_ids_len": output_ids.shape[0],
        "logits":         logits,
        "output_ids":     output_ids,
    }
