"""
baseline_model.py — OWNED BY TEAMMATE A
Implements FP16 Qwen3-0.6B with NO quantization and NO LoRA.

Teammate A: fill in load_model() and generate() below.
Do NOT add timing or memory measurement here — the backend handles that.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME_OR_PATH = os.path.join(_ROOT, "bf16", "best")


def load_model(device: str = "cuda"):
    """
    Load tokenizer and model into memory.
    Called once at server startup.

    Returns:
        tokenizer: PreTrainedTokenizer
        model:     PreTrainedModel  (bfloat16, on `device`)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
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
            "output_text":    str,           # generated text only (not prompt)
            "input_ids_len":  int,           # number of input tokens
            "output_ids_len": int,           # number of generated tokens
            "logits":         torch.Tensor,  # [output_ids_len, vocab_size], float32, CPU
                                             # set to None if memory is a concern
            "output_ids":     torch.Tensor,  # [output_ids_len], int64, CPU
                                             # set to None if logits is None
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
            output_scores=True,
            return_dict_in_generate=True,
        )

    # generated token ids (excluding prompt)
    output_ids = outputs.sequences[0, input_ids_len:].cpu()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # stack per-step logits → [T, vocab]
    logits = torch.stack(outputs.scores, dim=0).squeeze(1).cpu().float()  # [T, vocab]

    return {
        "output_text":    output_text,
        "input_ids_len":  input_ids_len,
        "output_ids_len": output_ids.shape[0],
        "logits":         logits,
        "output_ids":     output_ids,
    }
