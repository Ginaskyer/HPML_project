import time
import math
from typing import Callable, Optional

import torch

try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
        NVMLError,
    )
    nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gpu_memory_used_mb() -> float:
    """Returns current GPU memory usage in MB, or 0 if unavailable."""
    if not _NVML_AVAILABLE:
        return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    handle = nvmlDeviceGetHandleByIndex(0)
    return nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 2


def gpu_memory_used_mb() -> float:
    """Public wrapper — returns current total GPU memory used in MB via NVML."""
    return _gpu_memory_used_mb()


def model_memory_mb(model: torch.nn.Module) -> float:
    """
    Compute model weight memory in MB from parameter storage.

    Works for both BF16/FP32 and BnB int4 quantized models.
    For BnB Params4bit, numel() returns the packed uint8 count (half of
    original element count), so numel() * element_size() correctly
    reflects 0.5 bytes per 4-bit weight without any special casing.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / 1024**2, 2)


def _gpu_peak_memory_mb() -> float:
    """
    Returns the peak GPU memory allocated (in MB) since the last reset.

    Uses torch.cuda.max_memory_allocated() which tracks the high-water mark
    of *all* allocations (model weights + activations + KV-cache).  This is
    more meaningful than a before/after delta, which misses memory that was
    already allocated before the timed section.

    Falls back to current NVML usage if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return _gpu_memory_used_mb()


def _compute_perplexity(
    logits: torch.Tensor, output_ids: torch.Tensor
) -> Optional[float]:
    """
    logits    : [T, vocab_size]  float32 on CPU
    output_ids: [T]              int64   on CPU  (the generated token ids)
    Returns per-token perplexity of the generated sequence.
    """
    if logits is None or output_ids is None:
        return None
    try:
        shift_logits = logits[:-1]      # [T-1, vocab]
        shift_labels = output_ids[1:]   # [T-1]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.float(), shift_labels.long(), reduction="mean"
        )
        return math.exp(loss.item())
    except Exception:
        return None


def estimate_flops_per_token(
    model: torch.nn.Module, input_len: int
) -> Optional[float]:
    """
    Bit-width-aware GFLOPs/token estimate.

    Standard heuristic:  GFLOPs ≈ 2 × N_params (for one token).
    For quantized models we weight each parameter by its *actual* storage
    bit-width relative to FP16 (16 bits), so INT4 parameters count as
    4/16 = 0.25 of an FP16 parameter.  This makes INT4 and BF16 models
    directly comparable on a normalised compute scale.

    Note: this is still a theoretical lower bound; it does not account for
    attention FLOPs (O(seq_len²)) or per-layer activation sizes.
    """
    try:
        effective_params = 0.0
        for p in model.parameters():
            bits = p.element_size() * 8          # bytes → bits
            effective_params += p.numel() * (bits / 16)   # normalise to fp16

        gflops = 2 * effective_params / 1e9
        return round(gflops, 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warmup(
    generate_fn: Callable,
    tokenizer,
    model: torch.nn.Module,
    device: str = "cuda",
    max_new_tokens: int = 5,
) -> None:
    """
    Run one short inference pass to flush CUDA kernel compilation overhead.

    Call this once per model *before* benchmarking, so that latency numbers
    are not inflated by first-run JIT/kernel-launch costs.

    Example
    -------
    warmup(my_generate_fn, tokenizer, model)
    metrics = collect_metrics(my_generate_fn, tokenizer, model, prompt)
    """
    with torch.no_grad():
        generate_fn(tokenizer, model, "warmup", max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_metrics(
    generate_fn: Callable,
    tokenizer,
    model: torch.nn.Module,
    prompt: str,
    max_new_tokens: int = 128,
) -> dict:
    """
    Wraps generate_fn and returns all metrics.

    generate_fn signature (what teammates implement):
        generate_fn(tokenizer, model, prompt, max_new_tokens) -> {
            "output_text":    str,
            "input_ids_len":  int,
            "output_ids_len": int,
            "logits":         torch.Tensor | None,  # [T, vocab], float32, CPU
            "output_ids":     torch.Tensor | None,  # [T], int64, CPU
        }

    Returns
    -------
    {
        "output_text":     str,
        "latency_ms":      float,          # wall-clock inference time
        "throughput_tps":  float,          # output tokens / second
        "gpu_memory_mb":   float,          # peak GPU allocation during run
        "perplexity":      float | None,
        "gflops_per_tok":  float | None,   # bit-width-normalised estimate
    }

    Notes
    -----
    * Call warmup() once per model before collect_metrics() to avoid
      including CUDA kernel compilation in latency numbers.
    * gpu_memory_mb reflects the peak allocation since the last call to
      torch.cuda.reset_peak_memory_stats(), which is reset at the start
      of every collect_metrics() call.
    """
    # Reset peak tracker so we measure only this inference run.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = generate_fn(tokenizer, model, prompt, max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Peak memory during this inference pass (model + KV-cache + activations).
    gpu_memory_mb = _gpu_peak_memory_mb()

    n_out = result["output_ids_len"]
    throughput = n_out / elapsed if elapsed > 0 else 0.0

    perplexity = _compute_perplexity(
        result.get("logits"), result.get("output_ids")
    )

    gflops_per_tok = estimate_flops_per_token(model, result["input_ids_len"])

    return {
        "output_text":    result["output_text"],
        "latency_ms":     round(elapsed * 1000, 2),
        "throughput_tps": round(throughput, 2),
        "gpu_memory_mb":  round(gpu_memory_mb, 2),
        "perplexity":     round(perplexity, 4) if perplexity else None,
        "gflops_per_tok": gflops_per_tok,
    }
