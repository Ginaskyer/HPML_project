# QLoRA Finetuning on Qwen-0.6B - Implementation Plan

## Context
Course project for HPML. Goal: combine quantization + LoRA finetuning to accelerate inference while maintaining quality. No PEFT library - custom LoRA implementation from scratch.

**Environment**: codequant conda env on gh011, torch 2.10+cu126, bitsandbytes 0.49.1, transformers 5.5.0.dev0

## Project Structure
```
HPML_project/
├── README.md
├── requirements.txt
├── lora.py              # Custom LoRA layer implementation
├── quantize_utils.py    # INT4 quantization helpers (bitsandbytes)
├── model_utils.py       # Load Qwen + inject LoRA + quantize backbone
├── train.py             # QLoRA training on WikiText-2
├── evaluate.py          # Perplexity evaluation
├── config.py            # Hyperparameters & configs
└── outputs/             # Saved LoRA weights & logs
```

## Implementation Steps

### Step 1: Custom LoRA Layer (`lora.py`)
- Implement `LoRALinear` module: wraps a frozen `nn.Linear` (or quantized linear), adds trainable low-rank matrices A (rank x in) and B (out x rank) in BF16
- Forward: `y = frozen_linear(x) + scale * (x @ A^T @ B^T)`, where `scale = alpha / rank`
- A initialized with Kaiming uniform, B initialized with zeros (so LoRA starts as identity)
- Only A and B are trainable parameters

### Step 2: INT4 Quantization Utils (`quantize_utils.py`)
- Use `bitsandbytes.nn.Linear4bit` with NF4 quantization
- Function to replace target `nn.Linear` layers in the model with quantized versions
- Keep LM head and embeddings in original precision
- Backbone (attention + MLP linears) quantized to INT4

### Step 3: Model Loading & LoRA Injection (`model_utils.py`)
- Load Qwen2.5-0.6B with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`
- Inject LoRA adapters onto target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Freeze all original parameters, only LoRA A/B matrices trainable
- Function to save/load only LoRA weights
- Function to merge LoRA weights back into base model for evaluation

### Step 4: Training Script (`train.py`)
- Load WikiText-2 from HuggingFace datasets
- Tokenize with Qwen tokenizer, create causal LM sequences (block_size=1024)
- Load quantized model + inject LoRA
- Training config:
  - Optimizer: AdamW (paged_adamw_8bit from bitsandbytes for memory efficiency)
  - Learning rate: 2e-4 with cosine schedule
  - LoRA rank: 16, alpha: 32
  - Batch size: 4, gradient accumulation: 4
  - Epochs: 3
  - BF16 mixed precision via accelerate
- Save LoRA adapter weights at end

### Step 5: Evaluation Script (`evaluate.py`)
- Load base model (INT4 quantized) + load saved LoRA weights
- Option 1: Evaluate with LoRA as separate branch (no merge)
- Option 2: Merge LoRA into quantized weights then evaluate
- Compute perplexity on WikiText-2 test set
- Also evaluate base model (no LoRA) as baseline for comparison
- Report: base perplexity vs QLoRA perplexity, inference time comparison

### Step 6: Config (`config.py`)
- Centralize all hyperparameters, paths, model name, LoRA config

## Key Technical Details

**LoRA on quantized layers**: `bitsandbytes.nn.Linear4bit` stores weights in INT4 but computes in BF16. Our LoRA wrapper intercepts the forward pass - the frozen path goes through the quantized linear, while the LoRA branch (A, B matrices) stays in BF16.

**Merging for eval**: Since backbone is INT4, we can't directly merge FP LoRA weights into INT4. Two options:
1. Keep LoRA as separate branch during inference (adds minimal overhead)
2. Dequantize -> merge -> re-quantize (lossy but single-pass inference)

Recommend option 1 for accuracy, option 2 as experiment.

## Verification
1. `python train.py` - should run without OOM, loss decreasing
2. `python evaluate.py` - perplexity on WikiText-2 test, compare base vs QLoRA
3. Check that only LoRA params are trained (print trainable param count)
4. Measure inference time: base INT4 vs INT4+LoRA
