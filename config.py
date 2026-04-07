"""
Centralized configuration for QLoRA finetuning.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class QuantConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    block_size: int = 1024

    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Output
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42

    # Sub-configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
