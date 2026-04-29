"""
Centralized configuration.
"""

from dataclasses import dataclass

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

    # precision options: "bf16" or "fp32"
    precision: str = "bf16"

    # Output
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42