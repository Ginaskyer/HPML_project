"""
Custom LoRA (Low-Rank Adaptation) layer implementation.
No dependency on the PEFT library.
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Wraps a frozen linear layer (or quantized Linear4bit) with trainable
    low-rank matrices A and B.

    Forward: y = frozen_linear(x) + (x @ A^T @ B^T) * scaling
    where scaling = alpha / rank
    """

    def __init__(self, base_layer, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Infer dimensions from base layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Trainable low-rank matrices in BF16
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, dtype=torch.bfloat16)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.out_features, rank, dtype=torch.bfloat16)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros -> LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Frozen path (through quantized or original linear)
        base_out = self.base_layer(x)

        # LoRA path in BF16 — ensure A, B are on the same device as input
        lora_A = self.lora_A.to(x.device)
        lora_B = self.lora_B.to(x.device)

        x_bf16 = x.to(torch.bfloat16)
        lora_out = self.dropout(x_bf16) @ lora_A.t() @ lora_B.t()
        lora_out = lora_out * self.scaling

        return base_out + lora_out.to(base_out.dtype)

    def merge_and_unload(self):
        """
        Merge LoRA weights into the base layer (for non-quantized layers).
        Returns the base layer with merged weights.
        """
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
            if hasattr(self.base_layer, 'weight'):
                self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
        return self.base_layer
