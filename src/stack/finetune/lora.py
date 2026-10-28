"""Shape-aware low-rank adaptation and dense EMA teachers for Stack.

The two attention axes have very different widths. Ranks are configured by
biological operation rather than applying one rank to every linear layer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear map with an initially zero, mergeable low-rank update."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if not 0 < rank <= min(linear.in_features, linear.out_features):
            raise ValueError(f"Invalid rank {rank} for {tuple(linear.weight.shape)}")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.requires_grad_(False)
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(self.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.scaling * F.linear(
            F.linear(x, self.lora_A), self.lora_B
        )

    def effective_weight(self) -> torch.Tensor:
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)


def linear_group(name: str) -> str:
    """Name the operation, accounting for Stack's historical attribute names."""
    if ".gene_attn." in name:
        return "inter_cell"
    if ".cell_attn." in name:
        return "intra_cell"
    if name.startswith("layers.") and ".mlp." in name:
        return "token_mlp"
    if name.startswith("gene_reduction."):
        return "encoder"
    if name.startswith("output_mlp."):
        return "decoder"
    return "other"


def apply_lora(model: nn.Module, config: Mapping) -> dict:
    """Freeze the backbone, adapt requested groups, and train the ICL additions.

    ``ranks`` maps inter_cell/intra_cell/token_mlp/encoder/decoder to ranks;
    omitted groups stay frozen. ``alpha_over_rank`` defaults to one. The query
    embedding and confidence classifier remain dense trainable modules unless
    ``train_new_modules`` is false (useful when adapting an aligned model).
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("LoRA is already enabled")
    ranks = dict(config.get("ranks", {}))
    valid = {"inter_cell", "intra_cell", "token_mlp", "encoder", "decoder"}
    if set(ranks) - valid:
        raise ValueError(f"Unknown LoRA groups: {set(ranks) - valid}")
    if any(
        not isinstance(r, int) or isinstance(r, bool) or r <= 0 for r in ranks.values()
    ):
        raise ValueError("Configured ranks must be positive integers")
    scale = float(config.get("alpha_over_rank", 1.0))
    selected = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and linear_group(name) in ranks:
            group = linear_group(name)
            rank = ranks[group]
            if rank > min(module.in_features, module.out_features):
                raise ValueError(f"Rank {rank} exceeds dimensions of {name}")
            selected.append((name, module, group, rank))
    found = {entry[2] for entry in selected}
    if found != set(ranks):
        raise ValueError(f"No linear layers for groups: {set(ranks) - found}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("alpha_over_rank must be finite and positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = []
    for name, module, group, rank in selected:
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, LoRALinear(module, rank, scale * rank))
        modules.append(
            {
                "name": name,
                "group": group,
                "rank": rank,
                "shape": list(module.weight.shape),
            }
        )
    if config.get("train_new_modules", True):
        for name, parameter in model.named_parameters():
            if name == "query_pos_embedding" or name.startswith("cls."):
                parameter.requires_grad_(True)
    trainable = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    if not trainable:
        raise ValueError("Configuration has no trainable parameters")
    return {
        "config": dict(config),
        "modules": modules,
        "trainable": trainable,
        "trainable_parameters": sum(trainable.values()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export ordinary Stack keys, without modifying live adapter parameters."""
    state = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if not k.endswith((".lora_A", ".lora_B"))
    }
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name + ".weight"] = module.effective_weight().detach().clone()
    return state


@torch.no_grad()
def update_dense_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """EMA of effective weights, retaining full-rank history in the teacher.

    EMA(A) @ EMA(B) is not EMA(A @ B). Keeping a dense teacher avoids that
    approximation and keeps the original self-distillation interpretation.
    """
    if not 0 <= decay <= 1:
        raise ValueError("EMA decay must be in [0, 1]")
    params = dict(student.named_parameters())
    modules = dict(student.named_modules())
    for name, target in teacher.named_parameters():
        source = params[name]
        if name.endswith(".weight"):
            module = modules.get(name.rsplit(".", 1)[0])
            if isinstance(module, LoRALinear):
                source = module.effective_weight()
        if source.shape != target.shape:
            raise ValueError(f"Teacher shape mismatch at {name}")
        target.mul_(decay).add_(source, alpha=1 - decay)
    buffers = dict(student.named_buffers())
    for name, target in teacher.named_buffers():
        target.copy_(buffers[name])
