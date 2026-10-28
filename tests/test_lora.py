"""Numerical contracts for adapters, dense exports, and EMA history."""

import copy

import pytest
import torch
from torch import nn

from stack.finetune.lora import (
    LoRALinear,
    apply_lora,
    merged_state_dict,
    update_dense_teacher,
)


def test_zero_start_gradients_and_dense_export():
    torch.manual_seed(13)
    dense = nn.Linear(7, 11).double()
    adapter = LoRALinear(copy.deepcopy(dense), rank=3, alpha=6)
    x = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(adapter(x), dense(x), rtol=0, atol=0)
    adapter.lora_B.data.normal_()
    expected = dense(x) + 2 * (x @ adapter.lora_A.T @ adapter.lora_B.T)
    torch.testing.assert_close(adapter(x), expected)
    adapter(x).square().sum().backward()
    assert adapter.weight.grad is None and adapter.bias.grad is None
    assert adapter.lora_A.grad.abs().sum() > 0
    assert adapter.lora_B.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0
    wrapped = nn.Sequential(adapter)
    exported = nn.Sequential(copy.deepcopy(dense))
    exported.load_state_dict(merged_state_dict(wrapped), strict=True)
    torch.testing.assert_close(exported(x), adapter(x))


def test_ema_averages_effective_weights_across_multiple_updates():
    torch.manual_seed(14)
    teacher = nn.Sequential(nn.Linear(5, 9)).double()
    student = nn.Sequential(LoRALinear(copy.deepcopy(teacher[0]), 2, 2))
    expected = copy.deepcopy(teacher.state_dict())
    for _ in range(3):
        student[0].lora_A.data.normal_()
        student[0].lora_B.data.normal_()
        effective = merged_state_dict(student)
        expected = {
            k: 0.95 * value + 0.05 * effective[k] for k, value in expected.items()
        }
        update_dense_teacher(teacher, student, 0.95)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=1e-12, atol=1e-12)


def test_group_ranks_and_frozen_backbone():
    from stack.models.finetune.model import ICL_FinetunedModel

    model = ICL_FinetunedModel(
        n_genes=31, n_hidden=3, token_dim=8, n_layers=2, n_cells=8
    )
    summary = apply_lora(model, {"ranks": {"inter_cell": 4, "intra_cell": 2}})
    assert len(summary["modules"]) == 8
    assert model.layers[0].gene_attn.qkv.rank == 4
    assert model.layers[0].cell_attn.qkv.rank == 2
    assert not model.gene_reduction[0].weight.requires_grad
    assert model.query_pos_embedding.requires_grad
    assert all(p.requires_grad for p in model.cls.parameters())
    assert all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if ".lora_" not in n and n != "query_pos_embedding" and not n.startswith("cls.")
    )
    with pytest.raises(ValueError, match="already enabled"):
        apply_lora(model, {"ranks": {"inter_cell": 4}})


def test_invalid_group_and_rank_are_rejected_before_freezing():
    from stack.models.finetune.model import ICL_FinetunedModel

    model = ICL_FinetunedModel(
        n_genes=31, n_hidden=3, token_dim=8, n_layers=1, n_cells=8
    )
    with pytest.raises(ValueError, match="exceeds"):
        apply_lora(model, {"ranks": {"intra_cell": 16}})
    assert all(p.requires_grad for p in model.parameters())


def test_native_loader_roundtrips_adapter_checkpoint(tmp_path):
    from stack.model_loading import load_model_from_checkpoint
    from stack.models.finetune.model import ICL_FinetunedModel

    config = {"n_genes": 31, "n_hidden": 3, "token_dim": 8, "n_layers": 1, "n_cells": 8}
    model = ICL_FinetunedModel(**config)
    lora = {"ranks": {"inter_cell": 4}}
    apply_lora(model, lora)
    model.layers[0].gene_attn.qkv.lora_B.data.normal_()
    path = tmp_path / "adapter.ckpt"
    torch.save(
        {
            "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
            "hyper_parameters": {"model_config": config, "lora_config": lora},
        },
        path,
    )
    loaded = load_model_from_checkpoint(
        str(path), "ICL_FinetunedModel", device=torch.device("cpu")
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)


def test_lightning_initialization_optimizer_and_teacher_use_adapters():
    from stack.finetune.lightning import LightningFinetunedModel

    config = {"n_genes": 31, "n_hidden": 3, "token_dim": 8, "n_layers": 1, "n_cells": 8}
    module = LightningFinetunedModel(
        config,
        n_kept_cell=4,
        lora_config={"ranks": {"inter_cell": 4, "intra_cell": 2}},
    )
    module.on_fit_start()
    old = copy.deepcopy(module.teacher_model.state_dict())
    module.model.layers[0].gene_attn.qkv.lora_B.data.normal_()
    effective = merged_state_dict(module.model)
    module._ema_update_teacher()
    for key, value in module.teacher_model.state_dict().items():
        torch.testing.assert_close(value, 0.95 * old[key] + 0.05 * effective[key])
    optimizer = module.configure_optimizers()["optimizer"]
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in module.model.parameters() if p.requires_grad}
    assert actual == expected
    assert all(not p.requires_grad for p in module.teacher_model.parameters())
