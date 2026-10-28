"""Numerical smoke tests and fixed-parameter integrity checks."""

import hashlib
import json

from .common import ROOT


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_parent(spec):
    import torch
    from gentonik.io import atomic_json, sha256_file

    parent = torch.load(spec["parent"], map_location="cpu", weights_only=False)
    current = torch.load(spec["current"], map_location="cpu", weights_only=False)
    assert parent["state_dict"].keys() == current["state_dict"].keys()
    rows = []
    for key, value in current["state_dict"].items():
        old = parent["state_dict"][key]
        delta = value.double() - old.double()
        rows.append(
            {
                "parameter": key,
                "equal": torch.equal(value, old),
                "max_abs_error": float(delta.abs().max()),
                "relative_l2_error": float(
                    delta.norm() / old.double().norm().clamp_min(1e-30)
                ),
            }
        )
    result = {
        "status": "complete",
        "tensors": len(rows),
        "equal_tensors": sum(row["equal"] for row in rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "parameters": rows,
        "hashes": {p: sha256_file(p) for p in [spec["parent"], spec["current"]]},
    }
    atomic_json(ROOT / "verification" / (spec["id"] + ".json"), result)
    return result


def unit_tests(spec):
    import subprocess

    tests = subprocess.run(
        ["python", "-m", "pytest", "-q", "/opt/lora_tests"],
        check=False,
        text=True,
        capture_output=True,
    )
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    return {"status": "complete", "unit_tests": tests.stdout}


def smoke(spec):
    import copy

    import torch
    from gentonik.neural_adaptation.training import initialize

    from stack.finetune.lora import apply_lora, merged_state_dict, update_dense_teacher

    tests = unit_tests(spec)
    torch.manual_seed(41)
    model, config, _, init = initialize({"checkpoint": "aligned", "kind": "alignment"})
    model.model.eval()
    x = torch.rand(1, 16, config["n_genes"], device="cuda")
    ids = torch.arange(16, device="cuda")[None] % 2
    kwargs = {
        "observed_features": x,
        "ground_truth_features": x,
        "cell_type_ids": ids,
        "n_kept_cell": 4,
        "mask_genes": False,
        "return_loss": False,
    }
    # Forward still samples a diffusion time, so hold its RNG fixed.
    torch.manual_seed(42)
    before = model.model(**kwargs)
    summary = apply_lora(
        model.model,
        {"ranks": {"inter_cell": 8, "intra_cell": 2, "encoder": 8, "decoder": 8}},
    )
    torch.manual_seed(42)
    after = model.model(**kwargs)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    dense = copy.deepcopy(model.teacher_model)
    dense.load_state_dict(merged_state_dict(model.model), strict=True)
    update_dense_teacher(dense, model.model, 0.95)
    return {
        "status": "complete",
        "unit_tests": tests["unit_tests"],
        "zero_start": "bit_exact_fp32",
        "adapter_summary": summary,
        "initialization": init,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
    }


def checkpoint_audit(spec):
    """Independently reconstruct exports and every teacher update from files."""
    import torch
    from gentonik.io import atomic_json, sha256_file

    torch.set_num_threads(8)
    folder = ROOT / "training" / spec["training_id"]
    complete = json.loads((folder / "COMPLETE.json").read_text())
    resume = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    raw = resume["state_dict"]
    exported = torch.load(
        complete["final_checkpoint"], map_location="cpu", weights_only=False
    )["state_dict"]
    initial_path = complete["initialization"]["checkpoint"]
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    scale = (
        float(complete["spec"].get("lora", {}).get("alpha_over_rank", 1.0))
        if complete["spec"].get("lora")
        else 1.0
    )
    export_errors = []
    frozen_checks = 0
    change_rows = []
    for key, value in exported.items():
        expected = raw[key]
        if key.endswith(".weight"):
            prefix = key.removesuffix(".weight")
            if prefix + ".lora_A" in raw:
                update = raw[prefix + ".lora_B"].cuda() @ raw[prefix + ".lora_A"].cuda()
                expected = (expected.cuda() + scale * update).cpu()
        if not torch.equal(value, expected):
            export_errors.append(
                {"parameter": key, "max_error": float((value - expected).abs().max())}
            )
        if complete["spec"].get("lora") is not None and key not in {
            "model.query_pos_embedding",
            "model.cls.0.weight",
            "model.cls.0.bias",
            "model.cls.2.weight",
            "model.cls.2.bias",
        }:
            assert torch.equal(raw[key], initial[key]), f"Frozen base changed: {key}"
            frozen_checks += 1
        change_rows.append(
            {
                "parameter": key,
                "squared_change": float(
                    (value.double() - initial[key].double()).square().sum()
                ),
            }
        )
    assert not export_errors, export_errors
    assert resume["step"] == complete["updates"]
    optimizer_steps = sorted(
        {int(state["step"]) for state in resume["optimizer"]["state"].values()}
    )
    assert optimizer_steps == [resume["step"]], optimizer_steps
    # Reconstruct the full-rank EMA history from the actual merged students.
    expected_teacher = {key: value.cuda() for key, value in initial.items()}
    every = int(complete["spec"].get("teacher_every", 500))
    for step in range(every, resume["step"] + 1, every):
        student = torch.load(
            folder / f"step{step:05d}.ckpt", map_location="cpu", weights_only=False
        )["state_dict"]
        for key, target in expected_teacher.items():
            target.mul_(0.95).add_(student[key].cuda(), alpha=0.05)
        del student
    teacher_errors = []
    for key, expected in expected_teacher.items():
        actual = raw["teacher_model." + key.removeprefix("model.")]
        if not torch.equal(expected.cpu(), actual):
            teacher_errors.append(
                {
                    "parameter": key,
                    "max_error": float((expected.cpu() - actual).abs().max()),
                }
            )
    assert not teacher_errors, teacher_errors
    result = {
        "status": "complete",
        "training_id": spec["training_id"],
        "export_tensors_checked": len(exported),
        "frozen_tensors_checked": frozen_checks,
        "teacher_updates_reconstructed": resume["step"] // every,
        "teacher_tensors_checked": len(expected_teacher),
        "teacher_max_error": 0.0,
        "optimizer_steps": optimizer_steps,
        "parameter_changes": change_rows,
        "checkpoint_sha256": sha256_file(complete["final_checkpoint"]),
        "resume_sha256": sha256_file(folder / "resume.pt"),
    }
    atomic_json(ROOT / "verification" / (spec["id"] + ".json"), result)
    return result


def metric_audit(spec):
    """Recompute published effect metrics from retained counts and fixtures."""
    import numpy as np
    import pandas as pd
    from gentonik.io import atomic_json
    from gentonik.tf_timing.benchmarks import fixture_path
    from scipy.stats import pearsonr, spearmanr

    checks = 0
    sensitivities = []
    for benchmark_id in spec["benchmarks"]:
        folder = ROOT / "benchmarks" / benchmark_id
        frame = pd.read_csv(folder / "metrics.csv")
        for path in folder.rglob("*_profiles.npz"):
            task = path.name.removesuffix("_profiles.npz")
            saved = np.load(path)
            original = np.load(fixture_path(task + ".npz"))

            def normalize(counts, dtype):
                x = np.asarray(counts, dtype=dtype)
                return x / np.maximum(x.sum(1, keepdims=True), 1) * 1e4

            for dtype in [np.float32, np.float64]:
                predicted = normalize(saved["generated"], dtype).mean(0) - normalize(
                    saved["generated_control"], dtype
                ).mean(0)
                truth = normalize(original["truth"], dtype).mean(0) - normalize(
                    original["query"], dtype
                ).mean(0)
                genes = original["genes"]
                d, t = predicted[genes], truth[genes]
                top = np.argsort(-np.abs(t))[:100]
                calculated = {
                    "delta_pearson": float(pearsonr(d, t).statistic),
                    "delta_spearman": float(spearmanr(d, t).statistic),
                    "mse_over_unchanged": float(
                        np.mean((d - t) ** 2) / max(float(np.mean(t * t)), 1e-12)
                    ),
                    "top100_effect_sign_accuracy": float(
                        np.mean(np.sign(d[top]) == np.sign(t[top]))
                    ),
                    "top100_effect_mse_over_unchanged": float(
                        np.mean((d[top] - t[top]) ** 2)
                        / max(float(np.mean(t[top] ** 2)), 1e-12)
                    ),
                    "effect_norm_ratio": float(
                        np.linalg.norm(d) / max(np.linalg.norm(t), 1e-12)
                    ),
                }
                reported = frame.loc[
                    frame.task.eq(task) & frame.model.eq("STACK")
                ].iloc[0]
                for name, value in calculated.items():
                    if dtype == np.float32:
                        assert np.isclose(
                            value, reported[name], rtol=1e-6, atol=1e-7
                        ), (task, name, value, reported[name])
                        checks += 1
                    elif not np.isclose(value, reported[name], rtol=2e-4, atol=2e-5):
                        sensitivities.append(
                            {
                                "benchmark": benchmark_id,
                                "task": task,
                                "metric": name,
                                "reported": float(reported[name]),
                                "float64": value,
                            }
                        )
    result = {
        "status": "complete",
        "checks": checks,
        "float64_sensitivities": sensitivities,
        "benchmarks": spec["benchmarks"],
    }
    atomic_json(ROOT / "verification" / (spec["id"] + ".json"), result)
    return result
