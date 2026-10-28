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
    """Recompute all reported model/control effects from raw retained counts."""
    import numpy as np
    import pandas as pd
    from gentonik.io import atomic_json
    from gentonik.tf_timing.benchmarks import fixture_path
    from scipy.stats import pearsonr, spearmanr

    def normalized_mean(counts, dtype):
        x = np.asarray(counts, dtype=dtype)
        return (x / np.maximum(x.sum(1, keepdims=True), 1) * 1e4).mean(0)

    def metrics(predicted, truth, genes):
        d, t = predicted[genes], truth[genes]
        top = np.argsort(-np.abs(t))[:100]
        variable = np.std(d) > 0 and np.std(t) > 0
        return {
            "delta_pearson": float(pearsonr(d, t).statistic)
            if variable
            else float("nan"),
            "delta_spearman": float(spearmanr(d, t).statistic)
            if variable
            else float("nan"),
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

    checks = 0
    undefined = 0
    profile_checks = 0
    reported_metrics = 0
    sensitivities = []
    labels_checked = set()
    for benchmark_id in spec["benchmarks"]:
        folder = ROOT / "benchmarks" / benchmark_id
        frame = pd.read_csv(folder / "metrics.csv")
        profiles = list(folder.rglob("*_profiles.npz"))
        assert {p.name.removesuffix("_profiles.npz") for p in profiles} == set(
            frame.task
        )
        assert len(profiles) == frame.task.nunique(), benchmark_id
        reported_metrics += len(frame) * 6
        for path in profiles:
            task = path.name.removesuffix("_profiles.npz")
            saved = np.load(path)
            original = np.load(fixture_path(task + ".npz"))
            assert np.array_equal(saved["genes"], original["genes"])
            prompt_source = original
            prompt_record = path.parent / (task + "_prompt.json")
            if prompt_record.exists():
                replacement = json.loads(prompt_record.read_text())["prompt_task"]
                prompt_source = np.load(fixture_path(replacement + ".npz"))
            reported_rows = frame.loc[frame.task.eq(task)]
            assert not reported_rows.model.duplicated().any()
            for dtype in [np.float32, np.float64]:

                def mean(counts, dtype=dtype):
                    return normalized_mean(counts, dtype)

                truth = mean(original["truth"]) - mean(original["query"])
                predictions = {
                    "STACK": mean(saved["generated"])
                    - mean(saved["generated_control"]),
                    "other_context_mean_delta": mean(prompt_source["prompt_target"])
                    - mean(original["prompt_control"]),
                    "unchanged": np.zeros_like(truth),
                }
                if "component_a" in original.files:
                    predictions["measured_component_additive"] = (
                        mean(original["component_a"])
                        - mean(original["component_a_control"])
                        + mean(original["component_b"])
                        - mean(original["component_b_control"])
                    )
                if "single_a" in saved.files:
                    predictions["STACK_single_additive"] = (
                        mean(saved["single_a"])
                        - mean(saved["single_a_control"])
                        + mean(saved["single_b"])
                        - mean(saved["single_b_control"])
                    )
                    for order in ["ab", "ba"]:
                        predictions["STACK_chain_" + order] = mean(
                            saved["chain_" + order]
                        ) - mean(saved["chain_" + order + "_control"])
                if dtype == np.float32:
                    for field, rebuilt in {
                        "truth": truth,
                        "query": mean(original["query"]),
                        "predicted_delta": predictions["STACK"],
                        "mean_delta": predictions["other_context_mean_delta"],
                    }.items():
                        np.testing.assert_array_equal(saved[field], rebuilt)
                        profile_checks += 1
                for reported in reported_rows.to_dict("records"):
                    label = reported["model"]
                    assert label in predictions, (task, label)
                    labels_checked.add(label)
                    calculated = metrics(predictions[label], truth, original["genes"])
                    for name, value in calculated.items():
                        if dtype == np.float32:
                            assert np.isclose(
                                value,
                                reported[name],
                                rtol=1e-6,
                                atol=1e-7,
                                equal_nan=True,
                            ), (task, label, name, value, reported[name])
                            checks += 1
                            undefined += int(np.isnan(value))
                        elif not np.isclose(
                            value, reported[name], rtol=2e-4, atol=2e-5, equal_nan=True
                        ):
                            sensitivities.append(
                                {
                                    "benchmark": benchmark_id,
                                    "task": task,
                                    "model": label,
                                    "metric": name,
                                    "reported": float(reported[name]),
                                    "float64": value,
                                }
                            )
            saved.close()
            if prompt_source is not original:
                prompt_source.close()
            original.close()
    assert checks == reported_metrics
    result = {
        "status": "complete",
        "checks": checks,
        "exact_profile_reconstructions": profile_checks,
        "undefined_correlations_preserved": undefined,
        "models_checked": sorted(labels_checked),
        "float64_sensitivities": sensitivities,
        "benchmarks": spec["benchmarks"],
    }
    atomic_json(ROOT / "verification" / (spec["id"] + ".json"), result)
    return result


def study_audit(spec):
    """Reconcile requested runs, training coverage, checkpoint identity and ranks."""
    import numpy as np
    import pandas as pd
    from gentonik.io import atomic_json, sha256_file
    from gentonik.neural_adaptation.models import checkpoint_path
    from scipy.stats import kendalltau, rankdata

    hashes = {}

    def file_hash(path):
        key = str(path)
        if key not in hashes:
            hashes[key] = sha256_file(path)
        return hashes[key]

    training_rows = []
    data_hashes = set()
    for run_id in spec["training_ids"]:
        folder = ROOT / "training" / run_id
        done = json.loads((folder / "COMPLETE.json").read_text())
        config = done["spec"]
        curve = pd.read_csv(folder / "learning_curve.csv")
        train = curve.loc[curve.split.eq("train")]
        steps = int(config["updates"])
        sets = steps * int(config["microbatch"]) * int(config["accumulation"])
        assert done["status"] == "complete" and done["updates"] == steps
        assert train.step.tolist() == list(range(1, steps + 1))
        assert np.isfinite(train[["loss", "grad_norm", "seconds"]]).all().all()
        assert train.grad_norm.gt(0).all() and train.seconds.gt(0).all()
        assert done["teacher_updates"] == steps // int(config["teacher_every"])
        assert sum(done["coverage"]["study_sets"].values()) == sets
        assert done["cell_presentations"] == sets * 512
        for key in ["unique_target_cells", "unique_query_cells"]:
            assert 0 < done["coverage"][key] <= done["train_cells"]
        if config.get("lora") is not None:
            assert done["frozen_initial_sha256"] == done["frozen_final_sha256"]
        assert done["checkpoint_sha256"] == file_hash(done["final_checkpoint"])
        data_hashes.add(
            tuple(
                done["inputs"][key]
                for key in ["obs.parquet", "COMPLETE.json", "counts.npz", "mmap"]
            )
        )
        training_rows.append(
            {
                "id": run_id,
                "updates": steps,
                "teacher_updates": done["teacher_updates"],
                "sampled_sets": sets,
                "unique_target_cells": done["coverage"]["unique_target_cells"],
                "unique_query_cells": done["coverage"]["unique_query_cells"],
                "checkpoint_sha256": done["checkpoint_sha256"],
            }
        )
    assert len(data_hashes) == 1, "Training runs used different corpus inputs"
    benchmark_rows = []
    for benchmark_id in spec["benchmark_ids"]:
        folder = ROOT / "benchmarks" / benchmark_id
        done = json.loads((folder / "COMPLETE.json").read_text())
        table = pd.read_csv(folder / "metrics.csv")
        assert done["status"] == "complete"
        assert done["tasks"] == table.task.nunique()
        assert not table.duplicated(["task", "model"]).any()
        assert done["checkpoint_sha256"] == file_hash(checkpoint_path(done["spec"]))
        benchmark_rows.append(
            {"id": benchmark_id, "tasks": done["tasks"], "metric_rows": len(table)}
        )
    rank_rows = []
    frozen_inputs = set()
    for drug_id in spec["drug_ids"]:
        folder = ROOT / "drugs" / drug_id
        done = json.loads((folder / "COMPLETE.json").read_text())
        assert done["status"] == "complete"
        assert done["checkpoint_sha256"] == file_hash(checkpoint_path(done["spec"]))
        frozen_inputs.add(
            json.dumps(
                {
                    key: done[key]
                    for key in [
                        "frozen_readout_sha256",
                        "query_sha256",
                        "admission_sha256",
                    ]
                },
                sort_keys=True,
            )
        )
        context = pd.read_csv(folder / "context_scores.csv")
        assert not context.duplicated(
            ["screen_id", "source", "context", "scorer"]
        ).any()
        reconstructed = (
            context.groupby(["screen_id", "compound", "scorer"], dropna=False)
            .agg(
                maturity_delta=("maturity_delta", "mean"),
                phenotype=("phenotype", "first"),
            )
            .reset_index()
        )
        single = pd.read_csv(folder / "singleton_scores.csv")
        joined = single.merge(
            reconstructed,
            on=["screen_id", "compound", "scorer"],
            suffixes=("_saved", "_rebuilt"),
            validate="one_to_one",
        )
        assert len(joined) == len(single) == len(reconstructed)
        for metric in ["maturity_delta", "phenotype"]:
            np.testing.assert_allclose(
                joined[metric + "_saved"],
                joined[metric + "_rebuilt"],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        assert {row["scorer"] for row in done["ranking"]} == set(single.scorer)
        for row in done["ranking"]:
            group = single.loc[
                single.scorer.eq(row["scorer"]) & single.phenotype.notna()
            ]
            rho = float(
                np.corrcoef(rankdata(group.phenotype), rankdata(group.maturity_delta))[
                    0, 1
                ]
            )
            np.testing.assert_allclose(
                rho, row["rho"], atol=1e-10, rtol=0, equal_nan=True
            )
            np.testing.assert_allclose(
                kendalltau(group.phenotype, group.maturity_delta).statistic,
                row["kendall"],
                atol=1e-10,
                rtol=0,
                equal_nan=True,
            )
            order = group.sort_values("maturity_delta", ascending=False)
            overlap = len(
                set(order.head(4).screen_id)
                & set(group.nlargest(4, "phenotype").screen_id)
            )
            assert row["n"] == len(group) == 12
            assert (
                row["top1"] == order.iloc[0].compound and row["top4_overlap"] == overlap
            )
            rank_rows.append(
                {
                    "id": drug_id,
                    "scorer": row["scorer"],
                    "rho": rho,
                    "top1": row["top1"],
                    "top4_overlap": overlap,
                }
            )
    if spec["drug_ids"]:
        assert len(frozen_inputs) == 1, "Drug benchmarks changed frozen inputs"
    folder = ROOT / "verification" / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_rows).to_csv(folder / "training_checks.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(folder / "benchmark_checks.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(folder / "ranking_checks.csv", index=False)
    result = {
        "status": "complete",
        "training_runs": len(training_rows),
        "benchmarks": len(benchmark_rows),
        "ranking_checks": len(rank_rows),
        "files_hashed": hashes,
        "scope": "All requested outputs present; exact update/teacher/coverage accounting; frozen base identity; inference checkpoint hashes; context-to-singleton aggregation and independent rank calculations. Does not re-embed generated drug profiles.",
    }
    atomic_json(folder / "COMPLETE.json", result)
    return result
