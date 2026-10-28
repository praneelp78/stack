"""Validation-only pilot and common-checkpoint selection; no test labels loaded."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("STACK_LORA_RUN_DIR", "work/lora-study")).resolve()
META = RUN / "metadata"
NEURAL = [
    "Herring",
    "Velmeshev",
    "Jerber_development",
    "Jerber_rotenone",
    "HNOCA_development",
]


def read_json(path):
    return json.loads(path.read_text())


def transfer(path):
    grouped = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "STACK":
                grouped[row["dataset"]].append(row)

    def metric(source, name):
        return mean(float(row[name]) for row in grouped[source])

    return {
        "new_mse": 0.8 * metric("Lin2025_neural_intervention", "mse_over_unchanged")
        + 0.2 * metric("Lin2021_NGN2_time", "mse_over_unchanged"),
        "intervention_mse": metric("Lin2025_neural_intervention", "mse_over_unchanged"),
        "time_mse": metric("Lin2021_NGN2_time", "mse_over_unchanged"),
        "neural_mse": mean(metric(source, "mse_over_unchanged") for source in NEURAL),
        "replay_mse": metric("Replay_drug_transfer", "mse_over_unchanged"),
        "replay_r": metric("Replay_drug_transfer", "delta_pearson"),
        "tasks": sum(len(rows) for rows in grouped.values()),
    }


def retention(row, original):
    return (
        row["neural_mse"] <= original["neural_mse"] * 1.1
        and row["replay_mse"] <= original["replay_mse"] * 1.1
        and row["replay_r"] >= original["replay_r"] * 0.9
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pilot():
    specs = read_json(RUN / "pilot_specs.json")
    original = transfer(META / "benchmarks/baseline_aligned_val02/metrics.csv")
    rows = []
    for spec in specs:
        folder = META / "training" / spec["id"]
        if not (folder / "COMPLETE.json").exists():
            raise RuntimeError(f"Pilot not complete: {spec['id']}")
        result = read_json(folder / "COMPLETE.json")
        metrics = transfer(folder / "transfer/step00500/metrics.csv")
        with (folder / "learning_curve.csv").open() as handle:
            timings = [
                float(row["seconds"])
                for row in csv.DictReader(handle)
                if row["split"] == "train" and int(row["step"]) > 50
            ]
        row = {
            "id": spec["id"],
            "placement": spec["placement"],
            "rank": spec.get("rank", ""),
            "lr": spec["lr"],
            **metrics,
            "retention_pass": retention(metrics, original),
            "trainable_parameters": result["trainable_parameters"],
            "peak_container_gpu_gib": result["peak_gpu_bytes"] / 1024**3,
            "median_step_seconds": median(timings),
            "worker_seconds": result.get("seconds_worker", result["seconds"]),
            "frozen_verified": result["frozen_initial_sha256"]
            == result["frozen_final_sha256"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (not row["retention_pass"], row["new_mse"], row["id"]))
    write_csv(RUN / "pilot_comparison.csv", rows)
    candidates = []
    placements = set()
    by_id = {spec["id"]: spec for spec in specs}
    for row in rows:
        spec = by_id[row["id"]]
        adapter = spec.get("lora")
        if adapter is None or not adapter.get("ranks"):
            continue
        if row["placement"] in placements:
            continue
        candidates.append(row)
        placements.add(row["placement"])
        if len(candidates) == 2:
            break
    if len(candidates) != 2:
        raise RuntimeError("Fewer than two distinct LoRA placements completed")
    full = next(row for row in rows if row["placement"] == "full")
    chosen = candidates + [full]
    selection = {
        "stage": "pilot",
        "baseline": original,
        "chosen": chosen,
        "rule": "Retention-passing rows first, then weighted validation MSE; two distinct LoRA placements plus best full-FT LR. Failing rows, if needed, are exploratory. No phenotype/test access.",
    }
    (RUN / "pilot_selection.json").write_text(json.dumps(selection, indent=2))
    long_specs = []
    for row in chosen:
        source = by_id[row["id"]]
        for seed in [0, 1, 2]:
            name = source["id"].removeprefix("pilot_").rsplit("_s", 1)[0]
            spec = {
                **source,
                "id": f"long_{name}_s{seed}",
                "updates": 4000,
                "seed": seed,
                "validate_every": 500,
                "validation_sets_per_family": 64,
                "bench_steps": [500, 1000, 2000, 3000, 4000],
                "promoted_from": row["id"],
                "pilot_retention_pass": row["retention_pass"],
            }
            long_specs.append(spec)
    (RUN / "long_specs.json").write_text(json.dumps(long_specs, indent=2))
    print(json.dumps(selection, indent=2))


def long_selection():
    original = transfer(META / "benchmarks/baseline_aligned_val02/metrics.csv")
    specs = read_json(RUN / "long_specs.json")
    groups = defaultdict(list)
    for spec in specs:
        folder = META / "training" / spec["id"]
        assert (folder / "COMPLETE.json").exists(), spec["id"]
        groups[spec["promoted_from"]].append(spec)
    selected_models = []
    rows = []
    for arm, runs in groups.items():
        assert sorted(run["seed"] for run in runs) == [0, 1, 2]
        candidates = []
        for step in [500, 1000, 2000, 3000, 4000]:
            metrics = [
                transfer(
                    META
                    / "training"
                    / run["id"]
                    / "transfer"
                    / f"step{step:05d}/metrics.csv"
                )
                for run in runs
            ]
            record = {
                "arm": arm,
                "step": step,
                **{key: mean(metric[key] for metric in metrics) for key in metrics[0]},
            }
            record["retention_pass"] = retention(record, original)
            candidates.append(record)
        feasible = [record for record in candidates if record["retention_pass"]]
        selected = min(
            feasible or candidates, key=lambda row: (row["new_mse"], row["step"])
        )
        rows.extend(candidates)
        for run in runs:
            for step in sorted({selected["step"], 4000}):
                selected_models.append(
                    {
                        "arm": arm,
                        "training_id": run["id"],
                        "placement": run["placement"],
                        "rank": run.get("rank"),
                        "lr": run["lr"],
                        "lora": run.get("lora"),
                        "training_seed": run["seed"],
                        "step": step,
                        "id": run["id"] + f"_step{step:05d}",
                        "checkpoint": f"/results/training/{run['id']}/step{step:05d}.ckpt",
                        "roles": (
                            ["validation_selected"] if step == selected["step"] else []
                        )
                        + (["final_budget"] if step == 4000 else []),
                        "selection_retention_pass": bool(feasible),
                    }
                )
    write_csv(RUN / "long_validation.csv", rows)
    (RUN / "selected_models.json").write_text(json.dumps(selected_models, indent=2))
    print(
        json.dumps(
            {
                "models": len(selected_models),
                "selections": [
                    model for model in selected_models if model["training_seed"] == 0
                ],
            },
            indent=2,
        )
    )


def test_specs():
    config = read_json(HERE / "study_config.json")
    models = read_json(RUN / "selected_models.json")
    reference = {
        "id": "released_aligned",
        "checkpoint": "aligned",
        "roles": ["reference"],
    }
    specs = []
    for model in [reference, *models]:
        for panel, taskfiles in config["evaluation"]["test_panels"].items():
            specs.append(
                {
                    "id": model["id"] + "_test_" + panel,
                    "kind": "benchmark",
                    "checkpoint": model["checkpoint"],
                    "taskfiles": taskfiles,
                    "split": "test",
                    "combination_modes": True,
                    "seed": 0,
                }
            )
        if "validation_selected" in model["roles"] or "reference" in model["roles"]:
            specs.append(
                {
                    "id": model["id"] + "_permuted_recipe",
                    "kind": "benchmark",
                    "checkpoint": model["checkpoint"],
                    "taskfiles": ["tf_intervention"],
                    "split": "test",
                    "prompt_mode": "permuted_recipe",
                    "seed": 0,
                }
            )
            for seed in [1, 2]:
                specs.append(
                    {
                        "id": model["id"] + f"_rna_repeat{seed}",
                        "kind": "benchmark",
                        "checkpoint": model["checkpoint"],
                        "taskfiles": [
                            "tf_intervention",
                            "tf_ngn2_time",
                            "tf_ko",
                            "tian",
                            "chemical_matched",
                        ],
                        "split": "test",
                        "seed": seed,
                    }
                )
        for seed in [0, 1, 2]:
            specs.append(
                {
                    "id": model["id"] + f"_drugs{seed}",
                    "kind": "drugs",
                    "checkpoint": model["checkpoint"],
                    "seed": seed,
                    "contexts": 3,
                }
            )
    (RUN / "test_specs.json").write_text(json.dumps(specs, indent=2))
    print(json.dumps({"test_jobs": len(specs)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "long", "tests"])
    args = parser.parse_args()
    {"pilot": pilot, "long": long_selection, "tests": test_specs}[args.stage]()
