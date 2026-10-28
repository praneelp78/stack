"""Scientific plots from retained numerical artifacts."""

from .common import ROOT

COLORS = {
    "cross": "#275dad",
    "both": "#2b8a78",
    "cross_io": "#d47f1b",
    "within": "#7950a3",
    "io": "#b64c74",
    "all": "#8b6d32",
    "heads": "#7a8695",
    "full": "#252a34",
    "released": "#9aa0a6",
}
LABELS = {
    "cross": "Across-cell r8",
    "cross_io": "Across-cell + I/O r128",
    "full": "Full FT",
    "released": "Released",
}


def pilot_figures(spec):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from gentonik.io import atomic_json

    table = pd.DataFrame(spec["rows"])
    assert len(table) == table.id.nunique() == 27
    for column in ["rank", "lr", "new_mse"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["passed"] = table.retention_pass.astype(str).str.lower().eq("true")
    full = table.loc[table.placement.eq("full") & table.passed]
    reference = float(full.new_mse.min())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    placements = [
        ("cross", "Attention across cells"),
        ("both", "Both attention axes"),
        ("cross_io", "Across-cell attention + input/decoder"),
    ]
    colors = {2e-5: "#275dad", 1e-4: "#2b8a78"}
    for axis, (placement, title) in zip(axes, placements):
        subset = table.loc[table.placement.eq(placement)]
        for lr, color in colors.items():
            rows = subset.loc[subset.lr.eq(lr)].sort_values("rank")
            axis.plot(
                rows["rank"],
                rows.new_mse,
                color=color,
                linewidth=1.5,
                label=f"LR {lr:g}",
            )
            for row in rows.itertuples():
                axis.scatter(
                    row.rank,
                    row.new_mse,
                    s=48,
                    edgecolors=color,
                    facecolors=color if row.passed else "white",
                    linewidth=1.5,
                    zorder=3,
                )
        axis.axhline(
            reference,
            color="#333333",
            linestyle="--",
            linewidth=1,
            label="Full FT, selected LR",
        )
        axis.axhline(
            spec["released_score"],
            color="#999999",
            linestyle=":",
            linewidth=1,
            label="Released",
        )
        if spec.get("source_mean_score") is not None:
            axis.axhline(
                spec["source_mean_score"],
                color="#a5692c",
                linestyle="-.",
                linewidth=1,
                label="Source mean",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([8, 32, 128], ["8", "32", "128"])
        axis.set_xlabel("LoRA rank on wide layers")
        axis.set_title(title, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Weighted new-family validation relative MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3 if spec.get("source_mean_score") is not None else 4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "500-update pilot sweep\nFilled markers pass retention gates; lower validation MSE is better",
        fontsize=12,
    )
    fig.tight_layout()
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "pilot_sweep.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    result = {"status": "complete", "pilot_arms": 27, "file": "pilot_sweep.png"}
    atomic_json(out / "pilot_figures.json", result)
    return result


def diagnostic_figures(spec):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from gentonik.io import atomic_json

    sources = [
        ("aligned_update_spectra01", "bc_large_aligned.ckpt", "Pretrained → aligned"),
        ("tf_update_spectra01", "step00500.ckpt", "Aligned → TF/time, 500 updates"),
        ("tf_update_spectra01", "step04000.ckpt", "Aligned → TF/time, 4,000 updates"),
    ]
    groups = {
        "inter_cell": ("Attention across cells", "#275dad"),
        "encoder": ("Input projection", "#d47f1b"),
        "decoder": ("Decoder", "#2b8a78"),
    }
    ranks = [2, 4, 8, 16, 32, 64, 128]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    summaries = []
    for axis, (folder, checkpoint, title) in zip(axes, sources):
        frame = pd.read_csv(ROOT / "diagnostics" / folder / "spectra.csv")
        frame = frame.loc[frame.checkpoint.str.endswith(checkpoint)]
        for group, (label, color) in groups.items():
            values = frame.loc[frame.group.eq(group)]
            axis.plot(
                ranks,
                [values[f"energy_rank{rank}"].median() for rank in ranks],
                marker="o",
                markersize=4,
                linewidth=2,
                label=label,
                color=color,
            )
            summaries.append(
                {
                    "stage": title,
                    "group": group,
                    "median_rank90": float(values.rank90.median()),
                    "median_energy_rank16": float(values.energy_rank16.median()),
                    "median_relative_update_norm": float(
                        values.delta_over_weight_norm.median()
                    ),
                }
            )
        axis.axhline(0.9, color="#777777", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, [str(rank) for rank in ranks])
        axis.set_ylim(0, 1.03)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Rank retained")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Median fraction of update energy retained")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Full-fine-tuning updates have different spectra across model components",
        fontsize=13,
    )
    fig.tight_layout()
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "update_spectra.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "update_spectra.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(out / "update_spectra_summary.csv", index=False)
    result = {
        "status": "complete",
        "files": [
            "update_spectra.png",
            "update_spectra.pdf",
            "update_spectra_summary.csv",
        ],
        "interpretation": "Median across weight matrices in each group. Spectral compressibility of a full update does not prove LoRA trainability or functional equivalence.",
    }
    atomic_json(out / "diagnostic_figures.json", result)
    return result


def summarize(spec):
    """Build checkpoint-level and paired-seed tables without selecting on tests."""
    import json

    import pandas as pd
    from gentonik.io import atomic_json
    from gentonik.tf_timing import reporting, rna_repeats

    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    models = spec["models"]
    reference = {
        "id": "released_aligned",
        "checkpoint": "aligned",
        "arm": "released",
        "placement": "released",
        "roles": ["reference"],
        "training_seed": -1,
        "step": 0,
    }
    roster = [reference, *models]
    pd.DataFrame(roster).to_csv(out / "model_roster.csv", index=False)
    frames = []
    checkpoint_manifest = []
    for model in roster:
        pieces = []
        hashes = set()
        for panel in ["new", "old_neural", "old_perturb", "arlotta"]:
            path = (
                ROOT / "benchmarks" / (model["id"] + "_test_" + panel) / "metrics.csv"
            )
            frame = pd.read_csv(path)
            hashes.add(
                json.loads((path.parent / "COMPLETE.json").read_text())[
                    "checkpoint_sha256"
                ]
            )
            frame["panel"] = panel
            pieces.append(frame)
        frame = pd.concat(pieces, ignore_index=True)
        primary = frame.loc[frame.model.eq("STACK")]
        assert len(primary) == 186 and primary.task.nunique() == 186, model["id"]
        for key in ["id", "checkpoint", "arm", "placement", "training_seed", "step"]:
            frame[key] = model[key]
        frame["roles"] = ",".join(model["roles"])
        frames.append(frame)
        assert len(hashes) == 1, f"Checkpoint changed across panels: {model['id']}"
        checkpoint_manifest.append(
            {
                "model": model["id"],
                "checkpoint": model["checkpoint"],
                "sha256": next(iter(hashes)),
                "roles": ",".join(model["roles"]),
                "training_id": model.get("training_id", ""),
                "step": model["step"],
                "adapter": f"/results/training/{model['training_id']}/adapter{model['step']:05d}.pt"
                if model["placement"] not in {"full", "released"}
                else "",
            }
        )
    pd.DataFrame(checkpoint_manifest).to_csv(
        out / "checkpoint_manifest.csv", index=False
    )
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out / "rna_task_metrics.csv", index=False)
    numeric = [
        "delta_pearson",
        "delta_spearman",
        "mse_over_unchanged",
        "top100_effect_sign_accuracy",
        "top100_effect_mse_over_unchanged",
        "effect_norm_ratio",
    ]
    checkpoint = (
        raw.groupby(
            [
                "id",
                "checkpoint",
                "arm",
                "placement",
                "training_seed",
                "step",
                "roles",
                "dataset",
                "model",
            ]
        )[numeric]
        .mean()
        .reset_index()
    )
    checkpoint.to_csv(out / "rna_by_checkpoint.csv", index=False)
    expanded = []
    for role in ["reference", "validation_selected", "final_budget"]:
        subset = checkpoint.loc[
            checkpoint.roles.str.split(",").map(lambda roles, role=role: role in roles)
        ].copy()
        subset["role"] = role
        expanded.append(subset)
    expanded = pd.concat(expanded, ignore_index=True)
    summary = (
        expanded.groupby(["arm", "placement", "role", "dataset", "model"])
        .agg(
            r_mean=("delta_pearson", "mean"),
            r_min=("delta_pearson", "min"),
            r_max=("delta_pearson", "max"),
            mse_mean=("mse_over_unchanged", "mean"),
            mse_min=("mse_over_unchanged", "min"),
            mse_max=("mse_over_unchanged", "max"),
            effect_sign_mean=("top100_effect_sign_accuracy", "mean"),
            checkpoints=("id", "nunique"),
            finite_correlations=("delta_pearson", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out / "rna_by_arm.csv", index=False)
    # Average generation scores before ranking, as in the frozen parent analysis.
    reporting.ROOT = ROOT
    ensemble_folder = ROOT / "inference_diagnostics/generation_ensemble01"
    if not (ensemble_folder / "COMPLETE.json").exists():
        reporting.inference_diagnostics(
            {
                "id": "generation_ensemble01",
                "expected_checkpoints": [model["checkpoint"] for model in roster],
            }
        )
    drugs = pd.read_csv(ensemble_folder / "ensemble_rankings.csv")
    assert set(drugs.checkpoint) == {model["checkpoint"] for model in roster}
    metadata = pd.DataFrame(
        [
            {
                key: model[key]
                for key in [
                    "id",
                    "checkpoint",
                    "arm",
                    "placement",
                    "training_seed",
                    "step",
                ]
            }
            | {"roles": ",".join(model["roles"])}
            for model in roster
        ]
    )
    drugs = drugs.merge(metadata, on="checkpoint", validate="many_to_one")
    drugs.to_csv(out / "gentonik_generation_averages.csv", index=False)
    drug_rows = []
    for role in ["reference", "validation_selected", "final_budget"]:
        subset = drugs.loc[
            drugs.roles.str.split(",").map(lambda roles, role=role: role in roles)
        ].copy()
        subset["role"] = role
        drug_rows.append(subset)
    drug_summary = (
        pd.concat(drug_rows)
        .groupby(["arm", "placement", "role", "scorer"])
        .agg(
            rho_mean=("ensemble_rho", "mean"),
            rho_min=("ensemble_rho", "min"),
            rho_max=("ensemble_rho", "max"),
            top4_overlap_mean=("ensemble_top4_overlap", "mean"),
            checkpoints=("id", "nunique"),
            finite_rankings=("ensemble_rho", "count"),
        )
        .reset_index()
    )
    drug_summary.to_csv(out / "gentonik_by_arm.csv", index=False)
    rna_repeats.ROOT = ROOT
    repeats_folder = ROOT / "rna_inference_diagnostics/three_generation01"
    repeated_checkpoints = [
        model["checkpoint"]
        for model in roster
        if set(model["roles"]) & {"reference", "validation_selected"}
    ]
    if not (repeats_folder / "COMPLETE.json").exists():
        rna_repeats.rna_inference_diagnostics(
            {
                "id": "three_generation01",
                "checkpoints": repeated_checkpoints,
                "expected_tasks_per_checkpoint": 77,
            }
        )
    repeated = pd.read_csv(repeats_folder / "summary.csv")
    repeated.merge(metadata, on="checkpoint", validate="many_to_one").to_csv(
        out / "rna_generation_averages.csv", index=False
    )
    prompt_rows = []
    for model in roster:
        if not set(model["roles"]) & {"reference", "validation_selected"}:
            continue
        wrong = pd.read_csv(
            ROOT / "benchmarks" / (model["id"] + "_permuted_recipe") / "metrics.csv"
        )
        wrong = wrong.loc[wrong.model.eq("STACK")]
        correct = raw.loc[raw.id.eq(model["id"]) & raw.model.eq("STACK")]
        pairs = correct.merge(
            wrong, on="task", suffixes=("_correct", "_wrong"), validate="one_to_one"
        )
        assert len(pairs) > 0
        prompt_rows.append(
            {
                "id": model["id"],
                "arm": model["arm"],
                "training_seed": model["training_seed"],
                "tasks": len(pairs),
                "r_advantage": float(
                    (pairs.delta_pearson_correct - pairs.delta_pearson_wrong).mean()
                ),
                "mse_advantage": float(
                    (
                        pairs.mse_over_unchanged_wrong
                        - pairs.mse_over_unchanged_correct
                    ).mean()
                ),
            }
        )
    pd.DataFrame(prompt_rows).to_csv(out / "prompt_specificity.csv", index=False)
    efficiency = []
    training_ids = sorted({model["training_id"] for model in models})
    for training_id in training_ids:
        folder = ROOT / "training" / training_id
        complete = json.loads((folder / "COMPLETE.json").read_text())
        assert complete.get("memory_stats_reset"), "Per-run memory measurement required"
        curve = pd.read_csv(folder / "learning_curve.csv")
        times = curve.loc[curve.split.eq("train") & curve.step.gt(50), "seconds"]
        efficiency.append(
            {
                "training_id": training_id,
                "placement": complete["spec"]["placement"],
                "training_seed": complete["spec"]["seed"],
                "trainable_parameters": complete["trainable_parameters"],
                "median_step_seconds": float(times.median()),
                "mean_step_seconds": float(times.mean()),
                "peak_job_gpu_gib": complete["peak_gpu_bytes"] / 1024**3,
                "run_seconds_including_validation": complete["seconds"],
                "updates": complete["updates"],
            }
        )
    pd.DataFrame(efficiency).to_csv(out / "efficiency.csv", index=False)
    if spec.get("pilot_rows"):
        pd.DataFrame(spec["pilot_rows"]).to_csv(
            out / "pilot_comparison.csv", index=False
        )
    if spec.get("long_validation"):
        pd.DataFrame(spec["long_validation"]).to_csv(
            out / "long_validation.csv", index=False
        )
    make_plots(out, summary, drug_summary, pd.DataFrame(efficiency))
    learning_figures(out, models)
    result = {
        "status": "complete",
        "endpoint_models": len(roster),
        "primary_tasks_per_model": 186,
        "primary_model_task_pairs": len(roster) * 186,
        "rna_rows": len(raw),
        "gen_tonik_checkpoints": len(roster),
        "drug_generation_seeds": 3,
        "files": sorted(path.name for path in out.iterdir() if path.is_file()),
        "interpretation": "Ranges span training seeds, not biological replicates. GenTonik rho is computed after averaging generation scores per compound. Model selection was frozen before these tests.",
    }
    atomic_json(out / "SUMMARY.json", result)
    return result


def make_plots(out, rna, drugs, efficiency):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chosen = rna.loc[
        rna.model.eq("STACK") & rna.role.isin(["validation_selected", "reference"])
    ]
    arms = list(dict.fromkeys(chosen.arm))
    datasets = list(dict.fromkeys(chosen.dataset))
    labels = [name.replace("_", " ") for name in datasets]
    fig, axes = plt.subplots(
        1, 2, figsize=(13, max(7, len(datasets) * 0.6)), sharey=True
    )
    positions = np.arange(len(datasets))
    width = 0.75 / len(arms)
    for index, arm in enumerate(arms):
        frame = chosen.loc[chosen.arm.eq(arm)].set_index("dataset").loc[datasets]
        color = COLORS[frame.placement.iloc[0]]
        offset = (index - (len(arms) - 1) / 2) * width
        label = LABELS[frame.placement.iloc[0]]
        for axis, metric in zip(axes, ["r", "mse"]):
            values = frame[f"{metric}_mean"].to_numpy()
            error = np.array(
                [
                    values - frame[f"{metric}_min"].to_numpy(),
                    frame[f"{metric}_max"].to_numpy() - values,
                ]
            )
            axis.errorbar(
                values,
                positions + offset,
                xerr=np.maximum(error, 0),
                fmt="o",
                color=color,
                markersize=4,
                capsize=2,
                linewidth=1,
                label=label,
            )
    source_mean = (
        rna.loc[
            rna.arm.eq("released")
            & rna.role.eq("reference")
            & rna.model.eq("other_context_mean_delta")
        ]
        .set_index("dataset")
        .loc[datasets]
    )
    for axis, metric in zip(axes, ["r", "mse"]):
        axis.plot(
            source_mean[f"{metric}_mean"],
            positions,
            "x",
            color="#555555",
            markersize=5,
            label="Source mean",
        )
    axes[0].set_yticks(positions, labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Expression-change Pearson r (higher is better)")
    axes[0].axvline(0, color="#aaaaaa", linewidth=1)
    axes[1].set_xlabel("MSE / unchanged-prediction MSE (lower is better)")
    axes[1].axvline(1, color="#aaaaaa", linestyle="--", linewidth=1)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.12)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=len(arms) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Held-out RNA transfer: validation-selected checkpoints\nPoints are training-seed means; bars span their range",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "rna_transfer.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    scorers = sorted(drugs.scorer.unique())
    fig, axes = plt.subplots(
        1, len(scorers), figsize=(13, 4), sharey=True, squeeze=False
    )
    model_arms = list(dict.fromkeys(drugs.arm))
    for axis, scorer in zip(axes[0], scorers):
        for index, arm in enumerate(model_arms):
            group = drugs.loc[drugs.arm.eq(arm) & drugs.scorer.eq(scorer)]
            points = {}
            for role, marker, offset in [
                ("reference", "o", 0),
                ("validation_selected", "o", -0.08),
                ("final_budget", "s", 0.08),
            ]:
                row = group.loc[group.role.eq(role)]
                if row.empty:
                    continue
                row = row.iloc[0]
                points[role] = (index + offset, row.rho_mean)
                axis.errorbar(
                    index + offset,
                    row.rho_mean,
                    yerr=[
                        [max(0, row.rho_mean - row.rho_min)],
                        [max(0, row.rho_max - row.rho_mean)],
                    ],
                    fmt=marker,
                    capsize=3,
                    color=COLORS[row.placement],
                    markersize=6,
                )
            if "validation_selected" in points and "final_budget" in points:
                axis.plot(
                    *zip(points["validation_selected"], points["final_budget"]),
                    color="#aaaaaa",
                    linestyle=":",
                )
        axis.set_xticks(
            range(len(model_arms)),
            [
                LABELS[drugs.loc[drugs.arm.eq(arm), "placement"].iloc[0]]
                for arm in model_arms
            ],
            rotation=20,
        )
        axis.set_title(scorer.replace("_", " "), fontsize=11)
        axis.axhline(0, color="#bbbbbb", linewidth=1)
        axis.set_ylim(-1, 1)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Spearman rho with measured singleton phenotypes")
    fig.suptitle(
        "GenTonik ranking under all three frozen readouts\nCircle: validation selected; square: 4,000 updates. Bars span training seeds.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "gentonik_readouts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    stats = efficiency.groupby("placement").mean(numeric_only=True)
    for axis, metric, title in zip(
        axes,
        ["trainable_parameters", "median_step_seconds", "peak_job_gpu_gib"],
        [
            "Trainable parameters (millions)",
            "Median training-step seconds",
            "Peak job GPU memory (GiB)",
        ],
    ):
        values = stats[metric] / (1e6 if metric == "trainable_parameters" else 1)
        axis.bar(stats.index, values, color=[COLORS[name] for name in stats.index])
        axis.set_title(title, fontsize=11)
        axis.set_xticks(
            range(len(stats)),
            [LABELS[name] for name in stats.index],
            rotation=20,
            fontsize=8,
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Measured resource use on H100; identical data and update budgets", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def learning_figures(out, models):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from .review import transfer

    unique = {model["training_id"]: model for model in models}
    native, measured = [], []
    selected_steps = {}
    for model in models:
        if "validation_selected" in model["roles"]:
            selected_steps[model["arm"]] = model["step"]
    for training_id, model in unique.items():
        folder = ROOT / "training" / training_id
        curve = pd.read_csv(folder / "learning_curve.csv")
        values = (
            curve.loc[curve.teacher.eq("fixed_initial") & ~curve.family.eq("replay")]
            .groupby("step")
            .loss.mean()
        )
        for step, value in values.items():
            native.append(
                {
                    "arm": model["arm"],
                    "placement": model["placement"],
                    "seed": model["training_seed"],
                    "step": step,
                    "value": value,
                }
            )
        for step in [500, 1000, 2000, 3000, 4000]:
            score = transfer(folder / "transfer" / f"step{step:05d}/metrics.csv")
            measured.append(
                {
                    "arm": model["arm"],
                    "placement": model["placement"],
                    "seed": model["training_seed"],
                    "step": step,
                    "value": score["new_mse"],
                }
            )
    native, measured = pd.DataFrame(native), pd.DataFrame(measured)
    native.to_csv(out / "native_validation_by_seed.csv", index=False)
    measured.to_csv(out / "transfer_validation_by_seed.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, table, label in zip(
        axes,
        [native, measured],
        [
            "Fixed-teacher native validation loss",
            "Weighted new-family validation relative MSE",
        ],
    ):
        for arm, group in table.groupby("arm"):
            stats = group.groupby("step").value.agg(["mean", "min", "max"])
            placement = group.placement.iloc[0]
            color = COLORS[placement]
            axis.plot(
                stats.index,
                stats["mean"],
                color=color,
                linewidth=2,
                label=LABELS[placement],
            )
            axis.fill_between(
                stats.index.to_numpy(),
                stats["min"].to_numpy(),
                stats["max"].to_numpy(),
                color=color,
                alpha=0.15,
            )
            if axis is axes[1]:
                step = selected_steps[arm]
                axis.scatter(
                    step,
                    stats.loc[step, "mean"],
                    marker="*",
                    s=110,
                    color=color,
                    zorder=5,
                )
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    baseline = transfer(ROOT / "benchmarks/baseline_aligned_val02/metrics.csv")
    axes[1].axhline(
        baseline["new_mse"],
        color="#999999",
        linestyle="--",
        linewidth=1,
        label="Released",
    )
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Native alignment loss and measured transfer validation\nBands span training seeds; stars mark the common selected step",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
