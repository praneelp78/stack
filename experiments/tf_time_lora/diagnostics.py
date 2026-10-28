"""Full-update spectra, computed without using benchmark outcomes to set ranks."""

import json

from .common import ROOT, SOURCE_ROOT


def spectra(spec):
    import pandas as pd
    import torch
    from gentonik.io import atomic_json, sha256_file
    from gentonik.neural_adaptation.common import ALIGNED, BASE

    from stack.finetune.lora import linear_group

    torch.set_num_threads(8)
    paths = {"base": BASE, "aligned": ALIGNED}
    reference = paths.get(spec.get("reference", "aligned"), spec.get("reference"))
    reference_state = torch.load(reference, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    rows = []
    for checkpoint in spec["checkpoints"]:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        for key, tensor in state.items():
            if (
                not key.startswith("model.")
                or not key.endswith(".weight")
                or tensor.ndim != 2
            ):
                continue
            if key not in reference_state:
                continue
            delta = (tensor.float() - reference_state[key].float()).to(
                "cuda", torch.float64
            )
            energy = float(delta.square().sum())
            gram = (
                delta.T @ delta if delta.shape[0] >= delta.shape[1] else delta @ delta.T
            )
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
            cumulative = eigenvalues.cumsum(0)
            fractions = cumulative / max(energy, 1e-30)
            row = {
                "checkpoint": str(checkpoint),
                "reference": str(reference),
                "module": key,
                "group": linear_group(
                    key.removeprefix("model.").removesuffix(".weight")
                ),
                "rows": delta.shape[0],
                "cols": delta.shape[1],
                "delta_squared_norm": energy,
                "delta_over_weight_norm": energy**0.5
                / max(float(reference_state[key].norm()), 1e-30),
                "stable_rank": energy / max(float(eigenvalues[0]), 1e-30),
            }
            for rank in [2, 4, 8, 16, 32, 64, 128]:
                row[f"energy_rank{rank}"] = (
                    float(fractions[min(rank, len(fractions)) - 1]) if energy else 1.0
                )
            for level in [0.9, 0.95, 0.99]:
                row[f"rank{int(level * 100)}"] = (
                    int(torch.searchsorted(fractions, level)) + 1 if energy else 0
                )
            rows.append(row)
            print(json.dumps({"spectrum": key, "rank90": row["rank90"]}), flush=True)
        del state
    out = ROOT / "diagnostics" / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "spectra.csv", index=False)
    result = {
        "status": "complete",
        "rows": len(rows),
        "spec": spec,
        "checkpoint_hashes": {
            str(p): sha256_file(p) for p in [reference, *spec["checkpoints"]]
        },
    }
    atomic_json(out / "COMPLETE.json", result)
    return result


def inventory(spec):
    from collections import Counter

    import pandas as pd
    from gentonik.io import sha256_file
    from gentonik.tf_timing.benchmarks import fixture_path

    corpus = SOURCE_ROOT / "data/tf_corpus"
    obs = pd.read_parquet(corpus / "obs.parquet")
    fixtures = {}
    for name in spec["taskfiles"]:
        path = fixture_path(name + ".json")
        tasks = json.loads(path.read_text())
        fixtures[name] = {
            "sha256": sha256_file(path),
            "counts": dict(Counter(t["split"] for t in tasks)),
            "datasets": sorted({t["dataset"] for t in tasks}),
        }
    return {
        "status": "complete",
        "corpus": str(corpus),
        "admission": json.loads((corpus / "COMPLETE.json").read_text()),
        "obs_sha256": sha256_file(corpus / "obs.parquet"),
        "split_counts": obs.split.value_counts().to_dict(),
        "family_split_counts": {
            f"{f}/{s}": int(n)
            for (f, s), n in obs.groupby(["family", "split"]).size().items()
        },
        "fixtures": fixtures,
    }
