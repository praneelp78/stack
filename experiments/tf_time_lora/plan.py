"""Generate the prespecified pilot grid without launching or replacing runs."""

import copy
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pilot_specs(config):
    rows = []
    sweep = config["sweep"]
    for rank in sweep["ranks"]:
        for placement in ["cross", "both", "cross_io"]:
            ranks = {"inter_cell": rank}
            if placement == "both":
                ranks["intra_cell"] = sweep["intra_cell_rank"]
            if placement == "cross_io":
                ranks.update(encoder=rank, decoder=rank)
            for lr in sweep["learning_rates"]:
                rows.append(
                    {
                        "id": f"pilot_{placement}_r{rank}_lr{lr:g}_s0",
                        "lr": lr,
                        "lora": {"ranks": dict(ranks), "alpha_over_rank": 1.0},
                        "placement": placement,
                        "rank": rank,
                    }
                )
    for lr in [5e-6, 2e-5]:
        rows.append(
            {
                "id": f"pilot_full_lr{lr:g}_s0",
                "lr": lr,
                "placement": "full",
                "lora": None,
            }
        )
    controls = [
        ("within", {"intra_cell": 2}),
        ("io", {"encoder": 16, "decoder": 16}),
        (
            "all",
            {
                "inter_cell": 16,
                "intra_cell": 2,
                "token_mlp": 2,
                "encoder": 16,
                "decoder": 16,
            },
        ),
    ]
    for placement, ranks in controls:
        for lr in [2e-5, 1e-4]:
            rows.append(
                {
                    "id": f"pilot_{placement}_lr{lr:g}_s0",
                    "lr": lr,
                    "placement": placement,
                    "rank": 2 if placement == "within" else 16,
                    "lora": {"ranks": dict(ranks)},
                }
            )
    rows.append(
        {
            "id": "pilot_heads_lr0.0001_s0",
            "lr": 1e-4,
            "placement": "heads",
            "lora": {"ranks": {}},
        }
    )
    specs = []
    for row in rows:
        template = copy.deepcopy(config["training"])
        template.update(
            updates=sweep["pilot_updates"],
            validate_every=500,
            validation_sets_per_family=32,
            intermediate_benchmark=config["evaluation"]["pilot"],
            bench_steps=[500],
        )
        template.update(row)
        specs.append(template)
    assert len(specs) == len({spec["id"] for spec in specs}) == 27
    return specs


if __name__ == "__main__":
    config = json.loads((HERE / "study_config.json").read_text())
    folder = Path(os.environ.get("STACK_LORA_RUN_DIR", "work/lora-study"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "pilot_specs.json"
    specs = pilot_specs(config)
    if path.exists():
        assert json.loads(path.read_text()) == specs, (
            "Existing specs differ; preserve them and choose a new directory"
        )
    else:
        path.write_text(json.dumps(specs, indent=2))
    print(json.dumps({"pilot_arms": len(specs), "specs": str(path)}))
