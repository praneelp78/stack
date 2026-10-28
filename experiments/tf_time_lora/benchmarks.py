"""Use every new-family validation contrast and a fixed retention subset."""


def evaluate(model, spec, out):
    import pandas as pd
    from gentonik.io import atomic_json
    from gentonik.tf_timing.benchmarks import evaluate as native_evaluate

    if "panels" not in spec:
        return native_evaluate(model, spec, out)
    results = []
    frames = []
    for name, panel in spec["panels"].items():
        result = native_evaluate(
            model, {**panel, "seed": spec.get("seed", 0)}, out / name
        )
        results.append(result)
        frames.append(pd.read_csv(out / name / "metrics.csv"))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out / "metrics.csv", index=False)
    result = {
        "status": "complete",
        "spec": spec,
        "tasks": sum(r["tasks"] for r in results),
        "seconds": sum(r["seconds"] for r in results),
        "summary": frame.groupby(["dataset", "model"])
        .mean(numeric_only=True)
        .reset_index()
        .to_dict("records"),
    }
    atomic_json(out / "COMPLETE.json", result)
    return result
