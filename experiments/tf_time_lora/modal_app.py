"""CLI entry point; original corpus is read-only, results use a new Volume."""

import hashlib
import json
import os
from pathlib import Path

import modal

HERE = (
    Path(__file__).resolve().parent if modal.is_local() else Path("/opt/tf_time_lora")
)
REPO = HERE.parents[1] if modal.is_local() else Path("/opt/fork")
RUN = "stack_lora_tf_time_20261028"
app = modal.App(RUN.replace("_", "-"))
source_volume = modal.Volume.from_name("gentonik-data-20260925")
results_volume = modal.Volume.from_name(
    RUN.replace("_", "-"), create_if_missing=True, version=2
)

if modal.is_local():
    benchmark = Path(os.environ["STACK_BENCHMARK_SOURCE"]).resolve()
    # Match the completed campaign's CUDA stack; only the editable fork changes.
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "libgomp1")
        .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
        .pip_install(
            "numpy==1.26.4",
            "pandas==2.2.3",
            "anndata==0.11.4",
            "scanpy==1.11.2",
            "scvi-tools==1.3.3",
            "pytorch-lightning==2.5.2",
            "geomloss==0.2.6",
            "wandb==0.21.1",
            "pyarrow==19.0.1",
            "scikit-learn==1.6.1",
        )
        .run_commands(
            "git clone https://github.com/ArcInstitute/stack /opt/stack",
            "cd /opt/stack && git checkout cacc2e4b09435c3e536d46237d10b50f222dd144",
            "pip install --no-deps /opt/stack",
        )
        .pip_install("pytest==8.3.5")
        .env({"PYTHONPATH": "/opt/fork/src:/opt", "OMP_NUM_THREADS": "8"})
        .add_local_dir(REPO / "src", "/opt/fork/src", ignore=["**/__pycache__/**"])
        .add_local_dir(HERE, "/opt/tf_time_lora", ignore=["**/__pycache__/**"])
        .add_local_dir(REPO / "tests", "/opt/lora_tests", ignore=["**/__pycache__/**"])
        .add_local_dir(
            benchmark / "gentonik", "/opt/gentonik", ignore=["**/__pycache__/**"]
        )
        .add_local_dir(benchmark / "configs", "/workspace/configs")
        .add_local_dir(benchmark / "data/manifests", "/workspace/data/manifests")
        .add_local_file(
            benchmark / "outputs/gentonik-benchmark/human_sc_compound_matches.csv",
            "/workspace/data/matches.csv",
        )
    )
else:
    image = modal.Image.debian_slim()

VOLUMES = {
    "/data": source_volume.with_mount_options(read_only=True),
    "/results": results_volume,
}


def execute(spec):
    import resource
    import time
    import traceback

    from gentonik.io import atomic_json

    from tf_time_lora.common import ROOT

    results_volume.reload()
    started = time.monotonic()
    source_snapshot = None
    try:
        manifest = json.loads(
            Path("/opt/tf_time_lora/source_manifest.json").read_text()
        )
        source_snapshot = manifest["snapshot"]
        roots = {
            "src": Path("/opt/fork/src"),
            "tests": Path("/opt/lora_tests"),
            "tf_time_lora": Path("/opt/tf_time_lora"),
            "gentonik": Path("/opt/gentonik"),
        }
        for filename, expected_hash in manifest["files"].items():
            if filename == "study_config.json":
                path = Path("/opt/tf_time_lora/study_config.json")
            else:
                root, relative = filename.split("/", 1)
                path = roots[root] / relative
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Runtime source differs from frozen manifest: {filename}"
                )
        kind = spec["kind"]
        if kind == "train":
            from tf_time_lora.training import train

            result = train(spec, results_volume.commit)
        elif kind == "smoke":
            from tf_time_lora.verification import smoke

            result = smoke(spec)
        elif kind in {"spectra", "inventory"}:
            from tf_time_lora import diagnostics

            result = getattr(diagnostics, kind)(spec)
        elif kind in {
            "checkpoint_audit",
            "metric_audit",
            "unit_tests",
            "compare_parent",
        }:
            from tf_time_lora import verification

            result = getattr(verification, kind)(spec)
        elif kind == "diagnostic_figures":
            from tf_time_lora.analysis import diagnostic_figures

            result = diagnostic_figures(spec)
        elif kind == "summarize":
            from tf_time_lora.analysis import summarize

            result = summarize(spec)
        elif kind == "benchmark":
            from gentonik.io import sha256_file
            from gentonik.neural_adaptation.models import (
                checkpoint_path,
                load_inference,
            )

            from tf_time_lora.benchmarks import evaluate

            out = ROOT / "benchmarks" / spec["id"]
            if (out / "COMPLETE.json").exists():
                raise RuntimeError("Complete benchmark; use new identity")
            result = evaluate(load_inference(spec), spec, out)
            result["checkpoint_sha256"] = sha256_file(checkpoint_path(spec))
            atomic_json(out / "COMPLETE.json", result)
        elif kind == "drugs":
            from gentonik.tf_timing import drugs

            drugs.ROOT = ROOT
            result = drugs.drug_benchmark(spec, results_volume.commit)
        elif kind == "metadata":
            import tarfile

            files = [
                p
                for p in ROOT.rglob("*")
                if p.is_file()
                and p.suffix in {".json", ".csv"}
                and "provenance" not in p.parts
            ]
            with tarfile.open(ROOT / "metadata.tar.gz", "w:gz") as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
            result = {
                "status": "complete",
                "files": len(files),
                "archive": "/results/metadata.tar.gz",
            }
        else:
            raise ValueError(f"Unknown job kind {kind}")
    except Exception as error:  # noqa: BLE001 -- preserve remote failures as explicit receipts
        result = {
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    result.update(
        spec=spec,
        source_snapshot=source_snapshot,
        seconds_worker=time.monotonic() - started,
        peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    )
    receipt = ROOT / "jobs" / f"{spec['id']}_{time.time_ns()}.json"
    atomic_json(receipt, result)
    results_volume.commit()
    # Return plain JSON types so the CLI needs no training dependencies.
    return json.loads(receipt.read_text())


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    cpu=8,
    memory=196608,
    timeout=14400,
    max_containers=9,
)
def gpu_worker(spec):
    return execute(spec)


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=14400,
    max_containers=9,
)
def inference_worker(spec):
    return execute(spec)


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=7200,
    max_containers=1,
)
def diagnostic_worker(spec):
    return execute(spec)


@app.function(
    image=image, volumes=VOLUMES, cpu=4, memory=32768, timeout=3600, max_containers=2
)
def cpu_worker(spec):
    return execute(spec)


@app.local_entrypoint()
def main(specs: str = "", action: str = "submit", ledger: str = ""):
    import datetime

    folder = Path(os.environ.get("STACK_LORA_RUN_DIR", "work/lora-study")).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger) if ledger else folder / "jobs.jsonl"
    if action == "collect":
        records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        for record in records:
            receipt = folder / "receipts" / (record["call_id"] + ".json")
            if receipt.exists():
                continue
            try:
                result = modal.FunctionCall.from_id(record["call_id"]).get(timeout=0)
            except TimeoutError:
                continue
            except Exception as error:  # noqa: BLE001 -- preserve remote failures as explicit receipts
                result = {
                    "status": "failed",
                    "spec": record["spec"],
                    "error": str(error),
                    "exception_type": type(error).__name__,
                }
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "id": record["spec"]["id"],
                        "status": result["status"],
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )
        return
    if action != "submit":
        raise ValueError(action)
    rows = json.loads(Path(specs).read_text())
    hashes = {}
    for root in [REPO / "src", REPO / "tests", HERE, benchmark / "gentonik"]:
        for path in sorted(root.rglob("*.py")):
            hashes[str(path.relative_to(root.parent))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    hashes["study_config.json"] = hashlib.sha256(
        (HERE / "study_config.json").read_bytes()
    ).hexdigest()
    snapshot = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {"snapshot": snapshot, "files": hashes}
    # This file is sent with the image and hashed by every trainer.
    expected = json.loads((HERE / "source_manifest.json").read_text())
    if expected != manifest:
        raise RuntimeError(
            "Source changed; regenerate source_manifest.json before launching"
        )
    import tarfile

    bundle = folder / (snapshot + ".tar.gz")
    if not bundle.exists():
        with tarfile.open(bundle, "w:gz") as archive:
            for root in [REPO / "src", REPO / "tests", HERE, benchmark / "gentonik"]:
                for path in sorted(root.rglob("*.py")):
                    archive.add(path, arcname=str(path.relative_to(root.parent)))
            for name in ["study_config.json", "source_manifest.json"]:
                archive.add(HERE / name, arcname="tf_time_lora/" + name)
        with results_volume.batch_upload() as upload:
            upload.put_file(bundle, f"provenance/{snapshot}.tar.gz")
    cpu_kinds = {
        "compare_parent",
        "inventory",
        "metadata",
        "metric_audit",
        "diagnostic_figures",
        "unit_tests",
        "summarize",
    }
    for spec in rows:
        kind = spec["kind"]
        if kind in cpu_kinds:
            fn = cpu_worker
        elif kind == "train":
            fn = gpu_worker
        elif kind in {"benchmark", "drugs"}:
            fn = inference_worker
        else:
            fn = diagnostic_worker
        call = fn.spawn(spec)
        record = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "call_id": call.object_id,
            "spec": spec,
            "source_snapshot": snapshot,
        }
        with ledger_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
