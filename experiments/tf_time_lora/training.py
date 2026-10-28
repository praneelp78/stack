"""Full-parameter or group-specific LoRA alignment on the frozen TF/time corpus.

Student, EMA teacher, optimizer, all RNGs and sampler coverage are resumable.
Validation uses eval mode and fixed cells/masks; a frozen teacher diagnostic
keeps the target comparable when the native EMA teacher changes. Model choice
uses measured transfer validation separately from these training losses.
"""

import copy
import json
import math
import random
import time

import numpy as np
from gentonik.io import atomic_json, sha256_file
from gentonik.neural_adaptation.training import initialize, restore_rng, rng_state
from gentonik.neural_campaign.sampling import MixtureSampler

from stack.finetune.lora import apply_lora, merged_state_dict

from .common import ROOT, SOURCE_ROOT

KEYS = [
    "loss",
    "recon_loss",
    "mmd_loss",
    "sw_loss",
    "cls_loss",
    "cls_acc",
    "masked_mae",
    "masked_corr",
    "sw_predict",
    "mask_rate",
]


def load_corpus(name):
    import pandas as pd
    from gentonik.tf_timing.data import load_counts

    p = SOURCE_ROOT / "data" / name
    if not (p / "COMPLETE.json").exists():
        raise RuntimeError(f"Dataset not yet admitted: {name}")
    return load_counts(p), pd.read_parquet(p / "obs.parquet"), p


def train(spec, commit):
    import pandas as pd
    import torch

    torch.set_num_threads(8)
    seed = int(spec.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Modal reuses containers across jobs; measure this run's allocator peak.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = ROOT / "training" / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise RuntimeError("Complete training run; use new identity")
    counts, obs, datafolder = load_corpus(spec["dataset"])
    model, mc, post, init = initialize({**spec, "kind": "alignment"})
    assert post
    # Adapter initialization must not shift masks/times relative to matched full FT.
    initialization_rng = rng_state()
    adapter_summary = None
    if spec.get("lora") is not None:
        adapter_summary = apply_lora(model.model, spec["lora"])
    restore_rng(initialization_rng)
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    from .verification import frozen_hash

    frozen_initial = frozen_hash(model.model)
    atomic_json(
        out / "adapter_summary.json",
        adapter_summary
        or {"mode": "full", "trainable_parameters": sum(p.numel() for p in trainable)},
    )
    if spec.get("attention_backend") == "sdpa":
        from gentonik.neural_campaign.kernels import install_sdpa

        install_sdpa(model)
    fixed_teacher = copy.deepcopy(model.teacher_model)
    fixed_teacher.eval()
    weights = spec["weights"]
    valweights = spec.get("validation_weights", weights)
    sampler = MixtureSampler(
        counts,
        obs,
        "train",
        seed,
        weights,
        prompt_design=spec.get("prompt_design", "mixed"),
    )
    validation_seeds = {
        "Herring": 4242,
        "Velmeshev": 4243,
        "replay": 4244,
        "Jerber": 4245,
        "HNOCA": 4246,
        "Lin2025": 4247,
        "Lin2021": 4248,
    }
    validators = {
        f: MixtureSampler(counts, obs, "val", validation_seeds[f], {f: 1.0})
        for f in sorted(valweights)
        if valweights[f] > 0
    }
    # Evaluate an equal number per family regardless of training mixture.
    auditstate = copy.deepcopy(sampler.state())
    for _ in range(128):
        sampler.indices()
    audit = {
        "passed": True,
        "samples": 128,
        "training": sampler.summary,
        "validation": {k: v.summary for k, v in validators.items()},
        "audit_coverage": sampler.coverage(),
        "validation_seeds": validation_seeds,
    }
    sampler.restore(auditstate)
    atomic_json(out / "sampler_audit.json", audit)
    lr = float(spec["lr"])
    updates = int(spec["updates"])
    micro = int(spec.get("microbatch", 4))
    accum = int(spec.get("accumulation", 4))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.003)
    inputs = {n: sha256_file(datafolder / n) for n in ["obs.parquet", "COMPLETE.json"]}
    inputs["counts.npz"] = json.loads((datafolder / "COMPLETE.json").read_text())[
        "data_sha256"
    ]
    inputs["mmap"] = sha256_file(datafolder / "mmap/COMPLETE.json")
    inputs["config"] = sha256_file("/opt/tf_time_lora/study_config.json")
    inputs["sources"] = sha256_file("/opt/tf_time_lora/source_manifest.json")
    history = []
    start_step = 0
    best = float("inf")
    teacher_updates = 0
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        assert ck["spec"] == spec and ck["inputs"] == inputs
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        sampler.restore(ck["sampler_state"])
        history = ck["history"]
        start_step = ck["step"]
        best = ck["best"]
        teacher_updates = ck["teacher_updates"]
    started = time.monotonic()

    def forward(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model._forward_pass_with_teacher(batch)

    def numeric(r):
        return {k: float(v.detach()) for k, v in r.items() if k in KEYS}

    def validate(step):
        saved = rng_state()
        mode = model.model.training
        model.model.eval()
        model.teacher_model.eval()
        rows = []
        native_teacher = model.teacher_model
        for teachername in ["native_ema", "fixed_initial"]:
            model.teacher_model = (
                native_teacher if teachername == "native_ema" else fixed_teacher
            )
            for i, (family, val) in enumerate(validators.items()):
                old = copy.deepcopy(val.state())
                val.rng = np.random.default_rng(validation_seeds[family])
                torch.manual_seed(validation_seeds[family] + 101)
                torch.cuda.manual_seed_all(validation_seeds[family] + 101)
                pieces = []
                with torch.inference_mode():
                    for _ in range(
                        int(spec.get("validation_sets_per_family", 32)) // micro
                    ):
                        pieces.append(numeric(forward(val.batch(micro))))
                val.restore(old)
                rows.append(
                    {
                        "step": step,
                        "split": "val",
                        "family": family,
                        "teacher": teachername,
                        "teacher_updates": teacher_updates,
                        "validation_seed": validation_seeds[family],
                        **{
                            k: float(np.mean([r[k] for r in pieces])) for k in pieces[0]
                        },
                    }
                )
        model.teacher_model = native_teacher
        model.model.train(mode)
        restore_rng(saved)
        for row in rows:
            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"Invalid validation {row}")
        return rows

    def save(step, rows):
        nonlocal best
        score = float(
            np.mean(
                [
                    r["loss"]
                    for r in rows
                    if r["teacher"] == "fixed_initial" and r["family"] != "replay"
                ]
            )
        )
        student = {
            "state_dict": {
                "model." + k: v.detach().cpu()
                for k, v in merged_state_dict(model.model).items()
            },
            "hyper_parameters": {"model_config": mc},
            "adaptation_spec": spec,
            "step": step,
            "inputs": inputs,
            "lora": spec.get("lora"),
        }
        if adapter_summary is not None:
            adapter = {
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.model.state_dict().items()
                    if ".lora_" in k
                    or k == "query_pos_embedding"
                    or k.startswith("cls.")
                },
                "lora": spec["lora"],
                "initialization": init,
                "step": step,
            }
            torch.save(adapter, out / f"adapter{step:05d}.pt")
        name = f"step{step:05d}.ckpt"
        torch.save(student, out / (name + ".tmp"))
        (out / (name + ".tmp")).replace(out / name)
        if score < best:
            best = score
            atomic_json(
                out / "best_loss_checkpoint.json",
                {
                    "checkpoint": str(out / name),
                    "step": step,
                    "fixed_teacher_neural_loss": score,
                    "selection": "Loss diagnostic only; measured transfer validation selects final checkpoint.",
                },
            )
        resume = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(),
            "sampler_state": sampler.state(),
            "step": step,
            "spec": spec,
            "inputs": inputs,
            "history": history,
            "best": best,
            "teacher_updates": teacher_updates,
            "hyper_parameters": {"model_config": mc},
        }
        torch.save(resume, out / "resume.pt.tmp")
        (out / "resume.pt.tmp").replace(out / "resume.pt")
        pd.DataFrame(history).to_csv(out / "learning_curve.csv", index=False)
        atomic_json(out / "coverage.json", sampler.coverage())
        atomic_json(
            out / "progress.json",
            {
                "step": step,
                "updates": updates,
                "teacher_updates": teacher_updates,
                "latest_checkpoint": str(out / name),
                "validation": rows,
                "seconds": time.monotonic() - started,
            },
        )
        commit()
        if (
            step > 0
            and spec.get("intermediate_benchmark")
            and step in spec.get("bench_steps", [500, 1000, 2000, 4000])
        ):
            from .benchmarks import evaluate

            saved_rng = rng_state()
            was_training = model.model.training
            model.model.eval()
            try:
                # Every selection metric uses the exported inference artifact.
                from gentonik.neural_adaptation.models import load_inference

                benchmark_model = load_inference({"checkpoint": str(out / name)})
                evaluate(
                    benchmark_model,
                    {**spec["intermediate_benchmark"], "seed": 0},
                    out / "transfer" / f"step{step:05d}",
                )
                del benchmark_model
            finally:
                model.model.train(was_training)
                restore_rng(saved_rng)
            commit()

    if start_step == 0:
        rows = validate(0)
        history.extend(rows)
        save(0, rows)
    schedule_updates = int(spec.get("schedule_updates", updates))
    warmup = int(spec.get("warmup_updates", 50))
    for step in range(start_step + 1, updates + 1):
        factor = min(1.0, step / warmup) * (
            0.1
            + 0.9
            * 0.5
            * (1 + math.cos(math.pi * min(step, schedule_updates) / schedule_updates))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        optimizer.zero_grad(set_to_none=True)
        pieces = []
        tick = time.monotonic()
        for _ in range(accum):
            r = forward(sampler.batch(micro))
            loss = r["loss"] / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training at {step}")
            loss.backward()
            pieces.append(numeric(r))
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        if step % int(spec.get("teacher_every", 500)) == 0:
            # Identical parameter EMA and buffer copy to upstream hook (after
            # 500 completed student updates, before the next optimizer step).
            model._ema_update_teacher()
            teacher_updates += 1
        history.append(
            {
                "step": step,
                "split": "train",
                "lr": lr * factor,
                "grad_norm": float(norm),
                "teacher_updates": teacher_updates,
                "seconds": time.monotonic() - tick,
                **{k: float(np.mean([r[k] for r in pieces])) for k in pieces[0]},
            }
        )
        if step % int(spec.get("validate_every", 250)) == 0 or step == updates:
            rows = validate(step)
            history.extend(rows)
            save(step, rows)
            print(
                json.dumps(
                    {
                        "id": spec["id"],
                        "step": step,
                        "teacher_updates": teacher_updates,
                        "val": rows,
                    }
                ),
                flush=True,
            )
    frozen_final = frozen_hash(model.model)
    if frozen_final != frozen_initial:
        raise RuntimeError("Frozen backbone parameters changed")
    result = {
        "status": "complete",
        "trainable_parameters": sum(p.numel() for p in trainable),
        "adapter_summary": adapter_summary,
        "frozen_initial_sha256": frozen_initial,
        "frozen_final_sha256": frozen_final,
        "spec": spec,
        "initialization": init,
        "inputs": inputs,
        "updates": updates,
        "teacher_updates": teacher_updates,
        "best_fixed_teacher_neural_loss": best,
        "seconds": time.monotonic() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "memory_stats_reset": True,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cell_presentations": updates * micro * accum * 512,
        "train_cells": int(obs.split.eq("train").sum()),
        "coverage": sampler.coverage(),
        "final_checkpoint": str(out / f"step{updates:05d}.ckpt"),
        "checkpoint_sha256": sha256_file(out / f"step{updates:05d}.ckpt"),
        "parameters": sum(p.numel() for p in model.model.parameters()),
    }
    atomic_json(out / "COMPLETE.json", result)
    commit()
    return result
