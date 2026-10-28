"""Freeze the exact fork and external benchmark source before CLI submission."""

import hashlib
import json
import os
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[1]
benchmark = Path(os.environ["STACK_BENCHMARK_SOURCE"])
hashes = {}
for root in [repo / "src", repo / "tests", here, benchmark / "gentonik"]:
    for path in sorted(root.rglob("*.py")):
        hashes[str(path.relative_to(root.parent))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
hashes["study_config.json"] = hashlib.sha256(
    (here / "study_config.json").read_bytes()
).hexdigest()
snapshot = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
(here / "source_manifest.json").write_text(
    json.dumps({"snapshot": snapshot, "files": hashes}, indent=2)
)
print(snapshot)
