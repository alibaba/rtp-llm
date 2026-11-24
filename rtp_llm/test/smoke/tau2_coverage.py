"""Check pass@1 coverage independently of EvalScope's aggregate score."""

import json
from collections import Counter
from pathlib import Path


def require_all_reviews(work_dir, report_path, manifest_path, model):
    # work_dir is fresh for this invocation. Never credit a previous review.
    directories = list(Path(work_dir).glob("*/reviews"))
    if len(directories) != 1 or not Path(report_path).resolve().is_relative_to(
        directories[0].parent.resolve()
    ):
        raise ValueError("Expected the score and reviews from the same tau2 run")
    review_dir = directories[0] / model
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Empty tau2 task manifest")
    for domain, task_ids in manifest.items():
        expected = Counter(map(str, task_ids))
        if not expected or any(count != 1 for count in expected.values()):
            raise ValueError(f"Expected distinct pass@1 task IDs for {domain}")
        actual = Counter()
        with (review_dir / f"tau2_bench_{domain}.jsonl").open(encoding="utf-8") as rows:
            for row in rows:
                actual[str(json.loads(row)["sample_score"]["sample_metadata"]["id"])] += 1
        if actual != expected:
            raise ValueError(
                f"Incomplete or duplicate {domain} reviews: "
                f"missing={dict(expected - actual)}, extra={dict(actual - expected)}"
            )
