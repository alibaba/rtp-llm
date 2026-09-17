"""Validate the required task inventory before accepting a tau2 score."""

import json
from collections import Counter
from pathlib import Path


def validate_task_coverage(report_path: str, task_ids_path: str) -> dict[str, int]:
    report_file = Path(report_path)
    report = json.loads(report_file.read_text())
    task_ids = json.loads(Path(task_ids_path).read_text())
    if not isinstance(task_ids, dict) or not task_ids:
        raise ValueError("tau2 task inventory is empty or invalid")
    expected = {}
    for domain, ids in task_ids.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"tau2 {domain}: empty or invalid task inventory")
        counts = Counter(str(task_id) for task_id in ids)
        if any(count != 1 for count in counts.values()):
            raise ValueError(f"tau2 {domain}: duplicate required task IDs")
        expected[domain] = counts

    metrics = [m for m in report.get("metrics", []) if m.get("name") == "mean_acc"]
    if len(metrics) != 1:
        raise ValueError("tau2 report must contain one mean_acc coverage metric")
    metric = metrics[0]
    expected_counts = {domain: len(ids) for domain, ids in expected.items()}
    total = sum(expected_counts.values())
    actual_counts = {}
    for category in metric.get("categories", []):
        for subset in category.get("subsets", []):
            domain = subset.get("name")
            if domain in actual_counts:
                raise ValueError(f"tau2 report repeats subset {domain}")
            actual_counts[domain] = subset.get("num")
    if metric.get("num") != total or actual_counts != expected_counts:
        raise ValueError(
            f"tau2 incomplete execution: {metric.get('num')}/{total}; "
            f"scored={actual_counts}, required={expected_counts}"
        )

    # evalscope 1.6 writes one prediction per task. Check identities as well as
    # counts: duplicated/stale results must not hide a missing required task.
    predictions = report_file.parents[2] / "predictions" / report_file.parent.name
    for domain, required in expected.items():
        path = predictions / f"tau2_bench_{domain}.jsonl"
        actual = Counter()
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                actual[str(record.get("metadata", {}).get("id"))] += 1
        if actual != required:
            raise ValueError(
                f"tau2 {domain} task mismatch: "
                f"missing={list((required - actual).elements())}, "
                f"unexpected={list((actual - required).elements())}"
            )
    return expected_counts
