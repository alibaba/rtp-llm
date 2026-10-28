import json

import pytest

from rtp_llm.test.smoke.tau2_report import validate_task_coverage


def _write_results(tmp_path, counts, predicted_ids=None):
    required = {"airline": ["4", "5", "40"], "retail": [str(i) for i in range(17)], "telecom": ["easy", "hard"]}
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps(required))
    report = tmp_path / "reports" / "model" / "tau2_bench.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"score": 0.8, "metrics": [{
        "name": "mean_acc", "num": sum(counts.values()),
        "categories": [{"subsets": [{"name": k, "num": v} for k, v in counts.items()]}],
    }]}))
    predictions = tmp_path / "predictions" / "model"
    predictions.mkdir(parents=True)
    for domain, ids in (predicted_ids or required).items():
        (predictions / f"tau2_bench_{domain}.jsonl").write_text(
            "\n".join(json.dumps({"metadata": {"id": value}}) for value in ids)
        )
    return str(report), str(tasks)


def test_all_required_tasks_are_scored(tmp_path):
    counts = {"airline": 3, "retail": 17, "telecom": 2}
    assert validate_task_coverage(*_write_results(tmp_path, counts)) == counts


def test_twenty_of_twenty_two_cannot_pass_with_high_score(tmp_path):
    paths = _write_results(tmp_path, {"airline": 3, "retail": 17})
    with pytest.raises(ValueError, match="20/22"):
        validate_task_coverage(*paths)


def test_duplicate_prediction_cannot_replace_required_task(tmp_path):
    paths = _write_results(tmp_path, {"airline": 3, "retail": 17, "telecom": 2})
    prediction = tmp_path / "predictions" / "model" / "tau2_bench_telecom.jsonl"
    prediction.write_text('\n'.join([json.dumps({"metadata": {"id": "easy"}})] * 2))
    with pytest.raises(ValueError, match="task mismatch.*hard.*easy"):
        validate_task_coverage(*paths)


def test_missing_predictions_cannot_pass(tmp_path):
    paths = _write_results(tmp_path, {"airline": 3, "retail": 17, "telecom": 2})
    (tmp_path / "predictions" / "model" / "tau2_bench_telecom.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        validate_task_coverage(*paths)
