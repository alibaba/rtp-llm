"""CPU-only input integrity check and independent historical throughput calculation."""

import gzip
import hashlib
import json
import statistics
from pathlib import Path


def main():
    data = Path(__file__).resolve().parent
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["assets_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest, name
    long_runs = []
    for path in sorted((data / "results").glob("*.summary.json")):
        summary = json.loads(path.read_text())
        with gzip.open(data / "results" / summary["raw_requests"], "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == summary["cohort"]["requests"], path
        assert all(r["ok"] and not r.get("truncated") for r in rows), path
        assert all(
            r["usage"]["prompt_tokens"] == 24422
            and r["usage"]["prompt_tokens_details"]["video_tokens"] == 20240
            and r["aux_info"]["reuse_len"] == 0
            and r["aux_info"]["pd_sep"]
            for r in rows
        )
        for key, selected in [
            ("cohort", rows),
            (
                "steady",
                [r for r in rows if 30 <= r["ended_phase_s"] <= summary["dispatch_s"]],
            ),
        ]:
            metric = summary[key]
            inp = sum(r["usage"]["prompt_tokens"] for r in selected)
            out = sum(r["usage"]["completion_tokens"] for r in selected)
            assert len(selected) == metric["complete_success"]
            assert inp == metric["input_tokens"] and out == metric["output_tokens"]
            assert abs((inp + out) / metric["elapsed_s"] - metric["total_tps"]) < 1e-6
            assert abs(out / metric["elapsed_s"] - metric["output_tps"]) < 1e-6
        if path.name.startswith("long_"):
            assert summary["steady"]["elapsed_s"] >= 600
            long_runs.append(summary)
    assert len(long_runs) == 3
    assert sum(s["cohort"]["requests"] for s in long_runs) == 3360
    print(
        json.dumps(
            {
                "status": "PASSED",
                "historical_requests": 3360,
                "mean_steady_total_tps": statistics.mean(
                    s["steady"]["total_tps"] for s in long_runs
                ),
                "mean_steady_output_tps": statistics.mean(
                    s["steady"]["output_tps"] for s in long_runs
                ),
                "gpu_test_run": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
