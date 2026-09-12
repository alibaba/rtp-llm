"""Compare real fixed-cohort results with explicit goodput thresholds."""

import argparse
import json
from pathlib import Path


def read_run(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    cases = {
        int(p.stem[1:]): json.loads(p.read_text())
        for p in path.glob("b*.json")
        if p.stem[1:].isdigit()
    }
    manifest = json.loads((path / "corpus-manifest.json").read_text())
    return metadata, cases, manifest


def goodput(case, threshold):
    rows = [r for r in case["rows"] if not r["error"]]
    if not rows or case["status"] != "completed":
        return 0.0
    elapsed = max(r["decode_end"] for r in rows) - min(r["decode_start"] for r in rows)
    return (
        sum(r["measured_tokens"] for r in rows if r["tpot_ms"] <= threshold) / elapsed
    )


def compare_outputs(left, right):
    lhs = {r["row"]: r["output_ids"] for r in left["rows"]}
    rhs = {r["row"]: r["output_ids"] for r in right["rows"]}
    return {
        "batch": left["batch"],
        "requests": len(lhs),
        "exact_match_requests": sum(lhs[r] == rhs[r] for r in lhs),
        "compared_tokens": sum(map(len, lhs.values())),
        "first_divergence_by_request": {
            str(row): next(
                (i for i, (a, b) in enumerate(zip(ids, rhs[row])) if a != b),
                None if len(ids) == len(rhs[row]) else min(len(ids), len(rhs[row])),
            )
            for row, ids in lhs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--offload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-baseline")
    args = parser.parse_args()
    baseline, offload = read_run(args.baseline), read_run(args.offload)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("kv_mib", "context", "tokens", "warmup", "max_batch"):
        if baseline[0]["args"][key] != offload[0]["args"][key]:
            raise ValueError(f"comparison mismatch: {key}")
    shared = min(len(baseline[2]), len(offload[2]))
    assert [s["prompt_sha256"] for s in baseline[2][:shared]] == [
        s["prompt_sha256"] for s in offload[2][:shared]
    ]
    reference = max(
        (c for c in baseline[1].values() if c["status"] == "completed"),
        key=lambda c: c["batch"],
    )
    thresholds = {
        "45_ms": 45.0,
        "baseline_plus_10pct": reference["tpot_ms"] * 1.1,
        "baseline_plus_20pct": reference["tpot_ms"] * 1.2,
    }
    rows, parity = [], []
    for scheme, run in (("vanilla", baseline), ("offload", offload)):
        for batch, case in sorted(run[1].items()):
            rows.append(
                {
                    "scheme": scheme,
                    "batch": batch,
                    "status": case["status"],
                    "tpot_ms": case.get("tpot_ms"),
                    "decode_tokens_per_second": case.get("decode_tokens_per_second", 0),
                    "request_failure_rate": case["request_failure_rate"],
                    "fixed_cohort_capacity_rejected": case.get(
                        "fixed_cohort_capacity_rejected", False
                    ),
                    "gpu_peak_mib": case["gpu_peak_mib"],
                    "graph_replay_verified": bool(case.get("graph_replay_evidence")),
                    "goodput": {
                        name: goodput(case, threshold)
                        for name, threshold in thresholds.items()
                    },
                }
            )
    for batch in sorted(baseline[1].keys() & offload[1].keys()):
        left, right = baseline[1][batch], offload[1][batch]
        if left["status"] != "completed" or right["status"] != "completed":
            continue
        parity.append(compare_outputs(left, right))
    repeat_parity = []
    if args.repeat_baseline:
        repeated = read_run(args.repeat_baseline)
        shared = min(len(baseline[2]), len(repeated[2]))
        assert [s["prompt_sha256"] for s in baseline[2][:shared]] == [
            s["prompt_sha256"] for s in repeated[2][:shared]
        ]
        for batch in sorted(baseline[1].keys() & repeated[1].keys()):
            left, right = baseline[1][batch], repeated[1][batch]
            if left["status"] == right["status"] == "completed":
                repeat_parity.append(compare_outputs(left, right))
    report = {
        "baseline": args.baseline,
        "offload": args.offload,
        "thresholds_ms": thresholds,
        "reference_batch": reference["batch"],
        "reference_tpot_ms": reference["tpot_ms"],
        "rows": rows,
        "output_parity": parity,
        "repeat_baseline": args.repeat_baseline,
        "repeat_baseline_output_parity": repeat_parity,
    }
    (destination / "comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Measured Decode Results",
        "",
        f"Reference: largest successful vanilla batch B{reference['batch']}, TPOT {reference['tpot_ms']:.2f} ms.",
        f"Goodput thresholds: {json.dumps(thresholds)}",
        "",
        "| Scheme | Batch | TPOT ms | Token/s | Goodput @45ms | Goodput @ref+10% | Goodput @ref+20% | Error rate | Peak GPU GiB | Graph |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        latency = "-" if row["tpot_ms"] is None else f"{row['tpot_ms']:.2f}"
        gp = row["goodput"]
        lines.append(
            f"| {row['scheme']} | {row['batch']} | {latency} | {row['decode_tokens_per_second']:.2f} | "
            f"{gp['45_ms']:.2f} | {gp['baseline_plus_10pct']:.2f} | {gp['baseline_plus_20pct']:.2f} | "
            f"{row['request_failure_rate']:.0%} | {max(row['gpu_peak_mib']) / 1024:.2f} | "
            f"{'verified' if row['graph_replay_verified'] else '-'} |"
        )
    lines += [
        "",
        "Errors count benchmark response failures, never SLO misses. Fixed-cohort capacity rejection is an admission error imposed by this benchmark; normal serving can queue or shrink a batch. It is not physical CUDA OOM.",
        "",
        "Goodput excludes prefill: successful measured output tokens from requests meeting the per-request mean TPOT SLO, divided by decode wall time. These results do not measure mixed prefill/decode interference or end-to-end serving goodput.",
        "",
        "## Output Parity",
        "",
    ]
    for item in parity:
        lines.append(
            f"- B{item['batch']}: identical complete outputs for {item['exact_match_requests']}/{item['requests']} requests; {item['compared_tokens']} tokens compared."
        )
    if repeat_parity:
        lines += [
            "",
            "Vanilla-versus-vanilla output reference (not additional performance samples):",
            "",
        ]
        for item in repeat_parity:
            lines.append(
                f"- B{item['batch']}: {item['exact_match_requests']}/{item['requests']} identical complete outputs."
            )
        lines += [
            "",
            "Free-running greedy outputs also diverge between vanilla runs. This reference used different fixed-pool/reserve settings and uncontrolled request admission order; it does not isolate a numerical cause. Byte-preserving cache tests are independent evidence, not proof of whole-model quality equivalence.",
        ]
    (destination / "TABLE.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
