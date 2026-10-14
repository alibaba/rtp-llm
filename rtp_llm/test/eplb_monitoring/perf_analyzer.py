"""ABBA paired-run performance analyzer for EPLB monitoring overhead.

Compares monitoring OFF vs ON on the same GPU/model/requests.  Input runs
arrive in ABBA order per cycle (off, on, on, off), each run a dict:

    {
      "cycle": 0,                        # int, ABBA cycle index
      "mode": "off" | "on",
      "samples": [
        {"prefill_ms": float,
         "decode_ms_per_token": float,
         "throughput_tokens_per_s": float,
         "output_len": int,              # optional together with wall_s
         "wall_s": float},
        ...
      ],
      "window_count": int                # monitor flush windows observed
    }

Per cycle the two runs of each mode are merged. Both the median and the
amortized estimate are checked: arithmetic mean for latency, total output
tokens / total wall time for throughput. Legacy samples without output_len
and wall_s must have equal output token counts; their throughput estimate is
the harmonic mean. All samples must use the same metadata form.
Per-cycle paired regression for either estimate is:

    latency metrics (prefill_ms, decode_ms_per_token): on/off - 1
    throughput     (throughput_tokens_per_s):          off/on - 1

A performance conclusion requires >= 3 complete ABBA cycles AND every ON run
observing >= 2 flush windows; otherwise the result is "inconclusive" (never a
pass).  A metric is a "regression" only when a MAJORITY of cycles exceed the
threshold in the same (degrading) direction for either estimate; a minority is
reported as "noisy".  Raw samples are preserved in the result; there is no
bootstrap or other resampling. Amortized estimates retain infrequent flush
costs that the median can hide.

Malformed input (missing metrics, non-positive/non-finite values, bad modes)
fails explicitly instead of producing 0-division or NaN.

CLI:
    python perf_analyzer.py runs.json --threshold 0.03

runs.json is a list of run dicts as above.  Exit codes: 0 = ok,
2 = noisy or inconclusive, 3 = regression, 1 = invalid input/IO error.
"""

import argparse
import json
import math
import statistics
import sys
from typing import Any, Dict, List

METRICS = {
    "prefill_ms": "latency",
    "decode_ms_per_token": "latency",
    "throughput_tokens_per_s": "throughput",
}
MIN_COMPLETE_CYCLES = 3
MIN_ON_WINDOWS = 2


def _fail(errors: List[str]) -> Dict[str, Any]:
    return {"conclusion": "invalid", "ok": False, "errors": errors}


def _validate_records(records: Any) -> List[str]:
    errors: List[str] = []
    timed_samples = []
    if not isinstance(records, list):
        return ["records must be a list of runs"]
    for i, run in enumerate(records):
        if not isinstance(run, dict):
            errors.append(f"run[{i}]: not an object")
            continue
        cycle = run.get("cycle")
        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0:
            errors.append(f"run[{i}]: cycle must be a non-negative int")
        if run.get("mode") not in ("off", "on"):
            errors.append(f"run[{i}]: mode must be 'off' or 'on'")
        window_count = run.get("window_count")
        if (
            not isinstance(window_count, int)
            or isinstance(window_count, bool)
            or window_count < 0
        ):
            errors.append(f"run[{i}]: window_count must be a non-negative int")
        samples = run.get("samples")
        if not isinstance(samples, list) or not samples:
            errors.append(f"run[{i}]: samples must be a non-empty list")
            continue
        for j, sample in enumerate(samples):
            if not isinstance(sample, dict):
                errors.append(f"run[{i}].samples[{j}]: not an object")
                continue
            for metric in METRICS:
                v = sample.get(metric)
                if (
                    not isinstance(v, (int, float))
                    or isinstance(v, bool)
                    or not math.isfinite(v)
                    or v <= 0
                ):
                    errors.append(
                        f"run[{i}].samples[{j}].{metric}: must be a finite "
                        f"positive number, got {v!r}"
                    )
            has_tokens = "output_len" in sample
            has_wall = "wall_s" in sample
            timed_samples.append(has_tokens and has_wall)
            if has_tokens != has_wall:
                errors.append(
                    f"run[{i}].samples[{j}]: output_len and wall_s must be provided together"
                )
            elif has_tokens:
                tokens = sample["output_len"]
                wall = sample["wall_s"]
                if (
                    not isinstance(tokens, int)
                    or isinstance(tokens, bool)
                    or tokens <= 0
                ):
                    errors.append(
                        f"run[{i}].samples[{j}]: output_len must be a positive int"
                    )
                if (
                    not isinstance(wall, (int, float))
                    or isinstance(wall, bool)
                    or not math.isfinite(wall)
                    or wall <= 0
                ):
                    errors.append(
                        f"run[{i}].samples[{j}]: wall_s must be finite and positive"
                    )
    if any(timed_samples) and not all(timed_samples):
        errors.append(
            "all samples must provide output_len/wall_s, or none may provide them"
        )
    if errors:
        return errors
    cycle_modes: Dict[int, List[str]] = {}
    closed_cycles = set()
    previous_cycle = None
    for run in records:
        cycle = run["cycle"]
        if cycle != previous_cycle:
            if cycle in closed_cycles:
                errors.append(
                    f"cycle {cycle}: runs must form one contiguous ABBA block"
                )
            if previous_cycle is not None:
                closed_cycles.add(previous_cycle)
            previous_cycle = cycle
        cycle_modes.setdefault(cycle, []).append(run["mode"])
    for cycle, modes in cycle_modes.items():
        if len(modes) > 4 or (len(modes) == 4 and modes != ["off", "on", "on", "off"]):
            errors.append(f"cycle {cycle}: expected exactly off,on,on,off; got {modes}")
    return errors


def _median(values: List[float]) -> float:
    return statistics.median(values)


def _variance(values: List[float]) -> float:
    return statistics.variance(values) if len(values) >= 2 else 0.0


def _amortized(samples: List[Dict[str, Any]], metric: str) -> float:
    if METRICS[metric] == "latency":
        return statistics.fmean(sample[metric] for sample in samples)
    if "output_len" in samples[0]:
        return sum(sample["output_len"] for sample in samples) / math.fsum(
            sample["wall_s"] for sample in samples
        )
    return statistics.harmonic_mean(sample[metric] for sample in samples)


def analyze_abba(
    records: List[Dict[str, Any]], regression_threshold: float = 0.03
) -> Dict[str, Any]:
    """Analyze ABBA-paired OFF/ON runs.  See module docstring for semantics."""
    errors = _validate_records(records)
    if errors:
        return _fail(errors)
    if not (
        isinstance(regression_threshold, (int, float))
        and not isinstance(regression_threshold, bool)
        and math.isfinite(regression_threshold)
        and regression_threshold > 0
    ):
        return _fail(["regression_threshold must be a finite positive number"])

    # Group runs by cycle and mode (ABBA gives each mode 2 runs per cycle).
    cycles: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for run in records:
        cycles.setdefault(run["cycle"], {"off": [], "on": []})[run["mode"]].append(run)

    complete: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for cycle_id in sorted(cycles):
        modes = cycles[cycle_id]
        if len(modes["off"]) != 2 or len(modes["on"]) != 2:
            continue  # incomplete cycle: excluded from pairing
        merged = {}
        for mode in ("off", "on"):
            merged[mode] = [s for r in modes[mode] for s in r["samples"]]
        complete[cycle_id] = merged

    on_runs = [r for c in cycles.values() for r in c["on"]]
    min_on_windows = min((r["window_count"] for r in on_runs), default=0)

    result: Dict[str, Any] = {
        "conclusion": "inconclusive",
        "ok": False,
        "threshold": regression_threshold,
        "cycles_total": len(cycles),
        "cycles_complete": len(complete),
        "on_runs": len(on_runs),
        "min_on_window_count": min_on_windows,
        "metrics": {},
        "raw_records": records,
    }

    if len(complete) < MIN_COMPLETE_CYCLES or min_on_windows < MIN_ON_WINDOWS:
        reasons = []
        if len(complete) < MIN_COMPLETE_CYCLES:
            reasons.append(
                f"need >= {MIN_COMPLETE_CYCLES} complete ABBA cycles, have "
                f"{len(complete)}"
            )
        if min_on_windows < MIN_ON_WINDOWS:
            reasons.append(
                f"every ON run must span >= {MIN_ON_WINDOWS} flush windows, "
                f"min observed is {min_on_windows}"
            )
        result["reason"] = "; ".join(reasons)
        return result

    metric_verdicts: Dict[str, str] = {}
    for metric, kind in METRICS.items():
        per_cycle = []
        all_off_records: List[Dict[str, Any]] = []
        all_on_records: List[Dict[str, Any]] = []
        exceeding = 0
        median_exceeding = 0
        amortized_exceeding = 0
        for cycle_id, merged in complete.items():
            off_samples = [sample[metric] for sample in merged["off"]]
            on_samples = [sample[metric] for sample in merged["on"]]
            all_off_records.extend(merged["off"])
            all_on_records.extend(merged["on"])
            off_med = _median(off_samples)
            on_med = _median(on_samples)
            off_amortized = _amortized(merged["off"], metric)
            on_amortized = _amortized(merged["on"], metric)
            # off_med/on_med are validated positive, so no zero division.
            if kind == "latency":
                regression = on_med / off_med - 1.0
                amortized_regression = on_amortized / off_amortized - 1.0
            else:
                regression = off_med / on_med - 1.0
                amortized_regression = off_amortized / on_amortized - 1.0
            median_bad = regression > regression_threshold
            amortized_bad = amortized_regression > regression_threshold
            median_exceeding += median_bad
            amortized_exceeding += amortized_bad
            if median_bad or amortized_bad:
                exceeding += 1
            per_cycle.append(
                {
                    "cycle": cycle_id,
                    "off_median": off_med,
                    "on_median": on_med,
                    "regression": regression,
                    "off_amortized": off_amortized,
                    "on_amortized": on_amortized,
                    "amortized_regression": amortized_regression,
                }
            )
        n = len(complete)
        if exceeding * 2 > n:
            verdict = "regression"
        elif exceeding > 0:
            verdict = "noisy"
        else:
            verdict = "ok"
        metric_verdicts[metric] = verdict
        all_off = [sample[metric] for sample in all_off_records]
        all_on = [sample[metric] for sample in all_on_records]
        result["metrics"][metric] = {
            "kind": kind,
            "off_median": _median(all_off),
            "on_median": _median(all_on),
            "off_amortized": _amortized(all_off_records, metric),
            "on_amortized": _amortized(all_on_records, metric),
            "amortized_estimator": (
                "arithmetic_mean"
                if kind == "latency"
                else (
                    "aggregate_tokens_over_wall_time"
                    if "output_len" in all_off_records[0]
                    else "harmonic_mean_equal_tokens"
                )
            ),
            "off_variance": _variance(all_off),
            "on_variance": _variance(all_on),
            "off_sample_count": len(all_off),
            "on_sample_count": len(all_on),
            "per_cycle": per_cycle,
            "cycles_exceeding_threshold": exceeding,
            "median_cycles_exceeding_threshold": median_exceeding,
            "amortized_cycles_exceeding_threshold": amortized_exceeding,
            "verdict": verdict,
        }

    if any(v == "regression" for v in metric_verdicts.values()):
        bad = sorted(m for m, v in metric_verdicts.items() if v == "regression")
        result["conclusion"] = "regression"
        result["reason"] = (
            "majority of cycles exceed "
            f"{regression_threshold:.1%} degradation for: {', '.join(bad)}"
        )
    elif any(v == "noisy" for v in metric_verdicts.values()):
        noisy = sorted(m for m, v in metric_verdicts.items() if v == "noisy")
        result["conclusion"] = "noisy"
        result["reason"] = (
            "some but not a majority of cycles exceed "
            f"{regression_threshold:.1%} for: {', '.join(noisy)}; "
            "repeat measurement before claiming either way"
        )
    else:
        result["conclusion"] = "ok"
        result["reason"] = f"no metric exceeds {regression_threshold:.1%} in any cycle"
    result["ok"] = result["conclusion"] == "ok"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze ABBA-paired EPLB monitoring performance runs."
    )
    parser.add_argument("runs", help="JSON file with a list of run dicts")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="paired median/amortized regression threshold (default 0.03)",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.runs, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot load {args.runs}: {exc}", file=sys.stderr)
        return 1

    result = analyze_abba(records, regression_threshold=args.threshold)
    print(json.dumps(result, indent=2, sort_keys=True))
    conclusion = result["conclusion"]
    if conclusion == "ok":
        return 0
    if conclusion in ("noisy", "inconclusive"):
        return 2
    if conclusion == "regression":
        return 3
    return 1  # invalid


if __name__ == "__main__":
    raise SystemExit(main())
