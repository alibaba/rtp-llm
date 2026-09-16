"""Derive topology-specific healthy bands from independent baseline runs.

Only baseline observations set thresholds; saturation measurements never tune
the healthy contract. Confidence calculations are guard-band heuristics, not
independence claims about correlated requests within a run.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


def wilson_lower(successes, count, confidence):
    if not 0 <= successes <= count or count <= 0:
        raise ValueError("invalid hit-rate denominator")
    z = NormalDist().inv_cdf(confidence)
    p = successes / count
    return (
        p
        + z * z / (2 * count)
        - z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    ) / (1 + z * z / count)


def poisson_upper(events, seconds, confidence):
    if events < 0 or seconds <= 0:
        raise ValueError("invalid eviction observation exposure")
    if events == 0:
        return -math.log(1 - confidence) / seconds

    # Monotone inversion of Pr[N <= observed | mean] = 1-confidence.
    def cdf(mean):
        term = math.exp(-mean)
        result = term
        for k in range(1, events + 1):
            term *= mean / k
            result += term
        return result

    low, high = 0.0, float(events + 1)
    while cdf(high) > 1 - confidence:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if cdf(middle) > 1 - confidence:
            low = middle
        else:
            high = middle
    return high / seconds


def calibrate(paths, minimum_runs=3):
    if minimum_runs < 3:
        raise ValueError("at least three independent runs per topology are required")
    groups, seen = {}, set()
    for path in paths:
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            raise ValueError("duplicate calibration sample")
        seen.add(sha)
        row = json.loads(raw)
        if not all(
            row.get(k) is True
            for k in ("baseline_valid", "injection_valid", "recovery_valid")
        ):
            raise ValueError(f"invalid construction or recovery: {path}")
        result = json.loads(Path(path).with_name("result.json").read_text())
        failures = [
            (s["id"], c["id"])
            for s in result["stages"]
            for c in s["checks"]
            if c["status"] != "PASS"
        ]
        if any(
            stage != "healthy" and (stage, check) != ("validity", "calibrated")
            for stage, check in failures
        ):
            raise ValueError("hard failure cannot become a calibration sample")
        if result.get("error") or any(c["status"] != "PASS" for c in result["cleanup"]):
            raise ValueError("incomplete execution or failed cleanup")
        key = (row["profile"], row["p"])
        groups.setdefault(key, []).append((row, str(Path(path).resolve()), sha))
    output = {}
    for (profile, p), samples in sorted(groups.items()):
        if len(samples) < minimum_runs:
            raise ValueError(f"{profile}/P={p}: fewer than {minimum_runs} runs")

        def shape(config):
            return {
                k: v
                for k, v in config.items()
                if not k.startswith(("hit_", "eviction_", "holders_"))
                and k != "calibration_runs"
            }

        if any(
            shape(s[0]["config"]) != shape(samples[0][0]["config"]) for s in samples
        ):
            raise ValueError("mixed workload configurations in one calibration")
        baseline = [
            m
            for row, _, _ in samples
            for m in row["metrics"]
            if m["phase"] == "baseline"
        ]
        count = sum(m["hot_requests"] for m in baseline)
        hits = sum(m["hits"] for m in baseline)
        seconds = sum(m["duration_s"] for m in baseline)
        evictions = sum(m["eviction_count"] for m in baseline)
        bands = {"calibration_runs": len(samples)}
        for grade, confidence in (("strict", 0.90), ("normal", 0.95), ("loose", 0.99)):
            bands[f"hit_{grade}"] = round(wilson_lower(hits, count, confidence), 6)
            bands[f"eviction_{grade}"] = round(
                poisson_upper(evictions, seconds, confidence), 6
            )
            # Holder count is discrete. Preserve the measured baseline envelope;
            # no fractional/invented extra holder is introduced as headroom.
            bands[f"holders_{grade}"] = max(m["hot_holders"] for m in baseline)
        output[f"{profile}/p{p}"] = dict(
            parameters=bands,
            runs=len(samples),
            baseline_windows=len(baseline),
            baseline_requests=count,
            baseline_hits=hits,
            baseline_seconds=seconds,
            baseline_evictions=evictions,
            workload=shape(samples[0][0]["config"]),
            samples=[dict(path=path, sha256=sha) for _, path, sha in samples],
        )
    if not output:
        raise ValueError("no calibration samples")
    return dict(
        schema_version=1,
        method="baseline-only Wilson lower / Poisson upper / empirical holder maximum",
        topologies=output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples",
        nargs="+",
        type=Path,
        help="cache-storm-summary.json files from separate completed runs",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = calibrate(args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
