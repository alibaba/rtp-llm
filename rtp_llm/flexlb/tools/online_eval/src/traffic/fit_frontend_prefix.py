"""Fit an anonymized empirical prefix-lineage model; no raw tokens retained.

This first calibration preserves measured 512-token sharing exactly instead of
assuming a Zipf family model. It is an empirical event model, not a validated
small-parameter statistical generator. True arrivals are retained; playback
controls the rate. Output behavior is independently specified.
"""

import argparse, collections, gzip, hashlib, json, math, time
from pathlib import Path
try:
    from traffic.prefix_lineage import encode, write_trace
except ImportError:
    from prefix_lineage import encode, write_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-pods", type=int, required=True)
    p.add_argument("--output-tokens", type=int, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = {}
    for path in sorted(a.source.glob("pod-*.jsonl.gz")):
        summary = json.loads(
            path.with_name(path.name.replace(".jsonl.gz", ".summary.json")).read_text()
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["sha256"]:
            raise ValueError("capture checksum mismatch: " + str(path))
        sources[path.name] = summary
        for line in gzip.open(path, "rt"):
            row = json.loads(line)
            row["pod"] = path.name
            row["keys"] = [bytes.fromhex(k) for k in row["keys"]]
            if len(row["keys"]) != row["il"] // 512:
                raise ValueError("capture block count mismatch")
            rows.append(row)
    windows = {(v["start"], v["end"]) for v in sources.values()}
    if len(windows) != 1:
        raise ValueError("capture intervals differ")
    capture_start, capture_end = next(iter(windows))
    if (
        capture_end <= capture_start
        or a.expected_pods <= 0
        or a.output_tokens <= 0
    ):
        raise ValueError("invalid capture/workload parameters")
    present = {int(Path(name).name.split("-")[1].split(".")[0]) for name in sources}
    if not present <= set(range(a.expected_pods)):
        raise ValueError("capture pod index exceeds expected fleet")
    missing = sorted(set(range(a.expected_pods)) - present)
    rows.sort(key=lambda r: r["ts"])
    if not rows:
        raise ValueError("empty capture")
    seen = {}
    next_token = 1
    token_paths = []
    lengths = []
    status = collections.Counter()
    families = collections.Counter()
    windows = {}
    model = []
    potential = 0
    gaps = []
    source_origin = rows[0]["ts"]
    cross_duplicates = collections.Counter()
    plan = a.out / "input-plan.jsonl"
    with plan.open("w") as f:
        for i, row in enumerate(rows):
            keys = row["keys"]
            shared = 0
            parent = -1
            # Search deepest known cumulative prefix, then recover its previous owner.
            lo = 0
            hi = len(keys)
            while lo < hi:
                mid = (lo + hi) // 2
                if keys[mid] in seen:
                    lo = mid + 1
                else:
                    hi = mid
            shared = lo
            if shared:
                parent = seen[keys[shared - 1]]
                gaps.append(row["ts"] - rows[parent]["ts"])
            blocks = token_paths[parent][:shared] if parent >= 0 else []
            fresh = math.ceil(row["il"] / 512) - shared
            blocks.extend(range(next_token, next_token + fresh))
            next_token += fresh
            token_paths.append(blocks)
            for key in keys:
                seen[key] = i
            potential += shared * 512
            lengths.append(row["il"])
            status[str(row["status"])] += 1
            families[keys[7] if len(keys) >= 8 else ("short", row["tail_hash"])] += 1
            cross_duplicates[(row["rid"], row["ts"])] += 1
            t = int((row["ts"] - source_origin) // 60000)
            w = windows.setdefault(
                t, dict(requests=0, tokens=0, potential_tokens=0, errors=0)
            )
            w["requests"] += 1
            w["tokens"] += row["il"]
            w["potential_tokens"] += shared * 512
            w["errors"] += row["status"] != "OK"
            model.append([row["ts"] - source_origin, row["il"], parent, shared])
            if i % 20000 == 0:
                print("fit", i, "nodes", len(seen), flush=True)
    lengths.sort()
    gaps.sort()
    total = sum(lengths)
    model_path = a.out / "lineage-model.xz"
    raw = encode(model, dict(missing_pod_indices=missing,source_start=rows[0]['ts'],
        source_end=rows[-1]['ts'],capture_start=capture_start,capture_end=capture_end,
        source_pods=len(sources),expected_pods=a.expected_pods))
    model_path.write_bytes(raw)
    write_trace(plan, dict(path=str(model_path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
        count=len(model),output_tokens=a.output_tokens,priority=50),'frontend-fit:scale_in',a.out)
    digest = hashlib.sha256()
    with plan.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    report = dict(
        source_pods=len(sources),
        expected_pods=a.expected_pods,
        missing_pod_indices=missing,
        sources=sources,
        requests=len(rows),
        source_start=rows[0]["ts"],
        source_end=rows[-1]["ts"],
        source_qps=len(rows) * 1000 / (capture_end - capture_start),
        mean_input_tokens=total / len(rows),
        input_quantiles={
            str(q): lengths[int(q * (len(lengths) - 1))]
            for q in [0, 0.5, 0.9, 0.95, 0.99, 1]
        },
        status=dict(status),
        infinite_history_token_reuse=potential / total,
        distinct_prefix_blocks=len(seen),
        family_4k_count=len(families),
        family_4k_top5_share=sum(v for _, v in families.most_common(5)) / len(rows),
        nearest_prefix_repeat_gap_ms={
            str(q): gaps[int(q * (len(gaps) - 1))] if gaps else None
            for q in [0.5, 0.9, 0.99]
        },
        source_minute_windows=windows,
        cross_pod_duplicate_identity_count=sum(
            v - 1 for v in cross_duplicates.values() if v > 1
        ),
        plan_sha256=digest.hexdigest(),
        plan_bytes=plan.stat().st_size,
        model_bytes=(a.out / "lineage-model.xz").stat().st_size,
        realism="EMPIRICAL_PREFIX_STRUCTURE_PARTIAL_FLEET",
        limitations=[
            "Missing frontend pods: " + str(missing),
            "Capture interval: " + str((capture_start, capture_end)),
            "Output cap supplied independently; error-censored outputs are not fitted",
            "512-token equivalence preserved; sub-block token content not reconstructed",
            "Empirical prefix-lineage event model; no held-out statistical generalization claim",
            "True source timestamps retained; pacing controlled by the client",
        ],
    )
    (a.out / "fit-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["sources", "source_minute_windows"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
