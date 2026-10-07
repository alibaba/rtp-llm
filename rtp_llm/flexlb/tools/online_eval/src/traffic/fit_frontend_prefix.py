"""拟合匿名前缀流量；行契约见 traffic.capture_contract。

保留完整块共享、真实到达时间与精确总长度；残缺尾块不参与匹配。
"""

import argparse, collections, gzip, lzma, hashlib, json, math, time, sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic.capture_contract import BLOCK_SIZE, validate_metadata, validate_row
from traffic.codecs import codec_for_version
from traffic.datasets import build_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, required=True)
    p.add_argument("--namespace", default="fit", help="opaque request identity namespace")
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--source-info", type=Path, help="caller-supplied source metadata JSON")
    p.add_argument("--model-version", choices=("2", "3"), default="3")
    p.add_argument("--v2-reason", help="仅历史复拟合：说明显式生成 v2 的理由")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        fit(a)
    except Exception as exc:
        (a.out / "fit.summary.json").write_text(json.dumps(dict(
            complete=False, errors=[str(exc)]), indent=2) + "\n")
        raise
    (a.out / "fit.summary.json").write_text(json.dumps(dict(complete=True, errors=[])) + "\n")


def fit(a):
    if a.model_version == "2" and not (a.v2_reason and a.v2_reason.strip()):
        raise ValueError("--model-version 2 requires --v2-reason for historical refitting")
    codec = codec_for_version(int(a.model_version))
    encode, write_trace = codec.encode, codec.write_trace
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = {}
    for path in sorted(list(a.source.glob("*.jsonl.gz")) + list(a.source.glob("*.jsonl.xz"))):
        summary = json.loads(
            path.with_name(path.name.replace(".jsonl.gz", ".summary.json").replace(".jsonl.xz", ".summary.json")).read_text()
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["sha256"]:
            raise ValueError("capture checksum mismatch: " + str(path))
        validate_metadata(summary, str(path) + " summary")
        if summary.get("complete") is not True or summary.get("truncated") is not False:
            raise ValueError(f"{path}: incomplete capture; recapture a smaller window or increase budget")
        sources[path.name] = summary
        with (lzma.open if path.suffix == ".xz" else gzip.open)(path, "rt") as stream:
            for number, line in enumerate(stream, 1):
                location = f"{path}:{number}"
                try:
                    row = json.loads(line)
                except ValueError:
                    raise ValueError(f"{location}: invalid JSON") from None
                validate_row(row, location)
                row["keys"] = [bytes.fromhex(k) for k in row["keys"]]
                rows.append(row)
    windows = {(v["start"], v["end"]) for v in sources.values()}
    if len(windows) != 1:
        raise ValueError("capture intervals differ")
    capture_start, capture_end = next(iter(windows))
    if (
        capture_end <= capture_start
        or a.expected_shards <= 0
        or len(sources) > a.expected_shards
        or not isinstance(a.namespace, str) or not a.namespace
        or a.output_tokens <= 0
    ):
        raise ValueError("invalid capture/workload parameters")
    missing_count = a.expected_shards - len(sources)
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
    duplicate_counts = collections.Counter()
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
            fresh = math.ceil(row["il"] / BLOCK_SIZE) - shared
            blocks.extend(range(next_token, next_token + fresh))
            next_token += fresh
            token_paths.append(blocks)
            for key in keys:
                seen[key] = i
            potential += shared * BLOCK_SIZE
            lengths.append(row["il"])
            status[str(row["status"])] += 1
            families[keys[7] if len(keys) >= 8 else ("short", row["tail_hash"])] += 1
            duplicate_counts[(row["rid"], row["ts"])] += 1
            t = int((row["ts"] - source_origin) // 60000)
            w = windows.setdefault(
                t, dict(requests=0, tokens=0, potential_tokens=0, errors=0)
            )
            w["requests"] += 1
            w["tokens"] += row["il"]
            w["potential_tokens"] += shared * BLOCK_SIZE
            w["errors"] += row["status"] != "OK"
            model.append([row["ts"] - source_origin, row["il"], parent, shared])
            if i % 20000 == 0:
                print("fit", i, "nodes", len(seen), flush=True)
    lengths.sort()
    gaps.sort()
    total = sum(lengths)
    model_path = a.out / "lineage-model.xz"
    raw = encode(model, dict(missing_shard_count=missing_count,source_start=rows[0]['ts'],
        source_end=rows[-1]['ts'],capture_start=capture_start,capture_end=capture_end,
        source_shards=len(sources),expected_shards=a.expected_shards,
        **(dict(v2_reason=a.v2_reason) if a.model_version == "2" else {})))
    model_path.write_bytes(raw)
    source = json.loads(a.source_info.read_text()) if a.source_info else None
    model_path.with_suffix('.manifest.json').write_text(
        json.dumps(build_manifest(model_path, source), indent=2) + '\n')
    write_trace(plan, dict(path=str(model_path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
        count=len(model),output_tokens=a.output_tokens,priority=50),a.namespace,a.out)
    digest = hashlib.sha256()
    with plan.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    report = dict(
        source_shards=len(sources),
        expected_shards=a.expected_shards,
        missing_shard_count=missing_count,
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
        duplicate_identity_count=sum(v - 1 for v in duplicate_counts.values() if v > 1),
        plan_sha256=digest.hexdigest(),
        plan_bytes=plan.stat().st_size,
        model_bytes=(a.out / "lineage-model.xz").stat().st_size,
        realism="EMPIRICAL_PREFIX_STRUCTURE_PARTIAL_COVERAGE",
        limitations=[
            "Missing capture shards: " + str(missing_count),
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
