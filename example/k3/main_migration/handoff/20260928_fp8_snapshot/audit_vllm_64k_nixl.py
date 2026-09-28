#!/usr/bin/env python3
"""Independently check a warmed vLLM K3 64K NIXL PD target trace."""

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import statistics
import tarfile


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def metric_value(text, name):
    rows = re.findall(r"^" + re.escape(name) + r"\{[^}\n]*\} ([0-9.e+\-]+)$", text, re.M)
    if len(rows) != 1:
        raise ValueError(f"expected one {name} metric, found {len(rows)}")
    return float(rows[0])


def audit(run_dir: pathlib.Path, trace_tar: pathlib.Path,
          consumer_metrics: pathlib.Path, fixture_dir: pathlib.Path) -> dict:
    errors = []
    meta = json.loads((run_dir / "requests/input.json").read_text(),
                      object_pairs_hook=unique_keys)
    lines = [json.loads(line, object_pairs_hook=unique_keys)
             for line in (run_dir / "stdout.log").read_text().splitlines() if line.strip()]
    warmups = [row for row in lines if str(row.get("label", "")).startswith("warmup-")]
    profiled = [row for row in lines if str(row.get("label", "")).startswith("profiled-")]
    labels = [f"warmup-{i:02d}" for i in range(1, len(warmups) + 1)]
    labels += [f"profiled-{i:02d}" for i in range(1, 4)]
    if [row.get("label") for row in lines] != labels or len(warmups) < 10 or len(profiled) != 3:
        errors.append("request labels or warmup count differ")
    if (run_dir / "exit").read_text().strip() != "0":
        errors.append("runner exit status differs from zero")
    if not (meta.get("backend") == "vllm" and meta.get("model_layers") == 4
            and meta.get("input_tokens") == 65536
            and meta.get("submitted_prompt_tokens") == 65537
            and meta.get("vllm_nixl_holdback_token") is True):
        errors.append("four-layer vLLM 64K holdback configuration differs")
    http_stable = meta.get("warmup_stable") is True
    if not http_stable and not meta.get("warmup_exception"):
        errors.append("unstable HTTP warmup lacks an explicit exception")
    if len(meta.get("warmups", [])) != len(warmups) or len(meta.get("profiled_requests", [])) != 3:
        errors.append("request counts in input metadata differ")

    fixed = json.load(gzip.open(fixture_dir / "fixed.json.gz", "rt"))["input_ids"]
    if len(fixed) != 65536 or meta.get("token_ids_sha256") != digest_ids(fixed):
        errors.append("fixed 64K fixture hash differs")
    variant_start = meta.get("variant_start", 1)
    fixtures = {}
    for index, label in enumerate(labels):
        if label.startswith("profiled-"):
            fixtures[label] = fixed
        else:
            number = variant_start + index
            path = fixture_dir / f"warmup-variant-{number:02d}-20260926.json.gz"
            fixtures[label] = json.load(gzip.open(path, "rt"))["input_ids"]
    expected_files = {f"{label}.json" for label in labels} | {"input.json"}
    actual_files = {path.name for path in (run_dir / "requests").glob("*.json")}
    if actual_files != expected_files:
        errors.append("raw response file set differs from request labels")

    replacement_count = 0
    for row in lines:
        label = row.get("label")
        if label not in fixtures:
            continue
        try:
            ids = fixtures[label]
            original = digest_ids(ids)
            submitted = digest_ids([*ids, ids[-1]])
            if len(ids) != 65536 or row.get("token_ids_sha256") != original:
                raise ValueError("original 64K token digest differs")
            if row.get("submitted_token_ids_sha256") != submitted:
                raise ValueError("submitted holdback token digest differs")
            raw = (run_dir / "requests" / f"{label}.json").read_bytes()
            response = json.loads(raw.decode("utf-8", errors="strict"),
                                  object_pairs_hook=unique_keys)
            if row.get("status") != 200 or row.get("response_sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("HTTP status or response digest differs")
            usage = response["usage"]
            if usage.get("prompt_tokens") != 65537 or usage.get("completion_tokens") != 8:
                raise ValueError("response token counts differ")
            content = response["choices"][0]["text"]
            if not isinstance(content, str) or not content.strip() or "\x00" in content:
                raise ValueError("empty or NUL-containing response text")
            content.encode("utf-8", errors="strict")
            replacement_count += content.count("\ufffd")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    spans = [[] for _ in range(3)]
    with tarfile.open(trace_tar) as archive:
        members = [member for member in archive if member.isfile()]
        ranks = [re.search(r"rank([0-7])\.", member.name) for member in members]
        if len(members) != 8 or any(match is None for match in ranks) or \
                {int(match.group(1)) for match in ranks} != set(range(8)):
            errors.append("trace archive does not contain exactly eight ranks")
        else:
            for member in members:
                rank = int(re.search(r"rank([0-7])\.", member.name).group(1))
                events = json.load(archive.extractfile(member))["traceEvents"]
                scopes = sorted((event for event in events
                                 if event.get("cat") == "gpu_user_annotation"
                                 and event.get("name", "").startswith("execute_context_1(")),
                                key=lambda event: event["ts"])
                if len(scopes) != 3 or any(
                    event["name"] != "execute_context_1(65536)_generation_0(0)"
                    for event in scopes
                ):
                    errors.append(f"rank {rank}: target Prefill scopes or token count differ")
                    continue
                for index, event in enumerate(scopes):
                    spans[index].append(event["dur"] / 1000)
    worst_spans = [max(values) for values in spans if len(values) == 8]
    gpu_stable = False
    if len(worst_spans) == 3:
        median = statistics.median(worst_spans)
        gpu_stable = all(abs(value / median - 1) <= 0.05 for value in worst_spans)
    if not gpu_stable:
        errors.append("full-rank target GPU spans are absent or exceed ±5% of median")

    metrics = consumer_metrics.read_text()
    count = metric_value(metrics, "vllm:nixl_bytes_transferred_count")
    bytes_sum = metric_value(metrics, "vllm:nixl_bytes_transferred_sum")
    failures = metric_value(metrics, "vllm:nixl_num_failed_transfers_total")
    if count != len(lines) * 8 or bytes_sum <= 0 or failures != 0:
        errors.append("Decode NIXL transfer count, bytes, or failure metric differs")
    return {
        "passed": not errors,
        "http_warmup_stable": http_stable,
        "http_warmup_exception": meta.get("warmup_exception"),
        "warmup_count": len(warmups),
        "profiled_count": len(profiled),
        "prefill_tokens_per_rank": 65536 if len(worst_spans) == 3 else None,
        "worst_rank_target_gpu_span_ms": worst_spans,
        "target_gpu_span_stable": gpu_stable,
        "nixl_transfer_count": int(count),
        "nixl_bytes_transferred": int(bytes_sum),
        "nixl_failed_transfers": int(failures),
        "replacement_characters_observed": replacement_count,
        "semantic_answer_claim": False,
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--trace-tar", type=pathlib.Path, required=True)
    parser.add_argument("--consumer-metrics", type=pathlib.Path, required=True)
    parser.add_argument("--fixture-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent / "timeline-input-64k")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.trace_tar, args.consumer_metrics, args.fixture_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("passed", "warmup_count", "profiled_count", "target_gpu_span_stable", "errors")}))
    raise SystemExit(0 if result["passed"] else 1)
