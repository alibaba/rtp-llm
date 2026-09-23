#!/usr/bin/env python3
"""Sequential SSE TTFT for exact frozen prompts, without CUDA or RTP imports.

Client TTFT ends at the first token; server TTFT excludes only reported queue time."""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import math
import pathlib
import statistics
import time
import urllib.request
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REFERENCE_MANIFEST_SHA256 = (
    "0741223f7916c3ebc0c595e81f85e2b078d4eee5b8a984e6949a139507d50543"
)
REFERENCE_SOURCE_JSONL_SHA256 = (
    "a5d0ec89b42c5b31fabb47a6ec692e4d07e2403b3e4b36dedb4bb266cabc41bd"
)
REFERENCE_SOURCE_IDS_SHA256 = (
    "88582e2e19e3ff99f14de1fe08990915c30c2505f9aa75fcc86ac17f90fc610d"
)
METRIC_DEFINITIONS = {
    "client_ttft_ms": "dispatch to first parsed token-bearing SSE event",
    "client_response_ms": "dispatch to complete SSE consumption",
    "server_first_token_cost_ms": "native aux_info.first_token_cost_time",
    "server_wait_ms": "native aux_info.wait_time",
    "server_queue_excluded_ms": "first_token_cost_time - wait_time",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_hash(value) -> str:
    return sha256_bytes(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _token_ids(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("expected nonnegative token IDs for one sequence")
    return value


def load_reference_manifest(
    path: pathlib.Path,
    *,
    expected_sha256: str = REFERENCE_MANIFEST_SHA256,
    expected_count: int = 8,
    expected_input_len: int = 32768,
) -> Tuple[List[Dict], Dict]:
    path = pathlib.Path(path)
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        raise ValueError("reference manifest SHA256 mismatch")
    doc = json.loads(raw)
    if doc.get("schema_version") != 1 or doc.get("split") != "screening":
        raise ValueError("unexpected reference manifest schema/split")
    rows = doc.get("prompts")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(
            "reference manifest must contain exactly %d prompts" % expected_count
        )
    source_hashes = {x.get("sha256") for x in doc.get("sources", [])}
    if not {
        REFERENCE_SOURCE_JSONL_SHA256,
        REFERENCE_SOURCE_IDS_SHA256,
    }.issubset(source_hashes):
        raise ValueError("reference source hashes are not bound")
    out: List[Dict] = []
    text_hashes, id_hashes, ids_seen = set(), set(), set()
    for i, original in enumerate(rows):
        row = dict(original)
        text = row.get("prompt")
        ids = _token_ids(row.get("input_token_ids"))
        if not isinstance(text, str) or not text or ids is None:
            raise ValueError("prompt %d lacks text or token IDs" % i)
        if len(ids) != expected_input_len or row.get("input_len") != expected_input_len:
            raise ValueError(
                "prompt %d is not exactly %d tokens" % (i, expected_input_len)
            )
        th, ih = text_hash(text), canonical_json_hash(ids)
        if th != row.get("prompt_sha256") or ih != row.get("input_token_ids_sha256"):
            raise ValueError("prompt %d content/token digest mismatch" % i)
        if row.get("id") != "screening-%03d" % i:
            raise ValueError("reference prompt order/id mismatch at %d" % i)
        if th in text_hashes or ih in id_hashes or tuple(ids) in ids_seen:
            raise ValueError("reference prompts/token sequences must be distinct")
        text_hashes.add(th)
        id_hashes.add(ih)
        ids_seen.add(tuple(ids))
        out.append(row)
    return out, {
        "path": str(path.resolve()),
        "sha256": digest,
        "split": doc["split"],
        "count": len(out),
        "source_hashes": sorted(source_hashes),
    }


def plan_reference_requests(
    prompts: Sequence[Dict], *, warmup_runs: int = 2, measure_runs: int = 8
) -> List[Tuple[Dict, str]]:
    """Two declared warmups, then each of the eight prompts exactly once in order."""
    if warmup_runs != 2 or measure_runs != 8:
        raise ValueError(
            "reference protocol requires exactly two warmups and eight measures"
        )
    if len(prompts) != measure_runs:
        raise ValueError("measurement roster must equal the eight-prompt manifest")
    warmups = [(prompts[i % len(prompts)], "warmup-%d" % i) for i in range(warmup_runs)]
    measured = [(p, "measure-%d" % i) for i, p in enumerate(prompts)]
    return warmups + measured


class SSEDecoder:
    def __init__(self, max_bytes: int = 2 * 1024 * 1024):
        self.pending = b""
        self.data: List[bytes] = []
        self.size = 0
        self.max_bytes = max_bytes

    def feed(self, chunk: bytes) -> Iterable[str]:
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.rstrip(b"\r")
            self.size += len(line) + 1
            if self.size > self.max_bytes:
                raise ValueError("SSE event exceeds size limit")
            if not line:
                if self.data:
                    payload = b"\n".join(self.data).decode("utf-8")
                    self.data, self.size = [], 0
                    yield payload
                self.size = 0
            elif line.startswith(b"data:"):
                self.data.append(line[5:].lstrip(b" "))
        if len(self.pending) + self.size > self.max_bytes:
            raise ValueError("SSE event exceeds size limit")

    def finish(self) -> None:
        if self.pending.strip() or self.data:
            raise ValueError("truncated SSE frame at EOF")


def generation_config() -> Dict:
    return {
        "max_new_tokens": 1,
        "min_new_tokens": 1,
        "top_k": 1,
        "force_sp_accept": True,
        "return_output_ids": True,
        "return_input_ids": False,
        "is_streaming": True,
    }


def request_body(prompt: str) -> Dict:
    return {
        "prompt": prompt,
        "yield_generator": True,
        "top_k": 1,
        "generate_config": generation_config(),
    }


def consume_response(
    resp, start: float, clock: Callable[[], float] = time.perf_counter
) -> Dict:
    row = {
        "client_ttft_ms": None,
        "server_first_token_cost_ms": None,
        "server_wait_ms": None,
        "server_queue_excluded_ms": None,
        "client_response_ms": None,
        "output_ids": None,
        "finished": False,
        "text": "",
        "events_received": 0,
        "protocol_errors": [],
        "reuse_fields": {},
    }
    done = False

    def consume(obj: Dict, observed: float) -> None:
        if not isinstance(obj, dict):
            raise ValueError("native SSE event must be an object")
        if obj.get("error_code") or obj.get("error"):
            raise ValueError(
                "server error: %s" % obj.get("error_code_str", obj.get("error"))
            )
        if "response_batch" in obj:
            raise ValueError("unexpected batched response")
        aux = obj.get("aux_info") or {}
        if not isinstance(aux, dict):
            raise ValueError("expected one-sequence aux_info")
        ids = _token_ids(obj.get("output_ids"))
        has_token = bool(ids) or (aux.get("output_len", 0) or 0) > 0
        if has_token and row["client_ttft_ms"] is None:
            row["client_ttft_ms"] = (observed - start) * 1000
        row["events_received"] += 1
        for key in ("input_len", "output_len", "pd_sep", "iter_count"):
            if key in aux:
                row[key] = aux[key]
        for key, value in aux.items():
            if key.endswith("reuse_len"):
                row["reuse_fields"][key] = value
        if aux.get("first_token_cost_time") is not None:
            row["server_first_token_cost_ms"] = aux["first_token_cost_time"]
        if aux.get("wait_time") is not None:
            row["server_wait_ms"] = aux["wait_time"]
        if aux.get("cost_time") is not None:
            row["server_cost_ms"] = aux["cost_time"]
        if ids:
            row["output_ids"] = ids
        if has_token:
            row["text"] = obj.get("response", row["text"])
        if obj.get("finished") is True:
            row["finished"] = True

    try:
        if "text/event-stream" not in resp.headers.get("Content-Type", "").lower():
            raise ValueError("server did not return text/event-stream")
        parser = SSEDecoder()
        while True:
            data = resp.read1(65536)
            if not data:
                break
            for payload in parser.feed(data):
                if payload.strip().lower() == "[done]":
                    done = True
                    continue
                if done:
                    raise ValueError("data arrived after SSE done marker")
                consume(json.loads(payload), clock())
        parser.finish()
        if not done:
            row["protocol_errors"].append("missing native SSE [done] marker")
        if row["client_ttft_ms"] is None:
            row["protocol_errors"].append("no token-bearing SSE event")
    except (ValueError, OSError, TypeError) as exc:
        row["protocol_errors"].append(str(exc))
    row["client_response_ms"] = (clock() - start) * 1000
    first, wait = row.get("server_first_token_cost_ms"), row.get("server_wait_ms")
    if isinstance(first, (int, float)) and isinstance(wait, (int, float)):
        row["server_queue_excluded_ms"] = first - wait
    return row


def one_request(port: int, prompt: str, timeout: float = 1800.0) -> Dict:
    raw = json.dumps(request_body(prompt), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:%d/" % port,
        data=raw,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    with opener.open(req, timeout=timeout) as resp:
        return consume_response(resp, start)


def validate_result(row: Dict, prompt: Dict, *, expected_iter_count: int = 8) -> Dict:
    errors = list(row.get("protocol_errors", []))
    if row.get("finished") is not True:
        errors.append("response did not finish")
    if row.get("input_len") != prompt.get("input_len"):
        errors.append("server input_len does not match manifest")
    if row.get("output_len") != 1:
        errors.append("output_len must be exactly one")
    if row.get("pd_sep") is not False:
        errors.append("unexpected PD separation")
    if row.get("iter_count") != expected_iter_count:
        errors.append("iter_count must be %d" % expected_iter_count)
    if not row.get("reuse_fields") or any(
        value != 0 for value in row["reuse_fields"].values()
    ):
        errors.append("all reported reuse lengths must be zero")
    if not row.get("output_ids") or len(row["output_ids"]) != 1:
        errors.append("exactly one output token ID required")
    for metric in (
        "client_ttft_ms",
        "client_response_ms",
        "server_first_token_cost_ms",
        "server_wait_ms",
        "server_queue_excluded_ms",
    ):
        value = row.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append("missing/invalid %s" % metric)
            row[metric] = None
    row["validation_errors"] = errors
    row["ok"] = not errors
    return row


def metric_stats(values: Sequence[float]) -> Optional[Dict]:
    if not values:
        return None
    values = sorted(values)

    def percentile(p: float) -> float:
        index = (len(values) - 1) * p
        lo, hi = int(math.floor(index)), int(math.ceil(index))
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    trim = statistics.mean(values[1:-1]) if len(values) == 8 else None
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "six_middle_trimmed_mean": trim,
        "p95": percentile(0.95),
        "min": values[0],
        "max": values[-1],
        "samples_in_request_order": None,
    }


def summarize(
    results: Sequence[Dict], *, warmup_runs: int = 2, measure_runs: int = 8
) -> Dict:
    measured = list(results[warmup_runs:])
    valid = (
        len(results) == warmup_runs + measure_runs
        and all(row.get("ok") for row in results)
        and [row.get("tag") for row in measured]
        == ["measure-%d" % i for i in range(measure_runs)]
    )
    metrics = {}
    for metric in (
        "client_ttft_ms",
        "server_first_token_cost_ms",
        "server_wait_ms",
        "server_queue_excluded_ms",
        "client_response_ms",
    ):
        samples = [
            row[metric]
            for row in measured
            if row.get("ok") and row.get(metric) is not None
        ]
        stats = metric_stats(samples)
        if stats is not None:
            stats["samples_in_request_order"] = samples
        metrics[metric] = stats
    return {
        "valid_run": valid,
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "completed_measures": sum(bool(row.get("ok")) for row in measured),
        "failed_measures": sum(not row.get("ok") for row in measured),
        "warmup_failures": sum(not row.get("ok") for row in results[:warmup_runs]),
        "statistics_successes_only": metrics,
    }


@contextmanager
def benchmark_lock(port: int):
    with open("/tmp/rtp_cep4pp2_reference_%d.lock" % port, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another reference benchmark owns this port")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_reference_benchmark(
    *,
    port: int,
    manifest_path: pathlib.Path,
    output_path: pathlib.Path,
    timeout: float = 1800.0,
    warmup_runs: int = 2,
    measure_runs: int = 8,
    expected_iter_count: int = 8,
    request_fn: Callable[[int, str, float], Dict] = one_request,
) -> Dict:
    prompts, manifest = load_reference_manifest(manifest_path)
    plan = plan_reference_requests(
        prompts, warmup_runs=warmup_runs, measure_runs=measure_runs
    )
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = output_path.open("x")
    results: List[Dict] = []
    doc = {
        "schema": "cep4pp2-reference-ttft/1",
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "transport": "native-SSE",
        "metric_definitions": METRIC_DEFINITIONS,
        "prompt_manifest": manifest,
        "request_plan": [
            {
                "tag": tag,
                "prompt_id": p["id"],
                "prompt_sha256": p["prompt_sha256"],
                "input_token_ids_sha256": p["input_token_ids_sha256"],
            }
            for p, tag in plan
        ],
        "generation_config": generation_config(),
        "results": results,
        "no_retry": True,
    }

    def checkpoint() -> None:
        doc.update(
            summarize(results, warmup_runs=warmup_runs, measure_runs=measure_runs)
        )
        output.seek(0)
        json.dump(doc, output, indent=2, allow_nan=False)
        output.write("\n")
        output.truncate()
        output.flush()

    try:
        with benchmark_lock(port):
            checkpoint()
            for prompt, tag in plan:
                identity = {
                    "tag": tag,
                    "prompt_id": prompt["id"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "input_token_ids_sha256": prompt["input_token_ids_sha256"],
                }
                try:
                    row = request_fn(port, prompt["prompt"], timeout)
                    validate_result(
                        row, prompt, expected_iter_count=expected_iter_count
                    )
                except KeyboardInterrupt:
                    results.append(
                        dict(identity, ok=False, error="interrupted by operator")
                    )
                    doc["interrupted"] = True
                    checkpoint()
                    raise
                except Exception as exc:
                    row = {
                        "ok": False,
                        "error": "%s: %s" % (type(exc).__name__, str(exc)[:500]),
                    }
                row.update(identity)
                results.append(row)
                checkpoint()
    finally:
        checkpoint()
        output.close()
    return doc
