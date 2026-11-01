#!/usr/bin/env python3
"""Record serial real-model OpenAI SSE requests; compare operator-run A-B-B-A.

Stdlib only. Never starts a server, imports CUDA, or claims HTTP chunks are GPU
steps. Output includes every raw SSE data event, including errors and outliers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_EVENT_BYTES = 2 << 20
MAX_RESPONSE_BYTES = 64 << 20
LABEL_MODES = {"A1": "local-pinned", "B1": "external-egm", "B2": "external-egm", "A2": "local-pinned"}


class StreamError(ValueError):
    """Bounded public error code; never wrap raw native exceptions."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_number(value, minimum=0):
    return type(value) in (float, int) and math.isfinite(value) and value >= minimum


def token_count(value):
    return type(value) is int and value >= 0


def sse_events(stream, *, started, clock=time.perf_counter):
    """Yield complete SSE data events, timestamped when the separator is read.

    readline may itself be buffered. Timestamps describe client observation,
    not the network packet boundary or a server token/kernel completion.
    """
    data, size, total = [], 0, 0
    while True:
        line = stream.readline(MAX_EVENT_BYTES + 1)
        if not line:
            if data:
                raise StreamError("truncated_sse_event")
            return
        total += len(line)
        size += len(line)
        if len(line) > MAX_EVENT_BYTES or size > MAX_EVENT_BYTES or total > MAX_RESPONSE_BYTES:
            raise StreamError("response_size_limit")
        if not line.endswith(b"\n"):
            raise StreamError("truncated_sse_line")
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield {"received_ms": (clock() - started) * 1000, "data": "\n".join(data)}
            data, size = [], 0
            continue
        if line.startswith(b":"):
            continue
        field, sep, value = line.partition(b":")
        if value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            try:
                data.append(value.decode("utf-8"))
            except UnicodeDecodeError:
                raise StreamError("invalid_sse_utf8") from None


class Observation:
    def __init__(self):
        self.events = []
        self.first_event_ms = None
        self.first_text_ms = None
        self.last_text_ms = None
        self.text_event_times_ms = []
        self.content = []
        self.reasoning = []
        self.usage = None
        self.aux = None
        self.final_usage_received = False
        self.final_aux = None
        self.finish_reason = None
        self.done = False
        self.previous_tokens = 0
        self.prompt_tokens = None
        self.warnings = []

    def accept(self, event):
        self.events.append(event)
        if self.first_event_ms is None:
            self.first_event_ms = event["received_ms"]
        data = event["data"]
        if data.strip().lower() == "[done]":
            if self.finish_reason is None:
                raise StreamError("done_without_finish_reason")
            self.done = True
            return
        if self.done:
            raise StreamError("data_after_done")
        try:
            obj = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, TypeError):
            raise StreamError("invalid_sse_json") from None
        if not isinstance(obj, dict):
            raise StreamError("invalid_sse_object")
        if "error" in obj or "error_code" in obj:
            raise StreamError("server_inference_error")
        choices = obj.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamError("expected_single_choice")
        text = False
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index") != 0:
                raise StreamError("invalid_choice_index")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise StreamError("invalid_delta")
            if delta.get("tool_calls") or delta.get("function_call"):
                raise StreamError("unexpected_tool_output")
            for name, destination in (("content", self.content), ("reasoning_content", self.reasoning)):
                value = delta.get(name)
                if value is not None and not isinstance(value, str):
                    raise StreamError("invalid_text_delta")
                if value:
                    if self.finish_reason is not None:
                        raise StreamError("text_after_finish")
                    destination.append(value)
                    text = True
            reason = choice.get("finish_reason")
            if reason is not None:
                if self.finish_reason is not None or not isinstance(reason, str):
                    raise StreamError("invalid_finish_reason")
                if reason not in ("stop", "length"):
                    raise StreamError("unexpected_finish_reason")
                self.finish_reason = reason
        if text:
            stamp = event["received_ms"]
            if self.first_text_ms is None:
                self.first_text_ms = stamp
            self.last_text_ms = stamp
            self.text_event_times_ms.append(stamp)
        usage = obj.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise StreamError("invalid_usage")
            prompt, completion, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            if not all(token_count(x) for x in (prompt, completion, total)) or total != prompt + completion:
                raise StreamError("invalid_usage_counts")
            if completion < self.previous_tokens:
                raise StreamError("nonmonotonic_cumulative_usage")
            if prompt:
                if self.prompt_tokens is not None and prompt != self.prompt_tokens:
                    raise StreamError("prompt_token_count_changed")
                self.prompt_tokens = prompt
            self.previous_tokens, self.usage = completion, usage
            if self.finish_reason is not None:
                self.final_usage_received = True
        aux = obj.get("aux_info")
        if aux is not None:
            if not isinstance(aux, dict):
                raise StreamError("invalid_aux_info")
            self.aux = aux
            if self.finish_reason is not None:
                self.final_aux = aux

    def summary(self, *, eof_ms, expected_prompt_tokens, requested_tokens):
        if self.finish_reason is None:
            raise StreamError("eof_without_finish_reason")
        if not self.content or not "".join(self.content).strip():
            raise StreamError("missing_visible_content")
        if not self.final_usage_received or self.usage is None or self.usage["completion_tokens"] < 2:
            raise StreamError("missing_or_insufficient_final_usage")
        if self.usage["prompt_tokens"] != expected_prompt_tokens:
            raise StreamError("unexpected_prompt_token_count")
        count = self.usage["completion_tokens"]
        if count > requested_tokens:
            raise StreamError("completion_exceeds_requested_tokens")
        if count != requested_tokens:
            self.warnings.append("early_finish_not_fixed_length")
        aux = self.final_aux or {}
        cost, first, engine_count = aux.get("cost_time"), aux.get("first_token_cost_time"), aux.get("output_len")
        engine_tpot = None
        if (finite_number(cost) and finite_number(first) and first > 0 and cost >= first
                and token_count(engine_count) and engine_count > 1):
            if engine_count != count:
                self.warnings.append("engine_output_count_differs_from_usage")
            else:
                engine_tpot = (cost - first) / (engine_count - 1)
        if engine_tpot is None:
            self.warnings.append("engine_average_tpot_unavailable")
        times = self.text_event_times_ms
        return {"first_sse_event_ms": self.first_event_ms,
                "client_visible_text_ttft_ms": self.first_text_ms,
                "client_last_text_ms": self.last_text_ms, "client_eof_ms": eof_ms,
                "client_text_window_per_final_token_ms": (self.last_text_ms - self.first_text_ms) / (count - 1),
                "text_event_intervals_ms": [b - a for a, b in zip(times, times[1:])],
                "text_event_count": len(times), "sse_event_count": len(self.events),
                "usage": self.usage, "aux_info": self.final_aux, "finish_reason": self.finish_reason,
                "engine_stream_average_tpot_ms": engine_tpot, "warnings": self.warnings,
                "content": "".join(self.content), "reasoning_content": "".join(self.reasoning),
                "timing_boundary": "client observations and engine stream wall time; no GPU or individual decode-step timestamps"}


def payload(prompt, seed, tokens, *, temperature=0.7):
    return {"model": "deepseek_v41", "messages": [{"role": "user", "content": prompt}],
            "stream": True, "n": 1, "seed": seed, "max_tokens": tokens,
            "temperature": temperature, "thinking": {"type": "disabled"}, "aux_info": True,
            "extra_configs": {"min_new_tokens": tokens, "ignore_eos": True, "reuse_cache": False}}


def measure(endpoint, body, *, expected_prompt_tokens, timeout=600, opener=None, clock=time.perf_counter):
    observation = Observation()
    record = {"status": "error", "request": body, "events": observation.events}
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = clock()
    try:
        with opener.open(request, timeout=timeout) as response:
            record["http_status"] = response.status
            record["headers_received_ms"] = (clock() - started) * 1000
            if response.status != 200 or "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                record["unexpected_body"] = response.read(MAX_EVENT_BYTES + 1).decode("utf-8", errors="replace")
                raise StreamError("unexpected_http_response")
            for event in sse_events(response, started=started, clock=clock):
                observation.accept(event)
            summary = observation.summary(eof_ms=(clock() - started) * 1000,
                                          expected_prompt_tokens=expected_prompt_tokens, requested_tokens=body["max_tokens"])
            record.update(status="ok", metrics=summary)
    except urllib.error.HTTPError as error:
        record.update(status="error", error_code="http_error", http_status=error.code)
        try:
            record["error_body"] = error.read(MAX_EVENT_BYTES + 1).decode("utf-8", errors="replace")
        except Exception as body_error:
            # HTTPError itself is a response stream. Its body can independently
            # time out/reset: preserve this request and continue the run.
            record.update(error_body_read_code="body_read_failed", error_body_read_type=type(body_error).__name__)
        finally:
            try:
                error.close()
            except Exception as close_error:
                record.update(error_body_close_code="body_close_failed", error_body_close_type=type(close_error).__name__)
    except Exception as error:
        record.update(status="error", error_code=str(error) if isinstance(error, StreamError) else "transport_or_client_error",
                      error_type=type(error).__name__)
    record["client_completed_ms"] = (clock() - started) * 1000
    return record


def write_new(path, value):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def public_identity(report):
    """Whitelist public reproducibility fields. Never copy a private bundle."""
    checkpoint = report["checkpoint"]
    return {"storage_mode": report["storage_mode"], "model_revision": report["model_revision"],
            "config_sha256": checkpoint["config_sha256"], "tokenizer_sha256": checkpoint["tokenizer_sha256"],
            "index_sha256": checkpoint["index_sha256"], "native_sha256": report["runtime"]["native_sha256"],
            "shard_header_sha256": checkpoint["shard_header_sha256"],
            "experiment_environment": report["experiment_environment"],
            "full_layer_count": checkpoint["full_layer_count"], "engram_layers": checkpoint["engram_layers"],
            "consumer_has_table_entries": checkpoint["consumer_has_table_entries"]}


def run(args):
    parsed = urllib.parse.urlsplit(args.endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("explicit HTTP endpoint without credentials/query required")
    if args.tokens < 2 or args.expected_prompt_tokens < 1 or args.warmups < 1 or args.timeout <= 0:
        raise ValueError("positive timing parameters and at least one warmup required")
    if args.tokens + args.expected_prompt_tokens > 4096:
        raise ValueError("request exceeds fixed launch sequence capacity")
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
        raise ValueError("distinct nonnegative seeds required")
    identity = public_identity(json.loads(args.launch_report.read_text()))
    if identity["storage_mode"] != LABEL_MODES[args.label]:
        raise ValueError("A labels require local-pinned; B labels require external-egm")
    prompt = args.prompt_file.read_text()
    if not prompt.strip():
        raise ValueError("nonempty prompt required")
    request_template = payload(prompt, 0, args.tokens, temperature=args.temperature)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    manifest = {"schema": "rtp-engram-http-ab-v1", "cycle": args.cycle, "label": args.label,
                "identity": identity, "endpoint": args.endpoint, "started_unix_ns": time.time_ns(),
                "seeds": seeds, "warmups": args.warmups, "expected_prompt_tokens": args.expected_prompt_tokens,
                "request_template_sha256": fingerprint(request_template), "request_template": request_template,
                "note": "launch report is operator evidence, not runtime Graph attestation"}
    write_new(args.output / "manifest.json", manifest)
    failures = 0
    for index, seed in enumerate([seeds[0]] * args.warmups + seeds):
        warmup = index < args.warmups
        result = measure(args.endpoint, payload(prompt, seed, args.tokens, temperature=args.temperature),
                         expected_prompt_tokens=args.expected_prompt_tokens, timeout=args.timeout)
        result.update(index=index, warmup=warmup, seed=seed, label=args.label, cycle=args.cycle)
        write_new(args.output / ("request-{:05d}.json".format(index)), result)
        failures += result["status"] != "ok"
        print(json.dumps({"index": index, "warmup": warmup, "status": result["status"]}), flush=True)
    write_new(args.output / "complete.json", {"finished_unix_ns": time.time_ns(), "requests": len(seeds) + args.warmups,
                                             "failures": failures})
    return int(failures != 0)


def compare_runs(paths):
    """All inputs retained; incompatibilities invalidate, never silently filter."""
    manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    completions = [json.loads((path / "complete.json").read_text()) for path in paths]
    if [m["label"] for m in manifests] != list(LABEL_MODES):
        raise ValueError("runs must be ordered A1 B1 B2 A2")
    for before, after in zip(range(3), range(1, 4)):
        if completions[before]["finished_unix_ns"] > manifests[after]["started_unix_ns"]:
            raise ValueError("runs overlap or are out of chronological ABBA order")
    common_keys = ("cycle", "seeds", "warmups", "expected_prompt_tokens", "request_template_sha256")
    identity_keys = ("model_revision", "config_sha256", "tokenizer_sha256", "native_sha256", "experiment_environment",
                     "full_layer_count", "engram_layers")
    for m in manifests:
        if m["identity"]["storage_mode"] != LABEL_MODES[m["label"]]:
            raise ValueError("storage mode label mismatch")
        if m["identity"]["consumer_has_table_entries"] != (m["label"].startswith("A")):
            raise ValueError("checkpoint table presence contradicts storage mode")
        if any(m[k] != manifests[0][k] for k in common_keys):
            raise ValueError("request protocol differs across ABBA runs")
        if any(m["identity"][k] != manifests[0]["identity"][k] for k in identity_keys):
            raise ValueError("model/runtime/launch settings differ across ABBA runs")
    for first, second in ((0, 3), (1, 2)):
        for key in ("index_sha256", "shard_header_sha256"):
            if manifests[first]["identity"][key] != manifests[second]["identity"][key]:
                raise ValueError("same-mode checkpoint identity changed between repeats")
    all_runs = [[json.loads(file.read_text()) for file in sorted(path.glob("request-*.json"))] for path in paths]
    expected_requests = len(manifests[0]["seeds"]) + manifests[0]["warmups"]
    for records, done in zip(all_runs, completions):
        if len(records) != expected_requests or done["requests"] != expected_requests:
            raise ValueError("incomplete request files")
        if [r["index"] for r in records] != list(range(expected_requests)):
            raise ValueError("request indices missing or reordered")
    rows = []
    for index in range(expected_requests):
        quartet = [records[index] for records in all_runs]
        reasons = []
        warmup = index < manifests[0]["warmups"]
        expected_seed = manifests[0]["seeds"][0 if warmup else index - manifests[0]["warmups"]]
        for r, m in zip(quartet, manifests):
            if (r["warmup"] != warmup or r["seed"] != expected_seed or r["label"] != m["label"]
                    or r["cycle"] != m["cycle"]):
                raise ValueError("request identity mismatch")
            body = {**r["request"], "seed": 0}
            if fingerprint(body) != m["request_template_sha256"]:
                raise ValueError("request body differs from manifest")
        if any(r["status"] != "ok" for r in quartet):
            reasons.append("request_failure")
        else:
            metrics = [r["metrics"] for r in quartet]
            if len({(x["content"], x["reasoning_content"]) for x in metrics}) != 1:
                reasons.append("different_generated_text")
            if any(x["warnings"] for x in metrics):
                reasons.append("measurement_warnings")
            for key in ("prompt_tokens", "completion_tokens"):
                if len({x["usage"][key] for x in metrics}) != 1:
                    reasons.append("different_" + key)
            if any(not isinstance(x["aux_info"], dict) or not token_count(x["aux_info"].get("reuse_len")) for x in metrics):
                reasons.append("missing_reuse_length")
            elif len({x["aux_info"]["reuse_len"] for x in metrics}) != 1:
                reasons.append("different_reuse_length")
            if any(x["engine_stream_average_tpot_ms"] is None for x in metrics):
                reasons.append("missing_engine_mean")
        row = {"index": index, "seed": expected_seed, "warmup": warmup, "invalid_reasons": reasons}
        if not reasons:
            values = [r["metrics"]["engine_stream_average_tpot_ms"] * 1000 for r in quartet]
            # A1 B1 B2 A2; pairing at each side also exposes order drift.
            deltas = [values[1] - values[0], values[2] - values[3]]
            row.update(engine_request_mean_tpot_us=values, paired_request_mean_delta_us=deltas,
                       block_mean_delta_us=statistics.mean(deltas),
                       paired_request_mean_delta_exceeds_100us=[v > 100 for v in deltas],
                       text_equal=len({r["metrics"]["content"] for r in quartet}) == 1)
        rows.append(row)
    valid = all(not r["invalid_reasons"] for r in rows)
    return {"schema": "rtp-engram-http-ab-comparison-v1", "paths": [str(path) for path in paths],
            "comparable": valid, "rows": rows,
            "full_decode_step_100us_acceptance": "not_established_by_request_mean_or_http_chunks",
            "note": "No outliers removed. Means are engine stream wall time per output token; no per-step or GPU P99."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    req = commands.add_parser("run")
    req.add_argument("--endpoint", default="http://127.0.0.1:18630/v1/chat/completions")
    req.add_argument("--launch-report", type=Path, required=True)
    req.add_argument("--prompt-file", type=Path, required=True)
    req.add_argument("--expected-prompt-tokens", type=int, required=True)
    req.add_argument("--tokens", type=int, default=256)
    req.add_argument("--seeds", default="11,23,37,51,67")
    req.add_argument("--temperature", type=float, default=0.7)
    req.add_argument("--warmups", type=int, default=2)
    req.add_argument("--timeout", type=float, default=600)
    req.add_argument("--label", choices=tuple(LABEL_MODES), required=True)
    req.add_argument("--cycle", required=True)
    req.add_argument("--output", type=Path, required=True)
    cmp = commands.add_parser("compare")
    cmp.add_argument("runs", nargs=4, type=Path, help="A1 B1 B2 A2 output directories, in order")
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if not finite_number(args.temperature) or args.temperature > 2:
            parser.error("temperature must be finite in [0,2]")
        return run(args)
    result = compare_runs(args.runs)
    write_new(args.output, result)
    print(json.dumps({"comparable": result["comparable"], "output": str(args.output)}))
    return 0 if result["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
