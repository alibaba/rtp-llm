#!/usr/bin/env python3
"""Fixed-token 64K request runner; each result preserves the raw response."""

import argparse
import gzip
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


FIXTURE = Path(__file__).parent / "timeline-input-64k" / "fixed.json.gz"
TOKEN_SHA = "97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee"
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body, timeout):
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with NO_PROXY.open(request, timeout=timeout) as response:
        raw = response.read()
        status = response.status
    return time.perf_counter() - start, status, json.loads(raw) if raw.strip() else {}, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("rtp", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-layers", type=int, choices=(4, 93), required=True)
    parser.add_argument("--decode-ip", default="11.163.39.114")
    parser.add_argument("--decode-port", type=int, default=25200)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--allow-expensive-warmup", action="store_true",
                        help="for full 93-layer runs only: allow at least three converged full requests")
    parser.add_argument("--max-extra-warmups", type=int, default=4)
    parser.add_argument("--variant-start", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--profile-prefill", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--profile-requests", type=int, default=1)
    parser.add_argument("--trace-name", default="k3_64k_prefill")
    parser.add_argument("--profile-api-base", help="direct Prefill profile API when base-url is a vLLM PD proxy")
    parser.add_argument("--profile-decode", action="store_true")
    parser.add_argument("--no-reset-cache", action="store_true")
    parser.add_argument("--no-reuse-cache", action="store_true",
                        help="disable RTP prefix reuse for every warmup and measured request")
    parser.add_argument("--warmup-stability-field", choices=("http", "first-token"), default="http")
    parser.add_argument("--profile-after-unstable-http-warmup", action="store_true",
                        help="four-layer vLLM only: profile after all warmups even if PD HTTP latency varies; inspect GPU trace stability separately")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--legacy-feat-aux", action="store_true",
                        help="record the older feat/k3_dev PD response, which omits draft counters and PD handoff length")
    parser.add_argument("--expected-decode-reuse-len", type=int)
    parser.add_argument("--block-size", type=int, default=4096)
    args = parser.parse_args()
    required = max(3 if args.allow_expensive_warmup else 10,
                   args.warmups or 0)
    if args.allow_expensive_warmup and required < 3:
        parser.error("expensive warmup still needs at least three requests")
    if (args.max_extra_warmups < 0 or args.variant_start < 1
            or args.profile_steps < 1 or args.profile_requests < 1):
        parser.error("invalid warmup variant selection")
    if args.profile_requests != 1 and not (args.profile_prefill or args.profile_decode):
        parser.error("multiple profiled requests require an active profiler")
    if args.allow_expensive_warmup and args.model_layers != 93:
        parser.error("the shortened warmup exception is only for the full 93-layer model")
    if args.profile_decode and args.backend != "rtp":
        parser.error("--profile-decode is supported only for RTP PD")
    if args.profile_prefill and args.backend == "vllm" and not args.profile_api_base:
        parser.error("vLLM PD profiling needs the direct Prefill --profile-api-base")
    if args.disable_thinking and args.backend != "rtp":
        parser.error("--disable-thinking is supported only for RTP chat rendering")
    if args.legacy_feat_aux and (args.backend != "rtp" or not args.disable_thinking):
        parser.error("--legacy-feat-aux requires RTP and --disable-thinking for identical input tokens")
    if args.legacy_feat_aux and args.expected_decode_reuse_len is not None:
        parser.error("older feat/k3_dev does not expose Decode handoff length")
    if args.warmup_stability_field == "first-token" and args.backend != "rtp":
        parser.error("first-token warmup stability requires RTP aux_info")
    if args.profile_after_unstable_http_warmup and not (
        args.backend == "vllm" and args.model_layers == 4
        and args.warmup_stability_field == "http" and args.profile_prefill
    ):
        parser.error("unstable HTTP warmup exception requires four-layer vLLM Prefill profiling")

    fixture = json.load(gzip.open(FIXTURE, "rt"))
    ids = fixture["input_ids"]
    assert len(ids) == fixture["input_len"] == 65536
    assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == TOKEN_SHA
    variant_count = required + args.max_extra_warmups
    variants = [json.load(gzip.open(
        FIXTURE.parent / f"warmup-variant-{n:02d}-20260926.json.gz", "rt"
    )) for n in range(args.variant_start, args.variant_start + variant_count)]
    for variant in variants:
        assert variant["input_len"] == len(variant["input_ids"]) == 65536
    first_blocks = [tuple(item["input_ids"][:16]) for item in [fixture, *variants]]
    if len(set(first_blocks)) != len(first_blocks):
        raise ValueError("each request must have a distinct first 16-token cache block")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    if args.backend == "rtp":
        url = base + "/v1/chat/completions"
        payload = {
            "model": "kimi-k3", "messages": fixture["messages"],
            "max_tokens": args.max_tokens, "temperature": 0, "top_k": 1,
            "top_p": 0.95, "seed": 0, "stream": False, "debug_info": True,
            "extra_configs": {
                "ignore_eos": True,
                "reuse_cache": not args.no_reuse_cache,
                "role_addrs": [{"role": "DECODE", "ip": args.decode_ip,
                                "http_port": args.decode_port,
                                "grpc_port": args.decode_port + 1}],
            },
        }
    else:
        url = base + "/v1/completions"
        payload = {
            "model": "kimi-k3", "prompt": ids, "max_tokens": args.max_tokens,
            "temperature": 0, "seed": 0, "stream": False,
            "ignore_eos": True,
        }
    if args.disable_thinking:
        payload["enable_thinking"] = False

    meta = {"backend": args.backend, "url": url, "token_ids_sha256": TOKEN_SHA,
            "input_tokens": len(ids), "max_tokens": args.max_tokens,
            "model_layers": args.model_layers,
            "http_elapsed_scope": "complete HTTP response, not Prefill duration; use rank traces",
            "disable_thinking": args.disable_thinking,
            "legacy_feat_aux": args.legacy_feat_aux,
            "reuse_cache": not args.no_reuse_cache,
            "warmup_stability_field": args.warmup_stability_field,
            "profile_after_unstable_http_warmup": args.profile_after_unstable_http_warmup,
            "mtp_execution_verified": False if args.legacy_feat_aux else None,
            "expected_decode_reuse_len": args.expected_decode_reuse_len,
            "block_size": args.block_size,
            "profile_steps": args.profile_steps,
            "profile_requests": args.profile_requests,
            "trace_name": args.trace_name,
            "warmup_required": required,
            "expensive_warmup_exception": args.allow_expensive_warmup,
            "variant_start": args.variant_start,
            "cache_control": "unique_first_16_tokens" if args.no_reset_cache or args.backend == "rtp" else "reset_api_and_unique_first_16_tokens",
            "vllm_pd_runtime_verified": False if args.backend == "vllm" else None,
            "payload_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                separators=(",", ":")).encode()).hexdigest(), "warmups": []}
    (args.output_dir / "input.json").write_text(json.dumps(meta, indent=2))

    def request_one(label, candidate):
        request_payload = dict(payload)
        if args.backend == "rtp":
            request_payload["messages"] = candidate["messages"]
        else:
            request_payload["prompt"] = candidate["input_ids"]
        elapsed, status, body, raw = post(url, request_payload, args.timeout)
        (args.output_dir / f"{label}.json").write_bytes(raw)
        if status != 200 or "error" in body:
            raise RuntimeError(f"{label}: HTTP {status}: {raw[:1000]!r}")
        usage = body.get("usage", {})
        observed = usage.get("prompt_tokens")
        if args.backend == "vllm" and observed is not None and observed != 65536:
            raise RuntimeError(f"{label}: prompt_tokens={observed}, expected 65536")
        if args.backend == "rtp" and observed is not None and observed not in (65536, 65533):
            raise RuntimeError(f"{label}: prompt_tokens={observed}, expected 65536 or 65533")
        if usage.get("completion_tokens") != args.max_tokens:
            raise RuntimeError(
                f"{label}: completion_tokens={usage.get('completion_tokens')}, "
                f"expected {args.max_tokens} with ignore_eos"
            )
        aux = body.get("aux_info", {})
        if args.backend == "rtp":
            if aux.get("pd_sep") is not True or aux.get("prefill_total_reuse_len") != 0:
                raise RuntimeError(f"{label}: PD/reuse mismatch: {aux!r}")
            decode_reuse = aux.get("decode_total_reuse_len")
            if args.legacy_feat_aux:
                if decode_reuse != 0:
                    raise RuntimeError(f"{label}: older feat PD response changed: {aux!r}")
            elif args.expected_decode_reuse_len is not None:
                if decode_reuse != args.expected_decode_reuse_len:
                    raise RuntimeError(f"{label}: unexpected PD cache handoff: {aux!r}")
            elif not isinstance(decode_reuse, int) or not (65536 - args.block_size <= decode_reuse <= 65536):
                raise RuntimeError(f"{label}: missing or short PD cache handoff: {aux!r}")
            if aux.get("input_len") != 65536:
                raise RuntimeError(f"{label}: input_len={aux.get('input_len')}")
            actual_ids = body.get("debug_info", {}).get("input_ids")
            if actual_ids != candidate["input_ids"]:
                raise RuntimeError(f"{label}: input IDs differ from fixture")
            if not args.legacy_feat_aux and aux.get("speculative_draft_rounds", 0) <= 0:
                raise RuntimeError(f"{label}: native MTP did not execute a draft round")
        entry = {"label": label, "elapsed_s": elapsed, "status": status,
                 "token_ids_sha256": candidate.get("token_ids_sha256", TOKEN_SHA),
                 "prompt_tokens": observed, "output_tokens": usage.get("completion_tokens"),
                 "response_sha256": hashlib.sha256(raw).hexdigest(),
                 "reuse_len": aux.get("reuse_len"),
                 "prefill_total_reuse_len": aux.get("prefill_total_reuse_len"),
                 "decode_total_reuse_len": aux.get("decode_total_reuse_len"),
                 "first_token_cost_ms": aux.get("first_token_cost_time"),
                 "speculative_draft_rounds": aux.get("speculative_draft_rounds"),
                 "pd_sep": aux.get("pd_sep")}
        print(json.dumps(entry), flush=True)
        return entry

    if required + args.max_extra_warmups > len(variants):
        raise ValueError("requested warmups exceed the fixed 64K variants")
    for n in range(required + args.max_extra_warmups):
        if args.backend == "vllm" and n > 0 and not args.no_reset_cache:
            _, status, clear, _ = post(base + "/reset_prefix_cache", {}, 30)
            if status != 200 or not clear.get("success"):
                raise RuntimeError(f"vLLM reset_prefix_cache failed: {clear}")
        entry = request_one(f"warmup-{n+1:02d}", variants[n])
        meta["warmups"].append(entry)
        (args.output_dir / "input.json").write_text(json.dumps(meta, indent=2))
        field = "first_token_cost_ms" if args.warmup_stability_field == "first-token" else "elapsed_s"
        latencies = [x[field] for x in meta["warmups"][-3:]]
        if len(meta["warmups"]) >= required:
            if any(x is None for x in latencies):
                raise RuntimeError(f"warmup stability field {field} is missing")
            median = statistics.median(latencies)
            if max(abs(x - median) for x in latencies) / median <= 0.05:
                meta["warmup_stable"] = True
                break
    else:
        meta["warmup_stable"] = False
    (args.output_dir / "input.json").write_text(json.dumps(meta, indent=2))
    if not meta["warmup_stable"] and not args.profile_after_unstable_http_warmup:
        raise RuntimeError("64K warmup did not converge; profiler was not started")
    if not meta["warmup_stable"]:
        meta["warmup_exception"] = (
            "four-layer vLLM PD HTTP latency did not converge after all warmups; "
            "GPU Prefill trace stability must be checked before performance claims"
        )
        (args.output_dir / "input.json").write_text(json.dumps(meta, indent=2))
    if args.backend == "vllm" and not args.no_reset_cache:
        _, status, clear, _ = post(base + "/reset_prefix_cache", {}, 30)
        if status != 200 or not clear.get("success"):
            raise RuntimeError(f"vLLM reset_prefix_cache failed before profile: {clear}")
    if args.profile_decode:
        decode_base = f"http://{args.decode_ip}:{args.decode_port}"
        _, status, result, _ = post(decode_base + "/start_profile", {
            "trace_name": "k3_64k_decode_single", "start_step": 0,
            "num_steps": args.profile_steps, "enable_all_rank": True,
        }, 30)
        if status != 200 or result.get("status") != "ok":
            raise RuntimeError(f"RTP decode profiler failed: {result}")
    if args.profile_prefill:
        if args.backend == "rtp":
            _, status, result, _ = post(base + "/start_profile", {
                "trace_name": args.trace_name, "start_step": 0,
                "num_steps": args.profile_steps, "enable_all_rank": True,
            }, 30)
            if status != 200 or result.get("status") != "ok":
                raise RuntimeError(f"RTP profiler failed: {result}")
        else:
            _, status, result, _ = post(args.profile_api_base.rstrip("/") + "/start_profile", {}, 30)
            if status != 200:
                raise RuntimeError(f"vLLM profiler failed: {result}")
    if args.profile_prefill or args.profile_decode:
        entries = [
            request_one(f"profiled-{n+1:02d}", fixture)
            for n in range(args.profile_requests)
        ]
        meta["profiled_requests"] = entries
        if len(entries) == 1:
            meta["profiled"] = entries[0]
    else:
        meta["measured"] = request_one("measured-01", fixture)
    if args.profile_prefill and args.backend == "vllm":
        _, status, result, _ = post(args.profile_api_base.rstrip("/") + "/stop_profile", {}, 120)
        if status != 200:
            raise RuntimeError(f"vLLM stop profiler failed: {result}")
    (args.output_dir / "input.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
