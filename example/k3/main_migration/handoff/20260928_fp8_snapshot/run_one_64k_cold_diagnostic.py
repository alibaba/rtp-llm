#!/usr/bin/env python3
"""Run one recorded 64K PD request to materialize a cold K3 runtime path."""

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--decode-ip", required=True)
    parser.add_argument("--decode-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    fixture_path = HERE / "timeline-input-64k/warmup-variant-01-20260926.json.gz"
    fixture = json.load(gzip.open(fixture_path, "rt"))
    assert fixture["input_len"] == len(fixture["input_ids"]) == 65536
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": "kimi-k3", "messages": fixture["messages"],
        "max_tokens": 8, "temperature": 0, "top_k": 1, "top_p": 0.95,
        "seed": 0, "stream": False, "debug_info": True,
        "enable_thinking": False,
        "extra_configs": {
            "ignore_eos": True, "reuse_cache": False,
            "role_addrs": [{"role": "DECODE", "ip": args.decode_ip,
                            "http_port": args.decode_port,
                            "grpc_port": args.decode_port + 1}],
        },
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    (output / "request.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with opener.open(request, timeout=args.timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as response:
        status, raw = response.code, response.read()
    except Exception as exc:
        (output / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    (output / "response.json").write_bytes(raw)
    body = json.loads(raw)
    aux = body.get("aux_info", {})
    result = {
        "status": status, "cold_http_elapsed_s": time.monotonic() - start,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_input_sha256": hashlib.sha256(json.dumps(
            fixture["input_ids"], separators=(",", ":")).encode()).hexdigest(),
        "response_input_ids_match": body.get("debug_info", {}).get("input_ids") == fixture["input_ids"],
        "usage": body.get("usage"), "pd_sep": aux.get("pd_sep"),
        "prefill_total_reuse_len": aux.get("prefill_total_reuse_len"),
        "decode_total_reuse_len": aux.get("decode_total_reuse_len"),
        "speculative_draft_rounds": aux.get("speculative_draft_rounds"),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if (status == 200 and result["response_input_ids_match"]
                 and result["pd_sep"] is True
                 and isinstance(result["speculative_draft_rounds"], int)
                 and result["speculative_draft_rounds"] > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
