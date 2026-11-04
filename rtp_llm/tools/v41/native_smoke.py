"""Exercise the unchanged HTTP generation API of a native P/D service."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        "--decode-url",
        dest="url",
        required=True,
        help="native P HTTP entrance for the existing P-to-D RPC protocol",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--expected-input-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.max_new_tokens < 2 or args.requests < 1 or args.timeout <= 0:
        parser.error("require >=2 output tokens, >=1 request and a positive timeout")
    args.output.mkdir(parents=True, exist_ok=False)
    query = {
        "prompt": args.prompt,
        "generate_config": {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.max_new_tokens,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 42,
            "reuse_cache": False,
            "can_use_pd_separation": True,
            "return_input_ids": True,
            "return_output_ids": True,
            "return_cum_log_probs": True,
            "aux_info": True,
            "timeout_ms": int(args.timeout * 1000),
        },
    }
    (args.output / "query.json").write_text(
        json.dumps(query, ensure_ascii=False, indent=2) + "\n"
    )
    observed = []
    for request in range(args.requests):
        started = time.monotonic()
        post = urllib.request.Request(
            args.url.rstrip("/") + "/",
            data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(post, timeout=args.timeout) as response:
                raw = response.read().decode()
                status = response.status
        except urllib.error.HTTPError as error:
            raw, status = error.read().decode(), error.code
        (args.output / f"request-{request}.http.txt").write_text(raw)
        result = json.loads(raw)
        if (
            status != 200
            or "error_code" in result
            or result.get("finished") is not True
        ):
            raise RuntimeError(f"native HTTP generation failed: {result}")
        outputs = result.get("output_ids", [])
        if len(outputs) != 1 or len(outputs[0]) != args.max_new_tokens:
            raise RuntimeError(
                "HTTP output token count differs from P-first-token + D continuation"
            )
        if any(type(token) is not int or token < 0 for token in outputs[0]):
            raise RuntimeError("invalid generated token ids")
        aux = result.get("aux_info", {})
        if (
            args.expected_input_tokens is not None
            and aux.get("input_len") != args.expected_input_tokens
        ):
            raise RuntimeError(
                "native input token count differs from the requested boundary case"
            )
        if aux.get("pd_sep") is not True:
            raise RuntimeError("native request did not use P/D separation")
        scores = aux.get("cum_log_probs", [])
        if scores and not all(math.isfinite(value) for value in scores):
            raise RuntimeError("nonfinite native generation scores")
        observed.append(
            {
                "request": request,
                "seconds": time.monotonic() - started,
                "output_ids": outputs,
                "aux_info": aux,
            }
        )
        print(json.dumps(observed[-1], ensure_ascii=False), flush=True)
    report = {
        "http_pd_generation_ok": True,
        "url": args.url,
        "requests": observed,
        "native_cache_store_transport_requires_server_evidence": True,
        "numerical_alignment": "deferred",
        "nv_alignment": "deferred",
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
