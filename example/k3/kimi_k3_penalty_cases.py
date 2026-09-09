"""Build an immutable, audited replay manifest from exported failure cases.

This prepares requests; it never sends them. Unspecified generation defaults
remain unresolved until the effective server configurations are captured.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_request(
    request: dict[str, Any], *, case_type: str, protocol: str, model: str
) -> dict[str, Any]:
    """Preserve explicit sampling intent and document every transport change."""
    changes: list[dict[str, Any]] = []
    if protocol not in {"http_sync", "http_sse"}:
        raise ValueError(f"unsupported protocol: {protocol}")
    if case_type == "dash":
        unknown = set(request) - {"model", "input", "parameters"}
        if unknown or set(request.get("input", {})) != {"messages"}:
            raise ValueError(f"unsupported DashScope envelope: {sorted(unknown)}")
        params = copy.deepcopy(request.get("parameters", {}))
        messages = copy.deepcopy(request["input"]["messages"])
        changes.append({"source": "input.messages", "target": "messages"})
        prefix = "parameters."
    elif case_type == "openai":
        params = copy.deepcopy(request)
        messages = params.pop("messages")
        params.pop("model", None)
        prefix = ""
    else:
        raise ValueError(f"unsupported case type: {case_type}")
    if not isinstance(messages, list) or not messages:
        raise ValueError("nonempty messages are required")

    allowed = {
        "max_length",
        "max_tokens",
        "repetition_penalty",
        "result_format",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "enable_thinking",
        "thinking_budget",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unmapped request fields: {sorted(unknown)}")
    if "result_format" in params:
        if params.pop("result_format") != "message":
            raise ValueError("only result_format=message has a chat mapping")
        changes.append(
            {
                "source": prefix + "result_format",
                "target": None,
                "reason": "chat completion response envelope replaces DashScope format",
            }
        )
    if "max_length" in params:
        max_length = params.pop("max_length")
        if "max_tokens" in params and params["max_tokens"] != max_length:
            raise ValueError("max_length and max_tokens disagree")
        params["max_tokens"] = max_length
        changes.append(
            {
                "source": prefix + "max_length",
                "target": "max_tokens",
                "reason": "replay completion budget from exported case assertion",
                "production_gateway_mapping_verified": False,
            }
        )
    if "max_tokens" in params:
        value = params["max_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_tokens must be a positive integer")
    repetition = params.get("repetition_penalty")
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, (int, float))
        or not math.isfinite(repetition)
        or repetition <= 0
    ):
        raise ValueError("a finite positive repetition_penalty is required")

    stream = protocol == "http_sse"
    if "stream" in params and params["stream"] is not stream:
        raise ValueError("request stream field conflicts with exported protocol")
    if "stream" not in params:
        changes.append({"source": "index.protocol", "target": "stream"})
    params["stream"] = stream
    stream_options = params.pop("stream_options", None)
    if stream_options is not None:
        if not stream or stream_options != {"include_usage": True}:
            raise ValueError("unsupported stream_options")
        changes.append(
            {
                "source": prefix + "stream_options",
                "target": "vllm.stream_options",
                "reason": "RTP ChatCompletionRequest has no stream_options field",
            }
        )

    canonical = {"messages": messages, **params}
    rtp = {"model": model, **copy.deepcopy(canonical)}
    rtp["extra_configs"] = {"repetition_penalty": rtp.pop("repetition_penalty")}
    rtp["debug_info"] = True
    changes.extend(
        [
            {"source": "model", "target": "model", "value": model},
            {
                "source": prefix + "repetition_penalty",
                "target": "rtp.extra_configs.repetition_penalty",
                "reason": "RTP ChatCompletionRequest has no top-level repetition_penalty",
            },
            {
                "source": None,
                "target": "rtp.debug_info",
                "value": True,
                "reason": "request generated token IDs for trace correlation",
            },
        ]
    )
    vllm = {"model": model, **copy.deepcopy(canonical)}
    if stream:
        vllm["stream_options"] = {"include_usage": True}
        if stream_options is None:
            changes.append(
                {
                    "source": None,
                    "target": "vllm.stream_options.include_usage",
                    "value": True,
                    "reason": "collect final usage for SSE replay",
                }
            )
    # thinking_budget is an RTP extension, not a proven vLLM K3 mapping.
    if "thinking_budget" in vllm or "enable_thinking" in vllm:
        raise ValueError("thinking controls require an explicit vLLM renderer mapping")

    defaults = [
        name
        for name in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "enable_thinking",
            "thinking_budget",
        )
        if name not in canonical
    ]
    return {
        "canonical": canonical,
        "rtp_request": rtp,
        "vllm_request": vllm,
        "transformations": changes,
        "unresolved_defaults": defaults,
        "ready_for_comparable_replay": False,
        "remaining_gates": [
            "capture effective per-framework defaults and production gateway budget",
            "verify rendered prompt token IDs and stop token IDs",
            "verify actual backend repetition penalty and thinking settings",
        ],
    }


def build_manifest(source: Path, model: str) -> dict[str, Any]:
    source = source.resolve()
    index = source / "index.json"
    entries = json.loads(index.read_text())
    cases = []
    seen = set()
    for entry in entries:
        case_id = entry["directory"]
        case_dir = (source / case_id).resolve()
        if case_dir.parent != source or case_id in seen:
            raise ValueError(f"invalid or duplicate case directory: {case_id}")
        seen.add(case_id)
        path = case_dir / "request.json"
        request = json.loads(path.read_text())
        adapted = adapt_request(
            request,
            case_type=entry["case_type"],
            protocol=entry["protocol"],
            model=model,
        )
        declared = set(entry.get("penalties", {}).values())
        if declared != {adapted["canonical"]["repetition_penalty"]}:
            raise ValueError(f"index penalty differs from request: {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "original_protocol": entry["protocol"],
                "original_api": entry["case_type"],
                "source_request": str(path),
                "original_request": request,
                "source_files": {
                    file.name: sha256(file)
                    for file in sorted(case_dir.iterdir())
                    if file.is_file()
                },
                "online_observation": entry.get("observed_result"),
                "online_assertions": entry.get("errors"),
                **adapted,
            }
        )
    return {
        "schema_version": 1,
        "source_index": str(index),
        "source_index_sha256": sha256(index),
        "adapter_sha256": sha256(Path(__file__)),
        "case_count": len(cases),
        "case_order": "source index order; no deduplication",
        "endpoint": "/v1/chat/completions",
        "scope": "local-engine replay; production gateway is not reproduced",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.source, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(f"prepared {manifest['case_count']} cases: {args.output}")


if __name__ == "__main__":
    main()
