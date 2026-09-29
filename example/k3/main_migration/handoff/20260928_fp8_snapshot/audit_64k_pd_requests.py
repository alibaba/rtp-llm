#!/usr/bin/env python3
"""Independently audit recorded four-layer 64K RTP PD timeline requests."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


FIXED_INPUT_SHA = "97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee"


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def audit(directory: Path, decode_ip: str, decode_port: int) -> dict:
    errors = []
    meta = json.loads((directory / "requests/input.json").read_text(),
                      object_pairs_hook=unique_keys)
    legacy_feat_aux = meta.get("legacy_feat_aux") is True
    lines = [json.loads(line, object_pairs_hook=unique_keys)
             for line in (directory / "stdout.log").read_text().splitlines() if line.strip()]
    labels = [row.get("label") for row in lines]
    warmups = [row for row in lines if str(row.get("label", "")).startswith("warmup-")]
    expected = [f"warmup-{index:02d}" for index in range(1, len(warmups) + 1)]
    expected += [f"profiled-{index:02d}" for index in range(1, 17)]
    if labels != expected or len(warmups) < 10:
        errors.append("request labels are incomplete, duplicated, or out of order")
    if meta.get("model_layers") != 4 or meta.get("input_tokens") != 65536:
        errors.append("wrong four-layer 64K fixture")
    if meta.get("reuse_cache") is not False or meta.get("warmup_stable") is not True:
        errors.append("cache reuse or warmup convergence differs")
    if (directory / "exit").read_text().strip() != "0":
        errors.append("runner did not exit successfully")
    stability_field = ("first_token_cost_ms" if meta.get("warmup_stability_field") == "first-token"
                       else "elapsed_s")
    if len(warmups) >= 3:
        recent = [row.get(stability_field) for row in warmups[-3:]]
        if not all(isinstance(value, (float, int)) and value > 0 for value in recent):
            errors.append("invalid final warmup latencies")
        else:
            median = statistics.median(recent)
            if any(abs(value / median - 1) > 0.05 for value in recent):
                errors.append(f"final three {stability_field} warmups did not converge")
    files = {path.stem for path in (directory / "requests").glob("*.json")}
    if files != set(expected) | {"input"}:
        errors.append("raw response set differs from request labels")

    rows = []
    for line in lines:
        label = line.get("label")
        if label not in expected:
            continue
        row = {"label": label, "passed": False}
        try:
            raw = (directory / "requests" / f"{label}.json").read_bytes()
            body = json.loads(raw.decode("utf-8", errors="strict"),
                              object_pairs_hook=unique_keys)
            if line.get("status") != 200 or hashlib.sha256(raw).hexdigest() != line.get("response_sha256"):
                raise ValueError("HTTP status or raw response digest differs")
            ids = body["debug_info"]["input_ids"]
            digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            if len(ids) != 65536 or digest != line.get("token_ids_sha256"):
                raise ValueError("actual model input tokens differ from fixture digest")
            if label.startswith("profiled-") and digest != FIXED_INPUT_SHA:
                raise ValueError("profiled request did not use fixed 64K input")
            usage, aux = body["usage"], body["aux_info"]
            expected_prompt_tokens = (65533, 65536) if legacy_feat_aux else (65536,)
            if usage.get("prompt_tokens") not in expected_prompt_tokens or usage.get("completion_tokens") != 8:
                raise ValueError("input or output token count differs")
            if aux.get("pd_sep") is not True or aux.get("prefill_total_reuse_len") != 0:
                raise ValueError("PD flag or Prefill cache reuse differs")
            if legacy_feat_aux:
                if aux.get("decode_total_reuse_len") != 0:
                    raise ValueError("legacy Decode reuse field differs")
            elif aux.get("decode_total_reuse_len") != 61440:
                raise ValueError("Decode KV handoff differs from 15 complete pages")
            if not any(isinstance(addr, dict) and addr.get("ip") == decode_ip
                       and addr.get("http_port") == decode_port
                       for addr in aux.get("role_addrs", [])):
                raise ValueError("Decode route differs")
            draft = aux.get("speculative_draft_rounds")
            if not legacy_feat_aux and (not isinstance(draft, int) or draft <= 0):
                raise ValueError("native MTP draft did not execute")
            content = body["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip() or "\x00" in content:
                raise ValueError("empty or NUL-containing response text")
            content.encode("utf-8", errors="strict")
            row.update(passed=True, draft_rounds=draft,
                       replacement_characters=content.count("\ufffd"))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{label}: {row['error']}")
        rows.append(row)
    report = {
        "passed": not errors,
        "warmup_count": len(warmups),
        "profiled_count": sum(row["label"].startswith("profiled-") for row in rows),
        "replacement_characters_observed": sum(row.get("replacement_characters", 0) for row in rows),
        "semantic_answer_claim": False,
        "mtp_execution_claim": not legacy_feat_aux and not errors,
        "decode_handoff_length_claim": not legacy_feat_aux and not errors,
        "errors": errors,
        "requests": rows,
    }
    (directory / "independent-request-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decode-ip", default="11.163.39.112")
    parser.add_argument("--decode-port", type=int, default=27200)
    args = parser.parse_args()
    result = audit(args.directory, args.decode_ip, args.decode_port)
    print(json.dumps({key: result[key] for key in
                      ("passed", "warmup_count", "profiled_count", "replacement_characters_observed", "errors")}))
    raise SystemExit(0 if result["passed"] else 1)
