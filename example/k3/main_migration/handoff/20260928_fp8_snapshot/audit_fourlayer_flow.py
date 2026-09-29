#!/usr/bin/env python3
"""Independently recheck recorded four-layer K3 PD flow responses."""

import argparse
import json
from pathlib import Path


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def audit(directory: Path, decode_ip: str = "11.163.39.112", decode_port: int = 25200) -> dict:
    result = json.loads((directory / "result.json").read_text(), object_pairs_hook=unique_keys)
    errors = []
    cases = result.get("cases", [])
    names = [case.get("name") for case in cases]
    expected = {"chunkwise_rdma_flow_miss", "flow_hit_owner_0", "flow_mtp_draft"}
    expected.update(f"flow_uneven_{i}{suffix}" for i in range(4) for suffix in ("", "_hit"))
    if result.get("suite") != "flow" or set(names) != expected or len(names) != 11:
        errors.append("suite/case set differs from the 11-case four-layer flow")
    if result.get("passed") is not True or result.get("failures") or result.get("skipped_cases"):
        errors.append("runner reported a failure or skip")
    if result.get("formal_request_deadline_s") != 300:
        errors.append("case deadline differs from 300 seconds")
    if result.get("full_original_suite_passed") is not False:
        errors.append("flow incorrectly claims complete original suite")

    rows = []
    seen = set()
    for path in sorted((directory / "requests").glob("*.json")):
        record = json.loads(path.read_text(), object_pairs_hook=unique_keys)
        name = record.get("name")
        row = {"name": name, "file": path.name, "passed": False}
        try:
            if name in seen:
                raise ValueError("duplicate request artifact")
            seen.add(name)
            if name not in expected or record.get("phase") != "formal":
                raise ValueError("unexpected request")
            if record.get("passed") is not True or record.get("status") != 200:
                raise ValueError(f"record failed or HTTP {record.get('status')}")
            wire = json.loads(record["response_body"], object_pairs_hook=unique_keys)
            aux = wire["aux_info"]
            measured = record["result"]
            content = wire["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty or non-text output")
            content.encode("utf-8", errors="strict")
            # A four-layer checkpoint has random logits and can emit token
            # bytes that decode to U+FFFD. Count these for the full-model
            # answer audit, but do not treat them as a PD flow failure here.
            replacement_count = content.count("\ufffd")
            if "\x00" in content:
                raise ValueError("NUL character")
            if aux.get("pd_sep") is not True or measured.get("pd_sep") is not True:
                raise ValueError("PD handoff missing")
            input_len = measured.get("input_len")
            expected_handoff = ((input_len - 1) // result["block_size"]) * result["block_size"]
            if aux.get("decode_total_reuse_len") != expected_handoff:
                raise ValueError("Decode KV handoff differs from full-block count")
            if not any(isinstance(addr, dict) and addr.get("ip") == decode_ip
                       and addr.get("http_port") == decode_port
                       for addr in aux.get("role_addrs", [])):
                raise ValueError("Decode owner route differs")
            if not isinstance(aux.get("speculative_draft_rounds"), int) or aux["speculative_draft_rounds"] <= 0:
                raise ValueError("Native MTP draft did not execute")
            expected_len = record["expected"].get("input_len")
            if expected_len is not None and measured.get("input_len") != expected_len:
                raise ValueError("input length differs from fixture")
            elapsed = measured.get("elapsed_s")
            if not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 300:
                raise ValueError("request exceeded case deadline")
            row.update(passed=True, elapsed_s=elapsed, input_len=input_len,
                       output_len=measured.get("output_len"),
                       decode_handoff_len=aux["decode_total_reuse_len"],
                       mtp_draft_rounds=aux["speculative_draft_rounds"],
                       replacement_characters=replacement_count)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{name}: {row['error']}")
        rows.append(row)
    if seen != expected:
        errors.append("request artifacts differ from expected cases")
    report = {"passed": not errors, "checked": len(rows), "runner_cases": len(cases),
              "semantic_answer_claim": False,
              "replacement_characters_observed": sum(row.get("replacement_characters", 0) for row in rows),
              "errors": errors, "cases": rows}
    (directory / "independent-flow-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decode-ip", default="11.163.39.112")
    parser.add_argument("--decode-port", type=int, default=25200)
    args = parser.parse_args()
    summary = audit(args.directory, args.decode_ip, args.decode_port)
    print(json.dumps({key: summary[key] for key in ("passed", "checked", "errors")}))
    raise SystemExit(0 if summary["passed"] else 1)
