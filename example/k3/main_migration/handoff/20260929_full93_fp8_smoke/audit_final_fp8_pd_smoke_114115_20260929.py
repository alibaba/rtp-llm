#!/usr/bin/env python3
"""Independent raw-answer and PD/MTP audit for the capped full-model smoke."""

import argparse
import json
import pathlib
import re
import sys


sys.path.insert(0, "/data0/luohaocheng.lhc/artifacts/k3-fp8-main-20260926")
from independent_answer_recheck import check, unique_keys  # noqa: E402


DEFERRED = {
    "decode-page-cross-4096-cold",
    "decode-page-cross-4096-repeat",
    "decode-page-cross-8192-cold",
    "decode-page-cross-8192-repeat",
    "decode-page-cross-65536-cold",
    "decode-page-cross-65536-repeat",
}


def audit(directory):
    result = json.loads((directory / "result.json").read_text())
    errors = []
    rows = []
    result_names = [case["name"] for case in result.get("cases", [])]
    if len(result_names) != len(set(result_names)):
        errors.append("duplicate case names in result")
    if result.get("suite") != "main-text-64k-capped":
        errors.append("unexpected suite")
    if result.get("full_original_suite_passed") is not False:
        errors.append("original complete suite was misreported as passed")
    if result.get("formal_request_deadline_s") != 300:
        errors.append("formal request deadline is not 300 seconds")
    if not result.get("passed"):
        errors.append("runner did not pass")
    skipped = result.get("skipped_cases", [])
    if {case.get("name") for case in skipped} != DEFERRED or len(skipped) != len(DEFERRED):
        errors.append("deferred cases differ from the six authorized cases")
    if any(case.get("sent") is not False for case in skipped):
        errors.append("a deferred case was sent")

    seen = set()
    for path in sorted((directory / "requests").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("phase") != "formal":
            continue
        if record.get("skipped"):
            errors.append(f"{record.get('name')}: timed-out or skipped formal request")
            continue
        name = record.get("name")
        if name in seen:
            errors.append(f"{name}: duplicate request artifact")
            continue
        seen.add(name)
        row = {"name": name, "artifact": str(path), "passed": False}
        try:
            if record.get("status") != 200:
                raise ValueError(f"HTTP status {record.get('status')}")
            if float(record.get("result", {}).get("elapsed_s", 301)) > 300:
                raise ValueError("request exceeded 300 seconds")
            wire = json.loads(record["response_body"], object_pairs_hook=unique_keys)
            choice = wire["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"finish_reason={choice.get('finish_reason')!r}")
            message = choice["message"]
            answer = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if not isinstance(answer, str) or not isinstance(reasoning, str):
                raise ValueError("non-text output channel")
            for channel, value in (("content", answer), ("reasoning", reasoning)):
                value.encode("utf-8", errors="strict")
                if "\ufffd" in value or "\x00" in value:
                    raise ValueError(f"{channel} contains replacement or NUL characters")
                if re.search(r"([^\s])\1{7,}", value):
                    raise ValueError(f"{channel} contains repeated characters")
            prompt = record["request"]["messages"][-1]["content"]
            kind, expected, observed, correct = check(prompt, answer)
            if not correct:
                raise ValueError(f"{kind} answer differs from prompt-derived value")
            aux = wire["aux_info"]
            if aux.get("pd_sep") is not True:
                raise ValueError("PD handoff absent")
            if not any(isinstance(x, dict) and x.get("ip") == "11.163.39.115"
                       and int(x.get("http_port", -1)) == 27200
                       for x in aux.get("role_addrs", [])):
                raise ValueError("Decode route to 112 absent")
            output_len = int(aux.get("output_len", 0))
            iter_count = int(aux.get("iter_count", 0))
            if output_len < 1 or iter_count < 1:
                raise ValueError("invalid output or Decode iteration count")
            case = next(case for case in result["cases"] if case["name"] == name)
            if case.get("require_mtp") and output_len - iter_count <= 0:
                raise ValueError("required MTP acceptance absent")
            if case.get("require_mtp_draft") and int(aux.get("speculative_draft_rounds", 0)) <= 0:
                raise ValueError("required MTP draft rounds absent")
            row.update(passed=True, kind=kind, answer=observed,
                       output_len=output_len, mtp_accepted=output_len - iter_count,
                       mtp_draft_rounds=int(aux.get("speculative_draft_rounds", 0)))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{name}: {row['error']}")
        rows.append(row)

    if seen != set(result_names):
        errors.append("request/result case-name mismatch")
    if len(rows) < 115:
        errors.append(f"only {len(rows)} formal cases; expected at least 115")
    report = {"passed": not errors, "checked": len(rows), "runner_cases": len(result_names),
              "skipped": len(skipped), "errors": errors, "cases": rows}
    (directory / "independent-final-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    summary = audit(parser.parse_args().directory)
    print(json.dumps({key: summary[key] for key in ("passed", "checked", "runner_cases", "skipped", "errors")}, ensure_ascii=False))
    raise SystemExit(0 if summary["passed"] else 1)
