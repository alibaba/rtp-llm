#!/usr/bin/env python3
"""Check recorded feat/k3_dev four-layer PD flow against raw HTTP responses."""

import argparse
import json
import tarfile
from pathlib import Path


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-tar", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with tarfile.open(args.prefill_tar, "r:gz") as archive:
        def read(name):
            return json.loads(archive.extractfile(name).read(), object_pairs_hook=no_duplicate_keys)

        runner = read("accuracy.json")
        members = sorted(m.name for m in archive.getmembers()
                         if m.name.startswith("requests/") and m.name.endswith(".json"))
        errors = []
        rows = []
        for name in members:
            record = read(name)
            row = {"case": record.get("name"), "file": name, "passed": False}
            try:
                if record.get("status") != 200 or record.get("passed") is not True:
                    raise ValueError("request did not pass with HTTP 200")
                wire = json.loads(record["response_body"], object_pairs_hook=no_duplicate_keys)
                aux = wire["aux_info"]
                observed = record["result"]
                text = wire["choices"][0]["message"].get("content", "") or ""
                text += wire["choices"][0]["message"].get("reasoning_content", "") or ""
                if not text or "\x00" in text:
                    raise ValueError("empty text or NUL byte")
                if aux.get("pd_sep") is not True or observed.get("pd_sep") is not True:
                    raise ValueError("PD marker absent")
                role_addrs = aux.get("role_addrs", [])
                if not any(addr.get("role") == "DECODE"
                           and addr.get("ip") == "11.163.39.112"
                           and addr.get("http_port") == 26400 for addr in role_addrs):
                    raise ValueError("Decode route differs")
                if observed.get("output_len") != 16 or wire["usage"]["completion_tokens"] != 16:
                    raise ValueError("output length differs")
                if not 0 < observed["elapsed_s"] <= 300:
                    raise ValueError("outside 300-second case deadline")
                row.update(passed=True, input_len=observed["input_len"],
                           elapsed_s=observed["elapsed_s"],
                           replacement_characters=text.count("\ufffd"))
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(f"{name}: {row['error']}")
            rows.append(row)
    runner_names = [case["name"] for case in runner.get("cases", [])]
    artifact_names = [row["case"] for row in rows]
    if (runner.get("passed") is not True or runner.get("failures")
            or len(rows) != 10 or len(set(artifact_names)) != 10
            or set(runner_names) != set(artifact_names)):
        errors.append("runner status or 10-case artifact set differs")
    result = {"passed": not errors, "checked": len(rows),
              "semantic_answer_claim": False,
              "mtp_execution_claim": False,
              "decode_handoff_length_claim": False,
              "replacement_characters_observed": sum(
                  row.get("replacement_characters", 0) for row in rows),
              "errors": errors, "cases": rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "checked", "errors")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
