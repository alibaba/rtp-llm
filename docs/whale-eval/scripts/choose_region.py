#!/usr/bin/env python3
"""Rank regions with dashctl server previews. This script never submits a run."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess
import sys


DEFAULT_REGIONS = ("cn-zhongwei", "cn-wulanchabu", "cn-shanghai")
TEST_TYPES = {"func", "effective", "effective_small", "stab"}


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def summarize(preview, region, args):
    if not isinstance(preview, dict) or preview.get("capacity_checked") is not True:
        raise ValueError("server did not return a capacity check")
    subtasks = preview.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError("server did not return subtask previews")
    selected_tests = []
    groups = []
    pools = {}
    whale_gpu_keys = set()
    for child in subtasks:
        tests = child["test_types"]
        if not isinstance(tests, list) or not tests:
            raise ValueError("a child has no test types")
        selected_tests.extend(tests)
        coords = child["frozen_coordinates"]
        if (coords["baseline_path"], coords["baseline_commit"]) != (args.baseline_path, args.ref):
            raise ValueError("resolved baseline differs from the requested path or commit")
        refs = coords["resolved_refs"]
        suite = refs["ci-test"]["path"].removesuffix("/metadata.json")
        if suite != args.suite.removesuffix("/metadata.json"):
            raise ValueError("resolved CI-Test suite differs from the requested suite")
        if args.suite_commit and refs["ci-test"]["resolved_commit"] != args.suite_commit:
            raise ValueError("resolved CI-Test commit differs from the requested commit")
        if args.mint_path and refs["mint"]["path"] != args.mint_path:
            raise ValueError("resolved Mint baseline differs from the requested baseline")
        if args.mint_commit and refs["mint"]["resolved_commit"] != args.mint_commit:
            raise ValueError("resolved Mint commit differs from the requested commit")
        groups.append(tests)
        if child.get("capacity_checked") is not True or not child.get("resource_evaluation"):
            raise ValueError("a child has no resource evaluation")
        child_has_whale_gpu = False
        for row in child["resource_evaluation"]:
            pool, row_region, unit = row["pool"], row["region"], row["unit"]
            if not all(isinstance(v, str) and v for v in (pool, row_region, unit)):
                raise ValueError("a resource row has no pool, region, or unit")
            required, available = row["required"], row["available_at_evaluation"]
            if not nonnegative_integer(required) or not nonnegative_integer(available):
                raise ValueError(f"capacity is unknown for {pool}")
            gpu = row.get("gpu_type") or ""
            key = (row_region, pool, unit)
            if row["role"].startswith("whale") and unit == "gpu_cards" and required:
                if row_region != region:
                    raise ValueError("Whale GPU placement resolved to a different region")
                if not gpu:
                    raise ValueError(f"GPU type is unknown for {pool}")
                if args.gpu_type and gpu.casefold().removeprefix("nvidia ") != args.gpu_type:
                    raise ValueError(f"Whale GPU type {gpu} does not match requested type")
                whale_gpu_keys.add(key)
                child_has_whale_gpu = True
            current = pools.setdefault(key, {
                "region": row_region, "pool": pool, "unit": unit, "gpu_type": gpu,
                "required": 0, "available": available,
            })
            if current["gpu_type"] != gpu:
                raise ValueError(f"GPU type observations disagree for {pool}")
            current["required"] += required
            current["available"] = min(current["available"], available)
        if not child_has_whale_gpu:
            raise ValueError("a child has no positive Whale GPU demand")
    if sorted(selected_tests) != sorted(args.test.split(",")):
        raise ValueError("preview test types do not match the requested test types")
    for pool in pools.values():
        pool["spare"] = pool["available"] - pool["required"]
    return {
        "region": region,
        "groups": groups,
        "pools": list(pools.values()),
        "gpu_headroom": min(pools[key]["spare"] for key in whale_gpu_keys),
        "feasible": all(pool["spare"] >= 0 for pool in pools.values()),
    }


def preview_region(region, common, args):
    command = [*common, "--region", region]
    try:
        result = subprocess.run(
            [*command, "--dry-run=server", "-o", "json"],
            capture_output=True, text=True, check=False, timeout=args.timeout,
        )
        if result.returncode:
            # Backend diagnostics can contain raw configuration. Keep them out of the summary.
            return {"region": region, "feasible": False,
                    "error": f"dashctl preview exited {result.returncode}"}
        summary = summarize(json.loads(result.stdout), region, args)
        if summary["feasible"]:
            summary["submit_argv"] = command
        return summary
    except subprocess.TimeoutExpired:
        reason = "server preview timed out"
    except FileNotFoundError:
        reason = "dashctl is not on PATH"
    except json.JSONDecodeError:
        reason = "server response was not valid JSON"
    except (KeyError, TypeError, AttributeError):
        reason = "server response is missing required preview fields"
    except ValueError as exc:
        reason = str(exc)
    return {"region": region, "feasible": False, "error": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--ref", required=True, help="full immutable Model baseline commit")
    parser.add_argument("--suite", required=True, help="expected CI-Test path")
    parser.add_argument("--suite-commit", help="expected CI-Test commit SHA")
    parser.add_argument("--mint-path", help="expected Mint baseline path")
    parser.add_argument("--mint-commit", help="expected Mint baseline commit SHA")
    parser.add_argument("--region", action="append", help="allow-listed region, repeatable")
    parser.add_argument("--test", default="func,effective,stab")
    parser.add_argument("--replicas", action="append", default=[])
    parser.add_argument("--gpu-type", help="exact GPU type, e.g. L20D")
    parser.add_argument("--timeout", type=int, default=30, help="seconds per preview")
    args = parser.parse_args()

    if not args.baseline_path.startswith("model/") or not args.baseline_path.endswith(".json"):
        parser.error("--baseline-path must refer to a Model baseline JSON")
    for flag, commit in (("--ref", args.ref), ("--suite-commit", args.suite_commit),
                         ("--mint-commit", args.mint_commit)):
        if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
            parser.error(f"{flag} must be a full lowercase Git commit SHA")
    tests = args.test.split(",")
    if (set(tests) - TEST_TYPES or len(set(tests)) != len(tests)
            or {"effective", "effective_small"}.issubset(tests)):
        parser.error("--test must select distinct supported types")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.gpu_type:
        args.gpu_type = args.gpu_type.strip().casefold().removeprefix("nvidia ")
        if not args.gpu_type:
            parser.error("--gpu-type cannot be empty")

    regions = list(dict.fromkeys(args.region or DEFAULT_REGIONS))
    command = ["dashctl", "create", "cirun", "--baseline-path", args.baseline_path,
               "--ref", args.ref, "--test", args.test]
    for replica in args.replicas:
        command.extend(["--replicas", replica])

    with ThreadPoolExecutor(max_workers=min(3, len(regions))) as executor:
        candidates = list(executor.map(lambda region: preview_region(region, command, args), regions))
    feasible = [item for item in candidates if item["feasible"]]
    best = max(feasible, key=lambda item: item["gpu_headroom"], default=None)
    print(json.dumps({"selected": best and best["region"],
                      "submit_argv": best and best["submit_argv"],
                      "candidates": candidates}, ensure_ascii=False, indent=2))
    return 0 if best else 2


if __name__ == "__main__":
    sys.exit(main())
