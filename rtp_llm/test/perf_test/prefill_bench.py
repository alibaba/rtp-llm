"""Serial, single-token prefill measurements against an existing RTP-LLM server."""

import argparse
import hashlib
import http.client
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def load_cases(path):
    fixture = json.loads(Path(path).read_text())
    if not isinstance(fixture, dict):
        raise ValueError("Input fixture must be an object")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Input fixture requires a nonempty cases list")
    names = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        name = case.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Case names must be nonempty and unique")
        names.add(name)
        if type(case.get("input_len")) is not int or case["input_len"] <= 0:
            raise ValueError("input_len must be a positive integer")
        if not isinstance(case.get("prompt"), str) or not case["prompt"]:
            raise ValueError("Each case requires a nonempty prompt")
    return cases


def response_metrics(response, expected_input_len):
    if not isinstance(response, dict) or not isinstance(response.get("aux_info"), dict):
        raise ValueError("Response is missing aux_info")
    if response.get("finished", True) is not True:
        raise ValueError("Response is not complete")
    auxiliary = response["aux_info"]
    for name, expected in (("input_len", expected_input_len), ("output_len", 1)):
        if type(auxiliary.get(name)) is not int or auxiliary[name] != expected:
            raise ValueError(
                f"Unexpected {name}: {auxiliary.get(name)!r} != {expected}"
            )
    if auxiliary.get("reuse_len", 0) != 0:
        raise ValueError("Prefix reuse detected; disable reuse for full-prefill tests")
    timings = {}
    for name in ("first_token_cost_time", "wait_time", "cost_time"):
        value = auxiliary.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid timing: {name}={value!r}")
        timings[name] = float(value)
    if timings["first_token_cost_time"] < timings["wait_time"]:
        raise ValueError("First-token timing is smaller than queue time")
    if timings["cost_time"] < timings["first_token_cost_time"]:
        raise ValueError("Total timing is smaller than first-token timing")
    return {
        "prefill_ms": timings["first_token_cost_time"] - timings["wait_time"],
        "wait_ms": timings["wait_time"],
        "server_total_ms": timings["cost_time"] - timings["wait_time"],
    }


class PrefillClient:
    def __init__(self, endpoint, timeout=120):
        address = urlsplit(endpoint)
        if address.scheme not in ("http", "https") or not address.hostname:
            raise ValueError("Endpoint must be an HTTP(S) URL")
        if address.username or address.password or address.fragment:
            raise ValueError("Endpoint credentials and fragments are not supported")
        connection_type = (
            http.client.HTTPSConnection
            if address.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection_type(
            address.hostname, address.port, timeout=timeout
        )
        self.path = address.path or "/"
        if address.query:
            self.path += "?" + address.query

    def request(self, case):
        body = json.dumps(
            {
                "prompt": case["prompt"],
                "top_k": 1,
                "generate_config": {
                    "max_new_tokens": 1,
                    "min_new_tokens": 1,
                    "force_sp_accept": True,
                    "top_k": 1,
                },
            }
        ).encode()
        started = time.perf_counter_ns()
        self.connection.request(
            "POST", self.path, body, {"Content-Type": "application/json"}
        )
        response = self.connection.getresponse()
        content = response.read()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {content[:300]!r}")
        payload = json.loads(content)
        metrics = response_metrics(payload, case["input_len"])
        metrics["client_total_ms"] = elapsed_ms
        return metrics, payload

    def close(self):
        self.connection.close()


def describe(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize empty measurements")
    retained = ordered[1:-1] if len(ordered) > 2 else ordered
    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "sample_std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "trimmed_mean": statistics.mean(retained),
    }


def summarize(records):
    measured = [row for row in records if row["phase"] == "measure" and row["success"]]
    result = {}
    for name in sorted({row["case"] for row in measured}):
        selected = [row for row in measured if row["case"] == name]
        groups = {}
        for group in sorted({row["group"] for row in selected}):
            groups[str(group)] = describe(
                [row["prefill_ms"] for row in selected if row["group"] == group]
            )
        result[name] = {
            "input_len": selected[0]["input_len"],
            "prefill_ms": describe([row["prefill_ms"] for row in selected]),
            "client_total_ms": describe([row["client_total_ms"] for row in selected]),
            "wait_ms": describe([row["wait_ms"] for row in selected]),
            "groups": groups,
            "group_trimmed_means_ms": describe(
                [group["trimmed_mean"] for group in groups.values()]
            ),
        }
    return result


def run_benchmark(
    client, cases, output_dir, *, warmup=10, repeats=3, requests=100, metadata=None
):
    if warmup < 0 or repeats <= 0 or requests <= 0:
        raise ValueError(
            "warmup must be nonnegative; repeats and requests must be positive"
        )
    if not cases:
        raise ValueError("At least one input case is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    summary = {
        "status": "RUNNING",
        "started": datetime.now(timezone.utc).isoformat(),
        "warmup_per_case": warmup,
        "groups_per_case": repeats,
        "requests_per_group": requests,
        "independent_server_starts": 1,
        "metric": "aux_info.first_token_cost_time - aux_info.wait_time (ms)",
        "p95_method": "nearest rank",
        "metadata": metadata or {},
        "cases": [
            {
                "name": case["name"],
                "input_len": case["input_len"],
                "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            }
            for case in cases
        ],
    }
    try:
        with (destination / "requests.jsonl").open("x") as stream:
            for case in cases:
                schedule = [("warmup", 0, warmup)] + [
                    ("measure", group, requests) for group in range(1, repeats + 1)
                ]
                for phase, group, count in schedule:
                    for request_index in range(1, count + 1):
                        record = {
                            "case": case["name"],
                            "input_len": case["input_len"],
                            "phase": phase,
                            "group": group,
                            "request": request_index,
                            "unix_ns": time.time_ns(),
                            "success": False,
                        }
                        try:
                            metrics, payload = client.request(case)
                            record.update(metrics, success=True)
                            output = json.dumps(
                                payload.get("response"), sort_keys=True
                            ).encode()
                            record["output_sha256"] = hashlib.sha256(output).hexdigest()
                        except Exception as error:
                            record["error"] = repr(error)
                            raise
                        finally:
                            records.append(record)
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            stream.flush()
                    if phase == "measure":
                        metrics = summarize(records)[case["name"]]["groups"][str(group)]
                        print(
                            f"{case['name']} group {group}/{repeats}: "
                            f"mean={metrics['mean']:.3f} ms "
                            f"p95={metrics['p95']:.3f} ms",
                            flush=True,
                        )
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAILED"
        summary["error"] = repr(error)
        raise
    finally:
        summary["finished"] = datetime.now(timezone.utc).isoformat()
        summary["points"] = summarize(records)
        with (destination / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--inputs", required=True, help="JSON fixture with a cases list"
    )
    parser.add_argument("--output-dir", required=True, help="Must not already exist")
    parser.add_argument(
        "--metadata", help="Optional JSON describing the server configuration"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120)
    options = parser.parse_args()
    cases = load_cases(options.inputs)
    metadata = (
        json.loads(Path(options.metadata).read_text()) if options.metadata else {}
    )
    metadata["endpoint"] = options.endpoint
    client = PrefillClient(options.endpoint, options.timeout)
    try:
        run_benchmark(
            client,
            cases,
            options.output_dir,
            warmup=options.warmup,
            repeats=options.repeats,
            requests=options.requests,
            metadata=metadata,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
