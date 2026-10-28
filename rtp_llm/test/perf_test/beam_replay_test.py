"""Replay private Beam Search workloads through the normal perf-test server.

Run as a manual Bazel test. Request files and checkpoints are supplied at runtime;
no production prompts or model-specific paths belong in this target.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import resource
import statistics
import time
from pathlib import Path

import requests

from rtp_llm.test.perf_test.perf_utils import collect_timeline_files
from rtp_llm.test.perf_test.server import EngineServer


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def main():
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request_file", required=True)
    parser.add_argument("--concurrencies", default="1")
    parser.add_argument("--requests_per_point", type=int, default=32)
    parser.add_argument("--request_rate", type=float, default=0)
    parser.add_argument("--warmup_requests", type=int, default=3)
    parser.add_argument("--correctness_requests", type=int, default=3)
    parser.add_argument("--reference_path", default="")
    parser.add_argument(
        "--reference_mode", choices=["exact", "report"], default="exact"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_concurrency", type=int, default=1)
    parser.add_argument("--profile_steps", type=int, default=0)
    parser.add_argument("--restart_per_point", action="store_true")
    parser.add_argument("--diagnostic_aux_info", action="store_true")
    parser.add_argument("--diagnostic_repeats", type=int, default=0)
    parser.add_argument("--disable_core_dumps", action="store_true")
    parser.add_argument("--request_timeout", type=float, default=120)
    parser.add_argument("--server_start_timeout", type=int, default=1200)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--concurrency_limit", type=int, default=128)
    parser.add_argument(
        "--result_dir", default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    )
    args, remaining = parser.parse_known_args()
    if args.disable_core_dumps:
        _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0, hard_limit))
    args.partial = 0
    if not args.result_dir:
        parser.error("result_dir or TEST_UNDECLARED_OUTPUTS_DIR is required")
    points = [int(value) for value in args.concurrencies.split(",")]
    if min(points) < 1 or max(points) > args.concurrency_limit:
        parser.error("concurrencies must be within the server admission limit")
    if args.requests_per_point < max(points) or args.correctness_requests < 1:
        parser.error("provide enough requests to exercise each concurrency")
    if (
        args.request_rate < 0
        or not 1 <= args.profile_concurrency <= args.concurrency_limit
    ):
        parser.error("invalid request rate or profile concurrency")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.request_file).read_bytes()
    records = [json.loads(line) for line in source.splitlines() if line.strip()]
    if not records:
        raise ValueError("request file is empty")
    for record in records:
        config = record["generate_config"]
        beams = config.get("variable_num_beams", [])
        if not isinstance(record["prompt"], str) or not beams:
            raise ValueError("expected a text prompt and variable_num_beams")
        if config.get("max_new_tokens") != len(beams):
            raise ValueError("beam schedule must cover every generation step")

    references = (
        json.loads(Path(args.reference_path).read_text())
        if args.reference_path
        else None
    )
    EngineServer.propagate_engine_env(remaining)
    server = EngineServer(args, remaining)
    report = {
        "request_file_sha256": hashlib.sha256(source).hexdigest(),
        "request_count": len(records),
        "engine_args": remaining,
        "mode": (
            "paced_http_beam_replay"
            if args.request_rate
            else "closed_loop_http_beam_replay"
        ),
        "offered_request_rate": args.request_rate or None,
        "latency_boundary": (
            "scheduled_arrival_to_response"
            if args.request_rate
            else "http_request_to_response"
        ),
        "profiling_separate_from_timing": True,
        "restart_per_point": args.restart_per_point,
        "warmup_requests": args.warmup_requests,
        "correctness_requests": args.correctness_requests,
        "reference_mode": args.reference_mode,
        "reference_output_status": "NOT_COMPARED",
        "points": [],
        "correctness": [],
    }

    def send(index, profile=False, scheduled_at=None):
        record = records[index % len(records)]
        body = {
            "prompt": record["prompt"],
            "generate_config": dict(record["generate_config"]),
        }
        if args.diagnostic_aux_info:
            body["generate_config"]["aux_info"] = True
        if profile:
            body["generate_config"].update(
                gen_timeline=True,
                profile_step=args.profile_steps
                or len(body["generate_config"]["variable_num_beams"]),
            )
        started = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{server.port}",
            json=body,
            timeout=args.request_timeout,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            # Keep private error bodies out of shared stdout.
            error_path = result_dir / f"http_error_{index}.txt"
            error_path.write_text(response.text)
            raise AssertionError(f"HTTP {response.status_code}; details: {error_path}")
        output = response.json()
        texts = output.get("response")
        expected = record["generate_config"]["variable_num_beams"][-1]
        if output.get("finished") is not True:
            raise AssertionError("request did not return a terminal result")
        if not isinstance(texts, list) or len(texts) != expected:
            raise AssertionError(f"expected {expected} beam results")
        if not all(isinstance(text, str) and text for text in texts):
            raise AssertionError("empty or invalid beam result")
        timing = {
            "http_latency_ms": latency_ms,
            "client_queue_ms": (
                max(0.0, started - scheduled_at) * 1000
                if scheduled_at is not None
                else 0.0
            ),
        }
        return (
            latency_ms + timing["client_queue_ms"],
            texts,
            output.get("aux_info"),
            timing,
        )

    try:
        server.start(
            args.max_seq_len,
            args.concurrency_limit,
            server_start_timeout=args.server_start_timeout,
            use_batch_decode_scheduler=False,
        )
        for index in range(args.warmup_requests):
            send(index)
        for index in range(min(args.correctness_requests, len(records))):
            _, first, first_aux, _ = send(index)
            _, second, second_aux, _ = send(index)
            entry = {
                "index": index,
                "request_sha256": hashlib.sha256(
                    json.dumps(records[index], sort_keys=True).encode()
                ).hexdigest(),
                "responses": first,
                "repeat_equal": first == second,
                "first_aux_info": first_aux,
                "second_aux_info": second_aux,
            }
            report["correctness"].append(entry)
            if first != second:
                entry["repeat_responses"] = second
                entry["diagnostic_repeats"] = []
                for _ in range(args.diagnostic_repeats):
                    _, texts, aux, _ = send(index)
                    entry["diagnostic_repeats"].append(
                        {"responses": texts, "aux_info": aux}
                    )
                raise AssertionError(
                    f"beam outputs differ across repeats for input {index}; "
                    "check cache/execution state and the saved responses"
                )
            if references is not None:
                reference = references["correctness"][index]
                if reference["request_sha256"] != entry["request_sha256"]:
                    raise AssertionError("reference request identity mismatch")
                expected = reference["responses"]
                exact = expected == first
                entry["reference_comparison"] = {
                    "exact": exact,
                    "ordered_mismatch_count": sum(
                        a != b for a, b in zip(first, expected)
                    ),
                    "missing_count": len(set(expected) - set(first)),
                    "extra_count": len(set(first) - set(expected)),
                    "top1_equal": first[0] == expected[0],
                    "top10_overlap": len(set(first[:10]) & set(expected[:10])),
                    "top100_overlap": len(set(first[:100]) & set(expected[:100])),
                }
                if not exact:
                    report["reference_output_status"] = "DIFFERENT"
                elif report["reference_output_status"] != "DIFFERENT":
                    report["reference_output_status"] = "EXACT"
                if not exact and args.reference_mode == "exact":
                    raise AssertionError(
                        f"reference beam output mismatch for input {index}"
                    )
        for point_index, concurrency in enumerate(points):
            if point_index and args.restart_per_point:
                server.stop()
                server.start(
                    args.max_seq_len,
                    args.concurrency_limit,
                    server_start_timeout=args.server_start_timeout,
                    use_batch_decode_scheduler=False,
                )
                for index in range(args.warmup_requests):
                    send(index)
                for entry in report["correctness"]:
                    for _ in range(2):
                        _, texts, _, _ = send(entry["index"])
                        if texts != entry["responses"]:
                            raise AssertionError(
                                "beam output changed after server restart"
                            )
            latencies = []
            request_timings = []
            failures = []
            serial_reference = {
                entry["index"]: entry["responses"] for entry in report["correctness"]
            }
            started_unix = time.time()
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {}
                for i in range(args.requests_per_point):
                    scheduled_at = None
                    if args.request_rate:
                        scheduled_at = started + i / args.request_rate
                        delay = scheduled_at - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                    futures[pool.submit(send, i, scheduled_at=scheduled_at)] = i % len(
                        records
                    )
                for future in concurrent.futures.as_completed(futures):
                    try:
                        latency_ms, texts, _, timing = future.result()
                        latencies.append(latency_ms)
                        index = futures[future]
                        request_timings.append(
                            {
                                "request_index": index,
                                "latency_ms": latency_ms,
                                **timing,
                                "exact_to_serial_reference": (
                                    texts == serial_reference[index]
                                    if index in serial_reference
                                    else None
                                ),
                            }
                        )
                    except Exception as error:
                        failures.append(str(error))
            elapsed = time.perf_counter() - started
            point = {
                "concurrency": concurrency,
                "attempts": args.requests_per_point,
                "successes": len(latencies),
                "failures": failures,
                "wall_time_s": elapsed,
                "started_unix": started_unix,
                "ended_unix": time.time(),
                "latencies_ms": latencies,
                "request_timings": request_timings,
                "successful_requests_per_second": len(latencies) / elapsed,
                "p50_ms": percentile(latencies, 0.5) if latencies else None,
                "p90_ms": percentile(latencies, 0.9) if latencies else None,
                "p99_ms": percentile(latencies, 0.99) if latencies else None,
                "mean_ms": statistics.mean(latencies) if latencies else None,
            }
            report["points"].append(point)
            logging.info(
                "Beam replay metrics: %s",
                json.dumps(
                    {
                        key: value
                        for key, value in point.items()
                        if key not in ("latencies_ms", "request_timings")
                    }
                ),
            )
            if failures:
                raise AssertionError(
                    f"{len(failures)} failed requests at concurrency {concurrency}"
                )
        if args.profile:
            # Keep a separate loaded profile window; a single serial request
            # does not expose output-worker and scheduler stalls under load.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.profile_concurrency
            ) as pool:
                futures = [
                    pool.submit(send, i, profile=(i == 0))
                    for i in range(args.profile_concurrency * 3)
                ]
                for future in futures:
                    future.result()
            collect_timeline_files(str(result_dir))
            traces = sorted((result_dir / "timelines").glob("*.json"))
            report["profile_files"] = [path.name for path in traces]
            if not traces:
                raise AssertionError("profile request did not produce a timeline")
    finally:
        server.stop()
        (result_dir / "beam_replay_results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
