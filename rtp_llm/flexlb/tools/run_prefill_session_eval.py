#!/usr/bin/env python3
"""Real Master / Java Mock exercise. Requires the pinned online_eval runner and a frozen candidate jar.

Reports client TTFT and success-cohort engine cache hits; it makes no GPU throughput claim.
"""

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
import time
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def expected_rejection(row, hard_wait_s):
    if row.get("code") != 8431:
        return False
    error = row.get("error", "")
    return "Prefill queue overload" in error or (
        error.endswith("trigger=PREFILL placement unavailable")
        and row["ended_s"] - row["started_s"] >= hard_wait_s - 0.05
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--mock-jar", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--load", type=float, default=0.8)
    parser.add_argument(
        "--session-fraction",
        type=float,
        default=1.0,
        help="Fraction of reusable conversations carrying a session ID; cold requests normally have none.",
    )
    parser.add_argument("--session-for-cold", action="store_true")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning", action="store_true")
    parser.add_argument(
        "--removal-mode", choices=("coordinated", "immediate"), default="coordinated"
    )
    args = parser.parse_args()
    if args.load <= 0 or not 0 <= args.session_fraction <= 1 or args.seconds <= 0:
        parser.error("invalid traffic parameters")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    import os

    os.environ["FLEXLB_FT_MASTER_JAR"] = str(args.jar.resolve())
    os.environ.pop("FLEXLB_FT_MASTER_CONFIG_FILE", None)
    sys.path[:0] = [str(args.runner), str(args.runner / "src")]
    from runtime import paths

    paths.MOCK_JAR = args.mock_jar.resolve()
    from adapters import proto_utils
    from adapters.engine_ops import EngineOps
    from adapters.environment import EnvManager
    from adapters.environment_config import EnvSpec
    from runtime.network import http_post_json

    # Compile the current wire contract, not the pinned runner's historical copy.
    proto_utils.PROTO_DIR = (
        Path(__file__).resolve().parents[3] / "rtp_llm/cpp/model_rpc/proto"
    )
    config = {
        "schemaVersion": 3,
        "router": {
            "roles": {
                "prefill": {
                    "tokenBalance": {},
                    "executionTimeEstimator": {
                        "type": "FORMULA",
                        "expression": "sum(computeTokens) / 70",
                    },
                }
            }
        },
        "scheduler": {
            "type": "QUEUE",
            "ordering": {"type": "FIFO"},
            "decision": {"type": "SINGLE"},
            "queueTimeoutMs": 6000,
            "prefillQueueShedding": {
                "expectedWaitThresholdMs": 3000,
                "maxWaitMs": 6000,
            },
        },
        "dispatcher": {"type": "NON_BATCH", "maxInflightPerPrefillWorker": 32},
        "requestLifecycle": {
            "request": {"timeoutMs": 30000},
            "decision": {"lifetime": 2.0},
        },
        "workerRegistry": {
            "health": {
                "statusPollIntervalMs": 20,
                "statusRpcTimeoutMs": 200,
                "statusStaleAfterMs": 500,
                "cleanupIntervalMs": 200,
            }
        },
    }
    # The pinned mock reads only execution/dispatch settings and predates token routing.
    # Keep its compatible envelope separate; configure_master records the full candidate config.
    mock_config = json.loads(json.dumps(config))
    del mock_config["router"]["roles"]["prefill"]["tokenBalance"]
    del mock_config["scheduler"]["prefillQueueShedding"]
    if args.learning:
        del config["router"]["roles"]["prefill"]["executionTimeEstimator"]
    spec = EnvSpec(
        label="session-elastic",
        n_prefill=3,
        n_decode=6,
        discovery="discovery_file",
        master_profile="single-nonbatch",
        raw_config=json.dumps(mock_config),
        master_jvm_heap="2g",
        mock_heap="2g",
        prefill_cache_blocks=100_000,
    )
    spec.perf["prefill_expression"] = "sum(computeTokens) / 70"
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    os.environ["FLEXLB_FT_MASTER_CONFIG_FILE"] = str(
        (args.out_dir / "config.json").resolve()
    )
    manifest = {
        "jar": str(args.jar.resolve()),
        "sha256": hashlib.sha256(args.jar.read_bytes()).hexdigest(),
        "mock_sha256": hashlib.sha256(args.mock_jar.read_bytes()).hexdigest(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "limitations": "Mock linear 70k compute TPS/worker; no GPU/model validation. Load is nominal; actual measured separately.",
    }
    driver = Path(__file__).read_bytes()
    (args.out_dir / "driver.py").write_bytes(driver)
    manifest["driver_sha256"] = hashlib.sha256(driver).hexdigest()
    manifest["schedule_proto_sha256"] = hashlib.sha256(
        (proto_utils.PROTO_DIR / "flexlb_schedule_service.proto").read_bytes()
    ).hexdigest()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manager = EnvManager(args.out_dir / "environment")
    results, stages, topology = [], [], []
    ops = None
    randomizer = random.Random(args.seed)
    rid = 30_000

    def wait_topology(expected):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status, data = http_post_json(env.master_http("/rtp_llm/master/info"), {})
            row = (data or {}).get("worker_summary", {}).get("PREFILL", {})
            if (
                status == 200
                and row.get("alive") == expected
                and row.get("discovered") == expected
            ):
                topology.append(data)
                return
            time.sleep(0.1)
        raise AssertionError(f"Master topology did not converge to {expected}: {data}")

    def query(stage, identity, session, length, keys, planned):
        started = time.monotonic()
        row = dict(
            stage=stage,
            request_id=identity,
            session=session,
            input_tokens=length,
            planned_s=planned,
            started_s=started,
            pacing_lag_s=started - planned,
            success=False,
        )
        try:
            request = ops.build_schedule_request(
                identity, input_len=length, output_len=2, block_keys=keys
            )
            request.session_id = session
            stub = ops.schedule_pb2_grpc.FlexlbServiceStub(
                ops._channel(ops.master_target())
            )
            response = stub.Schedule(request, timeout=15)
            row.update(code=response.code, worker=ops.prefill_addr(response))
            if response.success and response.code == 200:
                stream = ops.start_stream(
                    response,
                    identity,
                    ops.build_generate_input(
                        identity, input_len=length, output_len=2, block_keys=keys
                    ),
                )
                ended = stream.wait_end(20)
                row.update(
                    success=ended
                    and stream.snap.completed
                    and not stream.snap.error
                    and not stream.snap.stream_error_code,
                    error=stream.snap.error,
                    stream_error_code=stream.snap.stream_error_code,
                )
                if stream.snap.first_received_s is not None:
                    row["ttft_s"] = stream.snap.first_received_s - started
            else:
                row["error"] = response.error_message
        except Exception as error:
            row["error"] = repr(error)
        row["ended_s"] = time.monotonic()
        return row

    def shape(index, cold=False):
        nonlocal rid
        rid += 1
        length = 65536 if index % 5 == 0 else 4096
        base = (rid + 100000 if cold else index + 1) * 1000
        keys = [base + block for block in range(length // 1024 - 1)] + [rid * 100000]
        session = (
            f"conversation-{index}"
            if not cold and index % 100 < args.session_fraction * 100
            else ""
        )
        if cold and args.session_for_cold:
            session = f"cold-conversation-{rid}"
        return rid, session, length, keys

    def remove_worker(worker):
        if args.removal_mode == "coordinated":
            # Planned scale-in must withdraw from discovery before stopping RPC admission.
            # Keep traffic flowing while Master observes withdrawal and existing engine work drains.
            discovery = json.loads(env.discovery_file.read_text())
            address = worker["http_addr"]
            assert any(address in addresses for addresses in discovery.values())
            for domain, addresses in discovery.items():
                discovery[domain] = [value for value in addresses if value != address]
            temporary = env.discovery_file.with_suffix(".withdraw.tmp")
            temporary.write_text(json.dumps(discovery))
            temporary.replace(env.discovery_file)
            wait_topology(3)
        return ops.remove_engine(
            worker["name"], mode="graceful", drain_timeout_ms=10000
        )

    try:
        env = manager.ensure(spec)
        ops = EngineOps(
            "127.0.0.1",
            env.master_http_port,
            env.mock_http_port,
            env.master_management_port,
        )
        wait_topology(3)
        # Repeat a conversation with entirely cold keys: only session routing can retain its owner.
        identity, _, length, keys = shape(1)
        first = query("affinity", identity, "probe", length, keys, time.monotonic())
        identity, _, length, keys = shape(1, True)
        second = query("affinity", identity, "probe", length, keys, time.monotonic())
        results.extend([first, second])
        assert (
            first["success"]
            and second["success"]
            and first["worker"] == second["worker"]
        ), (first, second)
        for index in range(100):
            row = query("warmup", *shape(index), time.monotonic())
            results.append(row)
            assert row["success"], row

        # 25% cold arrivals, 75% warm arrivals: nominal uncached tokens = .25*mean_input + .75*1024.
        qps = args.load * 3 * 70_000 / (0.25 * (0.2 * 65536 + 0.8 * 4096) + 0.75 * 1024)
        with concurrent.futures.ThreadPoolExecutor(max_workers=256) as pool:
            for stage in ("steady", "scale_out", "scale_in", "burst", "recovery"):
                resize = None
                if stage == "scale_out":
                    code, body = ops.add_engine("prefill")
                    assert code == 200, body
                    topology.append(body)
                    wait_topology(4)
                if stage == "scale_in":
                    worker = next(
                        e for e in ops.snapshot()["engines"] if e["role"] == "prefill"
                    )
                    resize = pool.submit(remove_worker, worker)
                start = time.monotonic()
                futures = []
                if stage == "burst":
                    for index in range(64):
                        futures.append(
                            pool.submit(query, stage, *shape(index, True), start)
                        )
                for index in range(max(1, int(qps * args.seconds))):
                    planned = start + index / qps
                    time.sleep(max(0, planned - time.monotonic()))
                    shape_args = shape(
                        randomizer.randrange(100), randomizer.random() < 0.25
                    )
                    futures.append(pool.submit(query, stage, *shape_args, planned))
                issued_end = time.monotonic()
                rows = [future.result() for future in futures]
                results.extend(rows)
                stages.append(
                    dict(
                        stage=stage,
                        start_s=start,
                        issued_end_s=issued_end,
                        drained_s=time.monotonic(),
                        qps=qps,
                    )
                )
                if resize is not None:
                    code, body = resize.result()
                    assert code == 200 and body.get("drained") is True, body
                    topology.append(body)
                    wait_topology(3)
                (args.out_dir / "requests.json").write_text(
                    json.dumps(results, indent=2)
                )
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "arrivals": len(rows),
                            "success": sum(r["success"] for r in rows),
                        }
                    ),
                    flush=True,
                )
    finally:
        try:
            if ops is not None:
                try:
                    (args.out_dir / "final-snapshot.json").write_text(
                        json.dumps(ops.snapshot(), indent=2)
                    )
                finally:
                    ops.close()
        finally:
            manager.teardown()
            (args.out_dir / "requests.json").write_text(json.dumps(results, indent=2))
            (args.out_dir / "topology.json").write_text(json.dumps(topology, indent=2))

    events = {}
    for path in (args.out_dir / "environment").rglob("engine_events.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "prefill_done" and not event.get("cancelled"):
                assert (
                    event["rid"] not in events
                ), "duplicate successful Prefill terminal"
                events[event["rid"]] = event
    report = []
    for stage in stages:
        rows = [row for row in results if row["stage"] == stage["stage"]]
        success = [row for row in rows if row["success"]]
        assert success, "zero successful requests cannot establish throughput"
        assert all(
            row["request_id"] in events for row in success
        ), "missing engine evidence"
        input_tokens = sum(row["input_tokens"] for row in success)
        hit_tokens = sum(
            events[row["request_id"]]["cache_hit_tokens"] for row in success
        )
        assert 0 <= hit_tokens <= input_tokens
        assert all(
            events[row["request_id"]]["input_len"] == row["input_tokens"]
            for row in success
        )
        elapsed = max(args.seconds, stage["drained_s"] - stage["start_s"])
        report.append(
            {
                **stage,
                "arrivals": len(rows),
                "completed": len(success),
                "errors": len(rows) - len(success),
                "offered_session_fraction": sum(bool(r["session"]) for r in rows)
                / len(rows),
                "input_tpm_including_drain": input_tokens * 60 / elapsed,
                "compute_tpm_including_drain": (input_tokens - hit_tokens)
                * 60
                / elapsed,
                "success_token_hit_rate": hit_tokens / input_tokens,
                "ttft_p50_s": percentile([r["ttft_s"] for r in success], 0.5),
                "ttft_p95_s": percentile([r["ttft_s"] for r in success], 0.95),
                "ttft_p99_s": percentile([r["ttft_s"] for r in success], 0.99),
                "max_pacing_lag_s": max(r["pacing_lag_s"] for r in rows),
            }
        )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    assert len({row["request_id"] for row in results}) == len(results)
    assert not any(
        not row["success"] and row.get("code") == 200 for row in results
    ), "accepted request failed in the engine/transport"
    assert all(
        row["success"]
        or expected_rejection(row, config["scheduler"]["queueTimeoutMs"] / 1000)
        for row in results
    ), "unexpected scheduling or transport failure"
    assert all(
        row["max_pacing_lag_s"] < 0.25 for row in report
    ), "load generator fell behind; nominal offered load is not established"
    if args.load <= 0.5:
        assert all(
            row["errors"] == 0
            for row in report
            if row["stage"] in {"steady", "scale_out", "scale_in", "recovery"}
        ), report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
