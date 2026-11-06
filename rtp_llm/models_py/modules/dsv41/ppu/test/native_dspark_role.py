"""A Bazel/GPU-locked native DSpark role using existing HTTP/CacheStore lifecycle.

L0 launches this target on P and D with matching immutable source/runtime;
after collecting both roles it creates each host's task-local control/stop.
"""

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from rtp_llm.models_py.modules.dsv41.ppu.test.native_ced_role import (
    healthy,
    stop_owned,
    wait_for,
    write_json,
)
from rtp_llm.tools.v41.launch import (
    DEFAULT_PREFILL_WAIT_TIMEOUT_MS,
    bounded_prefill_wait_timeout_ms,
)
from rtp_llm.tools.v41.native_smoke import make_query, run_request


def native_timeout_limits(env):
    def seconds(name, default, maximum):
        raw = env.get(name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite positive seconds") from error
        if (
            type(raw) is bool
            or not math.isfinite(value)
            or not 0.001 <= value <= maximum
        ):
            raise ValueError(f"{name} must be in [0.001, {maximum}] seconds")
        return value

    return dict(
        request_timeout_seconds=seconds(
            "RTP_V41_TEST_REQUEST_TIMEOUT_SECONDS", 3600, 86400
        ),
        prefill_wait_timeout_ms=bounded_prefill_wait_timeout_ms(
            env.get(
                "RTP_V41_TEST_PREFILL_WAIT_TIMEOUT_MS", DEFAULT_PREFILL_WAIT_TIMEOUT_MS
            )
        ),
        deadline_seconds=seconds("RTP_V41_TEST_DEADLINE_SECONDS", 12600, 172800),
    )


def remaining_request_timeout(request_seconds, deadline, *, now=None):
    now = time.monotonic() if now is None else now
    if (
        not all(math.isfinite(v) for v in (request_seconds, deadline, now))
        or request_seconds < 0.001
    ):
        raise ValueError("request timeout/deadline must be finite positive limits")
    remaining = min(request_seconds, deadline - now)
    if remaining < 0.001:
        raise TimeoutError(
            "global native deadline has no request millisecond remaining"
        )
    return remaining


def configure_contract_logging(env, suite, output, points, maximum, gamma):
    """Retain the finite contract run even if each model record fills a chunk."""
    if suite != "contracts":
        return None
    caps = [maximum] * (9 * len(points) + 18 + 1) + [64, 256]
    caps += [
        gamma + 2 if index % 3 else max(32, maximum * 4)
        for _ in range(2)
        for index in range(16)
    ]
    # Per PID, bound global rounds by summed output caps (one token minimum).
    # Count target/proposal/commit at three records per round, one extra round
    # and two prefill records per case, plus a terminal flush. This also bounds
    # an idle DP owner's alignment calls; batching can only reduce the count.
    # Python causal verify stages do not enqueue extra C++ logger records.
    record_bound = 3 * sum(caps) + 5 * len(caps) + 1
    backups = max(4096, 1 << (record_bound - 1).bit_length())
    # CP8 32K carries 4096 x 15360 BF16 target feature rows (120MiB).
    # The native logger keeps ordinary 64/256MiB defaults unless opted in.
    record_bytes, queue_bytes = 128 * 1024**2, 512 * 1024**2
    env.update(
        ENABLE_MODEL_INPUTS_LOG="1",
        LOG_FILE_BACKUP_COUNT=str(backups),
        RTP_LLM_MODEL_INPUTS_LOG_MAX_RECORD_BYTES=str(record_bytes),
        RTP_LLM_MODEL_INPUTS_LOG_MAX_QUEUE_BYTES=str(queue_bytes),
        DISABLE_ACCESS_LOG="0",
        RTP_V41_DSPARK_OBSERVE_DIR=str(output / "native-observations"),
    )
    return dict(
        cases=len(caps),
        output_token_cap=sum(caps),
        model_input_record_bound=record_bound,
        log_file_backup_count=backups,
        model_input_record_byte_limit=record_bytes,
        model_input_chunk_byte_limit=record_bytes,
        model_input_queue_byte_limit=queue_bytes,
        assumption="Existing eager DSpark, no warmup/retries; one record per chunk worst case. Per-record/queue byte limits and complete-trace gates remain mandatory.",
    )


def exercise_contract_cases(
    output,
    address,
    points,
    maximum,
    gamma,
    checkpoint,
    replay,
    process,
    deadline,
    parallel="cp",
    request_timeout_seconds=3600,
):
    """Existing HTTP API only; assertions use server observations after stop."""
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import (
        access_records,
    )

    checkpoint_config = json.loads((Path(checkpoint) / "config.json").read_text())
    eos = checkpoint_config.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("native EOS test requires actual checkpoint EOS IDs")
    records = []

    def issue(index, kind, length, count, **options):
        trace = f"v41-dspark-{output.name}-{kind}-{index}"
        cancel = options.pop("cancel_after_tokens", 0)
        expected_eos = options.pop("expected_eos", False)
        query = make_query(
            "hello " * (length - 1),
            count,
            trace_id=trace,
            timeout=request_timeout_seconds,
            reuse_cache=replay,
            stream=bool(cancel),
            **options,
        )
        if cancel:
            query["generate_config"]["ignore_eos"] = True
        host, port = address.rsplit(":", 1)
        p_owner = index % 8 if parallel == "dp" else 0
        timeout = remaining_request_timeout(request_timeout_seconds, deadline)
        query["generate_config"]["timeout_ms"] = int(timeout * 1000)
        record = run_request(
            f"http://{host}:{int(port) + p_owner * 9}",
            query,
            output / "contract-http",
            request=index,
            timeout=timeout,
            expected_input_tokens=length,
            eos_token_ids=eos if expected_eos else (),
            cancel_after_tokens=cancel,
        )
        record.update(
            case_kind=kind,
            query=query,
            expected_eos_token_ids=eos if expected_eos else [],
            prefill_http_owner=p_owner,
        )
        return record

    for length in points:
        for _ in range(9):
            records.append(issue(len(records), "greedy", length, maximum))
    for top_k, top_p, seed in ((32, 0.95, 17), (0, 0.8, 29)):
        for _ in range(9):
            records.append(
                issue(
                    len(records),
                    "random",
                    points[0],
                    maximum,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=0.8,
                    seed=seed,
                )
            )
    # Grammar constrains a finite phrase, then its ordinary EOS transition.
    # EOS IDs are expected values only: no per-request fake EOS override and
    # no stop_words path is used. A run without a real final EOS must fail.
    records.append(
        issue(
            len(records),
            "eos",
            points[0],
            64,
            min_new_tokens=0,
            regex="ok(?: ok){7}",
            expected_eos=True,
        )
    )
    cancelled = issue(len(records), "cancel", points[0], 256, cancel_after_tokens=2)
    records.append(cancelled)

    def terminal_access():
        return any(
            row["request"]["request_json"].get("generate_config", {}).get("trace_id")
            == cancelled["trace_id"]
            and row["response"].get("exception") is not None
            for row in access_records(output / "logs")
        )

    wait_for(
        terminal_access, process, deadline, "actual frontend disconnect terminal record"
    )
    records.append(issue(len(records), "recovery", points[0], maximum))
    for wave in range(2):
        gate = Barrier(16)
        first = len(records)

        def concurrent(index):
            gate.wait(timeout=120)
            length = points[index % len(points)]
            count = gamma + 2 if index % 3 else max(32, maximum * 4)
            return issue(
                first + index,
                f"concurrent-{wave}",
                length,
                count,
                seed=100 + wave * 16 + index,
            )

        with ThreadPoolExecutor(max_workers=16) as executor:
            records.extend(executor.map(concurrent, range(16)))
    write_json(
        output / "http-cases.json",
        dict(
            schema_version=1,
            cases=records,
            eos_token_ids=eos,
            checkpoint_config_sha256=hashlib.sha256(
                (Path(checkpoint) / "config.json").read_bytes()
            ).hexdigest(),
            coverage="HTTP results only; model/owner/first-anchor/vote/cache/cancel gates require actual server artifacts",
        ),
    )
    return records


def validate_contract_evidence(role, output, control, snapshot, deadline, text):
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import (
        access_records,
        assert_cancellation,
        assert_first_anchors,
        assert_native_handoff,
        assert_work_votes,
        read_model_inputs,
        read_observations,
        require,
    )

    headers, events, captures = read_observations(
        output / "native-observations", snapshot
    )
    model_inputs = read_model_inputs(output / "logs/model_inputs")
    require(
        {h["model"] for h in headers} == {"target", "draft"},
        "missing actual model observer",
    )
    if role == "prefill":
        require(
            {c["kind"] for c in captures} >= {"prefill", "commit"},
            "missing actual P publication byte captures",
        )
        result = dict(
            native_cache_captures=len(captures),
            model_input_records=len(model_inputs),
            full_decode_gates_require_peer=True,
            untouched_byte_comparisons=sum(
                c["byte_proof"]["untouched_byte_comparisons"] for c in captures
            ),
        )
    else:
        # L0 copies immutable task evidence only, after both owned servers exit.
        # These bytes are never supplied to a model or used as cache transport.
        while not (control / "peer-ready").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "L0 peer P evidence was not collected after native stop"
                )
            time.sleep(1)
        peer = control / "peer-evidence"
        cases = json.loads((peer / "http-cases.json").read_text())["cases"]
        _, _, p_captures = read_observations(peer / "native-observations", snapshot)
        require(
            {c["kind"] for c in captures} >= {"verify", "proposal", "commit"},
            "missing actual D cache lifecycle capture",
        )
        require(
            not re.search(
                r"(?:model.?inputs).*(?:dropped|queue.*full)", text, re.IGNORECASE
            ),
            "native model-input logger dropped evidence",
        )
        anchors = assert_first_anchors(model_inputs, events, cases)
        cancel = next(c for c in cases if c["case_kind"] == "cancel")
        recovery = next(c for c in cases if c["case_kind"] == "recovery")
        cancellation = assert_cancellation(
            cancel, recovery, access_records(peer / "logs"), text
        )
        result = dict(
            native_cache_captures=len(captures),
            model_input_records=len(model_inputs),
            work_votes=assert_work_votes(events),
            first_anchor_and_committed_input_chain=anchors,
            handoff=assert_native_handoff(p_captures, captures, events),
            cancellation=cancellation,
            eos_cases=[c["trace_id"] for c in cases if c.get("eos_observed")],
            evidence_transfer="Offline diagnostic artifacts after real P/D stop; never model inputs or cache transport",
            byte_proven_acceptance_classes={
                name: sum(
                    bool(
                        (count == 0)
                        if name == "zero"
                        else (
                            (0 < count < width)
                            if name == "partial"
                            else (count == width)
                        )
                    )
                    for capture in captures
                    if capture["kind"] == "verify"
                    for count, width in zip(
                        capture["accepted_input_rows"], capture["lengths"]
                    )
                )
                for name in ("zero", "partial", "all")
            },
            capture_limits=[h["limits"] for h in headers],
            untouched_byte_comparisons=sum(
                c["byte_proof"]["untouched_byte_comparisons"] for c in captures
            ),
            coverage="Only actually captured states/rows have byte evidence. Unobserved zero/partial/all states remain unqualified until covered by the combined S1 matrix.",
        )
        require(result["eos_cases"], "missing actual native EOS completion")
    write_json(output / "contract-evidence.json", result)
    return result


def main():
    env = dict(os.environ)
    # Reject invalid limits before source I/O or any model launch.
    limits = native_timeout_limits(env)
    suite = env.get("RTP_V41_TEST_DSPARK_SUITE", "smoke")
    if suite not in ("smoke", "contracts"):
        raise ValueError("RTP_V41_TEST_DSPARK_SUITE must be smoke or contracts")
    role = env["RTP_V41_DSPARK_ROLE"]
    if role not in ("prefill", "decode"):
        raise ValueError("RTP_V41_DSPARK_ROLE must be prefill or decode")
    source = Path(env["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(env["RTP_V41_TEST_RUNTIME"]).resolve()
    output = Path(env["RTP_V41_DSPARK_OUTPUT"]).resolve()
    control = Path(env["RTP_V41_DSPARK_CONTROL"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    runtime_snapshot = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != runtime_snapshot[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    manifest = json.loads((runtime / "native-runtime.json").read_text())
    if (
        manifest["source_sha256"] != snapshot["source_sha256"]
        or len(manifest["libraries"]) != 4
    ):
        raise ValueError(
            "DSpark needs all four libraries built from this immutable source"
        )
    for name, library in manifest["libraries"].items():
        path = runtime / library["runtime_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != library["sha256"]:
            raise ValueError(f"runtime library bytes changed: {name}")
    gamma = int(env.get("RTP_V41_TEST_DSPARK_GAMMA", "3"))
    points = [
        int(x) for x in env.get("RTP_V41_TEST_DSPARK_INPUTS", "129,257,1025").split(",")
    ]
    max_new = int(env.get("RTP_V41_TEST_DSPARK_OUTPUT_TOKENS", "8"))
    capacity = int(env.get("RTP_V41_TEST_MAX_SEQ_LEN", "65536"))
    if gamma not in (1, 3, 5) or min(points) < 2 or max_new < gamma + 2:
        raise ValueError(
            "require gamma1/3/5, positive prompt boundaries and more than one decode round"
        )
    if max(points) + max_new + gamma + 1 > capacity:
        raise ValueError(
            "input/output/verify headroom exceeds configured sequence capacity"
        )
    if (
        suite == "contracts"
        and max(points) + max(256, max_new * 4) + gamma + 1 > capacity
    ):
        raise ValueError(
            "contract cancellation/concurrency output headroom exceeds capacity"
        )
    confidence = env.get("RTP_V41_TEST_DSPARK_CONFIDENCE", "off")
    verify_mode = env.get("RTP_V41_TEST_DSPARK_VERIFY_MODE", "dense")
    budget = int(env.get("RTP_V41_TEST_DSPARK_BUDGET", "-1"))
    budget_mode = env.get("RTP_V41_TEST_DSPARK_BUDGET_MODE", "fixed")
    parallel = env.get("RTP_V41_TEST_PREFILL_PARALLEL", "cp")
    replay = env.get("RTP_V41_TEST_ENCODER_REPLAY", "0") == "1"
    ced = env.get("RTP_V41_TEST_CED_MODE", "bounded" if replay else "off")
    rdma = env.get("RTP_V41_TEST_RDMA", "0") == "1"
    checkpoint = env["RTP_V41_TEST_CHECKPOINT"]
    p_address = env["RTP_V41_DSPARK_PREFILL_ADDRESS"]
    d_address = env["RTP_V41_DSPARK_DECODE_ADDRESS"]
    address = p_address if role == "prefill" else d_address
    port = int(address.rsplit(":", 1)[1])
    env.update(
        PYTHONPATH=str(runtime) + ":" + env.get("PYTHONPATH", ""),
        LOG_PATH=str(output / "logs"),
        RTP_V41_PERF_EVIDENCE_DIR=str(output / "counters"),
    )
    model_input_retention = configure_contract_logging(
        env, suite, output, points, max_new, gamma
    )
    argv = [
        sys.executable,
        "-B",
        "-m",
        "rtp_llm.tools.v41.launch",
        role,
        "--checkpoint-path",
        checkpoint,
        "--prefill-address",
        p_address,
        "--decode-address",
        d_address,
        "--start-port",
        str(port),
        "--token-map-cache",
        str(output / "token-map.json"),
        "--max-seq-len",
        str(capacity),
        "--prefill-wait-timeout-ms",
        str(limits["prefill_wait_timeout_ms"]),
        "--prefill-parallel",
        parallel,
        "--ced-mode",
        ced,
        "--dspark-gamma",
        str(gamma),
        "--dspark-confidence",
        confidence,
        "--dspark-verify-mode",
        verify_mode,
        "--dspark-verify-budget",
        str(budget),
        "--dspark-budget-mode",
        budget_mode,
    ]
    if replay:
        argv.append("--encoder-swa-replay")
    if rdma:
        argv.append("--rdma")
    deadline = time.monotonic() + limits["deadline_seconds"]
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(
            argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        write_json(
            output / "launch.json",
            dict(
                pid=process.pid,
                argv=argv,
                cwd=str(runtime),
                role=role,
                snapshot_id=snapshot["snapshot_id"],
                gamma=gamma,
                confidence=confidence,
                verify_mode=verify_mode,
                budget=budget,
                budget_mode=budget_mode,
                capacity=capacity,
                points=points,
                suite=suite,
                model_input_retention=model_input_retention,
                timeouts=limits,
            ),
        )
        try:
            owner_count = 8 if role == "decode" or parallel == "dp" else 1
            local = [f"http://127.0.0.1:{port + row * 9}" for row in range(owner_count)]
            wait_for(
                lambda: all(map(healthy, local)),
                process,
                deadline,
                "all native local listeners ready",
            )
            write_json(output / "ready.json", {"role": role, "listeners": local})
            if role == "prefill":
                host, base = d_address.rsplit(":", 1)
                urls = [f"http://{host}:{int(base) + row * 9}" for row in range(8)]
                wait_for(
                    lambda: all(map(healthy, urls)),
                    process,
                    deadline,
                    "all native D owners ready",
                )
                if suite == "contracts":
                    exercise_contract_cases(
                        output,
                        p_address,
                        points,
                        max_new,
                        gamma,
                        checkpoint,
                        replay,
                        process,
                        deadline,
                        parallel,
                        request_timeout_seconds=limits["request_timeout_seconds"],
                    )
                for length in points if suite == "smoke" else []:
                    request_timeout = remaining_request_timeout(
                        limits["request_timeout_seconds"], deadline
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "rtp_llm.tools.v41.native_smoke",
                            "--url",
                            "http://" + p_address,
                            "--prompt",
                            "hello " * (length - 1),
                            "--expected-input-tokens",
                            str(length),
                            "--requests",
                            "9",
                            "--max-new-tokens",
                            str(max_new),
                            "--timeout",
                            str(request_timeout),
                            "--output",
                            str(output / f"http-{length}"),
                            *(["--reuse-cache"] if replay else []),
                        ],
                        cwd=runtime,
                        env=env,
                        check=True,
                        timeout=remaining_request_timeout(
                            limits["deadline_seconds"], deadline
                        ),
                    )
                write_json(
                    output / "requests-done.json",
                    dict(
                        points=points,
                        requests_per_point=9,
                        output_tokens=max_new,
                        transport="RDMA" if rdma else "TCP",
                        suite=suite,
                    ),
                )
            wait_for(
                lambda: (control / "stop").exists(),
                process,
                deadline,
                "L0 collected both roles",
            )
        finally:
            status = stop_owned(process, output)
            write_json(output / "server-exit.json", {"exit_code": status})
    if status != 0:
        raise RuntimeError(f"native DSpark role exit was {status}")
    logs = [output / "process.log", *(output / "logs").rglob("*.log")]
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    if (
        "layers=40" not in text
        or "PPU DSpark layers=3" not in text
        or "experts=128 topk=3" not in text
    ):
        raise AssertionError(
            "missing real40-layer target / three-layer128-expert draft evidence"
        )
    if role == "prefill":
        expected_cp = 8 if parallel == "cp" else 1
        if not re.search(
            r"capture_features=true ced_compaction=false.*layers=\(37, 38, 39\)", text
        ):
            raise AssertionError("missing complete target capture37/38/39 evidence")
        if "PPU CED compacted" in text:
            raise AssertionError("feature capture must not publish CED scatter holes")
        if not re.search(rf"PPU DSpark COMMIT .*cp=\d+/{expected_cp}", text):
            raise AssertionError("missing native draft commit publication path")
        evidence = {
            "capture_features": True,
            "ced_compaction": False,
            "draft_commit_only": True,
        }
    else:
        stages = {}
        for phase, pattern in (
            ("proposal", r"PPU DSpark PROPOSE dp=(\d+)[^\n]*fake=False"),
            ("verify", r"PPU target VERIFY dp=(\d+)"),
            ("accept", r"PPU target ACCEPT dp=(\d+)"),
        ):
            stages[phase] = sorted(set(map(int, re.findall(pattern, text))))
            if stages[phase] != list(range(8)):
                raise AssertionError(
                    f"missing {phase} on native owners: {stages[phase]}"
                )
        if "native PREFILL" in text:
            raise AssertionError(
                "D must restore native target/draft state without Prefill"
            )
        prefixes = [
            tuple(ast.literal_eval("(" + value + ")"))
            for value in re.findall(r"PPU target VERIFY .*prefixes=\(([^)]*)\)", text)
        ]
        if not all(any(point in row for row in prefixes) for point in points):
            raise AssertionError("missing verify anchor at the original prompt end")
        evidence = dict(
            stages=stages,
            prefill_executed=False,
            native_cache_writes="phase logs only; byte proof requires contracts suite",
        )
    if suite == "contracts":
        evidence["contracts"] = validate_contract_evidence(
            role, output, control, snapshot, deadline, text
        )
    write_json(
        output / "summary.json",
        dict(
            native_role_ok=True,
            role=role,
            snapshot_id=snapshot["snapshot_id"],
            transport="native RDMA" if rdma else "native TCP",
            gamma=gamma,
            verify_mode=verify_mode,
            budget=budget,
            budget_mode=budget_mode,
            evidence=evidence,
            numerical_alignment="deferred",
            nv_alignment="deferred",
        ),
    )


if __name__ == "__main__":
    main()
