"""Request workload and native evidence retention for DSpARK tests."""

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from rtp_llm.models_py.modules.dsv41.ppu.test.service_fixture import wait_for, write_json
from rtp_llm.tools.v41.native_smoke import make_query, run_request


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
