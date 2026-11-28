import asyncio
import json
from types import SimpleNamespace

import pytest
from mock_framework.config import ConfigError
from mock_framework.execution import ExecutionError
from mock_framework.execution.step.master import (
    MasterPvRequestsArgs,
    master_pv_requests,
)
from mock_framework.execution.step.step_registry import StepRegistry


def pv_context(tmp_path):
    events = []
    return SimpleNamespace(
        records=SimpleNamespace(directory=tmp_path, emit=lambda **v: events.append(v)),
        events=events,
    )


def write_pv_log(relative_path, records, tmp_path):
    path = tmp_path / "components" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"2026-11-24 19:00:00.000 [grpc-1] INFO  pvLogger - {json.dumps(record)}"
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_frontend_journal(component, round_name, events, tmp_path):
    path = (
        tmp_path / "components" / component / "playback" / round_name / "requests.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"version": 1, "frontend": component}] + events
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return path


def test_master_pv_requests_filters_requeue_origin_success_and_restarted_logs(tmp_path):
    write_pv_log(
        "master-0/logs/pv.log",
        [
            {
                "requestId": "101",
                "scheduleOrigin": "LOCAL_MASTER",
                "success": True,
                "code": 200,
                "decodeRequeueCount": 1,
            },
            {
                "requestId": "102",
                "scheduleOrigin": "LOCAL_MASTER",
                "success": True,
                "code": 200,
            },
            {
                "requestId": "103",
                "scheduleOrigin": "LOCAL_MASTER",
                "success": False,
                "code": 8431,
                "decodeRequeueCount": 2,
            },
        ],
        tmp_path,
    )
    write_pv_log(
        "master-0/restart-1/logs/pv.log",
        [
            {
                "requestId": "104",
                "scheduleOrigin": "LOCAL_FALLBACK",
                "success": True,
                "code": 200,
            }
        ],
        tmp_path,
    )
    ctx = pv_context(tmp_path)
    assert asyncio.run(master_pv_requests(ctx, MasterPvRequestsArgs())) == 4.0
    assert (
        asyncio.run(
            master_pv_requests(
                ctx,
                MasterPvRequestsArgs(decode_requeue_count_gte=1, success=True),
            )
        )
        == 1.0
    )
    assert ctx.events[-1]["sample_request_ids"] == ["101"]
    assert (
        asyncio.run(
            master_pv_requests(
                ctx,
                MasterPvRequestsArgs(schedule_origin="LOCAL_FALLBACK", code=200),
            )
        )
        == 1.0
    )
    assert (
        asyncio.run(
            master_pv_requests(
                ctx,
                MasterPvRequestsArgs(
                    decode_requeue_count_gte=2, success=False, code=8431
                ),
            )
        )
        == 1.0
    )


def test_master_pv_requests_cross_checks_frontend_terminal_outcome(tmp_path):
    write_pv_log(
        "master-0/logs/pv.log",
        [
            {
                "requestId": "201",
                "scheduleOrigin": "LOCAL_MASTER",
                "success": True,
                "code": 200,
                "decodeRequeueCount": 1,
            },
            {
                "requestId": "202",
                "scheduleOrigin": "LOCAL_MASTER",
                "success": True,
                "code": 200,
                "decodeRequeueCount": 1,
            },
            {
                "requestId": "203",
                "scheduleOrigin": "LOCAL_MASTER",
                "success": True,
                "code": 200,
                "decodeRequeueCount": 1,
            },
        ],
        tmp_path,
    )
    write_frontend_journal(
        "frontend-0",
        "1",
        [
            {"kind": "TERMINAL", "request_id": "201", "outcome": "success"},
            {"kind": "TERMINAL", "request_id": "202", "outcome": "timeout"},
            {"kind": "TERMINAL", "request_id": "203", "outcome": "success"},
            {"kind": "TERMINAL", "request_id": "203", "outcome": "error"},
        ],
        tmp_path,
    )
    ctx = pv_context(tmp_path)
    matched = asyncio.run(
        master_pv_requests(
            ctx,
            MasterPvRequestsArgs(
                decode_requeue_count_gte=1,
                success=True,
                require_frontend_outcome="success",
            ),
        )
    )
    assert matched == 1.0
    assert ctx.events[-1]["sample_request_ids"] == ["201"]


def test_master_pv_requests_requires_recorded_logs(tmp_path):
    (tmp_path / "components" / "master-0").mkdir(parents=True)
    ctx = pv_context(tmp_path)
    with pytest.raises(ExecutionError, match="pv.log"):
        asyncio.run(master_pv_requests(ctx, MasterPvRequestsArgs()))


@pytest.mark.parametrize(
    "values",
    [
        {"schedule_origin": " "},
        {"decode_requeue_count_gte": -1},
        {"require_frontend_outcome": "unknown"},
    ],
)
def test_master_pv_requests_args_are_validated(values):
    with pytest.raises((ValueError, ConfigError)):
        StepRegistry.standard().get("master/query/pv_requests").build_arguments(values)
