"""Project HA request and Master-state evidence into the case-selected view."""

import json
import math
from collections import defaultdict
from pathlib import Path

from reporting import bundle_path, details, links, table, write_bundle
from reporting.catalog import PALETTE


COLORS = {
    "A": PALETTE[0], "B": PALETTE[4],
    "sent": PALETTE[9], "success": PALETTE[1], "failed": PALETTE[3],
}
STATE_COLORS = {
    "scheduler_inflight": (PALETTE[0], PALETTE[4]),
    "prefill_inflight_requests": (PALETTE[5], PALETTE[6]),
    "decode_master_queued": (PALETTE[7], PALETTE[2]),
    "decode_confirmed_running": (PALETTE[1], PALETTE[8]),
}
STATE_FIELDS = {
    "http_up": "HTTP 可回读",
    "scheduler_inflight": "调度在飞",
    "prefill_inflight_requests": "Prefill 在飞请求",
    "decode_master_queued": "Decode Master 排队",
    "decode_confirmed_running": "Decode 已确认运行",
}


def _artifacts(payload):
    finish = next((stage for stage in payload["stages"] if stage["id"] == "finish"), {})
    paths = {Path(path).name: Path(path) for path in finish.get("artifacts", [])}
    return paths.get("client_events.jsonl"), paths.get("master_states.jsonl")


def _read_rows(path):
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _request_series(rows, anchor):
    buckets = defaultdict(lambda: {"sent": 0, "success": 0, "failed": 0})
    for row in rows:
        timestamp = row.get("send_start_epoch_ms")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError("HA request lacks finite send timestamp")
        bucket = math.floor(timestamp / 1000)
        counts = buckets[bucket]
        counts["sent"] += 1
        counts["success" if row.get("status") == "ok" else "failed"] += 1
    if not buckets:
        return {key: [] for key in ("sent", "success", "failed")}
    return {
        key: [dict(x=second - anchor, y=buckets[second][key])
              for second in range(min(buckets), max(buckets) + 1)]
        for key in ("sent", "success", "failed")
    }


def _state_series(rows, anchor):
    samples = defaultdict(list)
    for row in rows:
        timestamp = row.get("epoch_s")
        name = row.get("master")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or name not in {"A", "B"}:
            raise ValueError("HA Master state row lacks timestamp or identity")
        for field in STATE_FIELDS:
            value = row.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("invalid HA Master state value")
            samples[(name, field)].append(dict(x=timestamp - anchor, y=value))
    return samples


def build_spec(payload, presentation, default_path):
    anchor = payload["clock_anchor"]["epoch_s"]
    request_path, state_path = _artifacts(payload)
    requests = _read_rows(request_path)
    states = _read_rows(state_path)
    qps = _request_series(requests, anchor)
    inflight = _state_series(states, anchor)
    events = [dict(t=phase["epoch_s"] - anchor, name=presentation["events"][phase["stage"]])
              for phase in payload["phases"]
              if phase["event"] == "end" and phase["stage"] in presentation["events"]]
    panels = []
    for descriptor in presentation["panels"]:
        curves = []
        for field in descriptor["fields"]:
            if field in qps:
                curves.append(dict(
                    name={"sent": "发出", "success": "成功", "failed": "失败"}[field],
                    group="请求", axis="qps", color=COLORS[field],
                    points=qps[field],
                    provenance=dict(kind="derived", source=str(request_path) if request_path else None,
                                    calculation="terminal outcome counted by send_start_epoch_ms second"),
                ))
            else:
                for master in ("A", "B"):
                    curves.append(dict(
                        name=f"{master} · {STATE_FIELDS[field]}", group=master,
                        axis="up" if field == "http_up" else "count",
                        color=(COLORS[master] if field == "http_up" else
                               STATE_COLORS[field][master == "B"]),
                        points=inflight.get((master, field), []),
                        provenance=dict(kind="sampled", source=str(state_path) if state_path else None,
                                        endpoint="/rtp_llm/inflight_status", field=field,
                                        aggregation="sum across endpoints" if field.startswith(("prefill_", "decode_")) else "direct"),
                    ))
        panels.append(dict(
            id=descriptor["id"], title=descriptor["title"], caption=descriptor["caption"],
            overlay=True, timeX=True, axes={
                "qps": {"title": "requests / s", "position": "left"},
                "count": {"title": "requests", "position": "left"},
                "up": {"title": "HTTP 状态 (0/1)", "position": "right", "min": 0, "max": 1},
            }, series=curves, events=events,
        ))
    sections = [
        table("门禁检查", ["阶段 / 检查", "状态", "实际值", "门槛"], [
            [row["stage"] + "/" + row["id"], row["status"], row.get("actual"), row.get("expected")]
            for row in payload["checks"]
        ]),
        details("数据来源与完整性", dict(
            request_events=str(request_path) if request_path else None,
            request_count=len(requests),
            master_states=str(state_path) if state_path else None,
            state_samples=len(states),
            monitoring=payload["workload"].get("telemetry_completeness"),
            telemetry_errors=payload["workload"].get("telemetry_integrity_errors", []),
        )),
        links("其他报告视角", [dict(label="全量监控", href="../" + default_path.parent.name + "/report.html")]),
    ]
    maximum = max(
        [point["t"] for point in events]
        + [point["x"] for values in qps.values() for point in values]
        + [point["x"] for values in inflight.values() for point in values]
        + [1]
    )
    return dict(
        run_id=payload["id"], title=presentation["title"], subtitle=presentation["subtitle"],
        timeOriginLabel="秒；t=0 为 workload 运行开始", timeAxis=dict(min=0, max=maximum),
        kpis=[dict(label="Execution", value=payload["status"]),
              dict(label="Validity", value=payload["workload"]["runtime_validity"]),
              dict(label="请求数", value=len(requests))],
        panels=panels, sections=sections,
    )


def write_report(directory, payload, presentation):
    default_path = bundle_path(directory, "run", payload["id"]) / "report.html"
    spec = build_spec(payload, presentation, default_path)
    bundle = write_bundle(
        directory, "run", payload["id"] + "-ha-core",
        {"id": payload["id"], "view": "master_ha_core.yaml", "status": payload["status"]},
        spec, producer="workload-ha",
    )
    return bundle / "report.html"
