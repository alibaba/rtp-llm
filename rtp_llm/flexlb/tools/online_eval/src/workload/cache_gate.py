"""Offline cache-collapse adjudication from identity-preserving counter samples.

No HTTP, wall-clock waits or implicit threshold tuning. The same JSON supplies
both CI checks and the existing Chart.js renderer. Cache validity uses issued
load and completed Prefills; queue pressure and request failures are diagnostics.
"""

import argparse
import hashlib
from bisect import bisect_right
import json
import math
from pathlib import Path

from reporting.catalog import CACHE_METRICS, PALETTE
from reporting.view_config import view
from reporting import (
    details,
    table,
    run_meta,
    write_bundle,
)
from reporting.statistics import select_window, counter_delta

COUNTERS = (
    "hit_tokens_total",
    "context_tokens_total",
    "context_requests_total",
)


def align_send_counters(evidence, issued):
    """Use actual send times, not the time a buffered lifecycle journal was read.

    The caller must prove that every submitted send has an issued record.
    Terminal outcomes are independent of the offered-load measurement.
    """
    times = [row.get("send_start_epoch_ms") for row in issued]
    if not times or any(
        type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times
    ):
        raise ValueError("complete finite issued timestamps are required")
    times.sort()
    for row in evidence["samples"]:
        epoch = row.get("epoch_s")
        if type(epoch) not in (int, float) or not math.isfinite(epoch):
            raise ValueError("finite sample epoch is required")
        row.setdefault("journal_observed_started", row["started"])
        row["started"] = bisect_right(times, epoch * 1000)
    evidence["send_counter_alignment"] = dict(
        method="issued send_start_epoch_ms at each sample epoch",
        issued_count=len(times),
        first_send_epoch_ms=times[0],
        last_send_epoch_ms=times[-1],
    )


MEASUREMENT_POLICY = "survivor-cache-with-attributed-client-load"
DRAIN_DIAGNOSTICS = {
    "graceful drain timed out; removal introduced request loss",
    "intermediate graceful drain timed out; continuing observation",
}


def topology_ready(row, names, initial):
    """Serving population, not the set of processes still draining."""
    engines, names, initial = row["engines"], set(names), set(initial)
    return (row.get("master_p") == len(names)
            and names <= set(engines) <= initial
            and all(engines[n].get("admission_open") == 1 for n in names)
            and all(engines[n].get("admission_open") == 0 for n in set(engines) - names))


def attribute_client(evidence, snapshot):
    """Freeze send/terminal cohorts using the completed client journal.

    A missing route is unknown, never inferred from an error string or timing.
    Client load remains global; per-cohort counters expose victim contributions.
    """
    addresses = {}
    for row in evidence["samples"]:
        for name, engine in row["engines"].items():
            address = engine.get("grpc_addr")
            if address:
                addresses.setdefault(address, set()).add(name)
    cohorts = {group: {kind: [] for kind in ("sent", "terminal", "failed")}
               for group in ("survivor", "removed", "unknown")}
    failures, invalid, before_removal = [], [], []
    withdrawal = min((event["epoch_s"] * 1000 for event in evidence.get("events", [])
                      if event.get("name") in {"withdraw_start", "intermediate_withdraw_start"}),
                     default=None)
    closed_at = {r.get("engine"): r.get("admission", {}).get("admission_closed_epoch_ms")
                 for r in evidence.get("removals", [])}
    for record in snapshot.get("records", []):
        names = addresses.get(record.get("prefill"), set())
        name = next(iter(names)) if len(names) == 1 else None
        group = ("survivor" if name in evidence["survivors"] else
                 "removed" if name in evidence["initial_engines"] else "unknown")
        start, duration = record.get("send_start_epoch_ms"), record.get("total_ms")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (start, duration)):
            invalid.append(record.get("rid"))
            continue
        cohorts[group]["sent"].append(start)
        cohorts[group]["terminal"].append(start + duration)
        if record.get("status") != "ok":
            cohorts[group]["failed"].append(start + duration)
            cutoff = closed_at.get(name) or withdrawal
            if group == "removed" and (cutoff is None or start + duration < cutoff):
                before_removal.append(record.get("rid"))
            failures.append({key: record.get(key) for key in
                             ("rid", "request_id", "prefill", "status", "error",
                              "send_start_epoch_ms", "total_ms")} | {"cohort": group})
    for counts in cohorts.values():
        for times in counts.values():
            times.sort()
    for row in evidence["samples"]:
        row["client_cohorts"] = {group: {kind: bisect_right(times, row["epoch_s"] * 1000)
                                        for kind, times in counts.items()}
                                 for group, counts in cohorts.items()}
        row.setdefault("journal_observed_terminal", row["terminal"])
        row["terminal"] = sum(c["terminal"] for c in row["client_cohorts"].values())
    evidence["client_attribution"] = dict(
        complete=bool(snapshot.get("complete")) and not invalid,
        source="completed client journal; prefill matched to sampled grpc_addr",
        accounting=snapshot.get("status"), issued_count=len(snapshot.get("issued", [])),
        terminal_count=len(snapshot.get("records", [])),
        terminal_time="send_start_epoch_ms + total_ms",
        counts={group: {kind: len(times) for kind, times in counts.items()}
                for group, counts in cohorts.items()},
        failures=failures, invalid_timestamps=invalid,
        removed_failures_without_intervention=before_removal,
    )


def scope_contract(evidence):
    return dict(
        policy=MEASUREMENT_POLICY,
        baseline_engines=evidence["initial_engines"], post_engines=evidence["survivors"],
        baseline_scope="full initial population for predeclared warmup and threshold; survivor-filtered fields below apply post-removal",
        transition="withdraw_start through post_start excluded from cache adjudication",
        post_start="first sample with survivor admission open, victims closed/absent and Master count converged",
        hit=dict(scope="survivor-filtered", counters=["hit_tokens_total", "context_tokens_total"],
                 calculation="sum counter deltas / sum context token deltas"),
        completed=dict(scope="survivor-filtered", counters=["context_requests_total"]),
        waiting=dict(scope="survivor-filtered", counters=["waiting"], calculation="max sample sum"),
        running=dict(scope="survivor-filtered", counters=["running"], calculation="max sample sum; diagnostic"),
        sent_qps=dict(scope="aggregate-with-attribution", source="issued send_start_epoch_ms",
                      calculation="global offered load; cohort deltas shown separately, target unchanged"),
        terminal_qps=dict(scope="aggregate-with-attribution", source="client terminal journal",
                          calculation="global terminal delta; diagnostic, not successful prefill QPS"),
        client_pacing=dict(scope="aggregate-with-attribution", source="issued pacing_lag_ms", calculation="global maximum"),
        accounting="issued sends establish offered load; terminal outcomes and request failures are diagnostic only",
        drain="diagnostic only; no verdict contribution",
    )


def window(rows, start, end, names, max_gap_s):
    selected = select_window(rows, start, end, time=lambda r: r["t"], include_end=True)
    result = dict(
        start=start,
        end=end,
        hit=None,
        completed=0,
        sent_qps=None,
        terminal_qps=None,
        waiting=None,
        running=None,
        evictions=None,
        forward_ms=None,
        errors=[],
        diagnostic_gaps=[],
    )
    if (
        len(selected) < 2
        or selected[0]["t"] - start > max_gap_s
        or end - selected[-1]["t"] > max_gap_s
    ):
        result["errors"].append("incomplete window coverage")
        return result
    elapsed = selected[-1]["t"] - selected[0]["t"]
    if elapsed <= 0 or any(
        b["t"] - a["t"] > max_gap_s for a, b in zip(selected, selected[1:])
    ):
        result["errors"].append("sampling gap or non-increasing clock")
        return result
    result["source"] = dict(engines=list(names), first_sample_t=selected[0]["t"],
                            last_sample_t=selected[-1]["t"], elapsed_s=elapsed,
                            counters=list(COUNTERS), population="selected engines")
    if all("client_cohorts" in r for r in selected):
        result["client_cohorts"] = {
            group: {kind + "_qps": (selected[-1]["client_cohorts"][group][kind]
                                    - selected[0]["client_cohorts"][group][kind]) / elapsed
                    for kind in ("sent", "terminal", "failed")}
            for group in ("survivor", "removed", "unknown")}
    totals = dict.fromkeys(COUNTERS, 0)
    evictions = 0
    for name in names:
        values = [r["engines"].get(name) for r in selected]
        if (
            any(v is None for v in values)
            or len({(v.get("grpc_addr"), v.get("engine_incarnation")) for v in values})
            != 1
        ):
            result["errors"].append("missing or changed engine " + name)
            continue
        for field in COUNTERS:
            counter = [v.get(field) for v in values]
            delta, state = counter_delta(counter)
            if state == "MISSING_COUNTER":
                result["errors"].append("missing counter " + name + "/" + field)
            elif state == "COUNTER_RESET":
                result["errors"].append("counter reset " + name + "/" + field)
            else:
                totals[field] += delta
        counter = [v.get("cache_evictions") for v in values]
        delta, state = counter_delta(counter)
        if state != "AVAILABLE":
            result["diagnostic_gaps"].append(name + "/cache_evictions: " + state)
            evictions = None
        elif evictions is not None:
            evictions += delta
    counts = [r["started"] for r in selected]
    if any(b < a for a, b in zip(counts, counts[1:])):
        result["errors"].append("issued send counter reset")
    else:
        result["sent_qps"] = (counts[-1] - counts[0]) / elapsed
    counts = [r.get("terminal") for r in selected]
    if all(type(v) in (int, float) for v in counts) and all(
        b >= a for a, b in zip(counts, counts[1:])
    ):
        result["terminal_qps"] = (counts[-1] - counts[0]) / elapsed
    else:
        result["diagnostic_gaps"].append("terminal counter missing or reset")
    if not result["errors"]:
        tokens = totals["context_tokens_total"]
        hits = totals["hit_tokens_total"]
        if hits > tokens:
            result["errors"].append("hit tokens exceed context tokens")
        else:
            result["hit"] = hits / tokens if tokens else None
        result.update(
            completed=totals["context_requests_total"],
            context_tokens=tokens,
            evictions=evictions,
        )
        for field in ("waiting", "running"):
            if all(all(type(r["engines"][n].get(field)) in (int, float)
                       for n in names) for r in selected):
                result[field] = max(
                    sum(r["engines"][n][field] for n in names) for r in selected
                )
            else:
                result["diagnostic_gaps"].append(field + " missing")
        if all(type(selected[-1]["engines"][n].get("prefill_ms_avg")) in (int, float)
               for n in names):
            result["forward_ms"] = sum(
                selected[-1]["engines"][n]["prefill_ms_avg"] for n in names
            ) / len(names)
        else:
            result["diagnostic_gaps"].append("prefill_ms_avg missing")
    return result


def analyze(evidence):
    p, rows = evidence["criteria"], evidence["samples"]
    errors = [e for e in evidence.get("errors", []) if e not in DRAIN_DIAGNOSTICS]
    scoped = evidence.get("measurement_policy") == MEASUREMENT_POLICY
    if any(b["t"] <= a["t"] for a, b in zip(rows, rows[1:])):
        errors.append("sample clock must increase")
    baseline = window(
        rows,
        evidence["baseline_start"],
        evidence["baseline_end"],
        evidence["initial_engines"],
        p["max_gap_s"],
    )
    errors.extend(baseline["errors"])
    if baseline["hit"] is None or baseline["hit"] < p["baseline_min_hit"]:
        errors.append("baseline not warm")
    if baseline["completed"] < p["min_completed"]:
        errors.append("insufficient baseline prefill completions")
    if (
        baseline["sent_qps"] is None
        or abs(baseline["sent_qps"] / p["qps"] - 1) > p["qps_tolerance"]
    ):
        errors.append("baseline sending rate differs from target")
    start, end = evidence["post_start"], evidence["post_end"]
    if end - start < p["observe_s"] - p["max_gap_s"]:
        errors.append("observation too short")
    threshold = max(p["absolute_min_hit"], (baseline["hit"] or 0) - p["max_drop"])
    windows = []
    t = start + p["window_s"]
    while t <= end + 1e-8:
        w = window(rows, t - p["window_s"], t, evidence["survivors"], p["max_gap_s"])
        w["low"] = w["hit"] is not None and w["hit"] < threshold
        w["valid"] = (
            not w["errors"]
            and w["hit"] is not None
            and w["completed"] >= p["min_completed"]
            and w["sent_qps"] is not None
            and abs(w["sent_qps"] / p["qps"] - 1) <= p["qps_tolerance"]
        )
        windows.append(w)
        t += p["step_s"]
    if not windows or any(not w["valid"] for w in windows):
        errors.append(
            "missing samples, insufficient prefill completions or off-target load"
        )
    longest = run = 0.0
    first_low = first_collapse = recovery = None
    previous_end = None
    for w in windows:
        if w["valid"] and w["low"]:
            if first_low is None:
                first_low = w["end"]
            run = 0 if previous_end is None else run + w["end"] - previous_end
            if run >= p["sustain_s"] and first_collapse is None:
                first_collapse = w["end"]
            longest = max(longest, run)
            previous_end = w["end"]
        else:
            if first_collapse is not None and w["valid"] and recovery is None:
                recovery = w["end"]
            run, previous_end = 0, None
    verdict = (
        "INVALID" if errors else ("FAIL" if first_collapse is not None else "PASS")
    )
    return dict(
        schema_version=1,
        measurement_scope=scope_contract(evidence) if scoped else {
            "policy": "historical-detach-window", "drain": "diagnostic only"},
        excluded_drain_diagnostics=[e for e in evidence.get("errors", []) if e in DRAIN_DIAGNOSTICS],
        verdict=verdict,
        errors=sorted(set(errors)),
        threshold=threshold,
        baseline=baseline,
        windows=windows,
        longest_low_s=longest,
        first_low_s=first_low,
        first_collapse_s=first_collapse,
        first_recovery_s=recovery,
        semantics="completed-prefill token-weighted reuse on surviving P; request failures, queue pressure and later Master count changes are diagnostic only",
        criteria=p,
        events=evidence.get("events", []),
    )


def prepare_report(directory, evidence):
    from monitoring.session import archived_series
    from statistics import median

    rows = evidence["samples"]
    anchor = rows[0]["epoch_s"] - rows[0]["t"] if rows else 0
    series, sources, gaps, errors = archived_series(directory, anchor)
    metric_defs = CACHE_METRICS
    curves = []
    audit = []
    found = set()
    for key, points in series.items():
        _, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        if labels.get("role") == "decode":
            continue
        kind = "mock" if source == "mock" else "client" if source.startswith("client-") else "master"
        definition = metric_defs.get((kind, metric))
        if not definition:
            continue
        name, group, axis, unit, color, hidden = definition
        qualifiers = [str(value) for label, value in sorted(labels.items()) if label not in {"role"}]
        if qualifiers:
            name += " · " + ", ".join(qualifiers)
        found.add((kind, metric))
        visible_end = max((r["t"] for r in rows), default=0)
        valid_points = sum(
            value is not None and 0 <= timestamp <= visible_end
            for timestamp, value in points
        )
        expected = max(1, round((visible_end + 1) / max(sources[key].get("step", 1), 0.001)))
        coverage = min(1, valid_points / expected)
        curves.append(
            dict(
                name=name,
                group=group,
                axis=axis,
                unit=unit,
                color=color,
                points=[dict(x=t, y=v) for t, v in points],
                hidden=hidden,
                description=sources[key]["promql"],
            )
        )
        audit.append([name, f"{coverage:.0%}", "OK" if coverage >= 0.8 else "SPARSE", sources[key]["promql"]])
    for identity, definition in metric_defs.items():
        if identity == ("master", "schedule_responses_qps") and ("master", "completions_qps") in found:
            continue
        if identity not in found and identity in {
            ("mock", "context_completed_qps"),
            ("master", "schedule_responses_qps")
        }:
            audit.append([definition[0], "0%", "MISSING", "本次归档没有该监控序列"])

    by_name = {curve["name"]: curve for curve in curves}
    def values(name, start=None, end=None):
        return [
            point["y"]
            for point in by_name.get(name, {}).get("points", [])
            if point["y"] is not None
            and (start is None or point["x"] >= start)
            and (end is None or point["x"] <= end)
        ]
    baseline_start, baseline_end = evidence.get("baseline_start"), evidence.get("baseline_end")
    sent = values("Client sent QPS", baseline_start, baseline_end)
    success = values("Client success QPS", baseline_start, baseline_end)
    failures = values("Client error QPS", baseline_start, baseline_end)
    monitor_warnings = [str(error) for error in errors]
    if sent and success and median(sent) > 0 and median(success) / median(sent) < 0.9:
        monitor_warnings.append(
            f"缩容前客户端成功率偏低：success/send 中位数 {median(success) / median(sent):.1%}"
        )
    if sent and failures and median(sent) > 0 and median(failures) / median(sent) > 0.05:
        monitor_warnings.append(
            f"缩容前客户端错误率偏高：error/send 中位数 {median(failures) / median(sent):.1%}"
        )
    monitoring_status = "WARN" if monitor_warnings else "OK"
    return dict(curves=curves, audit=audit, sources=sources, gaps=gaps, errors=errors,
                monitoring_status=monitoring_status, monitor_warnings=monitor_warnings)


def report_panels(curves, presentation):
    """Project archived monitoring curves into independent presentation panels."""
    panels = []
    for descriptor in presentation["panels"]:
        selected = [dict(curve, hidden=False) for name in descriptor["names"]
                    for curve in curves
                    if curve["name"] == name or curve["name"].startswith(name + " · ")]
        missing = [name for name in descriptor["names"] if not any(
            curve["name"] == name or curve["name"].startswith(name + " · ")
            for curve in selected)]
        caption = descriptor["caption"] if selected else descriptor["empty_caption"]
        if selected and missing:
            caption += " 缺少监控序列：" + "、".join(missing) + "。"
        panels.append(dict(id=descriptor["id"], title=descriptor["title"],
                           overlay=True, timeX=True, axes=descriptor["axes"],
                           series=selected, caption=caption))
    return panels


def build_spec(directory, evidence, result, prepared):
    presentation = view("cache_scale_in_overview.yaml")
    rows = evidence["samples"]
    curves = list(prepared["curves"])
    curves.append(dict(name="Survivor window hit ratio", group="缓存", axis="ratio",
                       unit="", color=PALETTE[0],
                       description="冻结门禁窗口：仅 survivors 的 hit/context counter delta，x 为窗口结束时刻",
                       points=[dict(x=w["end"], y=w["hit"] if not w["errors"] else None)
                               for w in result["windows"]]))
    audit = prepared["audit"]
    sources, gaps, errors = (prepared[key] for key in ("sources", "gaps", "errors"))
    monitoring_status = prepared["monitoring_status"]
    monitor_warnings = prepared["monitor_warnings"]
    return dict(
        run_id="cache-scale-in",
        title=presentation["title"],
        subtitle=presentation["subtitle"].format(
            verdict=result["verdict"], monitoring_status=monitoring_status),
        meta=dict(
            params=evidence["criteria"],
            sampling=presentation["meta"]["sampling"],
            sources=dict(
                aggregate=presentation["meta"]["aggregate"],
                engineDist=presentation["meta"]["engine_dist"],
                runDir=str(Path(directory).resolve()),
            ),
        ),
        timeOriginLabel=presentation["time_origin"],
        events=evidence.get("events", []),
        kpis=[
            dict(label=presentation["kpis"]["verdict"], value=result["verdict"]),
            dict(label=presentation["kpis"]["monitoring"], value=monitoring_status,
                 tone="danger" if monitor_warnings else "success"),
        ],
        panels=report_panels(curves, presentation),
        sections=[
            table(presentation["sections"]["audit"], presentation["audit_columns"], audit),
            details(presentation["sections"]["monitoring"], dict(status=monitoring_status, warnings=monitor_warnings)),
            details(presentation["sections"]["verdict"], result),
            details("测量口径与请求归属", dict(scope=result["measurement_scope"],
                    client_attribution=evidence.get("client_attribution"),
                    removals=evidence.get("removals"),
                    reinterpretation=evidence.get("reinterpretation"))),
            details(presentation["sections"]["sources"], dict(queries=sources, gaps=gaps, errors=errors)),
        ],
        timeAxis=dict(min=0, max=max((r["t"] for r in rows), default=1)),
    )


def write_report(directory, evidence, result, prepared=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cache-gate-evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False)
    )
    prepared = prepared if prepared is not None else prepare_report(directory, evidence)
    spec = build_spec(directory, evidence, result, prepared)
    provenance = evidence.get("provenance", {})
    meta = run_meta(
        dict(id="cache-scale-in", instance=provenance.get("instance")),
        implementation=dict(
            files=provenance.get("files"), master=provenance.get("master_artifact")
        ),
        workload=provenance.get("trace"),
        configuration={
            k: provenance.get(k)
            for k in ("topology", "capacity", "performance", "master_config", "actual_master_config")
        },
        environment=provenance.get("client_environment"),
        evidence=[
            dict(
                path="../../../cache-gate-evidence.json",
                sha256=hashlib.sha256(
                    (directory / "cache-gate-evidence.json").read_bytes()
                ).hexdigest(),
            )
        ],
    )
    write_bundle(
        directory,
        "run",
        "cache-scale-in",
        result,
        spec,
        meta=meta,
        producer="cache-gate",
        role="gate",
    )
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reinterpret", action="store_true",
                        help="explicitly re-adjudicate historical evidence into a NEW output directory")
    parser.add_argument("--client-snapshot", type=Path,
                        help="complete java_flow evidence_snapshot JSON for historical attribution")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    if not args.reinterpret:
        parser.error("offline adjudication requires --reinterpret; comparisons use frozen bundles")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("reinterpretation requires an empty output directory; frozen reports cannot be overwritten")
    original_errors = list(evidence.get("errors", []))
    if args.client_snapshot:
        snapshot = json.loads(args.client_snapshot.read_text())
        if len(snapshot["issued"]) == snapshot.get("status", {}).get("submitted"):
            align_send_counters(evidence, snapshot["issued"])
        else:
            evidence.setdefault("errors", []).append("issued send accounting incomplete")
        attribute_client(evidence, snapshot)
    evidence["reinterpretation"] = dict(
        source=str(args.evidence.resolve()),
        source_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        client_snapshot=(dict(path=str(args.client_snapshot.resolve()),
                              sha256=hashlib.sha256(args.client_snapshot.read_bytes()).hexdigest())
                         if args.client_snapshot else None),
        original_errors=original_errors,
        excluded_drain_diagnostics=[e for e in original_errors if e in DRAIN_DIAGNOSTICS],
        window_policy="preserve archived baseline/post windows and all thresholds",
        limitations=[] if evidence.get("measurement_policy") == MEASUREMENT_POLICY else
            ["historical admission timing cannot be reconstructed; detach-based windows retained"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(evidence)
    write_report(args.output, evidence, result,
                 prepared=prepare_report(args.evidence.parent, evidence))
    print(result["verdict"])
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[result["verdict"]]


def report_series(directory, anchor_epoch_s):
    from monitoring.session import archived_series

    series, sources, _, _ = archived_series(directory, anchor_epoch_s)
    return series, sources


if __name__ == "__main__":
    raise SystemExit(main())
