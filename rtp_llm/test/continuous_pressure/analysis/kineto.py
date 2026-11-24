"""Validate native Kineto trace ranges against completed pressure steps."""

import math

from .nsys import unique_cuda_launch


def _range(event):
    start, duration = event.get("ts"), event.get("dur")
    if (
        event.get("ph") != "X"
        or not isinstance(start, (int, float))
        or not isinstance(duration, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None
    return start, start + duration


def _contains(outer, inner):
    a, b = _range(outer), _range(inner)
    return (
        a is not None
        and b is not None
        and outer.get("pid") == inner.get("pid")
        and outer.get("tid") == inner.get("tid")
        and a[0] <= b[0]
        and b[1] <= a[1]
    )


def validate_disk_native_evidence(
    scopes, *, disk_read_bytes, successful_disk, completed_writes
):
    required = {}
    observed_read = disk_read_bytes > 0 or successful_disk > 0
    if observed_read:
        required.update(
            {
                "cache.disk.read": "missing_native_disk_read",
                "cache.transfer.h2d": "missing_native_disk_h2d",
            }
        )
    if completed_writes:
        required["cache.disk.write"] = "missing_native_disk_write"
    if observed_read or completed_writes:
        required["cache.disk.staging_admission_callback"] = (
            "missing_native_staging_admission"
        )
    return [reason for name, reason in required.items() if not scopes.get(name, 0)]


REMOTE_NATIVE_READ_SCOPES = {
    "cache.remote_mock.load.submit": "missing_native_remote_load_submit",
    "cache.remote_mock.load.completion": "missing_native_remote_load_completion",
    "cache.remote_mock.crc": "missing_native_remote_crc",
    "cache.transfer.h2d": "missing_native_remote_h2d",
}
REMOTE_NATIVE_WRITE_SCOPES = {
    "cache.remote_mock.write.capture": "missing_native_remote_write_capture",
    "cache.transfer.d2h": "missing_native_remote_d2h",
    "cache.remote_mock.write.crc": "missing_native_remote_write_crc",
    "cache.remote_mock.write.submit": "missing_native_remote_write_submit",
    "cache.remote_mock.write.ack_callback": "missing_native_remote_write_ack",
}


def validate_remote_native_evidence(scopes, *, read_bytes, write_bytes):
    required = {}
    if read_bytes:
        required.update(REMOTE_NATIVE_READ_SCOPES)
    if write_bytes:
        required.update(REMOTE_NATIVE_WRITE_SCOPES)
    return [reason for name, reason in required.items() if not scopes.get(name, 0)]


def validate_kineto(trace, steps, requested_steps):
    """Join observed CPU markers, forward scopes, launches and CUDA kernels.

    Only the installed Kineto schema's cpu_op/user_annotation, cuda_runtime/
    cuda_driver and kernel categories are accepted. GPU pid/tid are not OS IDs.
    Sidecar timestamps are never compared to Kineto timestamps.
    """
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else []
    if (
        not isinstance(trace, dict)
        or not isinstance(events, list)
        or any(not isinstance(e, dict) for e in events)
    ):
        return dict(trace_valid=False, reasons=["invalid_trace_schema"])
    reasons = []
    if trace.get("schemaVersion", 1) != 1:
        reasons.append("unsupported_trace_schema")
    if any(not isinstance(e.get("args", {}), dict) for e in events):
        return dict(trace_valid=False, reasons=["invalid_trace_arguments"])
    cpu = [
        e for e in events if e.get("cat") in ("cpu_op", "user_annotation") and _range(e)
    ]
    forwards = [
        e for e in cpu if str(e.get("name", "")).startswith("executor.model_forward")
    ]
    kernels = [e for e in events if e.get("cat") == "kernel" and _range(e)]
    runtimes = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and _range(e)
    ]
    if not kernels:
        reasons.append("missing_gpu_kernels")
    if not forwards:
        reasons.append("missing_cpu_forward_ranges")
    markers = {}
    for event in cpu:
        name = str(event.get("name", ""))
        if not name.startswith("pressure.step/"):
            continue
        try:
            step_id = int(name.split("/", 1)[1])
            if step_id <= 0 or step_id in markers:
                raise ValueError()
            markers[step_id] = event
        except ValueError:
            reasons.append("invalid_step_identity")
    if not markers:
        reasons.append("missing_step_markers")
    sidecar = {int(s["step_id"]): s for s in steps}
    runtime_by_correlation = {}
    for call in runtimes:
        correlation = call.get("args", {}).get("correlation")
        if correlation is not None:
            runtime_by_correlation.setdefault(correlation, []).append(call)
    launches = {}
    for correlation, calls in runtime_by_correlation.items():
        index = unique_cuda_launch(
            [
                (
                    c["cat"].removeprefix("cuda_"),
                    (c.get("pid"), c.get("tid")),
                    *_range(c),
                )
                for c in calls
            ]
        )
        if index is not None:
            launches[correlation] = calls[index]
    work_by_launch = {}
    for kernel in kernels:
        correlation = kernel.get("args", {}).get("correlation")
        if correlation in launches and all(
            _range(call)[0] <= _range(kernel)[0]
            for call in runtime_by_correlation[correlation]
        ):
            work_by_launch.setdefault(id(launches[correlation]), []).append(kernel)
    scopes_by_step = {
        step_id: [f for f in forwards if _contains(marker, f) or _contains(f, marker)]
        for step_id, marker in markers.items()
    }
    launch_owners = {}
    for step_id, scopes in scopes_by_step.items():
        if len(scopes) == 1:
            for call in launches.values():
                if _contains(scopes[0], call) and _contains(markers[step_id], call):
                    launch_owners.setdefault(id(call), []).append(step_id)
    if any(len(owners) > 1 for owners in launch_owners.values()):
        reasons.append("ambiguous_step_forward")
    matched, linked = {}, set()
    for step_id, marker in markers.items():
        scopes = scopes_by_step[step_id]
        if len(scopes) != 1:
            reasons.append("ambiguous_step_forward")
            continue
        forward = scopes[0]
        calls = [r for r in launches.values() if launch_owners.get(id(r)) == [step_id]]
        work = [k for call in calls for k in work_by_launch.get(id(call), [])]
        linked.update(id(k) for k in work)
        if not work:
            reasons.append("missing_cuda_correlation")
        if step_id not in sidecar:
            reasons.append("missing_step_sidecar")
            continue
        matched[step_id] = (marker, forward, work, sidecar[step_id])
    non_forward_correlated = 0
    if forwards:
        low = min(_range(f)[0] for f in forwards)
        high = max(
            [_range(f)[1] for f in forwards]
            + [_range(k)[1] for k in kernels if id(k) in linked]
        )
        for kernel in kernels:
            if id(kernel) in linked or not low <= _range(kernel)[0] <= high:
                continue
            call = launches.get(kernel.get("args", {}).get("correlation"))
            if call is None or kernel not in work_by_launch.get(id(call), []):
                reasons.append("missing_cuda_correlation")
                continue
            overlaps = any(
                scope.get("pid") == call.get("pid")
                and scope.get("tid") == call.get("tid")
                and _range(scope)[0] < _range(call)[1]
                and _range(call)[0] < _range(scope)[1]
                for scope in [*forwards, *markers.values()]
            )
            if overlaps:
                reasons.append("missing_cuda_correlation")
            else:
                non_forward_correlated += 1
    observed = {
        i for i, (_, _, _, s) in matched.items() if s.get("context_batch", 0) > 0
    }
    if len(observed) < requested_steps:
        reasons.append("insufficient_observed_steps")
    usable, excluded = set(), {}
    first = min(markers, key=lambda i: _range(markers[i])[0]) if markers else None
    ends = [
        e["ts"]
        for e in events
        if e.get("name") == "Record Window End"
        and isinstance(e.get("ts"), (int, float))
    ]
    if len(ends) != 1 or not math.isfinite(ends[0]):
        reasons.append("missing_capture_trace_boundary")
    gathers = [e for e in cpu if e.get("name") == "executor.gather_model_input"]
    for step_id, (marker, forward, work, step) in matched.items():
        exclusion = None
        if not step.get("complete") or step.get("profile_boundary"):
            exclusion = "incomplete_or_boundary"
        elif not step.get("context_batch", 0) or step.get("decode_batch", 0):
            exclusion = "not_pure_prefill"
        elif step_id == first and not any(
            g.get("pid") == forward.get("pid")
            and g.get("tid") == forward.get("tid")
            and _range(g)[1] <= _range(forward)[0]
            for g in gathers
        ):
            exclusion = "first_step_gather_omitted"
        elif ends and max(
            [_range(marker)[1], _range(forward)[1]] + [_range(k)[1] for k in work]
        ) >= min(ends):
            exclusion = "flush_boundary"
        elif not work:
            exclusion = "missing_cuda_correlation"
        if exclusion:
            excluded[str(step_id)] = exclusion
        else:
            usable.add(step_id)
    pairs = sum(i + 1 in usable for i in usable)
    if not pairs:
        reasons.append("no_complete_internal_step_pair")
    return dict(
        trace_valid=not reasons,
        reasons=sorted(set(reasons)),
        observed_steps=len(observed),
        usable_steps=len(usable),
        usable_pairs=pairs,
        observed_step_ids=sorted(observed),
        usable_step_ids=sorted(usable),
        exclusions=excluded,
        non_forward_correlated=non_forward_correlated,
        native_cache_scopes={
            name: sum(e.get("name") == name for e in cpu)
            for name in (
                "cache.disk.read",
                "cache.disk.write",
                "cache.transfer.h2d",
                "cache.transfer.d2h",
                "cache.disk.staging_admission_callback",
                *REMOTE_NATIVE_READ_SCOPES,
                *REMOTE_NATIVE_WRITE_SCOPES,
            )
        },
    )
