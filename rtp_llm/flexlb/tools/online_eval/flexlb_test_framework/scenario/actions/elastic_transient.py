"""Abrupt Prefill scale-in: explicit capacities, windows and route ownership."""

import json
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..contracts import CheckResult, StageHandler, StageOutput
from . import elastic_balance as balance

P = ("prefill-0", "prefill-2")
D = ("decode-0", "decode-1")
SURVIVORS = P + D
VICTIM = "prefill-1"


def refs(fields):
    def validate(params, plan):
        from .elastic import _validate

        p = _validate(params, plan, set(fields), set(fields))
        for key, kind in fields.items():
            plan.reference(p[key], kind)
        return p

    return validate


def artifact(ctx, name, data):
    path = ctx.artifact_dir / f"elastic-transient-{name}-{time.time_ns()}.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def observe(ctx, params, deadline):
    from .elastic import ElasticMetrics

    class MasterMetrics(ElasticMetrics):
        def _sample_loop(self):
            while not self._stop.is_set():
                now = ctx.clock()
                if now >= self.end:
                    raise TimeoutError("master sampler exceeded lifetime")
                if ctx.env_epoch != self.env_epoch:
                    raise ValueError("master sampler environment epoch changed")
                try:
                    url = f"http://127.0.0.1:{ctx.ops.master_http_port}/rtp_llm/inflight_status"
                    with urllib.request.urlopen(
                        url, timeout=min(2, self.end - now)
                    ) as response:
                        raw = response.read(2_000_001)
                        if len(raw) > 2_000_000:
                            raise ValueError("master sample exceeds byte budget")
                        payload = json.loads(raw)
                    values = master_rows(payload)
                    with self._lock:
                        self.samples.append(dict(time_s=ctx.clock(), engines=values))
                except Exception as exc:
                    with self._lock:
                        self.errors.append(dict(time_s=ctx.clock(), error=repr(exc)))
                self._stop.wait(1)

        def stop(self, d):
            self._stop.set()
            try:
                if self.thread.ident is not None and not self.done.wait(d.remaining()):
                    raise TimeoutError("master sampler did not exit")
            finally:
                evidence = self.snapshot()
                evidence["complete"] = self.done.is_set() or self.thread.ident is None
                artifact(ctx, "master-metrics", evidence)
            if self.errors:
                raise ValueError(f"master acquisition errors: {self.errors}")

    deadline.check()
    result = balance.observe_start(ctx, {}, deadline)
    metrics = MasterMetrics(ctx, max_duration_s=1200)
    handle = ctx.register_resource("observation", metrics, cleanup=metrics.stop)
    try:
        metrics.thread.start()
    except BaseException:
        metrics.done.set()
        raise
    return StageOutput(output=dict(result.output, master_observation=handle))


def master_rows(payload):
    entries = payload.get("prefill_endpoints")
    if not isinstance(entries, list):
        raise ValueError("master sample has no Prefill endpoint list")
    result = {}
    for row in entries:
        address, value = row.get("ip_port"), row.get("inflight_requests")
        if (
            not isinstance(address, str)
            or not address
            or type(value) is not int
            or value < 0
        ):
            raise ValueError("master sample requires HTTP identity and request count")
        if address in result:
            raise ValueError("duplicate master HTTP endpoint")
        result[address] = dict(inflight_requests=value)
    return result


def prepare(ctx, params, deadline):
    from .elastic import _snapshot

    snap = _snapshot(ctx, deadline)
    addresses = set()
    for name in (VICTIM,) + SURVIVORS:
        row = snap.get(name, {})
        for field in ("http_addr", "grpc_addr"):
            value = row.get(field)
            if not isinstance(value, str) or not value or value in addresses:
                raise ValueError("snapshot lacks distinct HTTP and RPC identities")
            addresses.add(value)
        total, free = row.get("cache_blocks"), row.get("available_blocks")
        if (
            type(total) is not int
            or type(free) is not int
            or not 0 <= free <= total
            or total <= 0
        ):
            raise ValueError("snapshot lacks valid per-engine KV capacity")
    path = artifact(ctx, "pre-event", snap)
    return StageOutput(
        output=dict(snapshot=ctx.register_resource("snapshot", snap, historical=True)),
        artifacts=[path],
    )


def burst_start(ctx, params, deadline):
    from .elastic import RecordedRequests

    deadline.check()
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    pool = ThreadPoolExecutor(max_workers=15, thread_name_prefix="transient-burst")
    records.burst_pool, records.burst_jobs = pool, []
    records.burst_started = threading.Event()

    def cleanup(d):
        for future, event in records.burst_jobs:
            future.cancel()
        records.cancel_active("transient_burst_cleanup")
        try:
            for future, event in records.burst_jobs:
                if not event.wait(d.remaining()):
                    raise TimeoutError("burst consumer did not exit")
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            artifact(ctx, "burst-cleanup", records.snapshot_records())

    handle = ctx.register_resource("requests", records, cleanup=cleanup)
    for _ in range(30):
        record = records.issue(ctx.ops.next_request_id(), ctx.clock)
        event = threading.Event()

        def run(record=record):
            records.burst_started.set()
            rid = record["wire_request_id"]
            records.run(
                record,
                dict(input_len=2048, output_len=2, block_keys=[rid * 100 + 1]),
                timeout_s=75,
                schedule_timeout_s=30,
                stream_timeout_s=45,
            )

        future = pool.submit(run)
        future.add_done_callback(lambda f, event=event: event.set())
        records.burst_jobs.append((future, event))
    if not records.burst_started.wait(deadline.remaining()):
        raise TimeoutError("crossing burst never started")
    return StageOutput(output=dict(requests=handle))


def burst_collect(ctx, params, deadline):
    from .elastic import completeness

    records = ctx.resource(params["requests"], "requests")
    for future, event in records.burst_jobs:
        if not event.wait(deadline.remaining()):
            raise TimeoutError("burst exceeded two bounded request waves")
        future.result()
    records.burst_pool.shutdown(wait=False)
    rows = records.snapshot_records()
    detail = completeness(rows)
    path = artifact(ctx, "burst", dict(records=rows, summary=detail))
    return StageOutput(
        output=dict(settled_s=ctx.clock()),
        checks=[
            CheckResult(
                "complete",
                "PASS" if detail["result_complete"] and len(rows) == 30 else "FAIL",
                actual=detail,
            )
        ],
        artifacts=[path],
    )


def remove(ctx, params, deadline):
    from .elastic import _snapshot

    evidence = dict(engine=VICTIM, mode="abrupt", started_s=ctx.clock(), complete=False)
    try:
        deadline.check()
        request = urllib.request.Request(
            f"http://127.0.0.1:{ctx.ops.mock_http_port}/remove_engine",
            data=json.dumps(dict(engine=VICTIM, mode="abrupt")).encode(),
            headers={"Content-Type": "application/json"},
        )
        evidence["burst_before_remove"] = ctx.resource(
            params["requests"], "requests"
        ).snapshot_records()
        evidence["started_s"] = ctx.clock()
        with urllib.request.urlopen(
            request, timeout=min(5, deadline.remaining())
        ) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("remove response exceeds byte budget")
            reply = json.loads(raw)
        evidence["response"] = reply
        if (
            reply.get("status") != "ok"
            or reply.get("action") != "removed"
            or reply.get("engine") != VICTIM
            or reply.get("mode") != "abrupt"
        ):
            raise ValueError("abrupt removal lacks matching acknowledgement")
        after = _snapshot(ctx, deadline)
        if VICTIM in after or any(n not in after for n in SURVIVORS):
            raise ValueError("abrupt removal changed unexpected membership")
        evidence.update(after=after, returned_s=ctx.clock(), complete=True)
    finally:
        path = artifact(ctx, "remove", evidence)
    return StageOutput(
        output=dict(
            mutation=ctx.register_resource("snapshot", evidence, historical=True),
            started_s=evidence["started_s"],
        ),
        checks=[CheckResult("membership", "PASS")],
        artifacts=[path],
    )


def bounds(ctx, params, deadline):
    deadline.check()
    pre = ctx.resource(params["pre_event"], "snapshot")
    baseline = ctx.resource(params["baseline"], "snapshot")
    transient = ctx.resource(params["transient"], "snapshot")
    master = ctx.resource(params["master_observation"], "observation")
    mwin = dict(
        start_s=transient["start_s"], end_s=transient["end_s"], data=master.snapshot()
    )
    ppeak = peak(balance.points(transient, P, "mock_engine_waiting"))
    dpeak = peak(balance.points(transient, D, "mock_engine_waiting"))
    # Registry keys are HTTP addresses. The victim's RPC port is never a filter.
    mpeak = peak(
        balance.points(mwin, [pre[n]["http_addr"] for n in P], "inflight_requests")
    )
    occupancy = peak(balance.points(transient, SURVIVORS, "occupancy"))
    demand = pre[VICTIM]["cache_blocks"] - pre[VICTIM]["available_blocks"]
    free = sum(pre[n]["available_blocks"] for n in SURVIVORS)
    cap = math.ceil(max(0, demand - free))
    reject = sum(
        sum(balance.deltas(balance.points(transient, SURVIVORS, key)).values())
        for key in (
            "mock_engine_lack_mem_rejects_total",
            "mock_engine_kv_admission_fails_total",
        )
    )
    obs = dict(
        baseline_tps=cluster_tps(baseline, (VICTIM,) + SURVIVORS),
        transient_tps=cluster_tps(transient, SURVIVORS),
    )
    obs["tps_floor"] = (
        obs["baseline_tps"]["value"] * 4 / 5 * 0.85
        if obs["baseline_tps"]["value"] is not None
        else None
    )
    try:
        victim_peak = peak(
            balance.points(mwin, [pre[VICTIM]["http_addr"]], "inflight_requests")
        )
        obs["victim_master_inflight_peak"] = dict(
            value=victim_peak, unavailable_reason=None
        )
    except ValueError as exc:
        obs["victim_master_inflight_peak"] = dict(
            value=None, unavailable_reason=str(exc)
        )
    data = dict(
        window=[transient["start_s"], transient["end_s"]],
        prefill_waiting=ppeak,
        decode_waiting=dpeak,
        master_inflight=mpeak,
        occupancy=occupancy,
        rejects=reject,
        rejection_cap=cap,
        victim_occupied=demand,
        all_survivor_free=free,
        observations=obs,
    )
    return StageOutput(
        checks=[
            check(name, actual, limit)
            for name, actual, limit in [
                ("prefill_waiting", ppeak, 16),
                ("decode_waiting", dpeak, 128),
                ("master_inflight", mpeak, 64),
                ("occupancy", occupancy, 0.95),
                ("rejects", reject, cap),
            ]
        ],
        artifacts=[artifact(ctx, "bounds", data)],
    )


def peak(series):
    return max(v for seq in series.values() for _, v in seq)


def check(name, actual, limit):
    return CheckResult(
        name, "PASS" if actual <= limit else "FAIL", actual=actual, expected=limit
    )


def cluster_tps(window, names):
    try:
        values = []
        for name in names:
            metric = (
                "rtp_llm_context_tps"
                if name.startswith("prefill-")
                else "rtp_llm_generate_tps"
            )
            seq = balance.points(window, [name], metric)[name]
            values.append(sum(v for _, v in seq) / len(seq))
        return dict(value=sum(values), unavailable_reason=None)
    except ValueError as exc:
        return dict(value=None, unavailable_reason=str(exc))


def accepted_shares(window, names):
    delta = balance.deltas(balance.points(window, names, "mock_engine_accepted_total"))
    total = sum(delta.values())
    if total <= 0:
        raise ValueError("empty Prefill share denominator")
    return {name: value / total for name, value in delta.items()}


def steady(ctx, params, deadline):
    deadline.check()
    before = ctx.resource(params["baseline"], "snapshot")
    after = ctx.resource(params["steady"], "snapshot")
    tail = after["start_s"] + 40
    base_share = accepted_shares(before, (VICTIM,) + P)
    share = accepted_shares(after, P)
    cap = max(max(base_share.values()) + 0.1, 0.65)
    pbase = balance.spread(balance.points(before, (VICTIM,) + P, "occupancy"))
    dbase = balance.spread(balance.points(before, D, "occupancy"))
    pspread = balance.spread(balance.points(after, P, "occupancy", start=tail))
    dspread = balance.spread(balance.points(after, D, "occupancy", start=tail))
    wait = peak(balance.points(after, SURVIVORS, "mock_engine_waiting", start=tail))
    try:
        decode_share = dict(value=accepted_shares(after, D), unavailable_reason=None)
    except ValueError as exc:
        decode_share = dict(value=None, unavailable_reason=str(exc))
    data = dict(
        share=share,
        baseline_share=base_share,
        tail_start_s=tail,
        prefill_spread=pspread,
        decode_spread=dspread,
        waiting=wait,
        decode_share_observation=decode_share,
    )
    checks = [
        check("share_max", max(share.values()), cap),
        CheckResult(
            "share_min",
            "PASS" if min(share.values()) >= 0.1 else "FAIL",
            actual=min(share.values()),
            expected=0.1,
        ),
        check("prefill_spread", pspread, pbase + 0.05),
        check("decode_spread", dspread, dbase + 0.05),
        check("waiting", wait, 2),
    ]
    return StageOutput(checks=checks, artifacts=[artifact(ctx, "steady", data)])


def locality(ctx, params, deadline):
    from .elastic import completeness, request_success

    deadline.check()
    pre = ctx.resource(params["pre_event"], "snapshot")
    flow = ctx.resource(params["flow"], "flow")
    burst = ctx.resource(params["requests"], "requests")
    if not flow.done.is_set():
        raise ValueError("locality requires the whole stopped pump cohort")
    rows = flow.snapshot_records() + burst.snapshot_records()
    summary = completeness(rows)
    counts = dict(victim_failures=0, survivor_failures=0, unrouted_failures=0)
    for record in rows:
        ok = (
            request_success(record)
            and record["consumer_exit_s"] is not None
            and record["transport_terminal_s"] is not None
        )
        if not ok:
            address = record["prefill_addr"]
            key = (
                "unrouted_failures"
                if not address
                else (
                    "victim_failures"
                    if address == pre[VICTIM]["grpc_addr"]
                    else "survivor_failures"
                )
            )
            counts[key] += 1
    data = dict(records=rows, summary=summary, **counts)
    return StageOutput(
        checks=[
            CheckResult(
                "complete",
                "PASS" if summary["result_complete"] else "FAIL",
                actual=summary,
            ),
            CheckResult(
                "survivor_failures",
                "PASS" if counts["survivor_failures"] == 0 else "FAIL",
                actual=counts,
            ),
        ],
        artifacts=[artifact(ctx, "locality", data)],
    )


HANDLERS = [
    StageHandler(
        "elastic_transient_observe",
        balance.empty,
        observe,
        {
            "observation": "observation",
            "master_observation": "observation",
            "started_s": "number",
        },
    ),
    StageHandler(
        "elastic_transient_prepare", balance.empty, prepare, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "elastic_transient_burst", balance.empty, burst_start, {"requests": "requests"}
    ),
    StageHandler(
        "elastic_transient_collect",
        refs({"requests": "requests"}),
        burst_collect,
        {"settled_s": "number"},
        checks=frozenset({"complete"}),
    ),
    StageHandler(
        "elastic_transient_remove",
        refs({"requests": "requests"}),
        remove,
        {"mutation": "snapshot", "started_s": "number"},
        checks=frozenset({"membership"}),
    ),
    StageHandler(
        "elastic_transient_bounds",
        refs(
            {
                "pre_event": "snapshot",
                "baseline": "snapshot",
                "transient": "snapshot",
                "master_observation": "observation",
            }
        ),
        bounds,
        {},
        checks=frozenset(
            {
                "prefill_waiting",
                "decode_waiting",
                "master_inflight",
                "occupancy",
                "rejects",
            }
        ),
    ),
    StageHandler(
        "elastic_transient_steady",
        refs({"baseline": "snapshot", "steady": "snapshot"}),
        steady,
        {},
        checks=frozenset(
            {"share_max", "share_min", "prefill_spread", "decode_spread", "waiting"}
        ),
    ),
    StageHandler(
        "elastic_transient_locality",
        refs({"pre_event": "snapshot", "flow": "flow", "requests": "requests"}),
        locality,
        {},
        checks=frozenset({"complete", "survivor_failures"}),
    ),
]
