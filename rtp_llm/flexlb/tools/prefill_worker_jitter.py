"""Deterministic throughput changes on the Java mock's real execution formula.

Requires the mock's per-engine /prefill_formula endpoint. Durations change as
1 / throughput_factor; legacy jitter_pct instead perturbs duration randomly.
"""

import hashlib
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict

FACTORS = (0.9, 0.95, 1.0, 1.05, 1.1)
MODES = ("stable", "independent", "correlated")


def factor(seed, worker, slot, mode):
    """Each five-slot cycle covers both extremes, independently of request order."""
    if mode not in MODES or slot < 0:
        raise ValueError("invalid jitter mode or negative slot")
    if mode == "stable":
        return 1.0
    identity = worker if mode == "independent" else "all-workers"
    digest = hashlib.sha256(f"{seed}:{identity}:{slot // 5}".encode()).digest()
    choices = list(FACTORS)
    random.Random(int.from_bytes(digest, "big")).shuffle(choices)
    return choices[slot % 5]


def _http_get(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def _http_post(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


class JitterController:
    """Change each discovered Prefill worker once per elapsed monotonic slot.

    Formula changes apply to subsequent batches. Already running batches retain
    their duration. Control acknowledgements and independent GET readbacks are
    retained so an HTTP success alone never establishes that jitter was active.
    """

    def __init__(
        self,
        ops,
        mode="independent",
        seed=1,
        period_s=1.0,
        http_get=None,
        http_post=None,
    ):
        if mode not in MODES or not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("invalid jitter mode or period")
        self.ops, self.mode, self.seed, self.period_s = ops, mode, seed, period_s
        self.http_get = http_get or _http_get
        self.http_post = http_post or _http_post
        self.url = f"http://127.0.0.1:{ops.mock_http_port}/prefill_formula"
        self.records = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = None
        self._origin = None

    def _workers(self):
        return [
            worker
            for worker in self.ops.snapshot()["engines"]
            if worker["role"].lower() == "prefill" and not worker.get("stopped")
        ]

    def apply_slot(self, slot):
        """Synchronous control step, also useful for a deterministic unit test."""
        for worker in self._workers():
            name = worker["name"]
            multiplier = factor(self.seed, name, slot, self.mode)
            expression = f"sum(computeTokens) / 70 / {multiplier}"
            row = {
                "engine": name,
                "engine_incarnation": worker.get("engine_incarnation"),
                "slot": slot,
                "mode": self.mode,
                "factor": multiplier,
                "expression": expression,
                "request_wall_ms": time.time() * 1000,
                "request_mono_s": time.monotonic(),
            }
            if self._origin is not None:
                row["scheduled_mono_s"] = self._origin + slot * self.period_s
            self.records.append(row)
            status, response = self.http_post(
                self.url, {"engine": name, "expression": expression}
            )
            row.update(
                status=status,
                response=response,
                ack_wall_ms=time.time() * 1000,
                ack_mono_s=time.monotonic(),
            )
            if status != 200:
                # A worker may finish its graceful removal after the snapshot.
                if status in (400, 404) and name not in {
                    item["name"] for item in self._workers()
                }:
                    row["removed_before_apply"] = True
                    continue
                raise RuntimeError(f"formula update failed: {row}")
            state = (response or {}).get("engines", {}).get(name, {})
            if state.get("expression") != expression or not state.get(
                "runtime_override"
            ):
                raise RuntimeError(f"formula acknowledgement disagrees: {row}")
            readback = self.http_get(self.url)
            actual = (readback or {}).get("engines", {}).get(name)
            row["readback"] = actual
            row["readback_wall_ms"] = time.time() * 1000
            if actual is None and name not in {
                item["name"] for item in self._workers()
            }:
                row["removed_before_readback"] = True
                continue
            if (
                not actual
                or actual.get("expression") != expression
                or not actual.get("runtime_override")
            ):
                raise RuntimeError(f"formula readback disagrees: {row}")

    def start(self):
        if self._origin is not None:
            raise RuntimeError("jitter controller may only be started once")
        self._origin = time.monotonic()
        self.apply_slot(0)

        def run():
            slot = 1
            try:
                while not self._stop.wait(
                    max(0, self._origin + slot * self.period_s - time.monotonic())
                ):
                    elapsed_slot = int(
                        (time.monotonic() - self._origin) / self.period_s
                    )
                    if elapsed_slot > slot:
                        raise RuntimeError(
                            f"jitter control missed slot {slot}, now {elapsed_slot}"
                        )
                    self.apply_slot(slot)
                    slot += 1
            except Exception as error:
                self.errors.append(repr(error))
                self._stop.set()

        self._thread = threading.Thread(target=run, name="prefill-jitter", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise RuntimeError("jitter controller did not stop")
        if self.errors:
            raise RuntimeError("jitter controller failed: " + "; ".join(self.errors))
        return self.records


def audit_execution(
    events,
    records,
    mode,
    base_compute_tps=70_000,
    boundary_ms=10,
    min_checked_batches=5,
):
    """Audit actual modeled batch durations, not repeated member exec_ms values.

    The mock emits one completion timestamp for every member of an execution
    batch. All members, including cancelled ones, must be supplied. This audit
    assumes one execution lane per worker and the linear 70k-token/s formula;
    it rejects incomplete or colliding batch identities rather than guessing.
    Wall-clock time can exceed the model because this is a shared CPU process.
    """
    if mode not in MODES:
        raise ValueError("invalid jitter mode")
    controls = defaultdict(list)
    for row in records:
        if row.get("removed_before_apply"):
            continue
        assert row.get("status") == 200, f"unacknowledged control: {row}"
        assert row["mode"] == mode and row["factor"] in FACTORS, row
        expression = f"sum(computeTokens) / 70 / {row['factor']}"
        assert row["expression"] == expression, row
        assert row["ack_wall_ms"] >= row["request_wall_ms"], row
        if not row.get("removed_before_readback"):
            state = row.get("readback") or {}
            assert state.get("runtime_override") is True, row
            assert state.get("expression") == expression, row
        controls[(row["engine"], row.get("engine_incarnation"))].append(row)
    assert controls, "no successful jitter controls"
    for values in controls.values():
        values.sort(key=lambda row: row["request_wall_ms"])
    measurement_start = min(
        values[0]["request_wall_ms"] for values in controls.values()
    )
    batches = defaultdict(list)
    seen = set()
    for event in events:
        if event.get("event") != "prefill_done":
            continue
        identity = (event["engine_name"], event.get("engine_incarnation"))
        key = (*identity, event["prefill_done_ms"], event["exec_ms"])
        request = (*identity, event["rid"])
        assert request not in seen, f"duplicate Prefill terminal: {request}"
        seen.add(request)
        batches[key].append(event)
    rows = []
    for key, members in batches.items():
        identity = key[:2]
        timeline = controls.get(identity)
        assert timeline, f"worker generation never controlled: {identity}"
        start = min(member["prefill_start_ms"] for member in members)
        end_start = max(member["prefill_start_ms"] for member in members)
        if start < measurement_start:
            continue  # The driver starts controls after its explicit warmup.
        initial_uncontrolled = start < timeline[0]["request_wall_ms"]
        assert all(member["batch_size"] == len(members) for member in members), (
            "incomplete or ambiguous execution batch",
            key,
            len(members),
            [member["batch_size"] for member in members],
        )
        compute = 0
        for member in members:
            assert 0 <= member["cache_hit_tokens"] <= member["input_len"], member
            compute += member["input_len"] - member["cache_hit_tokens"]
        duration = members[0]["exec_ms"]
        baseline_ms = compute * 1000 / base_compute_tps
        allowed = (1.0,) if mode == "stable" else FACTORS
        possible = {
            value for value in allowed if duration == max(1, int(baseline_ms / value))
        }
        assert possible, (
            "execution duration does not match any configured throughput factor",
            key,
            compute,
            duration,
        )
        assert key[2] - end_start >= duration - 2, (
            "terminal precedes modeled execution duration",
            key,
            end_start,
        )
        preceding = [row for row in timeline if row["request_wall_ms"] <= start]
        active = preceding[-1] if preceding else None
        transitions = [
            row
            for row in timeline
            if row["request_wall_ms"] - boundary_ms <= end_start
            and row["ack_wall_ms"] + boundary_ms >= start
        ]
        ambiguous = bool(transitions) or active is None
        if initial_uncontrolled:
            assert 1.0 in possible, (
                "new worker did not execute its declared baseline before first control",
                key,
                sorted(possible),
            )
        if not ambiguous:
            assert active["factor"] in possible, (
                "actual batch duration disagrees with acknowledged formula",
                key,
                active,
                sorted(possible),
            )
        rows.append(
            {
                "engine": identity[0],
                "engine_incarnation": identity[1],
                "start_wall_ms": start,
                "done_wall_ms": key[2],
                "members": len(members),
                "compute_tokens": compute,
                "exec_ms": duration,
                "factor": None if ambiguous else active["factor"],
                "ambiguous_transition": ambiguous,
                "initial_uncontrolled": initial_uncontrolled,
                "possible_factors": sorted(possible),
            }
        )
    checked = [row for row in rows if not row["ambiguous_transition"]]
    assert len(checked) >= min_checked_batches, "too few audited execution batches"
    assert len(checked) >= len(rows) * 0.8, "too many ambiguous formula transitions"
    exercised = sorted({row["factor"] for row in checked})
    if mode != "stable":
        assert min(exercised) == 0.9 and max(exercised) == 1.1, (
            "actual executions did not exercise both +/-10% endpoints",
            exercised,
        )
    return {
        "mode": mode,
        "batches": len(rows),
        "checked_batches": len(checked),
        "ambiguous_batches": len(rows) - len(checked),
        "initial_uncontrolled_batches": sum(
            row["initial_uncontrolled"] for row in rows
        ),
        "factors_exercised": exercised,
        "executed_compute_tokens": sum(row["compute_tokens"] for row in rows),
        "modeled_busy_ms": sum(row["exec_ms"] for row in rows),
        "batch_evidence": rows,
    }
