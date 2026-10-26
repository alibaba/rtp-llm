"""Continuous Java traffic across a concurrent Prefill scale-in."""

import math
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

from adapters.mock_control import mock_json
from artifacts.gate import gate_check
from artifacts.gate_inputs import new_evidence
from artifacts.provenance import gate_provenance
from cases.cache_scale_in.analysis import MEASUREMENT_POLICY, align_send_counters, attribute_client, scope_contract, topology_ready, window
from cases.cache_scale_in.constants import SHARED_WARMUP_AFFINITY_MS
from cases.cache_scale_in.inputs import validate_criteria
from file_io import write_json
from runtime.observation import SampleBudget, poll_samples
from scenario.contracts import StageHandler, StageOutput
from scenario.parameters import reference_parameters, validate_fields


def validate(params, plan):
    grouped = validate_fields(
        params, plan, {"flow", "criteria", "gate_input", "observation"}, {"flow", "criteria", "gate_input", "observation"}
    )
    from cases.cache_scale_in.inputs import observation_contract

    durations = observation_contract(grouped["observation"])
    if any(grouped["criteria"].get(key) != value for key, value in durations.items()):
        raise ValueError("observation windows disagree with frozen criteria")
    validate_criteria(dict(grouped["criteria"], flow=grouped["flow"], gate_input=grouped["gate_input"]), plan)
    return grouped


def _master_count(ctx, deadline):
    from adapters.master_control import master_url
    from runtime.network import http_post_json

    code, data = http_post_json(master_url("127.0.0.1", ctx.env.master_http_port, "info"), {}, timeout=min(3, deadline.remaining()))
    if code != 200 or not isinstance(data, dict):
        raise ValueError("master topology HTTP unavailable")
    value = data["worker_summary"]["PREFILL"]["alive"]
    if type(value) is not int:
        raise ValueError("master topology missing")
    return value


def _affinity_transition(ctx, value, deadline):
    """Change only after a healthy baseline and independently verify the result."""
    from adapters.master_control import affinity_request

    def request(payload=None):
        return affinity_request(ctx, deadline, payload)

    before = request()
    if before.get("maxExtraTtftMs") != SHARED_WARMUP_AFFINITY_MS:
        raise ValueError("unexpected warmup affinity policy")
    update = request(dict(expectedMaxExtraTtftMs=SHARED_WARMUP_AFFINITY_MS, maxExtraTtftMs=value))
    readback = request()
    if (
        update.get("before") != SHARED_WARMUP_AFFINITY_MS
        or update.get("maxExtraTtftMs") != value
        or readback.get("maxExtraTtftMs") != value
    ):
        raise ValueError("affinity intervention readback mismatch")
    return dict(before=before, update=update, readback=readback)


class _ScaleInObservation:
    """Own the sample ledger and concurrent removals for one scale-in window."""

    def __init__(self, ctx, params: dict, deadline):
        self.ctx, self.deadline = ctx, deadline
        self.observation = params["observation"]
        self.criteria = dict(params["criteria"])
        self.gate_input = params["gate_input"]
        self.flow = self.ctx.resource(params["flow"], "java_flow")
        self.clock = self.ctx.start_observation()
        self.origin = self.clock.origin_monotonic_s
        self.budget = SampleBudget(self.observation["capture"])
        self.evidence = new_evidence(
            "cache_evidence_schema_version",
            self.clock,
            self.criteria,
            instance=self.ctx.instance["id"],
            env_epoch=self.ctx.env_epoch,
            measurement_policy=MEASUREMENT_POLICY,
            window_declarations=self.observation["windows"],
            observation_origin_epoch_s=self.clock.origin_epoch_s,
            gate_input=self.gate_input,
            events=[],
            initial_engines=[],
            survivors=[],
            baseline_start=0,
            baseline_end=0,
            post_start=0,
            post_end=0,
            max_pacing_lag_ms=None,
            capture_limits=self.observation["capture"],
        )
        self.path = ctx.artifact_dir / "cache-gate-evidence.json"
        self.pool: ThreadPoolExecutor | None = None
        self.futures: list[Future[dict]] = []

    def event(self, name: str, row: dict | None = None) -> None:
        timestamp = None if row is None else {key: row[key] for key in ("epoch_s", "monotonic_s")}
        recorded = self.ctx.record_event(name, timestamp=timestamp)
        self.evidence["events"].append(dict(recorded, t=recorded["monotonic_s"] - self.origin))

    def sample(self) -> dict:
        self.deadline.check()
        from cases.cache_scale_in.observation import engine_metric_snapshot

        engines = engine_metric_snapshot(self.ctx.monitor, self.gate_input["fields"], timeout=min(3, self.deadline.remaining()))
        state = self.flow.status()
        stamp = self.clock.stamp(self.ctx.clock())
        row = dict(
            t=stamp["elapsed_s"],
            **stamp,
            engines=engines,
            started=state["observed_started"],
            terminal=state["observed_terminal"],
            master_p=_master_count(self.ctx, self.deadline),
            waiting=sum(e["waiting"] for e in engines.values()),
            running=sum(e["running"] for e in engines.values()),
        )
        self.budget.append(row)
        self.evidence["samples"].append(row)
        if state["process_returncode"] is not None or state.get("state") != "SENDING":
            raise ValueError("Java traffic stopped before observation completed")
        return row

    def wait_for_baseline(self) -> None:
        """Require two warm, adequately loaded half-windows before intervention."""
        criteria = self.criteria
        first = self.sample()
        initial = sorted(first["engines"])
        self.evidence["initial_engines"] = initial
        self.evidence["survivors"] = initial[: criteria["target_p"]]
        if first["master_p"] != len(initial):
            raise ValueError("initial discovery not converged")
        self.event("warmup_start")
        for row in poll_samples(self.deadline, criteria["sample_s"], self.sample):
            t = row["t"]
            if t >= max(criteria["baseline_s"], criteria["warmup_min_s"]):
                from cases.cache_scale_in.observation import measurement_samples

                baseline_rows = measurement_samples(
                    self.ctx.monitor,
                    self.evidence,
                    self.deadline,
                    self.evidence["capture_limits"],
                    windows=[(t - criteria["baseline_s"], t)],
                )
                align_send_counters(dict(samples=baseline_rows), self.flow.issued_snapshot(self.deadline, fields=("send_start_epoch_ms",)))
                left = window(baseline_rows, t - criteria["baseline_s"], t - criteria["baseline_s"] / 2, initial, criteria["max_gap_s"])
                right = window(baseline_rows, t - criteria["baseline_s"] / 2, t, initial, criteria["max_gap_s"])
                ready = all(
                    not w["errors"]
                    and w["hit"] is not None
                    and w["hit"] >= criteria["baseline_min_hit"]
                    and w["completed"] >= criteria["min_completed"]
                    and w["sent_qps"] is not None
                    and abs(w["sent_qps"] / criteria["qps"] - 1) <= criteria["qps_tolerance"]
                    for w in (left, right)
                )
                if t >= criteria["warmup_min_s"] and ready and abs(left["hit"] - right["hit"]) <= criteria["baseline_max_spread"]:
                    self.evidence.update(baseline_start=t - criteria["baseline_s"], baseline_end=t)
                    self.evidence["readiness_sample_source"] = "prometheus_raw_scrape"
                    self.event("baseline_ready", row)
                    break
            if t >= criteria["warmup_timeout_s"]:
                raise ValueError("stable warm baseline not reached")

    def withdraw_and_wait(self) -> None:
        """Withdraw concurrently, then wait for survivor admission and Master topology."""
        criteria = self.criteria
        initial = self.evidence["initial_engines"]
        removed = initial[criteria["target_p"] :]
        self.evidence["affinity_transition"] = _affinity_transition(self.ctx, criteria["post_affinity_ms"], self.deadline)
        self.event("withdraw_start")
        # Every removal withdraws discovery before waiting. Parallel requests do
        # not serialize the drain periods; the Java mutation lock only covers rewrite.
        self.pool = ThreadPoolExecutor(max_workers=len(removed), thread_name_prefix="scale-in")
        self.futures = [
            self.pool.submit(
                mock_json,
                self.ctx.ops,
                "remove_engine",
                self.deadline,
                dict(engine=n, mode=criteria["removal_mode"], drain_timeout_ms=criteria["drain_timeout_ms"]),
            )
            for n in removed
        ]
        topology_end = self.ctx.clock() + criteria["topology_timeout_s"]
        for row in poll_samples(self.deadline, criteria["sample_s"], self.sample):
            if topology_ready(row, self.evidence["survivors"], self.evidence["initial_engines"]):
                self.evidence["post_start"] = row["t"]
                self.event("target_topology_observed", row)
                break
            if self.ctx.clock() >= topology_end:
                raise ValueError("target topology did not converge")

    def observe_survivors(self) -> None:
        """Keep sampling through the full post-removal window before stopping sends."""
        criteria = self.criteria
        end = self.evidence["post_start"] + criteria["observe_s"]
        for row in poll_samples(self.deadline, criteria["sample_s"], self.sample, until=self.origin + end):
            pass
        self.evidence["post_end"] = row["t"]
        self.event("observation_end", row)
        self.flow.stop_sending(self.deadline)
        self.event("sending_stopped")

    def finish_removals(self) -> None:
        """Join bounded drains and preserve every outcome, including failed observation."""
        if self.pool:
            # The underlying HTTP calls and server drains are deadline-bounded.
            self.pool.shutdown(wait=True, cancel_futures=True)
        # Preserve individual drain outcomes even if topology observation failed.
        self.evidence["removals"] = []
        for index, future in enumerate(self.futures):
            try:
                self.evidence["removals"].append(future.result())
            except Exception as exc:
                self.evidence.setdefault("removal_errors", []).append(dict(index=index, error=str(exc)))
        self.evidence["measurement_scope"] = scope_contract(self.evidence)
        write_json(self.path, self.evidence)

    def run(self) -> StageOutput:
        from cases.cache_scale_in import analysis as cache_gate

        try:
            self.evidence["provenance"].update(
                gate_provenance(self.ctx, self.flow, source_files=self.ctx.services.source_files(), analyzer_file=cache_gate.__file__)
            )
            self.wait_for_baseline()
            self.withdraw_and_wait()
            self.observe_survivors()
        except Exception as exc:
            self.evidence["errors"].append(str(exc))
        finally:
            self.finish_removals()
        return StageOutput(
            {"evidence": self.ctx.register_resource("gate_evidence", self.evidence, historical=True)}, artifacts=[str(self.path)]
        )


def observe(context, params: dict, deadline) -> StageOutput:
    """Collect baseline, intervention and survivor evidence without deciding the gate."""
    return _ScaleInObservation(context, params, deadline).run()


def check(ctx, params, deadline):
    evidence = ctx.resource(params["evidence"], "gate_evidence")
    flow = ctx.resource(params["flow"], "java_flow")
    # Keep the original poll ledger for topology and latency diagnostics.
    from cases.cache_scale_in.observation import measurement_samples
    from file_digest import sha256_file

    control_path = ctx.artifact_dir / "cache-control-evidence.json"
    write_json(control_path, evidence)
    evidence["control_evidence"] = dict(file=control_path.name, sha256=sha256_file(control_path))
    try:
        evidence["samples"] = measurement_samples(
            ctx.monitor, evidence, deadline, evidence["capture_limits"], directory=ctx.artifact_dir / "cache-measurement"
        )
        evidence["gate_sample_source"] = "prometheus_raw_scrape"
        evidence["raw_evidence"] = dict(
            path="cache-measurement/manifest.json", sha256=sha256_file(ctx.artifact_dir / "cache-measurement/manifest.json")
        )
    except TimeoutError:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        evidence["errors"].append("raw measurement unavailable: " + str(exc))
    snapshot = flow.evidence_snapshot()
    # Terminal request failures do not decide whether survivor cache counters
    # are measurable. Issued sends still need complete accounting for QPS.
    if len(snapshot["issued"]) != snapshot.get("status", {}).get("submitted"):
        evidence["errors"].append("issued send accounting incomplete")
    else:
        try:
            align_send_counters(evidence, snapshot["issued"])
        except ValueError as exc:
            evidence["errors"].append(str(exc))
    attribute_client(evidence, snapshot)
    evidence["client_attribution"]["artifacts"] = [
        dict(path=str(path.resolve()), sha256=sha256_file(path))
        for path in (flow.directory / "client_lifecycle.jsonl", flow.directory / "control/status.json")
        if path.is_file()
    ]
    times = [r["epoch_s"] for r in evidence["samples"]]
    issued = [r for r in snapshot["issued"] if times and times[0] * 1000 <= r.get("send_start_epoch_ms", -1) <= times[-1] * 1000]
    lags = [r.get("pacing_lag_ms") for r in issued]
    if lags and all(type(v) in (int, float) and math.isfinite(v) for v in lags):
        evidence["max_pacing_lag_ms"] = max(lags)
    write_json(ctx.artifact_dir / "cache-gate-evidence.json", evidence)
    evidence["curve_source"] = "prometheus"
    from artifacts.gate import commit_gate
    from cases.cache_scale_in.analysis import analyze

    result, path = commit_gate(ctx.artifact_dir, "cache", evidence, analyze)
    return StageOutput(
        checks=[gate_check("cache_stability", result, path, "valid survivor cache measurement without sustained hit collapse")],
        artifacts=[
            str(ctx.artifact_dir / name)
            for name in ("cache-control-evidence.json", "cache-gate-evidence.json", "cache-gate-result.json", "cache-gate-manifest.json")
        ],
    )


HANDLERS = [
    StageHandler("cache_scale_in_observe", validate, observe, {"evidence": "gate_evidence"}),
    StageHandler(
        "cache_scale_in_check",
        partial(reference_parameters, resources={"flow": "java_flow", "evidence": "gate_evidence"}),
        check,
        {},
        checks=frozenset({"cache_stability"}),
    ),
]
