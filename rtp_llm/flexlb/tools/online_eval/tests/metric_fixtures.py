"""Synthetic archives explicitly freeze the metric definitions they exercise."""

import hashlib
import json
import random
from pathlib import Path

from monitoring.archive import export_metrics
from scenario.registry import load_metric_plan as load_plan
from scenario.registry import view as view
from scenario.registry import write_views as write_views
from traffic.codecs import decode_model
from traffic.prefix_lineage import encode
from traffic.synthetic.derivation import derive_parameters


def freeze_metrics(directory, case):
    plan = load_plan(case + ".yaml")
    if case == "master_ha_failover":
        from monitoring.collection_plan import select_plan
        from scenario.registry import view as view

        gates = [identity for identity, spec in plan["produced"].items() if spec["value_kind"] == "scalar"]
        plan = select_plan(plan, gates, [view(case + ".yaml")])
    for path in Path(directory).glob("telemetry/*/queries.json"):
        data = json.loads(path.read_text())
        if "metric_plan" not in data:
            data.update(metric_plan=plan, query_plan=case + ".yaml")
            path.write_text(json.dumps(data))
    return export_metrics(directory, plan)


def selected_sources(plan):
    from monitoring.sources import selected_sources as select
    from scenario.registry import registry

    return select(plan, registry().sources)


def collection_plan(plan):
    from monitoring.collection_plan import collection_plan as collect
    from scenario.registry import registry

    return collect(plan, registry().sources)


def instance_plan(instance):
    from monitoring.collection_plan import selected_plan as read
    from scenario.registry import registry

    return read(instance.get("implementation", {}), instance["execution"]["monitoring"], registry().sources)


def frozen_plan(name, plan):
    from monitoring.collection_plan import frozen_plan as freeze
    from scenario.registry import registry

    return freeze(name, plan, registry().sources)


def publish(*args, producer, **kwargs):
    from monitoring.publication import publish as write
    from scenario.registry import registry

    kwargs.setdefault("describe", registry().metric_contracts.get(producer))
    return write(*args, producer=producer, **kwargs)


def PrometheusSession(*args, **kwargs):
    from monitoring.session import PrometheusSession as Session
    from scenario.registry import load_metric_plan, registry

    kwargs.setdefault("sources", registry().sources)
    if kwargs.get("query_plan") and "metric_plan" not in kwargs:
        kwargs["metric_plan"] = load_metric_plan(kwargs["query_plan"])
    return Session(*args, **kwargs)


def fidelity_fixture():
    rng = random.Random(17)
    events = [[i, 512 * 512, -1, 0] for i in range(32)]
    for i in range(32, 4032):
        blocks = rng.randint(20, 500)
        shared = 0 if i % 10 == 0 else int(blocks * 0.85)
        events.append([i, blocks * 512, i % 32 if shared else -1, shared])
    raw = encode(events, dict(source_start=0, source_end=4031))
    return raw, derive_parameters(decode_model(raw), {}, None, model_sha256=hashlib.sha256(raw).hexdigest())


def diagnostic_payload():
    return dict(
        schema_version=2,
        instance_id="process-1",
        snapshot_id="sample-1",
        status="ok",
        capture_started_at_ms=1,
        capture_finished_at_ms=2,
        capabilities={"workers": dict(version=1, features=["poll_health", "applied_observation", "cache_progress"])},
        components={
            "workers": dict(
                status="ok",
                capture_started_at_ms=1,
                capture_finished_at_ms=2,
                coverage=dict(scope="prefill_decode_endpoint_registry", complete=True, truncated=False, returned_count=1),
                rows=[
                    dict(
                        worker_id="prefill:a",
                        address="a",
                        role="prefill",
                        generation="9007199254740993",
                        lifecycle_state="active",
                        last_reported_alive=True,
                        poll_response_age_ms=10,
                        applied_observation_age_ms=20,
                        applied_status_version="1",
                        received_cache_version="2",
                        indexed_cache_version="2",
                        index_initialized=True,
                        cached_key_count=3,
                    )
                ],
            )
        },
    )


def workload_payload(series=None, gates=None):
    return dict(
        id="example",
        status="PASS",
        workload={"runtime_validity": "VALID"},
        implementation={},
        configuration={},
        configuration_sha256=None,
        clock_anchor={"epoch_s": 10},
        request_sources=[],
        series=series or {},
        statistic_sources={k: dict(unit="count") for k in series or {}},
        checks=[dict(stage="observe", id="cache", status="PASS")],
        iterations=[],
        traffic_manifests=[],
        gate_reports=gates or [],
    )


def performance_evidence():
    # Small arithmetic fixture only; never a runnable benchmark.
    c = {
        "warmup_s": 5,
        "measure_s": 30,
        "sample_s": 1,
        "max_gap_s": 3,
        "qps": 10,
        "qps_tolerance": 0.1,
        "min_requests": 250,
        "max_pacing_lag_ms": 100,
        "min_input_tps": 9000,
        "min_output_tps": 70,
        "min_goodput_rps": 9,
        "min_slo_fraction": 0.95,
        "max_error_rate": 0,
        "max_ttft_p99_ms": 1000,
        "max_e2e_p99_ms": 2000,
        "max_tpot_p99_ms": 150,
        "slo_ttft_ms": 1000,
        "slo_e2e_ms": 2000,
        "slo_tpot_ms": 150,
        "max_inflight_growth_rps": 0.2,
    }
    c.update(measure_s=10, min_requests=80, warmup_s=0)
    issued = []
    records = []
    for i in range(100):
        r = dict(rid=str(i), send_start_epoch_ms=100000 + i * 100, input_len=1024, output_len=8, pacing_lag_ms=0)
        issued.append(r)
        records.append(dict(r, status="ok", total_ms=100, ttft_ms=50, observed_output_tokens=8))
    return dict(
        performance_evidence_schema_version=2,
        criteria=c,
        errors=[],
        window=dict(start_epoch_ms=100000, end_epoch_ms=110000),
        samples=[dict(epoch_ms=100000 + i * 1000) for i in range(11)],
        flow=dict(complete=True, errors=[], issued=issued, records=records),
        provenance=dict(
            env_epoch=1,
            instance="master_performance::default::batch-window",
            configuration_sha256="f" * 64,
            master_artifact=dict(jar_sha256="a" * 64),
            mock_jar_sha256="b" * 64,
            actual_master_config=dict(dispatcher=dict(type="BATCH")),
            performance=dict(prefill=dict(fixed_ms=100)),
            topology=dict(prefill=2, decode=4),
            capacity=dict(prefill_cache_blocks=1000, decode_cache_blocks=1000),
            trace=dict(sha256="c" * 64, workload_sha256="d" * 64),
            client_environment=dict(FETCH_OUTPUT_STREAM="true"),
            analyzer_sha256="e" * 64,
        ),
    )


def cache_evidence(hit=0.8):
    rows = []
    for t in range(81):
        engines = {}
        for name in ["p0", "p1"] if t <= 20 else ["p0"]:
            engines[name] = dict(
                grpc_addr=name,
                hit_tokens_total=800 * min(t, 20) + hit * 1000 * max(t - 20, 0),
                context_tokens_total=1000 * t,
                context_requests_total=10 * t,
                cache_evictions=t,
                waiting=100 if t > 20 else 0,
                running=1,
                prefill_ms_avg=200,
                worker_status_qps=50,
                worker_status_rt_us=100,
            )
        rows.append(
            dict(
                t=t, epoch_s=t + 1000, engines=engines, started=t * 10, terminal=t * 8, master_p=2 if t < 20 else 1, waiting=100, running=1
            )
        )
    return dict(
        provenance=dict(instance="cache_scale_in::default::batch-window", env_epoch=1),
        samples=rows,
        criteria=dict(
            max_gap_s=2,
            baseline_min_hit=0.7,
            min_completed=10,
            qps=10,
            qps_tolerance=0.1,
            absolute_min_hit=0.4,
            max_drop=0.25,
            observe_s=60,
            window_s=10,
            step_s=1,
            sustain_s=15,
        ),
        baseline_start=10,
        baseline_end=20,
        initial_engines=["p0", "p1"],
        survivors=["p0"],
        post_start=20,
        post_end=80,
        max_pacing_lag_ms=0,
        events=[dict(id="withdraw_start", t=20, epoch_s=1020)],
    )
