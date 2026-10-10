"""Compile the shipped elastic YAML and execute its full stage/check contract.

Only external resources are faked. Windows, the high-hit guard, final verdict,
reference validation, output validation and fail-stop behavior use real code.
"""

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario import compile_scenarios
from scenario.actions import elastic as e
from scenario.contracts import CheckResult, StageOutput
from scenario.loader import load_scenarios
from scenario.runtime import execute_instance


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Backend:
    def setup(self, ctx, environment, deadline):
        return object(), NS(master_http_port=1)

    def teardown(self, ctx, deadline):
        pass


class ElasticRuntimeTests(unittest.TestCase):

    def source(self):
        source = load_scenarios(ROOT / "config/scenarios/elastic_lifecycle.yaml")[0][1]
        source["variants"] = [
            v for v in source["variants"] if v["id"].startswith("kv_skew_")
        ]
        return source

    def run_pilot(self, rate=1.0, flow_failed=False):
        clock = Clock()
        mutated = []

        def external(action, ctx, params, deadline):
            register = ctx.register_resource
            records = e.ClientRecords(ctx.env_epoch)
            for rid in range(20):
                record = records.issue(rid, clock)
                record["rpc_observation"] = dict(complete=True, legacy_success=True)
                records.update(
                    record,
                    business_finished=True,
                    schedule=dict(status="OK"),
                    stream=dict(status="OK"),
                    transport_terminal_s=clock(),
                    consumer_exit_s=clock(),
                    consumer_done=True,
                    consumer_completion_verified=True,
                )
            if action == "elastic_seed":
                return StageOutput(
                    output=dict(
                        requests=register("requests", records),
                        families=register("snapshot", dict(hot="p0", cold="p1")),
                    ),
                    checks=[
                        CheckResult("seed_success", "PASS"),
                        CheckResult("skew", "PASS"),
                    ],
                )
            if action == "elastic_metrics_start":
                samples = [
                    dict(
                        time_s=t,
                        engines={
                            name: dict(
                                role="prefill",
                                mock_engine_cache_key_hits_total=t * 10 * rate,
                                mock_engine_cache_keys_requested_total=t * 10,
                                rtp_llm_wait_stream_size=1,
                                rtp_llm_kv_cache_pool_total_blocks=100,
                                rtp_llm_kv_cache_pool_available_blocks=20,
                            )
                            for name in ("p0", "p1")
                        },
                    )
                    for t in range(121)
                ]
                metrics = NS(
                    skew_started_s=clock(),
                    snapshot=lambda: dict(
                        samples=samples, errors=[], env_epoch=ctx.env_epoch
                    ),
                )
                return StageOutput(
                    output=dict(observation=register("observation", metrics))
                )
            if action == "elastic_flow_start":
                return StageOutput(output=dict(flow=register("flow", records)))
            if action == "elastic_scale":
                mutated.append(clock())
                return StageOutput(
                    output=dict(
                        scale=register(
                            "snapshot",
                            dict(
                                started_s=clock(),
                                ended_s=clock(),
                                response=dict(drained=False),
                            ),
                        ),
                        drained=True,
                    ),
                    checks=[CheckResult("pre_scale_skew", "PASS")],
                )
            if action == "elastic_flow_stop":
                result = e.completeness(records.snapshot_records())
                if flow_failed:
                    result.update(zero_errors=False, failed_request_ids=[42])
                return StageOutput(
                    output=dict(
                        complete=True, issued=20, result=register("snapshot", result)
                    )
                )
            if action == "elastic_recovery":
                return StageOutput(
                    output=dict(
                        requests=register("requests", records), success_rate=1.0
                    )
                )
            raise AssertionError(action)

        handlers = {}
        for handler in e.HANDLERS:
            if handler.name in {
                "elastic_baseline",
                "elastic_window",
                "elastic_verdict",
            }:
                handlers[handler.name] = handler
            else:
                handlers[handler.name] = replace(
                    handler,
                    execute=lambda ctx, p, d, name=handler.name: external(
                        name, ctx, p, d
                    ),
                )
        plans = compile_scenarios([("pilot.yaml", self.source())], handlers=handlers)
        results = []
        with patch(
            "runtime.harness.http_post_json",
            return_value=(
                200,
                dict(worker_summary={"PREFILL": dict(discovered=1, alive=1)}),
            ),
        ):
            for plan in plans:
                clock.now = 0
                with tempfile.TemporaryDirectory() as root:
                    results.append(
                        execute_instance(
                            plan,
                            Backend(),
                            handlers=handlers,
                            artifact_dir=root,
                            clock=clock,
                            sleeper=clock.sleep,
                        )
                    )
        return results, mutated
