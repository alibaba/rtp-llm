"""Live BATCH eviction fixtures preserve deferred and serial survivor consumption."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario.actions import priority
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_priority import Clean


class LiveBackend(programs.Backend):
    def __init__(
        self,
        victim_code=8400,
        saw_victim=False,
        missing_metric=False,
        wrong_tags=False,
        missing_owner=False,
    ):
        super().__init__(terminals={3: victim_code})
        self.ops.batch = True
        self.missing_owner = missing_owner
        self.saw_victim, self.missing_metric, self.wrong_tags = (
            saw_victim,
            missing_metric,
            wrong_tags,
        )
        self.fetches = []
        fetch = self.ops.fetch

        def wrapped(req, timeout):
            self.fetches.append(
                dict(
                    rid=req["request_id"],
                    schedule_count=len(self.ops.responses),
                    timeout=timeout,
                )
            )
            return fetch(req, timeout)

        self.ops.fetch = wrapped

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_perf":
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        lifecycle = {str(i): dict(running_ms=i * 4000) for i in (1, 2, 4)}
        if self.saw_victim:
            lifecycle["3"] = dict(running_ms=3000)
        result = dict(
            engines=[
                dict(
                    name=f"{role[0]}{i}",
                    role=role,
                    stopped=False,
                    grpc_addr=f"{role}{i}",
                    running=1 if role == "prefill" else 0,
                    waiting=0,
                    inflight=0,
                    leak_detected=False,
                    request_lifecycle=lifecycle if role == "prefill" else {},
                )
                for role, count in [("prefill", 1), ("decode", 4)]
                for i in range(count)
            ]
        )

        if self.missing_owner:
            result["engines"].pop()
        return result

    def metrics(self, ctx, server, endpoint, deadline, **kwargs):
        if self.missing_metric:
            return 200, "jvm_threads_live_threads 1\n"
        incoming = "50" if self.wrong_tags else "70"
        return (
            200,
            f'flexlb_auto_tpm_victim_count{{stage="prefill_queued",victim_priority="30",incoming_priority="{incoming}"}} 1\nflexlb_auto_tpm_priority_preempt_count{{stage="prefill_queued"}} 1\n',
        )
