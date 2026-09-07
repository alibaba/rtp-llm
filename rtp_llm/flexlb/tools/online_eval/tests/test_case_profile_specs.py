"""Effective profile axes and environment isolation survive case migration."""

import json
import sys
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexlb_cfg import render_env
from flexlb_ft.cases import elastic
from flexlb_ft.harness import _elastic_env
from flexlb_ft.support import admission, engine_fault, master, priority, status


class CaseProfileSpecsTest(unittest.TestCase):
    def test_live_preemption_keeps_profile_specific_window(self):
        for factory in (priority._pq_live_spec, priority._dr_live_spec):
            for profile in ("batch-window", "single-batch"):
                spec = factory(SimpleNamespace(profile=profile))
                config = json.loads(
                    render_env(spec.master_profile, spec.config_overrides)
                )
                decision = config["scheduler"]["decision"]
                self.assertEqual(config["dispatcher"]["type"], "BATCH")
                if profile == "batch-window":
                    self.assertEqual(decision["type"], "FIXED_WINDOW")
                    self.assertEqual(decision["maxRequests"], 32)
                    self.assertEqual(decision["maxCollectionWaitMs"], 400)
                else:
                    self.assertEqual(decision, {"type": "SINGLE"})

    AXES = {
        "batch-window": ("FIXED_WINDOW", "BATCH"),
        "single-nonbatch": ("SINGLE", "NON_BATCH"),
        "single-batch": ("SINGLE", "BATCH"),
        "window-nonbatch": ("FIXED_WINDOW", "NON_BATCH"),
    }

    def test_specs_render_requested_axes_without_starting_processes(self):
        factories = [
            admission._slo_spec,
            admission._capacity_spec,
            admission._prefill_park_spec,
            admission._decode_park_spec,
            admission._incomer_spec,
            partial(admission._batcher_queue_spec, queue_timeout_ms=1500),
            admission._pool_wait_spec,
            admission._waiting_cap_spec,
            admission._lack_mem_spec,
            partial(
                admission._regroup_spec, max_batch_tokens=4096, max_batch_requests=2
            ),
            engine_fault._fault_spec,
            engine_fault._recovery_spec,
            master._quota_spec,
            status._status_spec,
            elastic._pending_drain_spec,
            elastic._full_shrink_spec,
            partial(elastic._skew_spec, variant="hot"),
            elastic._transient_spec,
            elastic._steady_recovery_spec,
        ]
        for profile, (decision, dispatcher) in self.AXES.items():
            ctx = SimpleNamespace(profile=profile)
            for factory in factories:
                with self.subTest(profile=profile, factory=factory):
                    spec = factory(ctx)
                    config = json.loads(
                        render_env(spec.master_profile, spec.config_overrides)
                    )
                    self.assertEqual(config["scheduler"]["decision"]["type"], decision)
                    self.assertEqual(config["dispatcher"]["type"], dispatcher)
                    if dispatcher == "NON_BATCH":
                        self.assertNotIn(
                            "maxInflightBatchesPerPrefillWorker", config["dispatcher"]
                        )

    def test_shared_fault_environment_uses_profile_and_keeps_ttl_budget(self):
        for profile, (decision, dispatcher) in self.AXES.items():
            ctx = SimpleNamespace(
                profile=profile,
                env_manager=SimpleNamespace(ensure=lambda spec: spec),
                engine_ops=lambda env: None,
            )
            spec, _ = _elastic_env(ctx)
            config = json.loads(render_env(spec.master_profile, spec.config_overrides))
            self.assertEqual(config["scheduler"]["decision"]["type"], decision)
            self.assertEqual(config["dispatcher"]["type"], dispatcher)
            self.assertNotIn("queueTimeoutMs", config["scheduler"])

    def test_mainline_elastic_isolation_discriminators_are_preserved(self):
        for profile in self.AXES:
            ctx = SimpleNamespace(profile=profile)
            specs = [
                elastic._full_shrink_spec(ctx),
                elastic._skew_spec(ctx, "hot"),
                elastic._skew_spec(ctx, "cold"),
                elastic._transient_spec(ctx),
                elastic._steady_recovery_spec(ctx),
            ]
            self.assertEqual(len({s.fingerprint() for s in specs}), len(specs))
            self.assertEqual(
                [s.master_env for s in specs],
                [
                    {"FLEXLB_FT_SPEC_ID": "kv_full_shrink"},
                    {"FLEXLB_FT_SPEC_VARIANT": "hot"},
                    {"FLEXLB_FT_SPEC_VARIANT": "cold"},
                    {"FLEXLB_FT_SPEC_ID": "transient_bound"},
                    {"FLEXLB_FT_SPEC_ID": "steady_recovery"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
