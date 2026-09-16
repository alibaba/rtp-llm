"""Execute shared-holder programs using a small cache/transport model, not Java."""

import copy
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import balance, elastic, engine_control, kv
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_kv import ROOT, Clock


class CacheModel:
    def __init__(
        self, collapse=False, foreign=False, missing=False, retain_eviction=False
    ):
        self.rid = 0
        self.collapse, self.foreign, self.missing, self.retain_eviction = (
            collapse,
            foreign,
            missing,
            retain_eviction,
        )
        self.keys, self.last, self.issued = {}, {}, Counter()
        self.evictions = 0
        self.removed = []

    def setup(self, ctx, environment, deadline):
        self.keys = {f"prefill-{i}": set() for i in range(environment["n_prefill"])}
        self.clock = ctx.clock
        return object(), SimpleNamespace(master_http_port=12345)

    def teardown(self, ctx, deadline):
        pass

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            rows = [
                dict(
                    name=name,
                    role="prefill",
                    grpc_addr=f'host:{100+int(name.rsplit("-",1)[1])}',
                    stopped=False,
                    completed=self.issued[name],
                    waiting=0,
                    running=1,
                    cache_key_set=sorted(keys),
                )
                for name, keys in self.keys.items()
            ]
            if self.missing:
                rows[0].pop("cache_key_set")
            return dict(engines=rows)
        name = body["engine"]
        port = 100 + int(name.rsplit("-", 1)[1])
        if endpoint == "cache_evict":
            self.evictions += 1
            if not self.retain_eviction:
                self.keys[name].difference_update(body["keys"])
        if endpoint == "remove_engine":
            self.removed.append(name)
            del self.keys[name]
            return dict(
                status="ok",
                engine=name,
                port=port,
                action="removed",
                mode="graceful",
                drained=True,
            )
        return dict(status="ok", engine=name, port=port)

    def master_info(self, *args, **kwargs):
        return 200, dict(worker_summary=dict(PREFILL=dict(alive=len(self.keys))))

    def select(self, wanted):
        def score(name):
            recently_issued = self.clock() - self.last.get(name, -100) < 2
            return (
                int(recently_issued),
                -len(wanted & self.keys[name]),
                self.issued[name],
                name,
            )

        return min(self.keys, key=score)

    def start_requests(self, ctx, params, deadline):
        records = ClientRecords(ctx.env_epoch)
        records.rows = []
        for _ in range(params["count"]):
            self.rid += 1
            wanted = set(params["block_keys"])

            name = self.select(wanted)
            if self.collapse and self.evictions >= 2:
                name = sorted(self.keys)[0]
            self.keys[name].update(wanted)
            self.last[name] = self.clock()
            self.issued[name] += 1
            row = records.issue(self.rid, ctx.clock)
            records.rows.append(row)
            address = f'host:{100+int(name.rsplit("-",1)[1])}'
            if self.foreign and self.rid == 3:
                address = "foreign:999"
            row["request_shape"] = {
                k: params[k] for k in ("input_len", "output_len", "block_keys")
            }
            records.update(row, schedule=dict(status="OK"), prefill_addr=address)
            if params.get("post_issue_delay_s"):
                deadline.sleep(params["post_issue_delay_s"])
        return ctx.register_resource("requests", records)

    def wait_requests(self, ctx, records, deadline):
        deadline.sleep(2.1)
        for row in records.rows:
            records.update(
                row,
                business_finished=True,
                stream=dict(status="OK"),
                transport_terminal_s=ctx.clock(),
                consumer_exit_s=ctx.clock(),
                consumer_done=True,
                consumer_completion_verified=True,
            )
        return dict(completed=True, error_count=0)


class GlobalKvTests(unittest.TestCase):
    def plans(self):
        return compile_scenarios(
            load_scenarios(ROOT / "scenarios/kv/cache_global_holders.yaml"),
            handlers=handlers(),
        )

    def run_plan(self, plan, backend):
        clock = Clock()
        with ExitStack() as stack:
            root = stack.enter_context(tempfile.TemporaryDirectory())
            for module in (kv, balance, elastic, engine_control):
                stack.enter_context(
                    patch.object(module, "_http", side_effect=backend.http)
                )
            stack.enter_context(
                patch(
                    "flexlb_test_framework.harness.http_post_json",
                    side_effect=backend.master_info,
                )
            )
            return execute_instance(
                plan,
                backend,
                artifact_dir=root,
                handlers=handlers(),
                clock=clock,
                sleeper=clock.sleep,
            )

    def test_twenty_full_programs_preserve_all_five_contracts(self):
        plans = self.plans()
        self.assertEqual(len(plans), 20)
        self.assertEqual(len({p["variant_id"] for p in plans}), 10)
        for plan in plans:
            with self.subTest(instance=(plan["variant_id"], plan["profile"])):
                model = CacheModel()
                result = self.run_plan(plan, model)
                self.assertEqual(
                    result["status"],
                    "PASS",
                    (
                        result["error"],
                        [
                            (s["id"], s["status"])
                            for s in result["stages"]
                            if s["status"] not in ("PASS",)
                        ],
                    ),
                )
                self.assertEqual(
                    len(model.removed), int(plan["variant_id"].startswith("down_"))
                )
                if plan["variant_id"].startswith(("release_", "mixed_")):
                    wave = next(s for s in plan["stages"] if s["id"] == "wave")
                    self.assertEqual(wave["params"]["post_issue_delay_s"], 0.12)
                    self.assertEqual(wave["params"]["count"], 20)

    def test_faults_do_not_pass_on_control_ack_or_partial_evidence(self):
        cases = [
            ("release_", CacheModel(collapse=True), "FAIL", "spread"),
            ("shared_", CacheModel(foreign=True), "FAIL", "holder_union"),
            ("redirect_", CacheModel(retain_eviction=True), "FAIL", "release_first"),
            ("shared_", CacheModel(missing=True), "ERROR", "seed_quiet"),
        ]
        for prefix, model, expected, stage in cases:
            with self.subTest(prefix=prefix, stage=stage):
                plan = next(
                    p for p in self.plans() if p["variant_id"].startswith(prefix)
                )
                result = self.run_plan(plan, model)
                self.assertEqual(result["status"], expected, result["error"])
                self.assertEqual(
                    next(s for s in result["stages"] if s["id"] == stage)["status"],
                    expected,
                )

    def test_duplicate_or_unobserved_expected_holders_are_errors(self):
        for expected in (
            [{"$ref": "stages.first_holder.output.engine"}] * 2,
            ["prefill-31"],
        ):
            plan = copy.deepcopy(self.plans()[0])
            next(s for s in plan["stages"] if s["id"] == "shared_holders")["params"][
                "holders"
            ] = expected
            result = self.run_plan(plan, CacheModel())
            self.assertEqual(result["status"], "ERROR")
            self.assertIn("distinct observed", result["error"])


if __name__ == "__main__":
    unittest.main()
