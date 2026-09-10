import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import balance, engine_control, kv
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Backend:
    def __init__(self, wrong_carve=False, continuation_hits=5, missing_keys=False):
        self.rid = 0
        self.b_keys = None
        self.keys = {"prefill-0": set(), "prefill-1": set()}
        self.wrong_carve, self.continuation_hits, self.missing_keys = (
            wrong_carve,
            continuation_hits,
            missing_keys,
        )

    def setup(self, ctx, environment, deadline):
        return object(), object()

    def teardown(self, ctx, deadline):
        pass

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            rows = [
                dict(
                    name=name,
                    role="prefill",
                    grpc_addr=f"host:{i+100}",
                    stopped=False,
                    completed=2,
                    waiting=0,
                    running=1,
                    cache_key_set=sorted(self.keys[name]),
                )
                for i, name in enumerate(self.keys)
            ]
            if self.missing_keys:
                rows[0].pop("cache_key_set")
            return {"engines": rows}
        name = body["engine"]
        if endpoint == "cache_evict":
            self.keys[name].difference_update(body["keys"])
            if self.wrong_carve and name == "prefill-0":
                self.keys[name].discard(810000)
        return dict(status="ok", engine=name, port=100 if name == "prefill-0" else 101)

    def start_requests(self, ctx, params, deadline):
        records = ClientRecords(ctx.env_epoch)
        records.rows = []
        for _ in range(params["count"]):
            self.rid += 1
            mode = ctx.instance["variant_id"].split("_")[0]
            if mode == "evict":
                second = (
                    self.rid > 2
                    and self.rid % 2 == 0
                    and not getattr(self, "collapse", False)
                )
            elif mode == "isolation":
                second = self.rid == 2 or params["block_keys"] == self.b_keys
                if self.rid == 2:
                    self.b_keys = list(params["block_keys"])
            else:
                second = self.rid == 2 or 3 <= self.rid < 3 + self.continuation_hits
            name = "prefill-1" if second else "prefill-0"
            self.keys[name].update(params["block_keys"])
            if mode == "isolation" and self.rid == 20 and getattr(self, "leak", False):
                self.keys["prefill-1"].add(811000)
            row = records.issue(self.rid, ctx.clock)
            records.rows.append(row)
            records.update(
                row,
                schedule=dict(status="OK"),
                prefill_addr="host:101" if second else "host:100",
            )
            if params.get("post_issue_delay_s"):
                deadline.sleep(params["post_issue_delay_s"])
        if params["consume"] == "immediate":
            self.wait_requests(ctx, records, deadline)
        return ctx.register_resource("requests", records)

    def wait_requests(self, ctx, records, deadline):
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


class KvScenarioTests(unittest.TestCase):
    def test_landing_can_reuse_identity_without_reusing_live_cache_observations(self):
        record = {"schedule": {"status": "OK"}, "prefill_addr": "host:100"}
        endpoint = {
            "name": "prefill-0",
            "role": "prefill",
            "grpc_addr": "host:100",
            "stopped": False,
        }
        records = NS(snapshot_records=lambda: [record])
        with tempfile.TemporaryDirectory() as tmp, patch.object(kv, "_http") as http:
            ctx = NS(
                resource=lambda ref, kind: (
                    records
                    if kind == "requests"
                    else {"engines": {"prefill-0": endpoint}}
                ),
                artifact_dir=Path(tmp),
            )
            params = {"requests": {}, "identity_snapshot": {}, "phase": "scheduled"}
            result = kv.landing(ctx, params, NS(check=lambda: None))
            self.assertEqual("prefill-0", result.output["engine"])
            http.assert_not_called()
            record["prefill_addr"] = "unknown:200"
            with self.assertRaises(ValueError):
                kv.landing(ctx, params, NS(check=lambda: None))

    def plans(self, grade="normal", prefix="continuity"):
        return [
            plan
            for plan in compile_scenarios(
                load_scenarios(ROOT / "scenarios/kv"), handlers=handlers(), grade=grade
            )
            if plan["variant_id"].startswith(prefix)
        ]

    def test_eviction_and_isolation_programs_execute_all_profiles(self):
        for mode in ("evict", "isolation"):
            plans = self.plans(prefix=mode)
            self.assertEqual(len(plans), 4)
            for plan in plans:
                result = self.run_plan(plan, Backend())
                self.assertEqual(result["status"], "PASS", result["error"])
                if mode == "evict":
                    wave = next(row for row in plan["stages"] if row["id"] == "wave")
                    self.assertEqual(wave["params"]["post_issue_delay_s"], 0.12)
                    self.assertEqual(wave["params"]["count"], 20)

    def test_stale_stickiness_and_foreign_cache_admission_fail_distinct_checks(self):
        for mode, attr, failed_stage in (
            ("evict", "collapse", "spread"),
            ("isolation", "leak", "final_b_isolation"),
        ):
            backend = Backend()
            setattr(backend, attr, True)
            result = self.run_plan(self.plans(prefix=mode)[0], backend)
            self.assertEqual(result["status"], "FAIL", result["error"])
            self.assertEqual(
                next(row for row in result["stages"] if row["id"] == failed_stage)[
                    "status"
                ],
                "FAIL",
            )

    def run_plan(self, plan, backend):
        clock = Clock()
        with tempfile.TemporaryDirectory() as root, patch.object(
            kv, "_http", side_effect=backend.http
        ), patch.object(
            engine_control, "_http", side_effect=backend.http
        ), patch.object(
            balance, "_http", side_effect=backend.http
        ):
            return execute_instance(
                plan,
                backend,
                artifact_dir=root,
                handlers=handlers(),
                clock=clock,
                sleeper=clock.sleep,
            )

    def test_full_shipped_continuity_program_uses_actual_handlers_all_profiles(self):
        plans = self.plans()
        self.assertEqual(len(plans), 4)
        for plan in plans:
            self.assertEqual(
                next(row for row in plan["stages"] if row["id"] == "first_pending")[
                    "timeout_s"
                ],
                6,
            )
            result = self.run_plan(plan, Backend())
            self.assertEqual(result["status"], "PASS", result["error"])
            self.assertEqual(len(result["stages"]), 31)
            for sid, expected in (("gap_prefix", 1), ("tail_prefix", 8)):
                self.assertEqual(
                    next(row for row in result["stages"] if row["id"] == sid)["output"][
                        "run"
                    ],
                    expected,
                )

    def test_wrong_prefix_shape_fails_and_blocks_continuations(self):
        backend = Backend(wrong_carve=True)
        result = self.run_plan(self.plans()[0], backend)
        self.assertEqual(result["status"], "FAIL", result["error"])
        self.assertEqual(backend.rid, 2)
        self.assertEqual(
            next(row for row in result["stages"] if row["id"] == "gap_prefix")[
                "status"
            ],
            "FAIL",
        )

    def test_missing_keys_are_unavailable_not_empty_or_quiet(self):
        result = self.run_plan(self.plans()[0], Backend(missing_keys=True))
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("typed cache key set", result["error"])
        self.assertEqual(
            next(row for row in result["stages"] if row["id"] == "seed_quiet")[
                "status"
            ],
            "ERROR",
        )

    def test_measured_concentration_respects_selected_grade(self):
        for grade, expected in (
            ("strict", "FAIL"),
            ("normal", "FAIL"),
            ("loose", "PASS"),
        ):
            result = self.run_plan(self.plans(grade)[0], Backend(continuation_hits=3))
            self.assertEqual(result["status"], expected, result["error"])
            affinity = next(row for row in result["stages"] if row["id"] == "affinity")
            self.assertEqual(affinity["output"]["share"], 0.6)
            self.assertEqual(result["grade"], grade)

    def test_duplicate_cohort_is_error_not_extra_samples(self):
        plan = copy.deepcopy(self.plans()[0])
        stage = next(row for row in plan["stages"] if row["id"] == "affinity")
        stage["params"]["requests"] = [stage["params"]["requests"][0]] * 5
        result = self.run_plan(plan, Backend())
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("duplicated", result["error"])


if __name__ == "__main__":
    unittest.main()
