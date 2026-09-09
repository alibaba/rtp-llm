"""Exercise formal YAML loading and real crossfire threads with fake services."""

import copy
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_concurrent as concurrent
from flexlb_test_framework.scenario.actions import elastic_lifecycle as life
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance


class ConcurrentTests(unittest.TestCase):
    def run_program(
        self,
        bad_master=False,
        bad_discovery=False,
        slow_remove=False,
        profile="batch-window",
        driver_factory=None,
    ):
        handlers = {h.name: h for h in e.HANDLERS}
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/concurrent_mutation.yaml"),
            handlers=handlers,
        )
        plan = next(p for p in plans if p["profile"] == profile)
        self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 65)
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        state = dict(
            engines={}, added=0, removed=0, active=0, max_active=0, cleaned=False
        )
        start = time.monotonic()
        # Real threads run against a faster monotonic clock; no ordering is
        # imposed by a synthetic serial stage executor.
        clock = lambda: (time.monotonic() - start) * 10
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "discovery.json"

            def write_file():
                d = {
                    f"mock.{role}.hosts.address": [
                        v["grpc_addr"]
                        for v in state["engines"].values()
                        if v["role"] == role
                    ]
                    for role in ("prefill", "decode")
                }
                if bad_discovery:
                    d["mock.prefill.hosts.address"].append("127.0.0.1:65500")
                file.write_text(json.dumps(d))

            class Backend:
                def setup(self, ctx, environment, deadline):
                    for role, count in [("prefill", 2), ("decode", 4)]:
                        for i in range(count):
                            state["engines"][f"{role}-{i}"] = dict(
                                role=role,
                                grpc_addr=f'127.0.0.1:{10000 + len(state["engines"])*2}',
                            )
                    write_file()
                    rid = iter(range(1, 10000))
                    ops = NS(
                        next_request_id=lambda: next(rid),
                        mock_http_port=1,
                        master_http_port=2,
                    )
                    if driver_factory is not None:
                        ops = driver_factory(ops, environment, state, clock)
                    return NS(discovery_file=file), ops

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            def http(ops, endpoint, deadline, body=None):
                if endpoint == "snapshot":
                    with lock:
                        self.assertEqual(state["active"], 0)
                        state["snapshot_s"] = clock()
                        return dict(
                            engines=copy.deepcopy(list(state["engines"].values()))
                        )
                with lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    first = endpoint == "add_engine" and state["added"] < 2
                try:
                    if first:
                        barrier.wait(timeout=2)
                    time.sleep(
                        0.9 if slow_remove and endpoint == "remove_engine" else 0.005
                    )
                    with lock:
                        if endpoint == "add_engine":
                            state["added"] += 1
                            port = 11000 + state["added"] * 2
                            state["engines"][str(port)] = dict(
                                role=body["role"], grpc_addr=f"127.0.0.1:{port}"
                            )
                            response = dict(status="ok", port=port)
                        else:
                            item = state["engines"].pop(str(body["port"]), None)
                            state["removed"] += int(item is not None)
                            state["last_remove_s"] = clock()
                            response = dict(status="ok" if item else "missing")
                        write_file()
                        return response
                finally:
                    with lock:
                        state["active"] -= 1

            def run(records, record, shape, **kwargs):
                records.update(
                    record,
                    business_finished=True,
                    schedule=dict(status="OK"),
                    stream=dict(status="OK"),
                    transport_terminal_s=clock(),
                    consumer_exit_s=clock(),
                )

            def master(*args):
                if bad_master:
                    raise ValueError("HTTP 503")
                return {}

            consumer_patch = (
                nullcontext()
                if driver_factory is not None
                else patch.object(e.RecordedRequests, "run", run)
            )
            with patch.object(
                concurrent, "mutation_http", side_effect=http
            ), patch.object(e, "_http", side_effect=http), consumer_patch, patch.object(
                life, "_master_get", side_effect=master
            ):
                result = execute_instance(
                    plan,
                    Backend(),
                    handlers=handlers,
                    artifact_dir=Path(temp) / "artifacts",
                    clock=clock,
                    sleeper=lambda s: time.sleep(s / 10),
                )
            self.assertTrue(state["cleaned"])
            self.assertFalse(
                any(
                    t.name.startswith("elastic-crossfire-") and t.is_alive()
                    for t in threading.enumerate()
                )
            )
            state["plan"] = plan
            state["request_artifacts"] = {
                p.name: json.loads(p.read_text())
                for p in (Path(temp) / "artifacts").glob("elastic-crossfire-*.json")
            }
            return result, state

    def test_mutation_http_preserves_add_remove_budgets(self):
        for endpoint, remaining, expected in [
            ("add_engine", 120, 10),
            ("remove_engine", 120, 95),
            ("remove_engine", 7, 7),
        ]:
            deadline = NS(check=lambda: None, remaining=lambda: remaining)
            with patch.object(
                concurrent.urllib.request,
                "urlopen",
                return_value=io.BytesIO(b'{"status":"ok"}'),
            ) as opening:
                concurrent.mutation_http(
                    NS(mock_http_port=1234), endpoint, deadline, {"port": 1235}
                )
                self.assertEqual(opening.call_args.kwargs["timeout"], expected)

    def test_slow_removal_finishes_after_window_before_discovery(self):
        result, state = self.run_program(slow_remove=True)
        self.assertEqual(result["status"], "PASS", result)
        self.assertGreater(state["last_remove_s"], 10)
        self.assertGreaterEqual(state["snapshot_s"], state["last_remove_s"])

    def test_four_real_workers_overlap_and_final_discovery_matches(self):
        result, state = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertGreaterEqual(state["max_active"], 2)
        self.assertGreater(state["removed"], 0)
        self.assertLessEqual(state["added"], 65)

    def test_master_health_failure_cannot_be_hidden_by_successful_requests(self):
        result, _ = self.run_program(bad_master=True)
        self.assertEqual(result["status"], "FAIL")
        row = next(s for s in result["stages"] if s["id"] == "crossfire")
        self.assertEqual(
            {c["id"]: c["status"] for c in row["checks"]},
            dict(workers_finished="PASS", master_http="FAIL"),
        )

    def test_discovery_count_mismatch_fails_after_workers_exit(self):
        result, _ = self.run_program(bad_discovery=True)
        self.assertEqual(result["status"], "FAIL")
        row = next(s for s in result["stages"] if s["id"] == "discovery")
        self.assertEqual(
            {c["id"]: c["status"] for c in row["checks"]},
            dict(parsable="PASS", counts_match="FAIL"),
        )


if __name__ == "__main__":
    unittest.main()
