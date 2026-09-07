"""Exercise formal YAML loading and real crossfire threads with fake services."""

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_ft.scenario import compile_scenarios
from flexlb_ft.scenario.actions import elastic as e
from flexlb_ft.scenario.actions import elastic_lifecycle as life
from flexlb_ft.scenario.loader import load_scenarios
from flexlb_ft.scenario.runtime import execute_instance


class ConcurrentTests(unittest.TestCase):
    def run_program(self, bad_master=False, bad_discovery=False):
        handlers = {h.name: h for h in e.HANDLERS}
        plan = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/concurrent_mutation.yaml"),
            handlers=handlers,
        )[0]
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
                    return NS(discovery_file=file), NS(
                        next_request_id=lambda: next(rid),
                        mock_http_port=1,
                        master_http_port=2,
                    )

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            def http(ops, endpoint, deadline, body=None):
                if endpoint == "snapshot":
                    with lock:
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
                    time.sleep(0.005)
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

            with patch.object(e, "_http", side_effect=http), patch.object(
                e.RecordedRequests, "run", run
            ), patch.object(life, "_master_get", side_effect=master):
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
            return result, state

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
