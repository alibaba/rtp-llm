"""Formal loader and real Schedule/consumer workers; only external I/O is fake."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.actions import priority as p
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.runtime import execute_instance
from test_scenario_backend import Ops


class Backend:
    def __init__(self, reverse=False, missing=False):
        self.ops = Ops(batch=False)
        self.ops.master_http_port = 1
        self.reverse, self.missing = reverse, missing
        self.ops.last_shapes = []
        original = self.ops.build_schedule_request

        def build(rid, **shape):
            self.ops.last_shapes.append(shape)
            return original(rid, **shape)

        self.ops.build_schedule_request = build

    def setup(self, ctx, environment, deadline):
        return NS(), self.ops

    def teardown(self, ctx, deadline):
        pass

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_perf":
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        lifecycle = {
            str(i): dict(running_ms=(8 - i if self.reverse else i) * 3000)
            for i in range(1, 8)
        }
        if self.missing:
            lifecycle.pop("4")
        return dict(
            engines=[
                dict(
                    name="p0",
                    role="prefill",
                    grpc_addr="prefill",
                    port=1234,
                    stopped=False,
                    request_lifecycle=lifecycle,
                )
            ]
        )


class Clean:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, n):
        return json.dumps(
            dict(
                scheduler_inflight=0,
                prefill_endpoints=[dict(inflight_batches=0)],
                decode_endpoints=[dict(total_load=0)],
            )
        ).encode()


class Tests(unittest.TestCase):
    def run_program(self, **kwargs):
        registry = handlers()
        registry.update({h.name: h for h in p.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority"), handlers=registry
        )
        self.assertEqual(len(plans), 1)
        backend = Backend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            p, "_http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ):
            result = execute_instance(
                plans[0], backend, handlers=registry, artifact_dir=tmp
            )
            records = json.loads(
                next(Path(tmp).glob("priority-wave-*.json")).read_text()
            )
        return result, records, backend

    def test_fifo_real_consumers(self):
        result, records, backend = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(backend.ops.generate_count, 7)
        self.assertEqual(backend.ops.fetch_count, 0)
        self.assertEqual(len(records), 7)
        self.assertTrue(
            all(
                r["consumer_done"] and r["consumer_completion_verified"]
                for r in records
            )
        )
        self.assertTrue(all(s["priority"] == 50 for s in backend.ops.last_shapes))
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_inverted_dispatch_fails(self):
        result, _, _ = self.run_program(reverse=True)
        self.assertEqual(result["status"], "FAIL", result)

    def test_missing_source_is_error(self):
        result, _, _ = self.run_program(missing=True)
        self.assertEqual(result["status"], "ERROR", result)

    def test_omitted_priority_stays_absent(self):
        value = p._wave_params(dict(requests=[dict(tag="unset")]), None)
        self.assertNotIn("priority", value["requests"][0])
        with self.assertRaises(ValueError):
            p._wave_params(dict(requests=[dict(tag="x", priority=True)]), None)


if __name__ == "__main__":
    unittest.main()
