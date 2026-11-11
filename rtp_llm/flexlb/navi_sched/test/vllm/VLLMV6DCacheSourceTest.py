"""Exercise the production V6D boundary without importing a GPU runtime."""

import ast
import copy
import json
import logging
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMEngineObserver.py"
if __name__ == "__main__" and len(sys.argv) == 2:
    SOURCE = Path(sys.argv.pop())


class VLLMV6DCacheSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text())
        cls.observer_class = next(
            n
            for n in cls.tree.body
            if isinstance(n, ast.ClassDef) and n.name == "VLLMEngineObserver"
        )

    def function(self, name):
        candidates = [
            n
            for n in self.tree.body + self.observer_class.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ]
        self.assertEqual(
            1, len(candidates), f"production V6D boundary is missing: {name}"
        )
        node = copy.deepcopy(candidates[0])
        node.decorator_list = []
        module = ast.parse("from __future__ import annotations\n")
        module.body.append(node)
        ns = dict(
            json=json,
            sys=sys,
            time=time,
            logger=logging.getLogger("v6d-cache-source-test"),
            _CACHE_SNAPSHOT_INTERVAL_NS=100_000_000,
            _read_bool_env=lambda *args: False,
            forward_request_type=lambda request: request,
        )
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
        return ns[name]

    def observer(self):
        observer = SimpleNamespace(
            _cache_policy=SimpleNamespace(group_roles=("tail", "tail", "tail", "full")),
            _bridge=SimpleNamespace(report_peer_cache_read_available=mock.Mock()),
            on_block_lifecycle_event=mock.Mock(),
            _forward_stop=threading.Event(),
            _v6d_scheduler=None,
            _v6d_snapshot_future=None,
            _next_v6d_snapshot_poll_ns=0,
        )
        observer.on_v6d_cache_snapshots = MethodType(
            self.function("on_v6d_cache_snapshots"), observer
        )
        return observer

    @staticmethod
    def snapshots(available=True):
        identity = json.dumps(dict(group_id=3, block_hashes=["01" * 32]))
        return [
            dict(
                peer_cache_read_available=available,
                objects=[
                    dict(
                        tier="vineyard",
                        sealedResident=True,
                        meta=dict(navi_sched_cache=identity),
                    )
                ],
            )
            for _ in range(4)
        ]

    def test_native_v6d_presence_controls_capability_without_a_navisched_switch(self):
        class Connector:
            def __init__(self):
                self.scheduler = SimpleNamespace(
                    _is_cross_node_tp=True, _client_context=None
                )

        module = SimpleNamespace(V6dObjectConnector=Connector)
        scheduler = SimpleNamespace(
            connector=SimpleNamespace(_connectors=[Connector()])
        )
        with mock.patch.dict(
            sys.modules, {"kvs_connector.v6d_object_connector": module}
        ):
            self.assertTrue(self.function("_v6d_peer_cache_enabled")(scheduler))
            scheduler.connector = None
            self.assertFalse(self.function("_v6d_peer_cache_enabled")(scheduler))

    def test_v6d_local_residency_reaches_the_external_cache_namespace(self):
        observer = self.observer()
        observer.on_v6d_cache_snapshots(self.snapshots(available=False))
        args, kwargs = observer.on_block_lifecycle_event.call_args
        self.assertEqual(("snapshot", None, {3: (bytes.fromhex("01" * 32),)}), args)
        self.assertEqual(1, kwargs["cache_key_type"])
        observer._bridge.report_peer_cache_read_available.assert_called_once_with(False)

    def test_native_peer_read_availability_is_forwarded_separately(self):
        observer = self.observer()
        snapshots = self.snapshots()
        observer.on_v6d_cache_snapshots(snapshots)
        observer._bridge.report_peer_cache_read_available.assert_called_with(True)
        snapshots[2]["peer_cache_read_available"] = False
        observer.on_v6d_cache_snapshots(snapshots)
        observer._bridge.report_peer_cache_read_available.assert_called_with(False)

    def test_eviction_in_one_native_storage_node_retracts_the_logical_hit(self):
        observer = self.observer()
        snapshots = self.snapshots()
        observer.on_v6d_cache_snapshots(snapshots)
        snapshots[2]["objects"] = []
        observer.on_v6d_cache_snapshots(snapshots)
        self.assertEqual(
            (), observer.on_block_lifecycle_event.call_args.args[2].get(3, ())
        )

    def test_unsealed_native_objects_are_not_advertised(self):
        observer = self.observer()
        snapshots = self.snapshots()
        snapshots[1]["objects"][0]["sealedResident"] = False
        observer.on_v6d_cache_snapshots(snapshots)
        self.assertEqual(
            (), observer.on_block_lifecycle_event.call_args.args[2].get(3, ())
        )

    def test_missing_native_readiness_is_not_interpreted_as_healthy(self):
        observer = self.observer()
        snapshots = self.snapshots()
        snapshots[0].pop("peer_cache_read_available")
        with self.assertRaises((KeyError, ValueError)):
            observer.on_v6d_cache_snapshots(snapshots)
        observer.on_block_lifecycle_event.assert_not_called()

    def test_poll_receives_native_data_and_does_not_queue_duplicate_reads(self):
        observer = self.observer()
        future = Future()
        observer._v6d_scheduler = SimpleNamespace(
            submit_local_cache_snapshots=mock.Mock(return_value=future)
        )
        poll = MethodType(self.function("_poll_v6d_cache_source"), observer)
        with mock.patch.object(time, "monotonic_ns", return_value=1_000_000_000):
            poll()
            poll()
            observer._v6d_scheduler.submit_local_cache_snapshots.assert_called_once()
            observer.on_block_lifecycle_event.assert_not_called()
            future.set_result(self.snapshots())
            poll()
        self.assertEqual(1, observer.on_block_lifecycle_event.call_count)

    def test_forward_loop_actually_drives_v6d_collection(self):
        observer = SimpleNamespace(
            _poll_v6d_cache_source=mock.Mock(),
            _drain_cancelled_requests=lambda *args: False,
            _forward_stop=SimpleNamespace(wait=lambda timeout: True),
        )
        modules = {
            "msgspec": SimpleNamespace(),
            "vllm.v1.engine": SimpleNamespace(
                EngineCoreRequest=object, EngineCoreRequestType=object
            ),
            "vllm.v1.serial_utils": SimpleNamespace(
                MsgpackDecoder=lambda request: object()
            ),
        }
        with mock.patch.dict(sys.modules, modules):
            self.function("_forward_batches")(observer)
        observer._poll_v6d_cache_source.assert_called_once()


if __name__ == "__main__":
    unittest.main()
