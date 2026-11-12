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

from navi_sched.util import naviSchedModule

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
            naviSchedModule=naviSchedModule,
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

    def test_native_metadata_merges_objects_and_intersects_every_pp_worker(self):
        def identity(group, *values):
            return json.dumps(dict(group_id=group, block_hashes=values))

        low, middle, high = "01" * 32, "80" * 32, "ff" * 32
        nodes = [
            [identity(3, high, low), identity(3, middle, low), identity(0, high)],
            [identity(3, middle, high), identity(0, low)],
            [identity(3, high, middle)],
            [identity(3, middle)],
        ]
        self.assertEqual(
            {0: (), 3: (bytes.fromhex(middle),)},
            naviSchedModule().decode_v6d_cache_metadata(nodes, 4),
        )

    def test_native_metadata_empty_first_worker_cannot_restore_evicted_hashes(self):
        identity = json.dumps(dict(group_id=3, block_hashes=["01" * 32]))
        self.assertEqual(
            {}, naviSchedModule().decode_v6d_cache_metadata([[], [identity]], 4)
        )
        self.assertEqual(
            {3: ()}, naviSchedModule().decode_v6d_cache_metadata([[identity], []], 4)
        )
        self.assertEqual(
            {3: (bytes.fromhex("01" * 32),)},
            naviSchedModule().decode_v6d_cache_metadata([[identity], [identity]], 4),
        )
        self.assertEqual(
            {3: ()}, naviSchedModule().decode_v6d_cache_metadata([[identity], []], 4)
        )

    def test_native_metadata_validates_later_workers_even_if_intersection_is_empty(self):
        malformed = [
            "not json",
            "[]",
            json.dumps(dict(group_id=True, block_hashes=["01" * 32])),
            json.dumps(dict(group_id=4, block_hashes=["01" * 32])),
            json.dumps(dict(group_id=-1, block_hashes=["01" * 32])),
            json.dumps(dict(group_id=3.0, block_hashes=["01" * 32])),
            json.dumps(dict(group_id=3, block_hashes=[])),
            json.dumps(dict(group_id=3, block_hashes=["01" * 31])),
            json.dumps(dict(group_id=3, block_hashes=["0 1" * 32])),
            json.dumps(dict(group_id=3, block_hashes=["gg" * 32])),
            json.dumps(dict(group_id=3, block_hashes=[None])),
            json.dumps(dict(group_id=3)),
        ]
        for identity in malformed:
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                naviSchedModule().decode_v6d_cache_metadata([[], [identity]], 4)

    def test_native_metadata_preserves_hex_whitespace_sorting_and_last_json_field(self):
        value = " \t" + "\n".join(["AB"] * 32) + "\r\v\f "
        identity = json.dumps(dict(group_id=3, block_hashes=[value, "01" * 32]))
        self.assertEqual(
            {3: (bytes.fromhex("01" * 32), bytes.fromhex("ab" * 32))},
            naviSchedModule().decode_v6d_cache_metadata([[identity]], 4),
        )
        duplicate = '{"group_id":0,"group_id":3,"block_hashes":["' + "01" * 32 + '"]}'
        self.assertEqual(
            {3: (bytes.fromhex("01" * 32),)},
            naviSchedModule().decode_v6d_cache_metadata([[duplicate]], 4),
        )

    def test_native_metadata_releases_gil_during_decode(self):
        identity = json.dumps(dict(group_id=3, block_hashes=["01" * 32] * 50000))
        ready = threading.Event()
        ran = threading.Event()
        allow_worker = False

        def worker():
            ready.set()
            while not allow_worker:
                time.sleep(0)
            ran.set()

        thread = threading.Thread(target=worker)
        thread.start()
        ready.wait()
        original_interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(60)
            allow_worker = True
            for _ in range(5):
                naviSchedModule().decode_v6d_cache_metadata([[identity]], 4)
            self.assertTrue(
                ran.is_set(), "metadata decoding blocked the other Python thread"
            )
        finally:
            sys.setswitchinterval(original_interval)
            allow_worker = True
            thread.join()



if __name__ == "__main__":
    unittest.main()
