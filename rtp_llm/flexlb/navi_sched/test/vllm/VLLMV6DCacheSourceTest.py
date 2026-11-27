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
        observer._drain_v6d_cache_touches = MethodType(
            self.function("_drain_v6d_cache_touches"), observer
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

    def test_workload_uses_resolved_native_cache_state_across_request_versions(self):
        names = {"_make_workload_contribution", "_native_cache_lookup_complete"}
        methods = [copy.deepcopy(n) for n in self.observer_class.body
                   if isinstance(n, ast.FunctionDef) and n.name in names]
        module = ast.parse("from __future__ import annotations\nclass Observer: pass\n")
        module.body[1].body = methods
        namespace = {"_WorkloadContribution": SimpleNamespace}
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        observer = namespace["Observer"]()
        observer._native_pipeline_state = None
        observer._pp_capacity_unified = False
        observer._predicted_cache_hit_for_request = lambda request: 48
        observer._forward_request_state = lambda request_id: SimpleNamespace(batch_id=77)
        cases = [
            ("new_resolved_local_before_allocation", dict(prefill_stats=SimpleNamespace(
                num_prompt_tokens=64, num_local_cached_tokens=24,
                num_external_cached_tokens=0)), 0, 24),
            ("new_resolved_external_before_allocation", dict(prefill_stats=SimpleNamespace(
                num_prompt_tokens=64, num_local_cached_tokens=16,
                num_external_cached_tokens=24)), 0, 40),
            ("new_resolved_cache_with_compute_inflight", dict(prefill_stats=SimpleNamespace(
                num_prompt_tokens=64, num_local_cached_tokens=24,
                num_external_cached_tokens=0)), 40, 32),
            ("new_resolved_counts_unavailable", dict(prefill_stats=SimpleNamespace(
                num_prompt_tokens=64, num_local_cached_tokens=None,
                num_external_cached_tokens=None)), 8, 0),
            ("new_miss", dict(prefill_stats=SimpleNamespace(num_prompt_tokens=64)), 8, 0),
            ("new_partial", dict(prefill_stats=SimpleNamespace(num_prompt_tokens=64)), 32, 24),
            ("new_unresolved", dict(prefill_stats=SimpleNamespace(num_prompt_tokens=0)), 8, 48),
            ("new_consumed", dict(prefill_stats=None), 32, 24),
            ("legacy_miss", dict(num_cached_tokens=0), 8, 0),
            ("legacy_unresolved", dict(num_cached_tokens=-1), 8, 48),
            ("new_resolved_old_unknown", dict(num_cached_tokens=-1,
                prefill_stats=SimpleNamespace(num_prompt_tokens=64)), 8, 0),
        ]
        for name, fields, computed, expected_context in cases:
            with self.subTest(name=name):
                request = SimpleNamespace(request_id="request", num_prompt_tokens=64,
                                          num_computed_tokens=computed, **fields)
                result = observer._make_workload_contribution(request, 8, None)
                self.assertEqual(expected_context, result.prefill_context_tokens)
                self.assertEqual(64 - expected_context, result.remaining_prefill_tokens)
                self.assertEqual(77, result.batch_id)

    def test_unscheduled_resolved_cache_lookup_refreshes_workload(self):
        names = {"_make_workload_contribution", "_native_cache_lookup_complete",
                 "_refresh_workload_contributions"}
        methods = [copy.deepcopy(n) for n in self.observer_class.body
                   if isinstance(n, ast.FunctionDef) and n.name in names]
        module = ast.parse("from __future__ import annotations\nclass Observer: pass\n")
        module.body[1].body = methods
        namespace = {"_WorkloadContribution": SimpleNamespace}
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        for fields, resolve, resolved_remaining in [
            (dict(prefill_stats=SimpleNamespace(num_prompt_tokens=0)),
             lambda r: setattr(r.prefill_stats, "num_prompt_tokens", 64), 64),
            (dict(num_cached_tokens=-1), lambda r: setattr(r, "num_cached_tokens", 0), 64),
            (dict(prefill_stats=SimpleNamespace(num_prompt_tokens=0)),
             lambda r: setattr(r, "prefill_stats", SimpleNamespace(
                 num_prompt_tokens=64, num_local_cached_tokens=24,
                 num_external_cached_tokens=0)), 40),
        ]:
            with self.subTest(fields=fields):
                observer = namespace["Observer"]()
                observer._native_pipeline_state = None
                observer._pp_capacity_unified = False
                observer._predicted_cache_hit_for_request = lambda r: 48
                observer._forward_request_state = lambda rid: SimpleNamespace(batch_id=77)
                observer._workload_contributions = {}
                observer._pending_cache_lookup_request_ids = set()
                remaining = 0

                def apply(contribution, sign):
                    nonlocal remaining
                    remaining += sign * contribution.remaining_prefill_tokens

                observer._workload_aggregate = SimpleNamespace(apply=apply)
                request = SimpleNamespace(request_id="r", num_prompt_tokens=64,
                                          num_computed_tokens=0, **fields)
                scheduler = SimpleNamespace(requests={"r": request})
                observer._refresh_workload_contributions(scheduler, {"r"}, {})
                self.assertEqual(16, remaining)
                observer._refresh_workload_contributions(scheduler, set(), {})
                self.assertEqual(16, remaining)
                resolve(request)
                observer._refresh_workload_contributions(scheduler, set(), {})
                self.assertEqual(resolved_remaining, remaining)
                self.assertFalse(observer._pending_cache_lookup_request_ids)
                request.num_computed_tokens = 24
                observer._refresh_workload_contributions(scheduler, {"r"}, {"r": 8})
                self.assertEqual(min(48, resolved_remaining), remaining)
                scheduler.requests.clear()
                observer._refresh_workload_contributions(scheduler, {"r"}, {})
                self.assertEqual(0, remaining)
                self.assertFalse(observer._workload_contributions)

    def test_cancelled_pending_cache_lookup_is_removed_without_scheduled_output(self):
        observer = SimpleNamespace(
            _pending_cache_lookup_request_ids={"cancelled"},
            _workload_contributions={"cancelled": SimpleNamespace(remaining_prefill_tokens=16)},
            _workload_aggregate=SimpleNamespace(apply=mock.Mock()),
        )
        self.function("_refresh_workload_contributions")(
            observer, SimpleNamespace(requests={}), set(), {}
        )
        self.assertFalse(observer._pending_cache_lookup_request_ids)
        self.assertFalse(observer._workload_contributions)
        self.assertEqual(-1, observer._workload_aggregate.apply.call_args.args[1])

    def test_native_request_readiness_audit_records_transitions_only(self):
        observer = SimpleNamespace(
            _native_request_state_audit_enabled=True,
            _pp_capacity_unified=False,
            _native_v6d_request_audit=lambda request: {},
            _native_request_states={},
            on_request_lifecycle_event=mock.Mock(),
            _native_cache_lookup_complete=self.function("_native_cache_lookup_complete"),
        )
        observe = self.function("_audit_native_request_state")
        request = SimpleNamespace(request_id="r", num_prompt_tokens=64,
                                  num_computed_tokens=0,
                                  status=SimpleNamespace(name="WAITING"),
                                  prefill_stats=SimpleNamespace(num_prompt_tokens=0,
                                      num_local_cached_tokens=0,
                                      num_external_cached_tokens=0))
        observe(observer, request, "waiting")
        first = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertFalse(first["cache_lookup_complete"])
        self.assertEqual("WAITING", first["native_status"])
        for _ in range(100):
            observe(observer, request, "waiting")
        observer.on_request_lifecycle_event.assert_called_once()
        request.prefill_stats.num_prompt_tokens = 64
        request.prefill_stats.num_external_cached_tokens = 48
        request.status.name = "WAITING_FOR_REMOTE_KVS"
        observe(observer, request, "waiting")
        second = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertTrue(second["cache_lookup_complete"])
        self.assertEqual("waiting", second["queue"])
        self.assertEqual("WAITING_FOR_REMOTE_KVS", second["native_status"])
        self.assertEqual(48, second["external_cached_tokens"])
        self.assertEqual(0, second["computed_tokens"])
        request.status.name = "RUNNING"
        request.num_computed_tokens = 56
        observe(observer, request, "running")
        self.assertEqual(3, observer.on_request_lifecycle_event.call_count)
        observer._native_request_state_audit_enabled = False
        request.status.name = "FINISHED_STOPPED"
        observe(observer, request, "running")
        self.assertEqual(3, observer.on_request_lifecycle_event.call_count)
        self.assertEqual(56, request.num_computed_tokens)
        self.assertEqual("FINISHED_STOPPED", request.status.name)

    def test_native_request_audit_cleans_removed_requests_during_workload_scan(self):
        partition = self.function("_partition_prefill_workload")
        partition.__globals__["_PrefillWorkloadSliceAggregate"] = lambda: SimpleNamespace(
            apply=mock.Mock(), snapshot=lambda: None
        )
        request = SimpleNamespace(request_id="r", num_prompt_tokens=64,
                                  num_computed_tokens=0, num_cached_tokens=-1)
        observer = SimpleNamespace(
            _native_request_state_audit_enabled=True,
            _pp_capacity_unified=False,
            _native_v6d_request_audit=lambda request: {},
            _native_request_states={"removed": ("waiting", "WAITING", False)},
            _workload_contributions={},
            on_request_lifecycle_event=mock.Mock(),
            _native_cache_lookup_complete=self.function("_native_cache_lookup_complete"),
        )
        observer._audit_native_request_state = MethodType(
            self.function("_audit_native_request_state"), observer
        )
        scheduler = SimpleNamespace(requests={"r": request}, running=[],
                                    waiting=[request], max_num_running_reqs=8)
        partition(observer, scheduler)
        self.assertEqual({"r"}, set(observer._native_request_states))
        partition(observer, scheduler)
        observer.on_request_lifecycle_event.assert_called_once()
        scheduler.requests.clear()
        scheduler.waiting.clear()
        partition(observer, scheduler)
        self.assertFalse(observer._native_request_states)
        observer.on_request_lifecycle_event.assert_called_once()

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

    def test_native_metadata_validates_later_workers_even_if_intersection_is_empty(
        self,
    ):
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

    def test_raw_snapshots_match_decoded_residency_and_capability_updates(self):
        snapshots = self.snapshots()
        variants = [
            snapshots,
            copy.deepcopy(snapshots),
            copy.deepcopy(snapshots),
            copy.deepcopy(snapshots),
        ]
        variants[1][2]["objects"] = []
        variants[2][1]["objects"][0]["sealedResident"] = False
        variants[3][3]["peer_cache_read_available"] = False
        for rows in variants:
            with self.subTest(rows=rows):
                decoded, raw = self.observer(), self.observer()
                decoded.on_v6d_cache_snapshots(rows)
                bodies = [json.dumps(row).encode() for row in rows]
                with mock.patch.object(
                    json,
                    "loads",
                    side_effect=AssertionError("Python snapshot decoding"),
                ):
                    raw.on_v6d_cache_snapshots(bodies)
                self.assertEqual(
                    decoded.on_block_lifecycle_event.call_args,
                    raw.on_block_lifecycle_event.call_args,
                )
                self.assertEqual(
                    decoded._bridge.report_peer_cache_read_available.call_args,
                    raw._bridge.report_peer_cache_read_available.call_args,
                )

    def test_raw_snapshot_ignores_unsealed_and_other_tier_metadata(self):
        snapshots = self.snapshots()
        for row in snapshots:
            row["objects"].extend(
                [
                    dict(
                        tier="tair",
                        sealedResident=True,
                        meta=dict(navi_sched_cache="bad metadata"),
                    ),
                    dict(
                        tier="vineyard",
                        sealedResident=False,
                        meta=dict(navi_sched_cache="bad metadata"),
                    ),
                    dict(tier="vineyard", sealedResident=True, meta={}),
                ]
            )
        observer = self.observer()
        observer.on_v6d_cache_snapshots([json.dumps(row).encode() for row in snapshots])
        self.assertEqual(
            {3: (bytes.fromhex("01" * 32),)},
            observer.on_block_lifecycle_event.call_args.args[2],
        )

    def test_raw_snapshot_validates_every_node_before_publishing(self):
        good = self.snapshots()[0]
        malformed = [
            b"not json",
            b'{"objects":[],"peer_cache_read_available":true}\x00junk',
            b"[]",
            b'{"objects":[]}',
            b'{"objects":[],"peer_cache_read_available":1}',
            b'{"objects":null,"peer_cache_read_available":true}',
            b'{"objects":[],"peer_cache_read_available":true,"extra":"\xff"}',
        ]
        for field, value in [("tier", None), ("sealedResident", None), ("meta", None)]:
            row = copy.deepcopy(good)
            row["objects"][0].pop(field)
            malformed.append(json.dumps(row).encode())
        row = copy.deepcopy(good)
        row["objects"][0]["meta"]["navi_sched_cache"] = dict(group_id=3)
        malformed.append(json.dumps(row).encode())
        empty = json.dumps(dict(peer_cache_read_available=True, objects=[])).encode()
        for body in malformed:
            with self.subTest(body=body), self.assertRaises(ValueError):
                observer = self.observer()
                observer.on_v6d_cache_snapshots([empty, body])
            observer.on_block_lifecycle_event.assert_not_called()
            observer._bridge.report_peer_cache_read_available.assert_not_called()

    def test_raw_snapshot_last_json_field_wins(self):
        body = b'{"objects":[],"peer_cache_read_available":false,"peer_cache_read_available":true}'
        self.assertEqual(
            ({}, True), naviSchedModule().decode_v6d_cache_snapshots([body], 4)
        )

    def test_raw_snapshot_releases_gil_while_parsing_outer_inventory(self):
        body = json.dumps(
            dict(
                peer_cache_read_available=True,
                objects=[dict(tier="tair", unused=list(range(300000)))],
            )
        ).encode()
        ready, ran = threading.Event(), threading.Event()
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
            naviSchedModule().decode_v6d_cache_snapshots([body], 4)
            self.assertTrue(ran.is_set(), "outer inventory parsing held the GIL")
        finally:
            sys.setswitchinterval(original_interval)
            allow_worker = True
            thread.join()


if __name__ == "__main__":
    unittest.main()
