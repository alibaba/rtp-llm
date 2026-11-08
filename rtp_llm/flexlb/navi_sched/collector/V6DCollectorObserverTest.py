import asyncio
import threading
import unittest
from enum import Enum
from types import SimpleNamespace as Obj

import navi_sched.adapters.vllm.V6DCollectorObserver as module

V6DCollectorObserver = module.V6DCollectorObserver


class Collector:
    def __init__(self):
        self.objects = {}
        self.stopped = False
        self.reject = False
        self.peer_read_states = []

    def replace_resident_objects(self, objects):
        self.objects = dict(objects)
        return not self.reject

    def object_resident(self, key, identity):
        if self.reject:
            return False
        if identity:
            self.objects[key] = identity
        else:
            self.objects.pop(key, None)
        return True

    def object_removed(self, key):
        if self.reject:
            return False
        self.objects.pop(key, None)
        return True

    def report_peer_cache_read_available(self, available):
        self.peer_read_states.append(available)
        return not self.reject

    def stop(self):
        self.stopped = True
        self.objects.clear()


class Peer:
    def __init__(self):
        self.config = Obj(tair_kvcm_p2p_enabled=False)
        self._lock = threading.RLock()
        self._object_info = {}
        self._object_to_creating_lease = {}
        self._object_fetching_keys = set()
        self._active_leases = {}
        self.names = set()
        self.lookups = []
        self._vineyard_peer = Obj(vineyard_server=Obj(get_names=self.get_names))
        self.seal_result = object()
        self.seal_error = None
        self.seal_calls = 0

    def get_names(self, keys, session):
        self.lookups.append(tuple(keys))
        return [Obj(id=1 if key in self.names else 0) for key in keys]

    def _add_object_info(self, object_key, size=0, tier="vineyard", meta=None, va=None):
        self._object_info[object_key] = Obj(tier=tier, meta=meta or {})

    def _track_vineyard_add(self, object_key):
        return None

    def _track_vineyard_remove(self, object_key):
        return None

    def _add_creating_objects(self, object_keys, creators):
        for key in object_keys:
            self._object_to_creating_lease[key] = "creating"

    def _add_fetching_objects(self, object_keys):
        self._object_fetching_keys.update(object_keys)

    def _remove_creating_objects(self, object_keys):
        for key in object_keys:
            self._object_to_creating_lease.pop(key, None)

    def _remove_fetching_objects(self, object_keys):
        self._object_fetching_keys.difference_update(object_keys)

    def _del_object_info(self, object_keys):
        for key in object_keys:
            self._track_vineyard_remove(key)
            self._object_info.pop(key, None)

    def mark_evicted(self, object_key):
        self._object_info[object_key].tier = "evicted"

    def _track_objects_in_local_vineyard(self, objects):
        for obj in objects:
            self._object_info[obj.key].tier = "vineyard"

    async def seal(self, lease_id, request_id=None, seal_object_keys=None):
        self.seal_calls += 1
        if self.seal_error is not None:
            raise self.seal_error
        keys = self._active_leases[lease_id].lease.object_key
        self.names.update(keys)
        self._remove_creating_objects(keys)
        self._remove_fetching_objects(keys)
        self._active_leases.pop(lease_id)
        return self.seal_result

    async def discard(self, lease_id, request_id=None):
        keys = self._active_leases[lease_id].lease.object_key
        self._del_object_info(keys)
        self._active_leases.pop(lease_id)

    async def acquire(self, object_keys, term=None, scope=None, metas=None, **kwargs):
        for key, meta in zip(
            object_keys, metas or [{}] * len(object_keys), strict=True
        ):
            self._object_info[key].meta.update(meta or {})
        return self.seal_result, [Obj(key=key) for key in object_keys]

    async def update_meta(self, object_keys, metas, request_id=None):
        for key, meta in zip(object_keys, metas, strict=True):
            self._object_info[key].meta.update(meta or {})

    def stop(self):
        return self.seal_result

    async def astop(self):
        if self.seal_error is not None:
            raise self.seal_error
        return self.seal_result


class V6DCollectorObserverTest(unittest.TestCase):
    def setUp(self):
        self.peer = Peer()
        self.collector = Collector()
        self.observer = None

    def tearDown(self):
        if self.observer:
            self.observer.stop()

    def install(self):
        self.observer = V6DCollectorObserver(
            self.peer, self.collector, pp_shard_count=4, invalid_object_id=0
        ).install()

    def add(self, key="a", identity="identity", *, named=True):
        self.peer._add_object_info(key, meta={"navi_sched_cache": identity})
        if named:
            self.peer.names.add(key)

    def test_non_pp_configuration_does_not_touch_peer_or_collector(self):
        native_seal = self.peer.seal.__func__
        for count in (0, 1, True, "4"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                V6DCollectorObserver(
                    self.peer, self.collector, pp_shard_count=count, invalid_object_id=0
                )
        self.assertEqual(native_seal, self.peer.seal.__func__)
        self.assertFalse(self.collector.stopped)
        self.assertFalse(hasattr(self.peer, "_navi_sched_collector_observer"))

    def test_bootstrap_only_sealed_resident_objects(self):
        for key in ("ready", "creating", "fetching", "missing", "evicted"):
            self.add(key, named=key != "missing")
        self.peer._object_to_creating_lease["creating"] = "lease"
        self.peer._object_fetching_keys.add("fetching")
        self.peer._object_info["evicted"].tier = "evicted"
        self.install()
        self.assertEqual({"ready": "identity"}, self.collector.objects)

    def test_seal_keeps_native_result_and_waits_for_native_readability(self):
        self.add(named=False)
        self.peer._object_to_creating_lease["a"] = "lease"
        self.peer._object_fetching_keys.add("a")
        self.peer._active_leases["lease"] = Obj(lease=Obj(object_key=["a"]))
        self.install()
        self.peer._track_vineyard_add("a")
        self.assertEqual({}, self.collector.objects)
        result = asyncio.run(self.peer.seal("lease"))
        self.assertIs(self.peer.seal_result, result)
        self.assertEqual(1, self.peer.seal_calls)
        self.assertEqual({"a": "identity"}, self.collector.objects)

    def test_failed_seal_preserves_exception_and_does_not_publish(self):
        self.add(named=False)
        self.peer._active_leases["lease"] = Obj(lease=Obj(object_key=["a"]))
        self.peer.seal_error = ValueError("native failure")
        self.install()
        with self.assertRaises(ValueError) as caught:
            asyncio.run(self.peer.seal("lease"))
        self.assertIs(self.peer.seal_error, caught.exception)
        self.assertEqual({}, self.collector.objects)

    def test_mark_evicted_retracts_before_native_blob_deletion(self):
        self.add()
        self.install()
        self.peer.mark_evicted("a")
        self.assertIn("a", self.peer.names)
        self.assertEqual({}, self.collector.objects)

    def test_remove_hook_does_not_readd_still_visible_native_info(self):
        self.add()
        self.install()
        self.peer._track_vineyard_remove("a")
        self.assertIn("a", self.peer._object_info)
        self.assertEqual({}, self.collector.objects)
        self.peer._del_object_info(["a"])
        self.assertEqual({}, self.collector.objects)

    def test_entering_fetch_or_create_retracts_existing_residency(self):
        self.add()
        self.install()
        self.peer._add_fetching_objects(["a"])
        self.assertEqual({}, self.collector.objects)
        self.peer._remove_fetching_objects(["a"])
        self.assertEqual({"a": "identity"}, self.collector.objects)
        self.peer._add_creating_objects(["a"], ["creator"])
        self.assertEqual({}, self.collector.objects)
        self.peer._remove_creating_objects(["a"])
        self.assertEqual({"a": "identity"}, self.collector.objects)

    def test_fetch_completion_and_late_read_metadata_are_observed(self):
        self.add(identity="")
        self.peer._object_fetching_keys.add("a")
        self.install()
        asyncio.run(self.peer.acquire(["a"], metas=[{"navi_sched_cache": "late"}]))
        self.assertEqual({}, self.collector.objects)
        self.peer._remove_fetching_objects(["a"])
        self.assertEqual({"a": "late"}, self.collector.objects)
        asyncio.run(self.peer.update_meta(["a"], [{"navi_sched_cache": "new"}]))
        self.assertEqual({"a": "new"}, self.collector.objects)

    def test_observation_failure_withdraws_stream_without_masking_native_result(self):
        self.add(named=False)
        self.peer._active_leases["lease"] = Obj(lease=Obj(object_key=["a"]))
        self.install()
        self.collector.reject = True
        with self.assertLogs(module.logger, level="ERROR"):
            result = asyncio.run(self.peer.seal("lease"))
        self.assertIs(self.peer.seal_result, result)
        self.assertTrue(self.collector.stopped)
        self.assertEqual(1, self.peer.seal_calls)

    def test_native_cancellation_is_preserved(self):
        self.add(named=False)
        self.peer._active_leases["lease"] = Obj(lease=Obj(object_key=["a"]))
        self.peer.seal_error = asyncio.CancelledError("cancel")
        self.install()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.peer.seal("lease"))
        self.assertEqual({}, self.collector.objects)

    def test_event_lookup_only_includes_changed_keys(self):
        for i in range(20):
            self.add(str(i))
        self.install()
        self.peer.lookups.clear()
        asyncio.run(self.peer.update_meta(["5"], [{"navi_sched_cache": "new"}]))
        self.assertEqual([("5",)], self.peer.lookups)

    def test_stop_restores_owned_hooks_and_preserves_other_wrappers(self):
        self.install()
        another_wrapper = lambda key: None
        self.peer.mark_evicted = another_wrapper
        self.observer.stop()
        self.assertIs(self.peer.mark_evicted, another_wrapper)
        self.assertEqual(Peer.seal, self.peer.seal.__func__)
        self.assertFalse(hasattr(self.peer, "_navi_sched_collector_observer"))

    def test_native_shutdown_result_is_preserved_and_stream_is_closed(self):
        self.add()
        self.install()
        self.assertIs(self.peer.seal_result, self.peer.stop())
        self.assertTrue(self.collector.stopped)
        self.assertEqual({}, self.collector.objects)

    def test_native_async_shutdown_failure_is_preserved_and_stream_is_closed(self):
        self.add()
        self.install()
        self.peer.seal_error = RuntimeError("native shutdown failure")
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(self.peer.astop())
        self.assertIs(self.peer.seal_error, caught.exception)
        self.assertTrue(self.collector.stopped)


class NativeCollectorBindingsTest(unittest.TestCase):
    def test_pp_agent_configuration_is_explicit_and_requires_all_shards(self):
        import navi_sched._navi_sched_core as core

        from navi_sched import NaviSchedAgent

        self.assertIsNone(core.NaviSchedAgentConfig().pp_cache_collector)
        config = core.PPCacheAgentConfig()
        config.pp_group = "pp-a"
        config.pipeline_parallel_size = 4
        config.policy.block_size = 4096
        config.policy.linear_step = 1
        self.assertFalse(config.valid())
        endpoints = []
        for shard in range(4):
            endpoint = core.PPCacheShardEndpoint()
            endpoint.shard_id = shard
            endpoint.biz_name = "pp_biz_" + str(shard)
            endpoint.host = "127.0.0.1"
            endpoint.grpc_port = 21000 + shard
            endpoints.append(endpoint)
        config.shards = endpoints
        self.assertTrue(config.valid())
        agent_config = core.NaviSchedAgentConfig()
        agent_config.pp_cache_collector = config
        self.assertTrue(agent_config.pp_cache_collector.valid())
        config.pipeline_parallel_size = 1
        self.assertFalse(config.valid())
        with self.assertRaisesRegex(ValueError, "PP cache collection"):
            NaviSchedAgent(
                "invalid_pp_biz",
                total_blocks=0,
                block_size=4096,
                linear_step=1,
                hybrid_tail_required=False,
                pp_cache_collector=config,
            )

    def test_native_collector_and_observer_fail_closed_on_invalid_metadata(self):
        import navi_sched._navi_sched_core as core

        config = core.NaviSchedCollectorConfig()
        config.service.name = "pp_collector_python_test"
        config.service.enable_kmon = False
        config.service.enable_snapshot = False
        config.service.wait_no_query = False
        config.service.gig.grpc_io_thread_num = 1
        config.runtime.biz_name = "pp_collector_python_biz"
        config.runtime.thread_num = 1
        config.pp_group = "pp-a"
        config.shard_count = 4
        config.group_roles = [core.PPCacheGroupRole.FULL]
        config.group_block_scales = [1]
        collector = core.NaviSchedCollector(config)
        self.assertTrue(collector.init())
        observer = None
        try:
            self.assertGreater(collector.get_grpc_port(), 0)
            peer = Peer()
            identity = '{"group_id":0,"block_hashes":["' + "0" * 63 + '8"]}'
            peer._add_object_info("a", meta={"navi_sched_cache": identity})
            peer.names.add("a")
            observer = V6DCollectorObserver(
                peer, collector, pp_shard_count=4, invalid_object_id=0
            ).install()
            peer.mark_evicted("a")
            peer._track_objects_in_local_vineyard([Obj(key="a")])
            self.assertTrue(
                collector.publish_terminal(10, 1, core.PPCacheRequestStatus.FAILED)
            )
            with self.assertLogs(module.logger, level="ERROR"):
                result = asyncio.run(
                    peer.update_meta(["a"], [{"navi_sched_cache": "invalid"}])
                )
            self.assertIsNone(result)
            self.assertTrue(observer._failed)
            self.assertFalse(
                collector.publish_terminal(10, 2, core.PPCacheRequestStatus.SUCCEEDED)
            )
        finally:
            if observer:
                observer.stop()
            collector.stop()


class PeerReadinessObserverTest(unittest.TestCase):
    class Mode(Enum):
        TRACKER_ONLY = 0
        DUAL_WRITE_READ_KVCM = 2
        KVCM_ONLY = 3

    class Tracker:
        def __init__(self, mode):
            self.mode = mode

        def update_mode(self, mode):
            self.mode = mode
            return mode

    class Breaker:
        def __init__(self):
            self._state = "closed"
            self.calls = 0
            self.error = None

        def record_success(self):
            self.calls += 1
            self._state = "closed"
            return self.calls

        def record_failure(self):
            self.calls += 1
            self._state = "open"
            if self.error is not None:
                raise self.error
            return self.calls

    class Client:
        def __init__(self):
            self._circuit_breaker = PeerReadinessObserverTest.Breaker()
            self.report_ready = False
            self.transfer_ready = False
            self.result = object()

        def init_report_event_runtime(self):
            self.report_ready = True
            return self.result

        def init_transfer_runtime(self):
            self.transfer_ready = True
            return self.result

    class Store:
        def __init__(self):
            self._client = PeerReadinessObserverTest.Client()
            self.ready_event = threading.Event()

        @property
        def circuit_breaker_state(self):
            return self._client._circuit_breaker._state

    def setUp(self):
        self.peer = Peer()
        self.peer.config.tair_kvcm_p2p_enabled = True
        self.peer._tracker = self.Tracker(self.Mode.KVCM_ONLY)
        self.peer._tair_kvcm_store = self.Store()
        self.client = self.peer._tair_kvcm_store._client
        self.readiness_reads = 0

        def report_ready():
            self.readiness_reads += 1
            return self.client.report_ready

        self.peer._is_tair_kvcm_report_event_ready = report_ready
        self.peer._is_tair_kvcm_cache_transfer_ready = (
            lambda: self.client.transfer_ready
        )
        self.collector = Collector()
        self.observer = V6DCollectorObserver(
            self.peer, self.collector, pp_shard_count=4, invalid_object_id=0
        ).install()
        self.addCleanup(self.observer.stop)

    def test_native_runtime_mode_and_breaker_changes_push_only_transitions(self):
        self.assertEqual([False], self.collector.peer_read_states)
        self.assertIs(self.client.result, self.client.init_report_event_runtime())
        self.assertEqual([False], self.collector.peer_read_states)
        self.assertIs(self.client.result, self.client.init_transfer_runtime())
        self.assertEqual([False, True], self.collector.peer_read_states)
        reads = self.readiness_reads
        for index in range(1000):
            self.assertEqual(index + 1, self.client._circuit_breaker.record_success())
        self.assertEqual(reads, self.readiness_reads)
        self.client._circuit_breaker.record_failure()
        self.assertEqual([False, True, False], self.collector.peer_read_states)
        self.assertEqual(
            self.Mode.TRACKER_ONLY,
            self.peer._tracker.update_mode(self.Mode.TRACKER_ONLY),
        )
        self.client._circuit_breaker.record_success()
        self.assertEqual([False, True, False], self.collector.peer_read_states)
        self.peer._tracker.update_mode(self.Mode.DUAL_WRITE_READ_KVCM)
        self.assertEqual([False, True, False, True], self.collector.peer_read_states)

    def test_ready_event_covers_initialization_already_in_flight_at_attachment(self):
        # An original native call may have started before the method was wrapped.
        self.client.report_ready = True
        self.Client.init_transfer_runtime(self.client)
        self.assertEqual([False], self.collector.peer_read_states)
        self.assertIsNone(self.peer._tair_kvcm_store.ready_event.set())
        self.assertEqual([False, True], self.collector.peer_read_states)

    def test_failed_status_report_preserves_native_exception_and_closes_stream(self):
        self.client.init_report_event_runtime()
        self.client.init_transfer_runtime()
        self.collector.reject = True
        error = RuntimeError("native breaker error")
        self.client._circuit_breaker.error = error
        with self.assertLogs(module.logger, level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                self.client._circuit_breaker.record_failure()
        self.assertIs(error, caught.exception)
        self.assertTrue(self.collector.stopped)

    def test_stop_restores_native_status_hooks(self):
        self.observer.stop()
        self.assertEqual(
            self.Tracker.update_mode, self.peer._tracker.update_mode.__func__
        )
        self.assertEqual(
            self.Breaker.record_success,
            self.client._circuit_breaker.record_success.__func__,
        )
        self.assertEqual(
            self.Client.init_transfer_runtime,
            self.client.init_transfer_runtime.__func__,
        )
        count = len(self.collector.peer_read_states)
        self.client.init_transfer_runtime()
        self.assertEqual(count, len(self.collector.peer_read_states))


if __name__ == "__main__":
    unittest.main()
