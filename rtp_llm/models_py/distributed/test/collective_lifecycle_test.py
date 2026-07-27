import sys
import tempfile
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, Mock, mock_open, patch

from rtp_llm.models_py.distributed import collective_torch as ct


class CollectiveLifecycleRegistryTest(unittest.TestCase):
    def _registry(self, calls):
        registry = ct._CollectiveLifecycleRegistry()
        for name in (
            "torch_process_groups",
            "cpu_tp_broadcaster",
            "comm_ops",
            "user_buffers",
        ):
            registry.register(
                name,
                rebuild=lambda name=name: calls.append(f"rebuild:{name}"),
                teardown=lambda name=name: calls.append(f"teardown:{name}"),
            )
        return registry

    def test_three_idempotent_teardown_rebuild_cycles_follow_dependency_order(self):
        calls = []
        registry = self._registry(calls)

        for _ in range(3):
            registry.rebuild()
            registry.rebuild()
            registry.teardown()
            registry.teardown()

        one_cycle = [
            "rebuild:torch_process_groups",
            "rebuild:cpu_tp_broadcaster",
            "rebuild:comm_ops",
            "rebuild:user_buffers",
            "teardown:user_buffers",
            "teardown:comm_ops",
            "teardown:cpu_tp_broadcaster",
            "teardown:torch_process_groups",
        ]
        self.assertEqual(calls, one_cycle * 3)
        self.assertEqual(registry.active_resources(), [])

    def test_failed_rebuild_reports_resource_and_retry_resumes(self):
        calls = []
        fail_once = [True]
        registry = ct._CollectiveLifecycleRegistry()
        registry.register(
            "torch_process_groups",
            rebuild=lambda: calls.append("rebuild:torch_process_groups"),
            teardown=lambda: None,
        )

        def rebuild_comm_ops():
            calls.append("rebuild:comm_ops")
            if fail_once[0]:
                fail_once[0] = False
                raise ValueError("injected")

        registry.register(
            "comm_ops",
            rebuild=rebuild_comm_ops,
            teardown=lambda: calls.append("cleanup:comm_ops"),
        )

        with self.assertRaisesRegex(RuntimeError, "comm_ops"):
            registry.rebuild()
        self.assertEqual(registry.active_resources(), ["torch_process_groups"])

        registry.rebuild()
        self.assertEqual(
            calls,
            [
                "rebuild:torch_process_groups",
                "rebuild:comm_ops",
                "cleanup:comm_ops",
                "rebuild:comm_ops",
            ],
        )

    def test_failed_teardown_does_not_destroy_its_dependency(self):
        calls = []
        fail_once = [True]
        registry = ct._CollectiveLifecycleRegistry()
        registry.register(
            "torch_process_groups",
            rebuild=lambda: None,
            teardown=lambda: calls.append("teardown:torch_process_groups"),
        )

        def teardown_user_buffers():
            calls.append("teardown:user_buffers")
            if fail_once[0]:
                fail_once[0] = False
                raise ValueError("injected")

        registry.register(
            "user_buffers", rebuild=lambda: None, teardown=teardown_user_buffers
        )
        registry.rebuild()

        with self.assertRaisesRegex(RuntimeError, "user_buffers"):
            registry.teardown()
        self.assertEqual(
            registry.active_resources(), ["torch_process_groups", "user_buffers"]
        )
        self.assertEqual(calls, ["teardown:user_buffers"])

        registry.teardown()
        self.assertEqual(
            calls,
            [
                "teardown:user_buffers",
                "teardown:user_buffers",
                "teardown:torch_process_groups",
            ],
        )

    def test_failed_teardown_reports_original_error(self):
        registry = ct._CollectiveLifecycleRegistry()
        registry.register(
            "torch_process_groups",
            rebuild=lambda: None,
            teardown=lambda: (_ for _ in ()).throw(
                ValueError("injected teardown detail")
            ),
        )
        registry.rebuild()

        with self.assertRaisesRegex(
            RuntimeError,
            "torch_process_groups.*ValueError: injected teardown detail",
        ) as raised:
            registry.teardown()

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual(registry.active_resources(), ["torch_process_groups"])


class Level3PhaseCoordinationTest(unittest.TestCase):
    class Store:
        def __init__(self, values=None, wait_error=None):
            self.values = dict(values or {})
            self.wait_error = wait_error
            self.set_calls = []
            self.wait_calls = []

        def set(self, key, value):
            self.set_calls.append((key, value))
            self.values[key] = value

        def wait(self, keys, timeout):
            self.wait_calls.append((keys, timeout))
            if self.wait_error is not None:
                raise self.wait_error
            missing = [key for key in keys if key not in self.values]
            if missing:
                raise RuntimeError(f"missing keys: {missing}")

        def get(self, key):
            return self.values[key]

    @staticmethod
    def _snapshot():
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(world_rank=0, world_size=2),
            timeout=17,
        )

    def setUp(self):
        ct._reset_multicast_keeper_phase_instance()

    def tearDown(self):
        ct._reset_multicast_keeper_phase_instance()

    def test_all_ranks_must_report_success(self):
        peer_key = "rtp_llm_level3/9/collectives_down/rank/1"
        store = self.Store({peer_key: b"1"})
        with patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(ct, "_get_or_create_lifecycle_store", return_value=store):
            self.assertTrue(ct.coordinate_level3_phase("collectives_down", 9, True))

        self.assertEqual(
            store.set_calls,
            [("rtp_llm_level3/9/collectives_down/rank/0", b"1")],
        )
        self.assertEqual(store.wait_calls[0][1].total_seconds(), 17)

    def test_remote_failure_fails_closed(self):
        peer_key = "rtp_llm_level3/3/collectives_ready/rank/1"
        store = self.Store({peer_key: b"0"})
        with patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(ct, "_get_or_create_lifecycle_store", return_value=store):
            self.assertFalse(ct.coordinate_level3_phase("collectives_ready", 3, True))

    def test_local_failure_is_published_before_waiting_for_peers(self):
        peer_key = "rtp_llm_level3/3/collectives_ready/rank/1"
        store = self.Store({peer_key: b"1"})
        with patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(ct, "_get_or_create_lifecycle_store", return_value=store):
            self.assertFalse(ct.coordinate_level3_phase("collectives_ready", 3, False))

        self.assertEqual(
            store.set_calls,
            [("rtp_llm_level3/3/collectives_ready/rank/0", b"0")],
        )
        self.assertEqual(len(store.wait_calls), 1)

    def _keeper_ping_exchange(self, response):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = f"{directory}/keeper.sock"
            ready = threading.Event()
            requests = []
            server_errors = []

            def serve():
                try:
                    with ct.socket.socket(
                        ct.socket.AF_UNIX, ct.socket.SOCK_SEQPACKET
                    ) as server:
                        server.bind(socket_path)
                        server.listen(1)
                        server.settimeout(2)
                        ready.set()
                        connection, _ = server.accept()
                        with connection:
                            requests.append(
                                connection.recv(ct._MULTICAST_REQUEST.size + 1)
                            )
                            connection.sendall(response)
                except Exception as error:
                    server_errors.append(error)
                    ready.set()

            thread = threading.Thread(target=serve)
            thread.start()
            self.assertTrue(ready.wait(timeout=2), "keeper test server did not start")
            result = ct._ping_multicast_keeper(socket_path)
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "keeper test server did not exit")
            self.assertEqual(server_errors, [])
            self.assertEqual(len(requests), 1)
            return result, ct._MULTICAST_REQUEST.unpack(requests[0])

    @staticmethod
    def _keeper_ping_response(*, version=ct._MULTICAST_PROTOCOL_VERSION):
        return ct._MULTICAST_RESPONSE.pack(
            ct._MULTICAST_PROTOCOL_MAGIC,
            version,
            ct._MULTICAST_OP_PING,
            ct._MULTICAST_RESPONSE.size,
            ct._MULTICAST_STATUS_OK,
            0,
            0x1234,
            0x5678,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def test_keeper_ping_accepts_only_a_live_compatible_holder(self):
        result, request = self._keeper_ping_exchange(self._keeper_ping_response())

        self.assertEqual(result, (True, "", (0x1234, 0x5678)))
        self.assertEqual(request[0], ct._MULTICAST_PROTOCOL_MAGIC)
        self.assertEqual(request[1], 3)
        self.assertEqual(request[2], ct._MULTICAST_OP_PING)
        self.assertEqual(request[3], ct._MULTICAST_REQUEST.size)

    def test_keeper_ping_fails_closed_on_incompatible_or_short_response(self):
        responses = (
            (self._keeper_ping_response(version=1), "incompatible response"),
            (self._keeper_ping_response()[:-1], "invalid response size"),
        )
        for response, expected_error in responses:
            with self.subTest(expected_error=expected_error):
                (ready, error, instance), _ = self._keeper_ping_exchange(response)
                self.assertFalse(ready)
                self.assertIsNone(instance)
                self.assertIn(expected_error, error)

    def test_keeper_failure_is_reported_at_resource_ready_gate(self):
        peer_key = "rtp_llm_level3/5/collective_rebuild_ready/rank/1"
        store = self.Store({peer_key: b"1"})
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
            },
            clear=False,
        ), patch.object(
            ct,
            "_multicast_keeper_ready",
            return_value=(False, "holder is dead", None),
        ) as keeper_ready, patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(
            ct, "_get_or_create_lifecycle_store", return_value=store
        ):
            self.assertFalse(
                ct.coordinate_level3_phase("collective_rebuild_ready", 5, True)
            )

        keeper_ready.assert_called_once_with()
        self.assertEqual(
            store.set_calls,
            [("rtp_llm_level3/5/collective_rebuild_ready/rank/0", b"0")],
        )

    def test_keeper_instance_is_pinned_until_successful_graph_recapture(self):
        instance = (0xAA, 0xBB)
        store = self.Store(
            {
                "rtp_llm_level3/7/collective_teardown_ready/rank/1": b"1",
                "rtp_llm_level3/7/collective_rebuild_ready/rank/1": b"1",
                "rtp_llm_level3/7/collective_rebuild_done/rank/1": b"1",
                "rtp_llm_level3/7/graph_recapture_ready/rank/1": b"1",
                "rtp_llm_level3/7/graph_recapture_done/rank/1": b"1",
            }
        )
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
            },
            clear=False,
        ), patch.object(
            ct, "_multicast_keeper_ready", return_value=(True, "", instance)
        ) as keeper_ready, patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(
            ct, "_get_or_create_lifecycle_store", return_value=store
        ):
            self.assertTrue(
                ct.coordinate_level3_phase("collective_teardown_ready", 7, True)
            )
            self.assertEqual(ct._multicast_keeper_pinned_epoch, 7)
            self.assertEqual(ct._multicast_keeper_pinned_instance, instance)
            self.assertTrue(
                ct.coordinate_level3_phase("collective_rebuild_ready", 7, True)
            )
            self.assertEqual(ct._multicast_keeper_pinned_epoch, 7)
            self.assertEqual(ct._multicast_keeper_pinned_instance, instance)
            self.assertTrue(
                ct.coordinate_level3_phase("collective_rebuild_done", 7, True)
            )
            self.assertEqual(ct._multicast_keeper_pinned_epoch, 7)
            self.assertEqual(ct._multicast_keeper_pinned_instance, instance)
            self.assertTrue(
                ct.coordinate_level3_phase("graph_recapture_ready", 7, True)
            )
            self.assertEqual(ct._multicast_keeper_pinned_epoch, 7)
            self.assertEqual(ct._multicast_keeper_pinned_instance, instance)
            self.assertTrue(ct.coordinate_level3_phase("graph_recapture_done", 7, True))

        self.assertIsNone(ct._multicast_keeper_pinned_epoch)
        self.assertIsNone(ct._multicast_keeper_pinned_instance)
        self.assertEqual(keeper_ready.call_count, 4)

    def test_graph_recapture_done_rejects_dead_or_replaced_keeper(self):
        original = (0xAA, 0xBB)
        failures = (
            ((False, "holder is dead", None), "holder is dead"),
            ((True, "", (0xCC, 0xDD)), "replaced holder"),
        )
        for keeper_result, label in failures:
            with self.subTest(label=label):
                ct._reset_multicast_keeper_phase_instance()
                ready, error = ct._validate_multicast_keeper_phase_instance(
                    "collective_teardown_ready", 7, original
                )
                self.assertTrue(ready, error)
                store = self.Store(
                    {"rtp_llm_level3/7/graph_recapture_done/rank/1": b"1"}
                )
                with patch.dict(
                    ct.os.environ,
                    {
                        "ENABLE_SLEEP_MODE": "1",
                        "SLEEP_MODE_LEVEL": "3",
                        "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
                    },
                    clear=False,
                ), patch.object(
                    ct, "_multicast_keeper_ready", return_value=keeper_result
                ), patch.object(
                    ct,
                    "_require_distributed_init_snapshot",
                    return_value=self._snapshot(),
                ), patch.object(
                    ct, "_get_or_create_lifecycle_store", return_value=store
                ):
                    self.assertFalse(
                        ct.coordinate_level3_phase("graph_recapture_done", 7, True)
                    )

                self.assertEqual(ct._multicast_keeper_pinned_epoch, 7)
                self.assertEqual(ct._multicast_keeper_pinned_instance, original)
                self.assertEqual(
                    store.set_calls[-1],
                    ("rtp_llm_level3/7/graph_recapture_done/rank/0", b"0"),
                )

    def test_replacement_keeper_is_rejected_and_original_identity_stays_pinned(self):
        original = (0xAA, 0xBB)
        replacement = (0xCC, 0xDD)
        store = self.Store(
            {
                "rtp_llm_level3/8/collective_teardown_ready/rank/1": b"1",
                "rtp_llm_level3/8/collective_rebuild_ready/rank/1": b"1",
            }
        )
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
            },
            clear=False,
        ), patch.object(
            ct,
            "_multicast_keeper_ready",
            side_effect=[(True, "", original), (True, "", replacement)],
        ), patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(
            ct, "_get_or_create_lifecycle_store", return_value=store
        ):
            self.assertTrue(
                ct.coordinate_level3_phase("collective_teardown_ready", 8, True)
            )
            self.assertFalse(
                ct.coordinate_level3_phase("collective_rebuild_ready", 8, True)
            )

        self.assertEqual(ct._multicast_keeper_pinned_epoch, 8)
        self.assertEqual(ct._multicast_keeper_pinned_instance, original)
        self.assertEqual(
            store.set_calls[-1],
            ("rtp_llm_level3/8/collective_rebuild_ready/rank/0", b"0"),
        )

    def test_incomplete_epoch_pin_rejects_next_sleep_epoch(self):
        instance = (0xAA, 0xBB)
        ready, error = ct._validate_multicast_keeper_phase_instance(
            "collective_teardown_ready", 9, instance
        )
        self.assertTrue(ready, error)

        ready, error = ct._validate_multicast_keeper_phase_instance(
            "collective_teardown_ready", 10, instance
        )
        self.assertFalse(ready)
        self.assertIn("incomplete epoch 9", error)
        self.assertEqual(ct._multicast_keeper_pinned_epoch, 9)
        self.assertEqual(ct._multicast_keeper_pinned_instance, instance)

    def test_graph_recapture_done_cannot_clear_a_stale_epoch_pin(self):
        instance = (0xAA, 0xBB)
        ready, error = ct._validate_multicast_keeper_phase_instance(
            "collective_teardown_ready", 11, instance
        )
        self.assertTrue(ready, error)
        store = self.Store({"rtp_llm_level3/12/graph_recapture_done/rank/1": b"1"})
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
            },
            clear=False,
        ), patch.object(
            ct, "_multicast_keeper_ready", return_value=(True, "", instance)
        ), patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(
            ct, "_get_or_create_lifecycle_store", return_value=store
        ):
            self.assertFalse(
                ct.coordinate_level3_phase("graph_recapture_done", 12, True)
            )

        self.assertEqual(ct._multicast_keeper_pinned_epoch, 11)
        self.assertEqual(ct._multicast_keeper_pinned_instance, instance)

    def test_timeout_is_reported_as_phase_failure(self):
        store = self.Store(wait_error=TimeoutError("peer missing"))
        with patch.object(
            ct, "_require_distributed_init_snapshot", return_value=self._snapshot()
        ), patch.object(ct, "_get_or_create_lifecycle_store", return_value=store):
            with self.assertRaisesRegex(
                RuntimeError, "collectives_ready.*peer missing"
            ):
                ct.coordinate_level3_phase("collectives_ready", 4, True)

    def test_invalid_identity_is_rejected_before_store_access(self):
        with self.assertRaises(ValueError):
            ct.coordinate_level3_phase("bad/phase", 1, True)
        with self.assertRaises(ValueError):
            ct.coordinate_level3_phase("collectives_ready", 0, True)

    def test_process_group_teardown_clears_stale_globals(self):
        old_group_map = ct._group_map
        old_parallelism_config = ct._parallelism_config
        old_initialized = ct._initialized
        try:
            ct._group_map = {ct.Group.DP_AND_TP: object()}
            ct._parallelism_config = SimpleNamespace(world_rank=0)
            ct._initialized = True
            with patch.object(
                ct.torch.distributed, "is_initialized", return_value=False
            ), patch.object(ct, "_get_rocm_rccl", return_value=None):
                ct._destroy_torch_process_groups()

            self.assertEqual(ct._group_map, {})
            self.assertIsNone(ct._parallelism_config)
            self.assertFalse(ct._initialized)
        finally:
            ct._group_map = old_group_map
            ct._parallelism_config = old_parallelism_config
            ct._initialized = old_initialized

    def test_process_group_teardown_destroys_symm_mem_before_all_groups(self):
        old_group_map = ct._group_map
        old_snapshot = ct._distributed_init_snapshot
        old_parallelism_config = ct._parallelism_config
        old_initialized = ct._initialized
        quiesce_group = object()
        calls = []
        symm_mem = SimpleNamespace(
            destroy_symm_mem_communicator=lambda: calls.append("symm_mem")
        )
        ct._distributed_init_snapshot = SimpleNamespace(
            parallelism_config=SimpleNamespace(local_rank=0)
        )
        try:
            with patch.object(
                ct.torch.distributed,
                "is_initialized",
                side_effect=[True, False] * 3,
            ), patch.object(
                ct.torch.distributed,
                "barrier",
                side_effect=lambda **kwargs: calls.append(
                    ("barrier", kwargs.get("group"))
                ),
            ), patch.object(
                ct.torch.distributed,
                "destroy_process_group",
                side_effect=lambda: calls.append("process_groups"),
            ), patch.object(
                ct, "_get_rocm_rccl", return_value=None
            ), patch.object(
                ct, "_get_symm_mem", return_value=symm_mem
            ), patch.object(
                ct.torch.cuda, "is_available", return_value=False
            ):
                for _ in range(3):
                    ct._group_map = {
                        ct.Group.DP_AND_TP: object(),
                        ct.Group.SLEEP_QUIESCE: quiesce_group,
                    }
                    ct._parallelism_config = SimpleNamespace(world_rank=0)
                    ct._initialized = True
                    ct._destroy_torch_process_groups()

            self.assertEqual(
                calls,
                [
                    ("barrier", quiesce_group),
                    "symm_mem",
                    "process_groups",
                ]
                * 3,
            )
        finally:
            ct._group_map = old_group_map
            ct._distributed_init_snapshot = old_snapshot
            ct._parallelism_config = old_parallelism_config
            ct._initialized = old_initialized

    def test_process_group_rebuild_initializes_symm_mem_after_world_group(self):
        old_group_map = ct._group_map
        old_snapshot = ct._distributed_init_snapshot
        old_parallelism_config = ct._parallelism_config
        old_initialized = ct._initialized
        old_generation = ct._process_group_generation
        world_group = object()
        calls = []
        parallelism_config = SimpleNamespace(
            world_rank=0,
            world_size=2,
            local_rank=0,
            tp_size=2,
            dp_size=1,
        )
        ct._distributed_init_snapshot = SimpleNamespace(
            parallelism_config=parallelism_config,
            nccl_comm_config=SimpleNamespace(nccl_ip="127.0.0.1"),
            nccl_init_port=12345,
            backend="nccl",
            timeout=10,
        )
        ct._initialized = False
        try:
            fake_symm_mem = SimpleNamespace(
                init_symm_mem_communicator=lambda group: calls.append(
                    ("symm_mem", group)
                )
            )
            with patch.object(ct, "_normalize_parallelism_ranks"), patch.object(
                ct,
                "_make_cpu_tp_broadcaster_base_path",
                return_value="/tmp/test-broadcaster",
            ), patch.object(
                ct.torch.distributed, "is_initialized", return_value=False
            ), patch.object(
                ct, "_get_or_create_lifecycle_store", return_value=object()
            ), patch.object(
                ct, "_wait_for_process_group_generation_ready"
            ), patch.object(
                ct, "_get_rocm_rccl", return_value=None
            ), patch.object(
                ct.torch.distributed,
                "PrefixStore",
                side_effect=lambda _prefix, store: store,
            ), patch.object(
                ct.torch.cuda, "is_available", return_value=False
            ), patch.object(
                ct.torch.distributed,
                "init_process_group",
                side_effect=lambda **_kwargs: calls.append("process_group"),
            ), patch.object(
                ct.torch.distributed,
                "group",
                SimpleNamespace(WORLD=world_group),
            ), patch.object(
                ct.torch.distributed, "barrier"
            ), patch.object(
                ct, "_get_symm_mem", return_value=fake_symm_mem
            ), patch.object(
                ct, "_maybe_create_sleep_quiesce_group"
            ):
                ct._rebuild_torch_process_groups()

            self.assertEqual(calls, ["process_group", ("symm_mem", world_group)])
        finally:
            ct._group_map = old_group_map
            ct._distributed_init_snapshot = old_snapshot
            ct._parallelism_config = old_parallelism_config
            ct._initialized = old_initialized
            ct._process_group_generation = old_generation

    def test_process_group_teardown_ignores_post_destroy_cleanup_failures(self):
        old_group_map = ct._group_map
        old_parallelism_config = ct._parallelism_config
        old_initialized = ct._initialized
        try:
            for failed_operation in ("gc.collect", "ipc_collect"):
                with self.subTest(failed_operation=failed_operation):
                    ct._group_map = {ct.Group.DP_AND_TP: object()}
                    ct._parallelism_config = SimpleNamespace(world_rank=0)
                    ct._initialized = True
                    gc_collect = Mock()
                    ipc_collect = Mock()
                    empty_cache = Mock()
                    cleanup = {
                        "gc.collect": gc_collect,
                        "ipc_collect": ipc_collect,
                    }
                    cleanup[failed_operation].side_effect = RuntimeError(
                        "CUDA error: invalid argument"
                    )
                    registry = ct._CollectiveLifecycleRegistry()
                    registry.register(
                        "torch_process_groups",
                        rebuild=lambda: None,
                        teardown=ct._destroy_torch_process_groups,
                    )
                    registry.rebuild()
                    with patch.object(
                        ct.torch.distributed,
                        "is_initialized",
                        return_value=False,
                    ), patch.object(
                        ct, "_get_rocm_rccl", return_value=None
                    ), patch.object(
                        ct.gc, "collect", gc_collect
                    ), patch.object(
                        ct.torch.cuda, "is_available", return_value=True
                    ), patch.object(
                        ct.torch.cuda, "ipc_collect", ipc_collect
                    ), patch.object(
                        ct.torch.cuda, "empty_cache", empty_cache
                    ), self.assertLogs(
                        level="WARNING"
                    ) as logs:
                        registry.teardown()

                    gc_collect.assert_called_once_with()
                    ipc_collect.assert_called_once_with()
                    empty_cache.assert_not_called()
                    self.assertTrue(
                        any(failed_operation in line for line in logs.output)
                    )
                    self.assertEqual(ct._group_map, {})
                    self.assertIsNone(ct._parallelism_config)
                    self.assertFalse(ct._initialized)
                    self.assertEqual(registry.active_resources(), [])
        finally:
            ct._group_map = old_group_map
            ct._parallelism_config = old_parallelism_config
            ct._initialized = old_initialized

    def test_process_group_teardown_rejects_incomplete_destroy(self):
        old_group_map = ct._group_map
        old_snapshot = ct._distributed_init_snapshot
        old_parallelism_config = ct._parallelism_config
        old_initialized = ct._initialized
        ct._group_map = {ct.Group.DP_AND_TP: object()}
        ct._distributed_init_snapshot = SimpleNamespace(
            parallelism_config=SimpleNamespace(local_rank=0)
        )
        ct._parallelism_config = SimpleNamespace(world_rank=0)
        ct._initialized = True
        try:
            with patch.object(
                ct.torch.distributed, "is_initialized", return_value=True
            ), patch.object(ct.torch.distributed, "barrier"), patch.object(
                ct.torch.distributed, "destroy_process_group"
            ), patch.object(
                ct, "_get_rocm_rccl", return_value=None
            ), patch.object(
                ct,
                "_get_symm_mem",
                return_value=SimpleNamespace(
                    destroy_symm_mem_communicator=lambda: None
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "remained initialized"):
                    ct._destroy_torch_process_groups()
        finally:
            ct._group_map = old_group_map
            ct._distributed_init_snapshot = old_snapshot
            ct._parallelism_config = old_parallelism_config
            ct._initialized = old_initialized

    def test_lifecycle_store_is_cpu_only_and_reused_across_generations(self):
        old_store = ct._lifecycle_store
        store = object()
        snapshot = SimpleNamespace(
            timeout=45,
            nccl_init_port=12345,
            nccl_comm_config=SimpleNamespace(nccl_ip="127.0.0.1"),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=2),
        )
        ct._lifecycle_store = None
        try:
            with patch.object(
                ct.torch.distributed, "TCPStore", return_value=store
            ) as tcp_store:
                self.assertIs(ct._get_or_create_lifecycle_store(snapshot), store)
                self.assertIs(ct._get_or_create_lifecycle_store(snapshot), store)

            tcp_store.assert_called_once_with(
                host_name="127.0.0.1",
                port=12345,
                world_size=2,
                is_master=True,
                timeout=ct.timedelta(seconds=45),
                wait_for_workers=True,
                multi_tenant=True,
            )
        finally:
            ct._lifecycle_store = old_store

    def test_generation_ready_barrier_waits_for_every_rank(self):
        store = Mock()
        timeout = ct.timedelta(seconds=41)

        ct._wait_for_process_group_generation_ready(
            store,
            generation=2,
            world_rank=0,
            world_size=3,
            timeout=timeout,
        )

        store.set.assert_called_once_with("rtp_llm_pg_ready/2/rank/0", b"1")
        store.wait.assert_called_once_with(
            [
                "rtp_llm_pg_ready/2/rank/0",
                "rtp_llm_pg_ready/2/rank/1",
                "rtp_llm_pg_ready/2/rank/2",
            ],
            timeout,
        )

    def test_generation_ready_barrier_keys_are_generation_scoped(self):
        store = Mock()
        timeout = ct.timedelta(seconds=12)

        for generation in (1, 2):
            ct._wait_for_process_group_generation_ready(
                store,
                generation=generation,
                world_rank=1,
                world_size=2,
                timeout=timeout,
            )

        self.assertEqual(
            [call.args[0] for call in store.set.call_args_list],
            ["rtp_llm_pg_ready/1/rank/1", "rtp_llm_pg_ready/2/rank/1"],
        )
        generation_one_keys = store.wait.call_args_list[0].args[0]
        generation_two_keys = store.wait.call_args_list[1].args[0]
        self.assertTrue(all("/1/" in key for key in generation_one_keys))
        self.assertTrue(all("/2/" in key for key in generation_two_keys))
        self.assertTrue(set(generation_one_keys).isdisjoint(generation_two_keys))

    def test_generation_ready_barrier_world_one_waits_for_own_key(self):
        store = Mock()
        timeout = ct.timedelta(seconds=8)

        ct._wait_for_process_group_generation_ready(
            store,
            generation=1,
            world_rank=0,
            world_size=1,
            timeout=timeout,
        )

        store.set.assert_called_once_with("rtp_llm_pg_ready/1/rank/0", b"1")
        store.wait.assert_called_once_with(["rtp_llm_pg_ready/1/rank/0"], timeout)

    def test_generation_ready_barrier_propagates_timeout_with_context(self):
        timeout_error = RuntimeError("Wait timeout after 9 seconds")
        store = Mock()
        store.wait.side_effect = timeout_error
        timeout = ct.timedelta(seconds=9)

        with self.assertRaisesRegex(
            RuntimeError,
            "generation 7 ready barrier failed for rank 1 of 2.*Wait timeout",
        ) as raised:
            ct._wait_for_process_group_generation_ready(
                store,
                generation=7,
                world_rank=1,
                world_size=2,
                timeout=timeout,
            )

        self.assertIs(raised.exception.__cause__, timeout_error)
        store.wait.assert_called_once_with(
            ["rtp_llm_pg_ready/7/rank/0", "rtp_llm_pg_ready/7/rank/1"],
            timeout,
        )

    def test_generation_ready_failure_precedes_cuda_and_nccl_init(self):
        old_snapshot = ct._distributed_init_snapshot
        old_generation = ct._process_group_generation
        snapshot = SimpleNamespace(
            timeout=13,
            nccl_init_port=12345,
            nccl_comm_config=SimpleNamespace(nccl_ip="127.0.0.1"),
            parallelism_config=SimpleNamespace(
                world_rank=0,
                world_size=2,
                local_rank=0,
                tp_size=2,
            ),
            backend="nccl",
        )
        ct._distributed_init_snapshot = snapshot
        ct._process_group_generation = 1
        try:
            with patch.object(ct, "_normalize_parallelism_ranks"), patch.object(
                ct, "_make_cpu_tp_broadcaster_base_path", return_value="/tmp/test"
            ), patch.object(ct, "_get_rocm_rccl", return_value=None), patch.object(
                ct.torch.distributed, "is_initialized", return_value=False
            ), patch.object(
                ct, "_get_or_create_lifecycle_store", return_value=Mock()
            ), patch.object(
                ct,
                "_wait_for_process_group_generation_ready",
                side_effect=RuntimeError("ready timeout"),
            ) as ready, patch.object(
                ct.torch.cuda, "is_available"
            ) as cuda_available, patch.object(
                ct.torch.distributed, "init_process_group"
            ) as init_process_group:
                with self.assertRaisesRegex(RuntimeError, "ready timeout"):
                    ct._rebuild_torch_process_groups()

            ready.assert_called_once_with(
                ANY,
                generation=2,
                world_rank=0,
                world_size=2,
                timeout=ct.timedelta(seconds=13),
                namespace="",
            )
            cuda_available.assert_not_called()
            init_process_group.assert_not_called()
        finally:
            ct._distributed_init_snapshot = old_snapshot
            ct._process_group_generation = old_generation

    def test_level_three_disables_multicast_before_initial_lifecycle_rebuild(self):
        old_snapshot = ct._distributed_init_snapshot
        old_initialized = ct._initialized
        ct._distributed_init_snapshot = None
        ct._initialized = False
        observed = []
        try:
            with patch.dict(
                ct.os.environ,
                {
                    "ENABLE_SLEEP_MODE": "1",
                    "SLEEP_MODE_LEVEL": "3",
                    "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "0",
                    "NCCL_NVLS_ENABLE": "1",
                    "TORCH_SYMM_MEM_DISABLE_MULTICAST": "0",
                },
                clear=False,
            ), patch.object(
                ct, "_ensure_collective_resources_registered"
            ), patch.object(
                ct._collective_lifecycle,
                "active_resources",
                return_value=[],
            ), patch.object(
                ct._collective_lifecycle,
                "rebuild",
                side_effect=lambda: observed.append(
                    (
                        ct.os.environ.get("NCCL_NVLS_ENABLE"),
                        ct.os.environ.get("TORCH_SYMM_MEM_DISABLE_MULTICAST"),
                    )
                ),
            ):
                with self.assertLogs(level="WARNING") as logs:
                    ct.init_distributed_environment(
                        SimpleNamespace(), SimpleNamespace(), 12345
                    )

                self.assertEqual(observed, [("0", "1")])
                self.assertTrue(
                    any("overrides NCCL_NVLS_ENABLE=1" in line for line in logs.output)
                )
                self.assertTrue(
                    any(
                        "overrides TORCH_SYMM_MEM_DISABLE_MULTICAST=0" in line
                        for line in logs.output
                    )
                )
        finally:
            ct._distributed_init_snapshot = old_snapshot
            ct._initialized = old_initialized

    def test_non_level_three_preserves_user_multicast_settings(self):
        for enabled, level in (("0", "3"), ("1", "2")):
            with self.subTest(enabled=enabled, level=level), patch.dict(
                ct.os.environ,
                {
                    "ENABLE_SLEEP_MODE": enabled,
                    "SLEEP_MODE_LEVEL": level,
                    "NCCL_NVLS_ENABLE": "1",
                    "TORCH_SYMM_MEM_DISABLE_MULTICAST": "0",
                },
                clear=False,
            ):
                ct._enforce_level3_multicast_disabled()
                self.assertEqual(ct.os.environ["NCCL_NVLS_ENABLE"], "1")
                self.assertEqual(ct.os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"], "0")

    def test_level_three_sets_unconfigured_multicast_environment(self):
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "0",
            },
            clear=False,
        ):
            ct.os.environ.pop("NCCL_NVLS_ENABLE", None)
            ct.os.environ.pop("TORCH_SYMM_MEM_DISABLE_MULTICAST", None)
            ct._enforce_level3_multicast_disabled()
            self.assertEqual(ct.os.environ["NCCL_NVLS_ENABLE"], "0")
            self.assertEqual(ct.os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"], "1")

    def test_level_three_keeper_enables_nvls_and_symmetric_multicast(self):
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
                "NCCL_NVLS_ENABLE": "0",
                "TORCH_SYMM_MEM_DISABLE_MULTICAST": "1",
            },
            clear=False,
        ), patch.object(ct, "_multicast_keeper_ready", return_value=(True, "", (1, 2))):
            ct._configure_level3_multicast()
            self.assertEqual(ct.os.environ["NCCL_NVLS_ENABLE"], "1")
            self.assertEqual(ct.os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"], "0")

    def test_keeper_ready_prefers_exact_socket_environment(self):
        exact_socket = "/run/rtp-llm/custom-keeper.sock"
        with patch.dict(
            ct.os.environ,
            {
                "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET": exact_socket,
                "NEKYIA_KEEPER_DIR": "/ignored/keeper-dir",
            },
            clear=False,
        ), patch.object(
            ct.os,
            "stat",
            return_value=SimpleNamespace(st_mode=ct.stat.S_IFSOCK),
        ) as socket_stat, patch(
            "builtins.open",
            mock_open(read_data="7f00-8000 r-xp mc_shim_unified.so\n"),
        ), patch.object(
            ct, "_ping_multicast_keeper", return_value=(True, "", (1, 2))
        ):
            self.assertEqual(ct._multicast_keeper_ready(), (True, "", (1, 2)))
            socket_stat.assert_called_once_with(exact_socket)

    def test_keeper_ready_falls_back_to_nekyia_default_socket(self):
        with patch.dict(
            ct.os.environ,
            {
                "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET": "",
                "NEKYIA_KEEPER_DIR": "/run/rtp-llm/keeper",
            },
            clear=False,
        ), patch.object(
            ct.os,
            "stat",
            return_value=SimpleNamespace(st_mode=ct.stat.S_IFSOCK),
        ) as socket_stat, patch(
            "builtins.open",
            mock_open(read_data="7f00-8000 r-xp mc_shim_unified.so\n"),
        ), patch.object(
            ct, "_ping_multicast_keeper", return_value=(True, "", (1, 2))
        ):
            self.assertEqual(ct._multicast_keeper_ready(), (True, "", (1, 2)))
            socket_stat.assert_called_once_with("/run/rtp-llm/keeper/mcsk.sock")

    def test_level_three_keeper_request_fails_closed_when_shim_is_missing(self):
        with patch.dict(
            ct.os.environ,
            {
                "ENABLE_SLEEP_MODE": "1",
                "SLEEP_MODE_LEVEL": "3",
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
            },
            clear=False,
        ), patch.object(
            ct,
            "_multicast_keeper_ready",
            return_value=(False, "mc_shim_unified.so is not loaded", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "not loaded"):
                ct._configure_level3_multicast()

    def test_wake_reasserts_multicast_environment_before_lifecycle_rebuild(self):
        old_snapshot = ct._distributed_init_snapshot
        old_initialized = ct._initialized
        ct._distributed_init_snapshot = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_rank=0)
        )
        ct._initialized = False
        observed = []
        try:
            with patch.dict(
                ct.os.environ,
                {
                    "ENABLE_SLEEP_MODE": "1",
                    "SLEEP_MODE_LEVEL": "3",
                    "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "0",
                    "NCCL_NVLS_ENABLE": "1",
                    "TORCH_SYMM_MEM_DISABLE_MULTICAST": "0",
                },
                clear=False,
            ), patch.object(
                ct._collective_lifecycle,
                "active_resources",
                return_value=[],
            ), patch.object(
                ct._collective_lifecycle,
                "rebuild",
                side_effect=lambda: observed.append(
                    (
                        ct.os.environ.get("NCCL_NVLS_ENABLE"),
                        ct.os.environ.get("TORCH_SYMM_MEM_DISABLE_MULTICAST"),
                    )
                ),
            ):
                ct.rebuild_distributed_environment()
                self.assertEqual(observed, [("0", "1")])
                self.assertEqual(ct.os.environ["NCCL_NVLS_ENABLE"], "0")
                self.assertEqual(ct.os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"], "1")
        finally:
            ct._distributed_init_snapshot = old_snapshot
            ct._initialized = old_initialized

    def test_user_buffer_teardown_restores_reinitializable_global(self):
        old_snapshot = ct._distributed_init_snapshot
        fake_arch = ModuleType("rtp_llm.models_py.utils.arch")
        fake_arch.is_cuda = lambda: True
        fake_user_buffers = ModuleType("rtp_llm.models_py.distributed.user_buffers")
        fake_user_buffers._global_communicator = object()

        def destroy():
            del fake_user_buffers._global_communicator

        fake_user_buffers.destroy_user_buffers_communicator = destroy
        ct._distributed_init_snapshot = SimpleNamespace(
            parallelism_config=SimpleNamespace(use_ub_comm=True)
        )
        try:
            with patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.utils.arch": fake_arch,
                    "rtp_llm.models_py.distributed.user_buffers": fake_user_buffers,
                },
            ):
                ct._destroy_user_buffers()
            self.assertIsNone(fake_user_buffers._global_communicator)
        finally:
            ct._distributed_init_snapshot = old_snapshot


class Level3KeyNamespaceTest(unittest.TestCase):
    """FIX #6: coordination keys must be namespaced by PD role + generation."""

    @staticmethod
    def _config(role_name=None):
        role = SimpleNamespace(name=role_name) if role_name is not None else None
        return SimpleNamespace(role_type=role, world_rank=0, world_size=2)

    def setUp(self):
        self._saved_generation = ct.os.environ.get(ct._SLEEP_INSTANCE_GENERATION_ENV)
        ct.os.environ.pop(ct._SLEEP_INSTANCE_GENERATION_ENV, None)

    def tearDown(self):
        if self._saved_generation is None:
            ct.os.environ.pop(ct._SLEEP_INSTANCE_GENERATION_ENV, None)
        else:
            ct.os.environ[ct._SLEEP_INSTANCE_GENERATION_ENV] = self._saved_generation

    def test_missing_role_keeps_legacy_empty_namespace(self):
        # Backward compatible: no role, no generation -> legacy role-less keys.
        self.assertEqual(ct._level3_key_namespace(self._config()), "")

    def test_role_is_included_in_namespace(self):
        self.assertEqual(ct._level3_key_namespace(self._config("PREFILL")), "PREFILL")
        self.assertEqual(ct._level3_key_namespace(self._config("DECODE")), "DECODE")

    def test_prefill_and_decode_namespaces_do_not_collide(self):
        self.assertNotEqual(
            ct._level3_key_namespace(self._config("PREFILL")),
            ct._level3_key_namespace(self._config("DECODE")),
        )

    def test_instance_generation_is_included_when_published(self):
        ct.os.environ[ct._SLEEP_INSTANCE_GENERATION_ENV] = "gen-abc123"
        self.assertEqual(
            ct._level3_key_namespace(self._config("PREFILL")),
            "PREFILL/gen-abc123",
        )
        # Distinct generations of the same role must not share a namespace.
        prefill_gen1 = ct._level3_key_namespace(self._config("PREFILL"))
        ct.os.environ[ct._SLEEP_INSTANCE_GENERATION_ENV] = "gen-def456"
        prefill_gen2 = ct._level3_key_namespace(self._config("PREFILL"))
        self.assertNotEqual(prefill_gen1, prefill_gen2)

    def test_namespace_components_are_sanitized(self):
        ct.os.environ[ct._SLEEP_INSTANCE_GENERATION_ENV] = "bad/gen key"
        namespace = ct._level3_key_namespace(self._config("RoleType.PREFILL"))
        # No characters that would break the "/"-delimited store key layout.
        self.assertNotIn(" ", namespace)
        self.assertEqual(namespace, "RoleType.PREFILL/bad_gen_key")

    def test_phase_keys_are_namespaced_by_role_and_generation(self):
        ct.os.environ[ct._SLEEP_INSTANCE_GENERATION_ENV] = "gen-1"
        snapshot = SimpleNamespace(
            parallelism_config=self._config("DECODE"), timeout=17
        )
        peer_key = "rtp_llm_level3/DECODE/gen-1/9/collectives_down/rank/1"
        store = Level3PhaseCoordinationTest.Store({peer_key: b"1"})
        with patch.object(
            ct, "_require_distributed_init_snapshot", return_value=snapshot
        ), patch.object(ct, "_get_or_create_lifecycle_store", return_value=store):
            self.assertTrue(ct.coordinate_level3_phase("collectives_down", 9, True))
        # The published key carries the role + generation namespace, so a shared
        # TCPStore cannot collide with another role or a prior generation.
        self.assertEqual(
            store.set_calls,
            [("rtp_llm_level3/DECODE/gen-1/9/collectives_down/rank/0", b"1")],
        )

    def test_generation_ready_keys_are_namespaced(self):
        store = Level3PhaseCoordinationTest.Store()
        ct._wait_for_process_group_generation_ready(
            store,
            generation=4,
            world_rank=0,
            world_size=1,
            timeout=ct.timedelta(seconds=1),
            namespace="PREFILL/gen-1",
        )
        self.assertEqual(
            store.set_calls,
            [("rtp_llm_pg_ready/PREFILL/gen-1/4/rank/0", b"1")],
        )


class MulticastHolderEnginePublishTest(unittest.TestCase):
    """The pinned keeper holder must be handed to the C++ engine so GetSleepStatus
    can report it into the durable checkpoint manifest, and only when the multicast
    keeper is enabled (L1/L2 and non-keeper deployments must stay untouched)."""

    def _fake_engine_module(self):
        module = ModuleType("libth_transformer")
        module.set_multicast_holder_instance = Mock()
        module.clear_multicast_holder_instance = Mock()
        return module

    def tearDown(self):
        ct._reset_multicast_keeper_phase_instance()

    def test_publish_is_noop_when_keeper_disabled(self):
        module = self._fake_engine_module()
        with patch.dict(sys.modules, {"libth_transformer": module}), patch.dict(
            ct.os.environ,
            {"RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "0"},
            clear=False,
        ):
            ct._publish_holder_instance_to_engine((0xAA, 0xBB))
        module.set_multicast_holder_instance.assert_not_called()
        module.clear_multicast_holder_instance.assert_not_called()

    def test_publish_sets_and_clears_when_keeper_enabled(self):
        module = self._fake_engine_module()
        with patch.dict(sys.modules, {"libth_transformer": module}), patch.dict(
            ct.os.environ,
            {"RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1"},
            clear=False,
        ):
            ct._publish_holder_instance_to_engine((0xAA, 0xBB))
            module.set_multicast_holder_instance.assert_called_once_with(0xAA, 0xBB)
            ct._publish_holder_instance_to_engine(None)
            module.clear_multicast_holder_instance.assert_called_once_with()

    def test_pinning_and_reset_publish_to_engine(self):
        module = self._fake_engine_module()
        ct._reset_multicast_keeper_phase_instance()
        with patch.dict(sys.modules, {"libth_transformer": module}), patch.dict(
            ct.os.environ,
            {"RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1"},
            clear=False,
        ):
            ready, error = ct._validate_multicast_keeper_phase_instance(
                "collective_teardown_ready", 5, (0x11, 0x22)
            )
            self.assertTrue(ready, error)
            module.set_multicast_holder_instance.assert_called_once_with(0x11, 0x22)
            ct._reset_multicast_keeper_phase_instance()
            module.clear_multicast_holder_instance.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
