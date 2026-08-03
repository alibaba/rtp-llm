import importlib.util
import inspect
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "trt_allreduce.py"
_SPEC = importlib.util.spec_from_file_location(
    "rtp_llm_trt_allreduce_host_test_module", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load TRT allreduce module from {_MODULE_PATH}")
trt_allreduce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trt_allreduce)


class TrtllmDistEnvLifecycleTest(unittest.TestCase):
    def _handle(self):
        handle = MagicMock()
        handle.get_barrier_handle.return_value = b"barrier"
        handle.get_data_handle.return_value = b"data"
        return handle

    def _bare_env(self, handle=None, *, device_id=0):
        env = object.__new__(trt_allreduce.TrtllmDistEnv)
        env.handle = self._handle() if handle is None else handle
        env.disabled = False
        env.device_id = device_id
        env.rank = 0
        env.world_size = 2
        env.control_group = object()
        env._is_captured = False
        env._capture_handles_pending = False
        return env

    @contextmanager
    def _patched_runtime(
        self,
        handle,
        gather,
        *,
        world_size=2,
        data_ranks=(0, 1),
        control_ranks=(0, 1),
    ):
        constructor = MagicMock(return_value=handle)
        data_group = object()
        control_group = object()

        def ranks(group):
            return data_ranks if group is data_group else control_ranks

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    trt_allreduce, "_get_handle_class", return_value=constructor
                )
            )
            stack.enter_context(
                patch.object(trt_allreduce.dist, "get_rank", return_value=0)
            )
            stack.enter_context(
                patch.object(
                    trt_allreduce.dist, "get_world_size", return_value=world_size
                )
            )
            stack.enter_context(
                patch.object(
                    trt_allreduce.dist,
                    "get_process_group_ranks",
                    side_effect=ranks,
                )
            )
            all_gather = stack.enter_context(
                patch.object(
                    trt_allreduce.dist, "all_gather_object", side_effect=gather
                )
            )
            barrier = stack.enter_context(patch.object(trt_allreduce.dist, "barrier"))
            stack.enter_context(patch.object(trt_allreduce.torch.cuda, "set_device"))
            stack.enter_context(patch.object(trt_allreduce.torch.cuda, "synchronize"))
            yield SimpleNamespace(
                constructor=constructor,
                all_gather=all_gather,
                barrier=barrier,
                data_group=data_group,
                control_group=control_group,
            )

    def test_remote_initialization_failure_uses_two_phase_release(self):
        handle = self._handle()

        def gather(output, local, group):
            del group
            output[:] = [local, False]

        with self._patched_runtime(handle, gather) as runtime:
            env = trt_allreduce.TrtllmDistEnv(
                group=runtime.data_group,
                control_group=runtime.control_group,
                device_id=0,
            )

        self.assertTrue(env.disabled)
        self.assertIsNone(env.handle)
        handle.close_peer_mappings.assert_called_once()
        self.assertEqual(runtime.all_gather.call_count, 1)
        self.assertEqual(runtime.barrier.call_count, 2)

    def test_unpublished_handle_failure_releases_local_exports(self):
        handle = self._handle()
        handle.get_barrier_handle.side_effect = RuntimeError("handle export failed")

        def gather(output, local, group):
            del group
            output[:] = [local, False]

        with self._patched_runtime(handle, gather) as runtime:
            env = trt_allreduce.TrtllmDistEnv(
                group=runtime.data_group,
                control_group=runtime.control_group,
                device_id=0,
            )

        self.assertTrue(env.disabled)
        self.assertIsNone(env.handle)
        handle.close_peer_mappings.assert_called_once_with()
        handle.release_local_exports.assert_called_once_with()
        handle.get_data_handle.assert_not_called()
        self.assertEqual(runtime.all_gather.call_count, 1)

    def test_open_failure_closes_peers_before_local_export_release(self):
        handle = self._handle()
        handle.open_barrier_handles.side_effect = RuntimeError("open failed")
        gathered_locals = []

        def gather(output, local, group):
            del group
            gathered_locals.append(local)
            output[:] = [False, True] if local is False else [local, local]

        with self._patched_runtime(handle, gather) as runtime:
            env = trt_allreduce.TrtllmDistEnv(
                group=runtime.data_group,
                control_group=runtime.control_group,
                device_id=0,
            )

        self.assertTrue(env.disabled)
        self.assertIsNone(env.handle)
        self.assertEqual(runtime.all_gather.call_count, 4)
        self.assertEqual(gathered_locals, [True, b"barrier", b"data", False])
        self.assertEqual(runtime.barrier.call_count, 3)
        handle.close_peer_mappings.assert_called_once()
        handle.open_data_handles.assert_not_called()

    def test_control_group_failure_quarantines_local_exports(self):
        handle = self._handle()
        try:
            with self._patched_runtime(
                handle, MagicMock(side_effect=RuntimeError("control failed"))
            ) as runtime:
                with self.assertRaisesRegex(RuntimeError, "process must be rebuilt"):
                    trt_allreduce.TrtllmDistEnv(
                        group=runtime.data_group,
                        control_group=runtime.control_group,
                        device_id=0,
                    )
            handle.close_peer_mappings.assert_called_once_with()
            handle.release_local_exports.assert_not_called()
            self.assertIs(trt_allreduce._unrecoverable_workspaces[-1], handle)
        finally:
            if (
                trt_allreduce._unrecoverable_workspaces
                and trt_allreduce._unrecoverable_workspaces[-1] is handle
            ):
                trt_allreduce._unrecoverable_workspaces.pop()

    def test_release_barrier_failure_retains_handle(self):
        handle = self._handle()
        env = self._bare_env(handle)
        with patch.object(trt_allreduce.torch.cuda, "set_device"), patch.object(
            trt_allreduce.torch.cuda, "synchronize"
        ), patch.object(env, "_barrier", side_effect=RuntimeError("timeout")):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                env._release_workspace_two_phase()
        self.assertIs(env.handle, handle)
        self.assertTrue(env.disabled)
        handle.close_peer_mappings.assert_called_once()
        handle.release_local_exports.assert_not_called()

    def test_release_drains_local_device_before_closing_peer_mappings(self):
        events = []
        handle = self._handle()
        handle.close_peer_mappings.side_effect = lambda: events.append("close-peers")
        handle.release_local_exports.side_effect = lambda: events.append(
            "release-exports"
        )
        env = self._bare_env(handle, device_id=3)
        with patch.object(
            trt_allreduce.torch.cuda,
            "set_device",
            side_effect=lambda device: events.append(f"set-device:{device}"),
        ), patch.object(
            trt_allreduce.torch.cuda,
            "synchronize",
            side_effect=lambda device: events.append(f"synchronize:{device}"),
        ), patch.object(
            env, "_barrier", side_effect=lambda: events.append("barrier")
        ):
            env._release_workspace_two_phase()

        self.assertEqual(
            events,
            [
                "set-device:3",
                "synchronize:3",
                "close-peers",
                "barrier",
                "release-exports",
                "barrier",
            ],
        )

    def test_constructor_exposes_real_max_size_contract(self):
        parameter = inspect.signature(trt_allreduce.TrtllmDistEnv).parameters[
            "max_size_in_bytes"
        ]
        self.assertEqual(parameter.default, 16384 * 16384)

        handle = self._handle()

        def gather(output, local, group):
            del group
            output[:] = [local, local]

        with self._patched_runtime(handle, gather) as runtime:
            env = trt_allreduce.TrtllmDistEnv(
                group=runtime.data_group,
                control_group=runtime.control_group,
                device_id=0,
                max_size_in_bytes=123456,
            )
        self.assertEqual(env.max_size_in_bytes, 123456)
        self.assertEqual(runtime.constructor.call_args.args[-2], 123456)

    def test_destructor_quarantines_when_collective_shutdown_was_skipped(self):
        env = self._bare_env(device_id=3)
        env.rank = 2
        handle = env.handle
        quarantine_size = len(trt_allreduce._unrecoverable_workspaces)
        try:
            with patch.object(trt_allreduce.logging, "error") as error:
                env.__del__()
            error.assert_called_once()
            self.assertIsNone(env.handle)
            self.assertIs(trt_allreduce._unrecoverable_workspaces[-1], handle)
        finally:
            del trt_allreduce._unrecoverable_workspaces[quarantine_size:]

    def test_second_release_barrier_failure_retains_handle_for_retry(self):
        handle = self._handle()
        env = self._bare_env(handle)
        with patch.object(trt_allreduce.torch.cuda, "set_device"), patch.object(
            trt_allreduce.torch.cuda, "synchronize"
        ), patch.object(
            env, "_barrier", side_effect=[None, RuntimeError("second timeout")]
        ):
            with self.assertRaisesRegex(RuntimeError, "second timeout"):
                env._release_workspace_two_phase()
        self.assertIs(env.handle, handle)
        self.assertTrue(env.disabled)
        handle.close_peer_mappings.assert_called_once()
        handle.release_local_exports.assert_called_once()

        with patch.object(trt_allreduce.torch.cuda, "set_device"), patch.object(
            trt_allreduce.torch.cuda, "synchronize"
        ), patch.object(env, "_barrier") as barrier:
            env._release_workspace_two_phase()
        self.assertIsNone(env.handle)
        self.assertEqual(barrier.call_count, 2)
        self.assertEqual(handle.close_peer_mappings.call_count, 2)
        self.assertEqual(handle.release_local_exports.call_count, 2)

    def test_data_and_control_rank_membership_must_match(self):
        with self._patched_runtime(
            self._handle(),
            MagicMock(),
            data_ranks=(0, 1),
            control_ranks=(1, 0),
        ) as runtime:
            with self.assertRaisesRegex(RuntimeError, "rank order"):
                trt_allreduce.TrtllmDistEnv(
                    group=runtime.data_group,
                    control_group=runtime.control_group,
                    device_id=0,
                )
            runtime.constructor.assert_not_called()

    def test_finish_capture_session_consumes_after_uniform_pending_consensus(self):
        env = self._bare_env()
        env._is_captured = True

        def gather(output, local, group):
            self.assertIs(group, env.control_group)
            output[:] = [local, local]

        with patch.object(
            trt_allreduce.torch.cuda,
            "is_current_stream_capturing",
            return_value=False,
        ), patch.object(
            trt_allreduce.dist, "all_gather_object", side_effect=gather
        ), patch.object(
            env, "_consume_capture"
        ) as consume:
            env.finish_capture_session()
        consume.assert_called_once_with()

    def test_finish_capture_session_rejects_mixed_rank_participation(self):
        env = self._bare_env()
        env._is_captured = True

        def gather(output, local, group):
            del local, group
            output[:] = [True, False]

        with patch.object(
            trt_allreduce.torch.cuda,
            "is_current_stream_capturing",
            return_value=False,
        ), patch.object(
            trt_allreduce.dist, "all_gather_object", side_effect=gather
        ), patch.object(
            env, "_barrier"
        ) as barrier:
            with self.assertRaisesRegex(RuntimeError, "differed across TP ranks"):
                env.finish_capture_session()
        env.handle.capture_clear.assert_called_once_with()
        barrier.assert_called_once_with()
        self.assertFalse(env._is_captured)
        self.assertFalse(env._capture_handles_pending)

    def test_supported_world_size_boundaries(self):
        for world_size, should_initialize in (
            (1, False),
            (2, True),
            (3, False),
            (4, True),
            (8, True),
            (16, False),
        ):
            with self.subTest(world_size=world_size):
                handle = self._handle()

                def gather(output, local, group):
                    del group
                    output[:] = [local] * world_size

                ranks = tuple(range(world_size))
                with self._patched_runtime(
                    handle,
                    gather,
                    world_size=world_size,
                    data_ranks=ranks,
                    control_ranks=ranks,
                ) as runtime:
                    env = trt_allreduce.TrtllmDistEnv(
                        group=runtime.data_group,
                        control_group=runtime.control_group,
                        device_id=0,
                    )
                self.assertEqual(runtime.constructor.called, should_initialize)
                self.assertEqual(env.disabled, not should_initialize)


class TrtllmCommManagerIdentityTest(unittest.TestCase):
    def setUp(self):
        self.old_manager = trt_allreduce._trtllm_comm_manager
        trt_allreduce._trtllm_comm_manager = trt_allreduce.TrtllmCommManager()

    def tearDown(self):
        trt_allreduce._trtllm_comm_manager = self.old_manager

    def _ready(self, *, generation=7):
        manager = trt_allreduce._trtllm_comm_manager
        manager.group = object()
        manager.control_group = object()
        manager.device_id = 0
        manager.generation = generation
        manager.dist_env = SimpleNamespace(
            disabled=False,
            shutdown=MagicMock(),
            allreduce_op=MagicMock(),
            allreduce_add_rms_fused=MagicMock(return_value=(1, 2, 3)),
        )
        manager.initialized = True
        return manager

    def test_no_argument_ensure_preserves_graph_generation_and_control_group(self):
        manager = self._ready()
        with patch.object(manager, "initialize") as initialize:
            self.assertTrue(
                trt_allreduce.ensure_trtllm_comm_initialized(manager.group, 0)
            )
        initialize.assert_not_called()
        self.assertEqual(manager.generation, 7)

    def test_generation_or_control_mismatch_does_not_reinitialize_graph_workspace(self):
        manager = self._ready()
        with patch.object(manager, "initialize") as initialize:
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                trt_allreduce.ensure_trtllm_comm_initialized(
                    manager.group, 0, generation=8, control_group=manager.control_group
                )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                trt_allreduce.ensure_trtllm_comm_initialized(
                    manager.group, 0, generation=7, control_group=object()
                )
        initialize.assert_not_called()

    def test_both_allreduce_routes_reject_identity_mismatch_without_initialize(self):
        manager = self._ready()
        other_group = object()
        with patch.object(manager, "initialize") as initialize:
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                trt_allreduce.allreduce(MagicMock(), other_group, 0)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                trt_allreduce.allreduce_residual_rmsnorm(
                    MagicMock(), MagicMock(), MagicMock(), other_group, 0
                )
        initialize.assert_not_called()

    def test_capture_never_lazily_initializes_workspace(self):
        group = object()
        manager = trt_allreduce._trtllm_comm_manager
        with patch.object(
            trt_allreduce.torch.cuda, "is_current_stream_capturing", return_value=True
        ), patch.object(manager, "initialize") as initialize:
            with self.assertRaisesRegex(RuntimeError, "before stream capture"):
                trt_allreduce.allreduce(MagicMock(), group, 0)
        initialize.assert_not_called()

    def test_cleanup_resets_identity_and_readiness(self):
        manager = self._ready()
        dist_env = manager.dist_env
        manager.cleanup()
        dist_env.shutdown.assert_called_once()
        self.assertFalse(manager.initialized)
        self.assertIsNone(manager.group)
        self.assertIsNone(manager.control_group)
        self.assertIsNone(manager.device_id)
        self.assertIsNone(manager.generation)
        self.assertIsNone(manager.dist_env)
        self.assertFalse(trt_allreduce.is_trt_allreduce_ready())

    def test_cleanup_retains_identity_for_collective_shutdown_retry(self):
        manager = self._ready()
        dist_env = manager.dist_env
        dist_env.shutdown.side_effect = RuntimeError("injected shutdown failure")

        with self.assertRaisesRegex(RuntimeError, "injected shutdown failure"):
            manager.cleanup()

        self.assertFalse(manager.initialized)
        self.assertIsNotNone(manager.group)
        self.assertIsNotNone(manager.control_group)
        self.assertEqual(manager.device_id, 0)
        self.assertEqual(manager.generation, 7)
        self.assertIs(manager.dist_env, dist_env)
        self.assertFalse(trt_allreduce.is_trt_allreduce_ready())

        dist_env.shutdown.side_effect = None
        manager.cleanup()
        self.assertIsNone(manager.dist_env)
        self.assertEqual(dist_env.shutdown.call_count, 2)


if __name__ == "__main__":
    unittest.main()
