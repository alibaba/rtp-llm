import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from rtp_llm.server.backend_manager import BackendManager


class _FakeEngine:
    def __init__(self, exc=None):
        self.exc = exc
        self.stopped = False

    def stop(self):
        self.stopped = True
        if self.exc is not None:
            raise self.exc


class _FakeNfsManager:
    def __init__(self, exc=None):
        self.exc = exc
        self.unmounted = False

    def unmount_all(self):
        self.unmounted = True
        if self.exc is not None:
            raise self.exc


class BackendManagerStopTest(unittest.TestCase):
    def test_stop_unmounts_nfs_after_engine_stop(self):
        manager = BackendManager.__new__(BackendManager)
        engine = _FakeEngine()
        nfs_manager = _FakeNfsManager()
        manager.engine = engine

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_stop_unmounts_nfs_even_when_engine_stop_raises(self):
        manager = BackendManager.__new__(BackendManager)
        engine_error = RuntimeError("engine stop failed")
        engine = _FakeEngine(engine_error)
        nfs_manager = _FakeNfsManager()
        manager.engine = engine

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            with self.assertRaisesRegex(RuntimeError, "engine stop failed"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)


class BackendManagerDeepEPTest(unittest.TestCase):
    def test_initializes_only_consumed_deepep_backend(self):
        for is_v41, strategy, expected_calls in (
            (True, "auto", 0),
            (False, "mega_moe", 0),
            (False, "mega_moe_se", 0),
            (False, "auto", 1),
        ):
            with self.subTest(is_v41=is_v41, strategy=strategy):
                manager = BackendManager.__new__(BackendManager)
                manager.py_env_configs = MagicMock()
                manager._distributed_server = MagicMock()
                engine_config = MagicMock()
                engine_config.parallelism_config.world_size = 2
                engine_config.moe_config = SimpleNamespace(
                    use_all_gather=False,
                    use_deepep_moe=True,
                    use_mori_ep=False,
                    moe_strategy=strategy,
                )
                model_config = MagicMock()
                model_config.expert_num = 384
                model_config.is_deepseek_v41 = is_v41
                with ExitStack() as stack:
                    for name in (
                        "init_distributed_environment",
                        "get_world_info",
                        "update_worker_addrs",
                        "validate_hw_kernel_group_args",
                    ):
                        stack.enter_context(
                            patch(f"rtp_llm.server.backend_manager.{name}")
                        )
                    stack.enter_context(
                        patch(
                            "rtp_llm.server.backend_manager.EngineConfig.create",
                            return_value=engine_config,
                        )
                    )
                    factory = stack.enter_context(
                        patch("rtp_llm.server.backend_manager.ModelFactory")
                    )
                    factory.create_model_config.return_value = model_config
                    init_deepep = stack.enter_context(
                        patch(
                            "rtp_llm.models_py.distributed.deepep_wrapper.init_deepep_wrapper"
                        )
                    )
                    manager.start()
                    self.assertEqual(init_deepep.call_count, expected_calls)
                    if expected_calls:
                        init_deepep.assert_called_once_with(engine_config, model_config)
                    factory.from_model_configs.assert_called_once()


class BackendManagerCommInitTest(unittest.TestCase):
    def _check_start(self, world_size, cp_enabled, expected_init):
        config = SimpleNamespace(
            world_size=world_size,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: cp_enabled),
        )
        manager = BackendManager.__new__(BackendManager)
        manager.py_env_configs = SimpleNamespace(
            parallelism_config=config,
            server_config=object(),
            distribute_config=SimpleNamespace(dist_comm_timeout=60),
            ft_disable_custom_ar_override=None,
        )
        manager._distributed_server = Mock()
        engine_config = SimpleNamespace(parallelism_config=config)
        boundary = RuntimeError("stop before model loading")
        with (
            patch(
                "rtp_llm.server.backend_manager.EngineConfig.create",
                return_value=engine_config,
            ),
            patch(
                "rtp_llm.server.backend_manager.init_distributed_environment"
            ) as initialize,
            patch(
                "rtp_llm.server.backend_manager.get_world_info", side_effect=boundary
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                manager.start()
        self.assertEqual(initialize.call_count, int(expected_init))
        if expected_init:
            self.assertIs(initialize.call_args.args[0], config)
            self.assertEqual(initialize.call_args.kwargs["backend"], "nccl")

    def test_cp_single_rank_initializes_communication(self):
        self._check_start(world_size=1, cp_enabled=True, expected_init=True)

    def test_plain_single_rank_keeps_existing_startup(self):
        self._check_start(world_size=1, cp_enabled=False, expected_init=False)

    def test_multiple_ranks_keep_existing_initialization(self):
        self._check_start(world_size=2, cp_enabled=False, expected_init=True)


if __name__ == "__main__":
    unittest.main()
