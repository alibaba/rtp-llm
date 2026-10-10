import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.server.backend_manager import BackendManager


class BackendManagerModulePreflightTest(unittest.TestCase):
    def _start(self, fail_draft=False):
        manager = BackendManager.__new__(BackendManager)
        manager.py_env_configs = MagicMock()
        manager._distributed_server = MagicMock()
        engine_config = MagicMock()
        engine_config.parallelism_config.world_size = 1
        engine_config.module_dispatch.mode = "auto"
        target = SimpleNamespace(expert_num=0)
        draft = object()
        contexts = [object(), object()]
        events = []

        def prepare(config, *args, **kwargs):
            if config is target:
                events.append("target_preflight")
                return contexts[0]
            self.assertIs(config, draft)
            self.assertEqual(kwargs["namespace"], "propose")
            events.append("draft_preflight")
            if fail_draft:
                raise ValueError("draft contract mismatch")
            return contexts[1]

        def load(**kwargs):
            self.assertIs(kwargs["propose_model_config"], draft)
            self.assertIs(engine_config.module_build_context, contexts[0])
            self.assertIs(engine_config.propose_module_build_context, contexts[1])
            events.append("weights")
            return MagicMock()

        with patch(
            "rtp_llm.server.backend_manager.EngineConfig.create",
            return_value=engine_config,
        ), patch("rtp_llm.server.backend_manager.get_world_info"), patch(
            "rtp_llm.server.backend_manager.update_worker_addrs"
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.create_model_config",
            return_value=target,
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.update_engine_config_from_model_config"
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.create_propose_model_config",
            return_value=draft,
        ), patch(
            "rtp_llm.models_py.pluggable.worker.prepare_worker_model_context",
            side_effect=prepare,
        ), patch(
            "rtp_llm.device.get_current_device"
        ) as device, patch(
            "rtp_llm.server.backend_manager.ModelFactory.from_model_configs",
            side_effect=load,
        ) as factory:
            device.return_value.prepare_model_runtime.side_effect = (
                lambda *a: events.append("runtime")
            )
            if fail_draft:
                with self.assertRaisesRegex(ValueError, "draft contract mismatch"):
                    manager.start()
                device.assert_not_called()
                factory.assert_not_called()
                self.assertEqual(events, ["target_preflight", "draft_preflight"])
            else:
                manager.start()
                self.assertEqual(
                    events,
                    ["target_preflight", "draft_preflight", "runtime", "weights"],
                )

    def test_target_and_draft_preflight_before_runtime_and_weights(self):
        self._start()

    def test_draft_rejection_prevents_runtime_initialization(self):
        self._start(fail_draft=True)


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


if __name__ == "__main__":
    unittest.main()
