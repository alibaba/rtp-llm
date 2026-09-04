"""Unit tests for the optional Epsilon/sCR template helpers.

These tests deliberately use tensor-like doubles. A real CUDA device (and a
loaded Epsilon native agent) is not required to verify feature gating,
registration, worker mapping, or the daemon trigger path.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.utils import scr_template_utils as scr


class _FakeTensor:
    def __init__(self, pointer: int, nbytes: int = 8, numel: int = 1) -> None:
        self.device = "cuda:0"
        self._pointer = pointer
        self.nbytes = nbytes
        self._numel = numel

    def data_ptr(self) -> int:
        return self._pointer

    def numel(self) -> int:
        return self._numel


class _FakeEpsilon:
    def __init__(self, snap_enabled: bool = True) -> None:
        self.snap_enabled = snap_enabled
        self.calls = []
        self.before_callback = None
        self.snap_calls = []
        self.snap_result = 0
        self.raise_on_snap = False

    def is_snapstart_enable(self):
        return self.snap_enabled

    def register_model(self, model):
        self.calls.append(("model", model))
        return 0

    def register_kv_caches(self, tensors):
        self.calls.append(("cache", tensors))
        return 0

    def register_before_checkpoint_func(self, callback):
        self.calls.append(("before", callback))
        self.before_callback = callback
        return 0

    def register_after_restore_func(self, callback):
        self.calls.append(("after", callback))
        return 0

    def snapstart_checkpoint(self, **kwargs):
        if self.raise_on_snap:
            raise RuntimeError("test checkpoint failure")
        self.snap_calls.append(kwargs)
        return self.snap_result


class ScrTemplateUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        scr._reset_for_test()

    def tearDown(self) -> None:
        scr._reset_for_test()

    def test_feature_gate_defaults_and_alias(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(scr.is_scr_enabled())

    def test_participant_manifest_assigns_one_contiguous_quorum(self) -> None:
        manifest = scr.build_scr_participant_manifest(
            [
                ("start_server", "0"),
                ("backend_manager", "0"),
                ("backend_rank", "0"),
                ("frontend", "0:0"),
                ("dash_sc", "0:0"),
            ]
        )
        self.assertEqual(manifest.worker_num, 5)
        self.assertEqual(manifest.worker_id("backend_rank", "0"), 2)
        manifest.validate()

    def test_unified_switch_configures_shim_but_not_controller_phase(self) -> None:
        with mock.patch.dict(os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True):
            self.assertTrue(scr.is_scr_enabled())
            self.assertEqual(os.environ[scr.SCR_SHIM_ENABLE_ENV], "1")
            self.assertNotIn(scr.SCR_PHASE_ENV, os.environ)

        with mock.patch.dict(
            os.environ,
            {scr.SCR_ENABLE_ENV: "1", scr.SCR_PHASE_ENV: scr.SCR_PHASE_RESTORE},
            clear=True,
        ):
            self.assertTrue(scr.is_scr_enabled())
            self.assertEqual(os.environ[scr.SCR_SHIM_ENABLE_ENV], "1")
            self.assertEqual(os.environ[scr.SCR_PHASE_ENV], scr.SCR_PHASE_RESTORE)

    def test_backend_mode_keeps_app_gate_and_shim_gate_separate(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scr.epsilon_backend_mode(), "disabled")
        with mock.patch.dict(
            os.environ, {scr.SCR_SHIM_ENABLE_ENV: "1"}, clear=True
        ):
            self.assertFalse(scr.is_scr_enabled())
            self.assertEqual(scr.epsilon_backend_mode(), "disabled")
        with mock.patch.dict(
            os.environ,
            {scr.SCR_ENABLE_ENV: "1", scr.SCR_SHIM_ENABLE_ENV: "0"},
            clear=True,
        ):
            self.assertEqual(scr.epsilon_backend_mode(), "wheel-native")
        with mock.patch.dict(
            os.environ,
            {scr.SCR_ENABLE_ENV: "1", scr.SCR_SHIM_ENABLE_ENV: "1"},
            clear=True,
        ), mock.patch.object(scr.os.path, "isdir", return_value=True), mock.patch.object(
            scr.platform, "release", return_value="6.6.1-aarch64"
        ):
            self.assertEqual(scr.epsilon_backend_mode(), "external-shim")

        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ALIAS_ENV: "yes"}, clear=True
        ):
            self.assertTrue(scr.is_scr_enabled())

        # The primary spelling wins when both variables are present.
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_ENABLE_ENV: "0",
                scr.SCR_ENABLE_ALIAS_ENV: "1",
            },
            clear=True,
        ):
            self.assertFalse(scr.is_scr_enabled())

    def test_register_kv_cache_and_deduplicate_region_and_scale_tensors(self) -> None:
        model = SimpleNamespace()
        base = _FakeTensor(100)
        region = _FakeTensor(200)
        scale = _FakeTensor(300, nbytes=4)
        empty = _FakeTensor(400, numel=0)
        model.kv_cache = SimpleNamespace(
            # Region and flat views can alias one another. Only the region
            # view should be traversed when it is available.
            kv_cache_base_by_layer_region=[[base, base, empty], [region]],
            kv_cache_base_by_layer=[[empty]],
            kv_scale_base_by_layer_region=[[scale, scale]],
            kv_scale_base_by_layer=[[empty]],
        )
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        epsilon = _FakeEpsilon()

        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr,
            "_is_tensor",
            side_effect=lambda value: isinstance(value, _FakeTensor),
        ):
            self.assertTrue(scr.register_for_scr(engine))

        self.assertEqual(
            [call[0] for call in epsilon.calls], ["cache", "before"]
        )
        self.assertFalse(any(call[0] == "model" for call in epsilon.calls))
        registered = epsilon.calls[0][1]
        self.assertEqual(registered, [base, region, scale])
        self.assertTrue(callable(epsilon.before_callback))
        registration = scr._registrations[id(engine)]
        self.assertEqual(registration.tensors, (base, region, scale))

    def test_registration_is_inert_when_epsilon_is_not_active(self) -> None:
        model = SimpleNamespace(kv_cache=SimpleNamespace())
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        epsilon = _FakeEpsilon(snap_enabled=False)

        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            self.assertFalse(scr.register_for_scr(engine))
        self.assertEqual(epsilon.calls, [])
        self.assertNotIn(id(engine), scr._registrations)

    def test_checkpoint_uses_worker_id_override_and_local_world_size(self) -> None:
        epsilon = _FakeEpsilon()
        epsilon.snap_result = 7
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_ENABLE_ENV: "1",
                "LOCAL_RANK": "3",
                "RANK": "11",
                "LOCAL_WORLD_SIZE": "4",
                scr.SCR_WORKER_ID_ENV: "3",
            },
            clear=True,
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            result = scr.start_scr_checkpoint(timeout=12, inactivity_timeout=2)

        self.assertEqual(result, 7)
        self.assertEqual(
            epsilon.snap_calls,
            [
                {
                    "wait_mode": 1,
                    "worker_id": 3,
                    "worker_num": 4,
                    "timeout": 12,
                    "inactivity_timeout": 2,
                }
            ],
        )

        # An explicit argument is used when no environment override is set.
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_ENABLE_ENV: "1",
                "LOCAL_RANK": "3",
                "LOCAL_WORLD_SIZE": "4",
            },
            clear=True,
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            scr.start_scr_checkpoint(worker_id=2, worker_num=5)
        self.assertEqual(epsilon.snap_calls[-1]["worker_id"], 2)
        self.assertEqual(epsilon.snap_calls[-1]["worker_num"], 5)

        # A frozen launcher manifest must win over a stale process-level ID.
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_ENABLE_ENV: "1",
                scr.SCR_WORKER_ID_ENV: "0",
            },
            clear=True,
        ), mock.patch.object(scr.importlib, "import_module", return_value=epsilon):
            scr.start_scr_checkpoint(worker_id=3, worker_num=5)
        self.assertEqual(epsilon.snap_calls[-1]["worker_id"], 3)

    def test_triggered_daemon_waiter_and_fail_open_exception(self) -> None:
        epsilon = _FakeEpsilon()
        with tempfile.TemporaryDirectory() as temporary_dir:
            trigger = Path(temporary_dir) / "checkpoint.trigger"
            with mock.patch.dict(
                os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
            ), mock.patch.object(
                scr.importlib, "import_module", return_value=epsilon
            ):
                waiter = scr.start_scr_checkpoint_thread(
                    trigger_file=str(trigger),
                    poll_interval_seconds=0.01,
                    timeout=3,
                )
                self.assertIsNotNone(waiter)
                time.sleep(0.04)
                self.assertEqual(epsilon.snap_calls, [])
                trigger.touch()
                waiter.join(timeout=1)

            self.assertFalse(waiter.is_alive())
            self.assertEqual(len(epsilon.snap_calls), 1)
            self.assertEqual(epsilon.snap_calls[0]["wait_mode"], 1)

        # Exceptions from the optional daemon are swallowed and do not escape
        # into the caller's startup thread.
        epsilon.raise_on_snap = True
        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            waiter = scr.start_scr_checkpoint_thread(poll_interval_seconds=0.01)
            waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())

    def test_invalid_worker_mapping_fails_open_without_calling_epsilon(self) -> None:
        epsilon = _FakeEpsilon()
        with mock.patch.dict(
            os.environ,
            {scr.SCR_ENABLE_ENV: "1", scr.SCR_WORKER_ID_ENV: "4"},
            clear=True,
        ), mock.patch.object(scr.importlib, "import_module", return_value=epsilon):
            self.assertIsNone(scr.start_scr_checkpoint(worker_num=4))
        self.assertEqual(epsilon.snap_calls, [])

    def test_failed_registration_can_retry_when_cache_becomes_ready(self) -> None:
        model = SimpleNamespace(kv_cache=SimpleNamespace())
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        epsilon = _FakeEpsilon()
        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr,
            "_is_tensor",
            side_effect=lambda value: isinstance(value, _FakeTensor),
        ):
            self.assertFalse(scr.register_for_scr(engine))
            model.kv_cache.kv_cache_base_by_layer = [[_FakeTensor(123)]]
            self.assertTrue(scr.register_for_scr(engine))
        self.assertEqual(
            [call[0] for call in epsilon.calls],
            ["before", "cache", "before"],
        )

    def test_before_callback_uses_captured_device(self) -> None:
        epsilon = _FakeEpsilon()
        model = SimpleNamespace(
            kv_cache=SimpleNamespace(kv_cache_base_by_layer=[[_FakeTensor(1)]])
        )
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr,
            "_is_tensor",
            side_effect=lambda value: isinstance(value, _FakeTensor),
        ), mock.patch.object(
            scr, "_capture_cuda_device", return_value=5
        ), mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.synchronize"
        ) as synchronize:
            self.assertTrue(scr.register_for_scr(engine))
            epsilon.before_callback()
        synchronize.assert_called_once_with(device=5)

    def test_waiter_enters_common_quorum_when_registration_fails(self) -> None:
        epsilon = _FakeEpsilon()
        engine = SimpleNamespace(
            model=SimpleNamespace(py_model=SimpleNamespace(kv_cache=SimpleNamespace()))
        )
        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            waiter = scr.start_scr_checkpoint_thread(engine=engine, poll_interval_seconds=0.01)
            waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(epsilon.snap_calls), 1)


if __name__ == "__main__":
    unittest.main()
