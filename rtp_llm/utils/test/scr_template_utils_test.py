"""Unit tests for the rank-local Epsilon registration helpers."""

from __future__ import annotations

import os
import unittest
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

    def is_snapstart_enable(self):
        return self.snap_enabled

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


class ScrTemplateUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        scr._reset_for_test()

    def tearDown(self) -> None:
        scr._reset_for_test()

    def test_feature_gate_defaults_and_alias(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(scr.is_scr_enabled())

        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ALIAS_ENV: "yes"}, clear=True
        ):
            self.assertTrue(scr.is_scr_enabled())
            self.assertEqual(os.environ[scr.SCR_SHIM_ENABLE_ENV], "1")

    def test_unified_switch_does_not_choose_controller_phase(self) -> None:
        with mock.patch.dict(
            os.environ,
            {scr.SCR_ENABLE_ENV: "1", scr.SCR_PHASE_ENV: scr.SCR_PHASE_RESTORE},
            clear=True,
        ):
            self.assertTrue(scr.is_scr_enabled())
            self.assertEqual(os.environ[scr.SCR_SHIM_ENABLE_ENV], "1")
            self.assertEqual(os.environ[scr.SCR_PHASE_ENV], scr.SCR_PHASE_RESTORE)

    def test_checkpoint_control_is_not_exposed_by_rtp_llm(self) -> None:
        self.assertFalse(hasattr(scr, "start_scr_checkpoint"))
        self.assertFalse(hasattr(scr, "start_scr_checkpoint_thread"))
        self.assertFalse(hasattr(scr, "ScrParticipantManifest"))

    def test_backend_mode_keeps_app_gate_and_shim_gate_separate(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scr.epsilon_backend_mode(), "disabled")
        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1", scr.SCR_SHIM_ENABLE_ENV: "0"}, clear=True
        ):
            self.assertEqual(scr.epsilon_backend_mode(), "wheel-native")
        with mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1", scr.SCR_SHIM_ENABLE_ENV: "1"}, clear=True
        ), mock.patch.object(scr.os.path, "isdir", return_value=True), mock.patch.object(
            scr.platform, "release", return_value="6.6.1-aarch64"
        ):
            self.assertEqual(scr.epsilon_backend_mode(), "external-shim")

    def test_registers_all_unique_kv_cache_regions_and_scales(self) -> None:
        model = SimpleNamespace()
        base = _FakeTensor(100)
        region = _FakeTensor(200)
        scale = _FakeTensor(300, nbytes=4)
        empty = _FakeTensor(400, numel=0)
        model.kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=[[base, base, empty], [region]],
            kv_cache_base_by_layer=[[empty]],
            kv_scale_base_by_layer_region=[[scale, scale]],
            kv_scale_base_by_layer=[[empty]],
        )
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        epsilon = _FakeEpsilon()

        with mock.patch.dict(os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ):
            self.assertTrue(scr.register_for_scr(engine))

        self.assertEqual([call[0] for call in epsilon.calls], ["cache", "before"])
        self.assertEqual(epsilon.calls[0][1], [base, region, scale])
        self.assertTrue(callable(epsilon.before_callback))
        self.assertEqual(scr._registrations[id(engine)].tensors, (base, region, scale))

    def test_registration_is_inert_when_epsilon_is_not_active(self) -> None:
        model = SimpleNamespace(kv_cache=SimpleNamespace())
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        epsilon = _FakeEpsilon(snap_enabled=False)

        with mock.patch.dict(os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            self.assertFalse(scr.register_for_scr(engine))
        self.assertEqual(epsilon.calls, [])
        self.assertNotIn(id(engine), scr._registrations)

    def test_registration_failure_can_retry_when_cache_becomes_ready(self) -> None:
        model = SimpleNamespace(kv_cache=SimpleNamespace())
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        epsilon = _FakeEpsilon()
        with mock.patch.dict(os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ):
            self.assertFalse(scr.register_for_scr(engine))
            model.kv_cache.kv_cache_base_by_layer = [[_FakeTensor(123)]]
            self.assertTrue(scr.register_for_scr(engine))
        self.assertEqual([call[0] for call in epsilon.calls], ["before", "cache", "before"])

    def test_before_callback_uses_captured_device(self) -> None:
        epsilon = _FakeEpsilon()
        model = SimpleNamespace(
            kv_cache=SimpleNamespace(kv_cache_base_by_layer=[[_FakeTensor(1)]])
        )
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        with mock.patch.dict(os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ), mock.patch.object(scr, "_capture_cuda_device", return_value=5), mock.patch(
            "torch.cuda.is_available", return_value=True
        ), mock.patch("torch.cuda.synchronize") as synchronize:
            self.assertTrue(scr.register_for_scr(engine))
            epsilon.before_callback()
        synchronize.assert_called_once_with(device=5)

    def test_after_restore_callback_is_only_registered(self) -> None:
        epsilon = _FakeEpsilon()
        tensor = _FakeTensor(1)
        model = SimpleNamespace(kv_cache=SimpleNamespace(kv_cache_base_by_layer=[[tensor]]))
        engine = SimpleNamespace(model=SimpleNamespace(py_model=model))
        callback = mock.Mock()
        with mock.patch.dict(os.environ, {scr.SCR_ENABLE_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ):
            self.assertTrue(scr.register_for_scr(engine, after_restore=callback))
        self.assertEqual([call[0] for call in epsilon.calls], ["cache", "before", "after"])
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
