"""Unit tests for rtp_llm.model_loader.weight_memory_saver.

GPU-free: torch_memory_saver is faked via sys.modules injection (or forced
to be un-importable with a None sys.modules entry), so the tests validate:
  - default-off behavior (env switch unset -> everything is a no-op)
  - graceful degradation when torch_memory_saver is unavailable
  - region(tag="weights", enable_cpu_backup=True) call forwarding
  - region re-entrancy (nested regions enter the real region once)

Weight pause/resume itself is driven from the C++ sleep controller
(VmmBackend::pause/resume("weights")), not from this module, so there is no
Python pause/resume API to test here.
"""

import contextlib
import os
import sys
import types
import unittest
from typing import Any, Dict, Iterator, List, Optional

from rtp_llm.model_loader import weight_memory_saver as wms

_TMS_MODULE = "torch_memory_saver"


class _FakeTms:
    """Records region calls like the torch_memory_saver singleton."""

    def __init__(self) -> None:
        self.region_calls: List[Dict[str, Any]] = []
        self.region_depth: int = 0
        self.max_region_depth: int = 0

    @contextlib.contextmanager
    def region(
        self, tag: Optional[str] = None, enable_cpu_backup: bool = False
    ) -> Iterator[None]:
        self.region_calls.append({"tag": tag, "enable_cpu_backup": enable_cpu_backup})
        self.region_depth += 1
        self.max_region_depth = max(self.max_region_depth, self.region_depth)
        try:
            yield
        finally:
            self.region_depth -= 1


class _FakeTorchMemorySaverModule(types.ModuleType):
    def __init__(self, fake_tms: _FakeTms) -> None:
        super().__init__(_TMS_MODULE)
        self.torch_memory_saver = fake_tms
        self.configure_subprocess_enter_count = 0
        self.configure_subprocess_exit_count = 0

    @contextlib.contextmanager
    def configure_subprocess(self) -> Iterator[None]:
        self.configure_subprocess_enter_count += 1
        old_value = os.environ.get("LD_PRELOAD")
        os.environ["LD_PRELOAD"] = "fake_torch_memory_saver_preload.so"
        try:
            yield
        finally:
            self.configure_subprocess_exit_count += 1
            if old_value is None:
                os.environ.pop("LD_PRELOAD", None)
            else:
                os.environ["LD_PRELOAD"] = old_value


class _FakeProcess:
    def __init__(self) -> None:
        self.started = False
        self.ld_preload_at_start: Optional[str] = None

    def start(self) -> None:
        self.started = True
        self.ld_preload_at_start = os.environ.get("LD_PRELOAD")


class WeightMemorySaverTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            wms.ENV_SWITCH: os.environ.get(wms.ENV_SWITCH),
            wms.LEGACY_ENV_SWITCH: os.environ.get(wms.LEGACY_ENV_SWITCH),
        }
        self._saved_module = sys.modules.get(_TMS_MODULE)
        self._had_module = _TMS_MODULE in sys.modules
        wms._reset_for_testing()

    def tearDown(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if self._had_module:
            sys.modules[_TMS_MODULE] = self._saved_module
        else:
            sys.modules.pop(_TMS_MODULE, None)
        wms._reset_for_testing()

    def _inject_fake_tms(self) -> _FakeTms:
        fake = _FakeTms()
        module = _FakeTorchMemorySaverModule(fake)
        sys.modules[_TMS_MODULE] = module
        return fake

    def _get_fake_module(self) -> _FakeTorchMemorySaverModule:
        module = sys.modules[_TMS_MODULE]
        self.assertIsInstance(module, _FakeTorchMemorySaverModule)
        return module  # type: ignore[return-value]

    def _make_tms_unimportable(self) -> None:
        # A None entry in sys.modules makes `import torch_memory_saver`
        # raise ImportError deterministically, even if the real package
        # happens to be installed in the environment.
        sys.modules[_TMS_MODULE] = None  # type: ignore[assignment]


class DefaultDisabledTest(WeightMemorySaverTestBase):
    """Env switch off (default): everything must be a strict no-op."""

    def setUp(self) -> None:
        super().setUp()
        os.environ.pop(wms.ENV_SWITCH, None)
        os.environ.pop(wms.LEGACY_ENV_SWITCH, None)

    def test_disabled_flags(self) -> None:
        self.assertFalse(wms.is_enabled())
        self.assertFalse(wms.is_available())

    def test_region_is_noop_and_tms_untouched(self) -> None:
        fake = self._inject_fake_tms()
        executed = False
        with wms.weights_region():
            executed = True
        self.assertTrue(executed)
        self.assertEqual(fake.region_calls, [])

    def test_configure_subprocess_disabled_is_noop(self) -> None:
        self._inject_fake_tms()
        module = self._get_fake_module()
        with wms.configure_subprocess():
            self.assertNotEqual(
                os.environ.get("LD_PRELOAD"), "fake_torch_memory_saver_preload.so"
            )
        self.assertEqual(module.configure_subprocess_enter_count, 0)

    def test_explicit_zero_is_disabled(self) -> None:
        os.environ[wms.ENV_SWITCH] = "0"
        self.assertFalse(wms.is_enabled())
        self.assertFalse(wms.is_available())

    def test_runtime_override_enables_without_env(self) -> None:
        os.environ.pop(wms.ENV_SWITCH, None)
        wms.configure_from_runtime(True)
        self.assertTrue(wms.is_enabled())

    def test_runtime_override_disables_even_when_env_is_set(self) -> None:
        os.environ[wms.ENV_SWITCH] = "1"
        wms.configure_from_runtime(False)
        self.assertFalse(wms.is_enabled())


class UnavailableTest(WeightMemorySaverTestBase):
    """Env switch on but torch_memory_saver not importable: graceful no-op."""

    def setUp(self) -> None:
        super().setUp()
        os.environ[wms.ENV_SWITCH] = "1"
        self._make_tms_unimportable()

    def test_is_available_false_with_warning(self) -> None:
        with self.assertLogs(level="WARNING") as logs:
            self.assertFalse(wms.is_available())
        self.assertTrue(any("torch_memory_saver" in m for m in logs.output))

    def test_region_is_noop(self) -> None:
        executed = False
        with wms.weights_region():
            executed = True
        self.assertTrue(executed)

    def test_import_failure_is_cached(self) -> None:
        with self.assertLogs(level="WARNING"):
            self.assertFalse(wms.is_available())
        # Second call must not re-attempt the import (no second warning).
        self.assertFalse(wms.is_available())

    def test_configure_subprocess_unavailable_is_noop(self) -> None:
        with self.assertLogs(level="WARNING") as logs:
            with wms.configure_subprocess():
                self.assertNotEqual(
                    os.environ.get("LD_PRELOAD"),
                    "fake_torch_memory_saver_preload.so",
                )
        self.assertTrue(any("configure_subprocess" in m for m in logs.output))


class FakeTmsForwardingTest(WeightMemorySaverTestBase):
    """Env switch on + fake torch_memory_saver: verify call forwarding."""

    def setUp(self) -> None:
        super().setUp()
        os.environ[wms.ENV_SWITCH] = "1"
        self.fake = self._inject_fake_tms()

    def test_is_available(self) -> None:
        self.assertTrue(wms.is_enabled())
        self.assertTrue(wms.is_available())

    def test_configure_subprocess_forwards_and_restores_env(self) -> None:
        module = self._get_fake_module()
        old_value = os.environ.get("LD_PRELOAD")
        with wms.configure_subprocess():
            self.assertEqual(
                os.environ.get("LD_PRELOAD"), "fake_torch_memory_saver_preload.so"
            )
        self.assertEqual(module.configure_subprocess_enter_count, 1)
        self.assertEqual(module.configure_subprocess_exit_count, 1)
        self.assertEqual(os.environ.get("LD_PRELOAD"), old_value)

    def test_start_configured_process_forwards_preload(self) -> None:
        module = self._get_fake_module()
        old_value = os.environ.get("LD_PRELOAD")
        process = _FakeProcess()

        wms.start_configured_process(process)

        self.assertTrue(process.started)
        self.assertEqual(
            process.ld_preload_at_start, "fake_torch_memory_saver_preload.so"
        )
        self.assertEqual(module.configure_subprocess_enter_count, 1)
        self.assertEqual(module.configure_subprocess_exit_count, 1)
        self.assertEqual(os.environ.get("LD_PRELOAD"), old_value)

    def test_region_params(self) -> None:
        with wms.weights_region():
            self.assertEqual(self.fake.region_depth, 1)
        self.assertEqual(
            self.fake.region_calls,
            [{"tag": wms.WEIGHTS_TAG, "enable_cpu_backup": True}],
        )
        self.assertEqual(self.fake.region_depth, 0)

    def test_region_reentrant_enters_once(self) -> None:
        with wms.weights_region():
            with wms.weights_region():
                self.assertEqual(self.fake.region_depth, 1)
        self.assertEqual(len(self.fake.region_calls), 1)
        self.assertEqual(self.fake.max_region_depth, 1)
        # After full exit a new region can be entered again.
        with wms.weights_region():
            pass
        self.assertEqual(len(self.fake.region_calls), 2)

    def test_region_depth_restored_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with wms.weights_region():
                raise RuntimeError("boom")
        self.assertEqual(self.fake.region_depth, 0)
        with wms.weights_region():
            pass
        self.assertEqual(len(self.fake.region_calls), 2)


class LegacyEnvForwardingTest(WeightMemorySaverTestBase):
    """Low-level developer override remains available without sleep endpoints."""

    def setUp(self) -> None:
        super().setUp()
        os.environ.pop(wms.ENV_SWITCH, None)
        os.environ[wms.LEGACY_ENV_SWITCH] = "1"
        self.fake = self._inject_fake_tms()

    def test_legacy_env_enables_weight_saver(self) -> None:
        self.assertTrue(wms.is_enabled())
        self.assertTrue(wms.is_available())
        with wms.weights_region():
            pass
        self.assertEqual(
            self.fake.region_calls,
            [{"tag": wms.WEIGHTS_TAG, "enable_cpu_backup": True}],
        )


if __name__ == "__main__":
    unittest.main()
