"""Unit tests for rtp_llm.model_loader.weight_memory_saver (Sleep/wake_up M6).

GPU-free: torch_memory_saver is faked via sys.modules injection (or forced
to be un-importable with a None sys.modules entry), so the tests validate:
  - default-off behavior (env switch unset -> everything is a no-op)
  - graceful degradation when torch_memory_saver is unavailable
  - region(tag="weights", enable_cpu_backup=True) call forwarding
  - pause/resume forwarding + is_paused state machine (idempotency)
  - region re-entrancy (nested regions enter the real region once)
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
    """Records region/pause/resume calls like the torch_memory_saver singleton."""

    def __init__(self) -> None:
        self.region_calls: List[Dict[str, Any]] = []
        self.region_depth: int = 0
        self.max_region_depth: int = 0
        self.pause_calls: List[Optional[str]] = []
        self.resume_calls: List[Optional[str]] = []

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

    def pause(self, tag: Optional[str] = None) -> None:
        self.pause_calls.append(tag)

    def resume(self, tag: Optional[str] = None) -> None:
        self.resume_calls.append(tag)


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
        module = types.ModuleType(_TMS_MODULE)
        module.torch_memory_saver = fake  # type: ignore[attr-defined]
        sys.modules[_TMS_MODULE] = module
        return fake

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
        self.assertFalse(wms.is_paused())

    def test_region_is_noop_and_tms_untouched(self) -> None:
        fake = self._inject_fake_tms()
        executed = False
        with wms.weights_region():
            executed = True
        self.assertTrue(executed)
        self.assertEqual(fake.region_calls, [])

    def test_pause_resume_noop(self) -> None:
        fake = self._inject_fake_tms()
        self.assertFalse(wms.pause_weights())
        self.assertFalse(wms.is_paused())
        self.assertFalse(wms.resume_weights())
        self.assertFalse(wms.is_paused())
        self.assertEqual(fake.pause_calls, [])
        self.assertEqual(fake.resume_calls, [])

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

    def test_pause_resume_warn_but_do_not_raise(self) -> None:
        with self.assertLogs(level="WARNING") as logs:
            self.assertFalse(wms.pause_weights())
            self.assertFalse(wms.resume_weights())
        self.assertFalse(wms.is_paused())
        self.assertTrue(any("pause_weights" in m for m in logs.output))
        self.assertTrue(any("resume_weights" in m for m in logs.output))

    def test_import_failure_is_cached(self) -> None:
        with self.assertLogs(level="WARNING"):
            self.assertFalse(wms.is_available())
        # Second call must not re-attempt the import (no second warning).
        self.assertFalse(wms.is_available())


class FakeTmsForwardingTest(WeightMemorySaverTestBase):
    """Env switch on + fake torch_memory_saver: verify call forwarding."""

    def setUp(self) -> None:
        super().setUp()
        os.environ[wms.ENV_SWITCH] = "1"
        self.fake = self._inject_fake_tms()

    def test_is_available(self) -> None:
        self.assertTrue(wms.is_enabled())
        self.assertTrue(wms.is_available())

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

    def test_pause_resume_forwarding_and_state_machine(self) -> None:
        self.assertFalse(wms.is_paused())

        self.assertTrue(wms.pause_weights())
        self.assertTrue(wms.is_paused())
        self.assertEqual(self.fake.pause_calls, [wms.WEIGHTS_TAG])

        # Idempotent: second pause does not call tms.pause again.
        self.assertTrue(wms.pause_weights())
        self.assertTrue(wms.is_paused())
        self.assertEqual(self.fake.pause_calls, [wms.WEIGHTS_TAG])

        self.assertTrue(wms.resume_weights())
        self.assertFalse(wms.is_paused())
        self.assertEqual(self.fake.resume_calls, [wms.WEIGHTS_TAG])

        # Idempotent: second resume does not call tms.resume again.
        self.assertTrue(wms.resume_weights())
        self.assertFalse(wms.is_paused())
        self.assertEqual(self.fake.resume_calls, [wms.WEIGHTS_TAG])

    def test_resume_without_pause_is_noop(self) -> None:
        self.assertTrue(wms.resume_weights())
        self.assertFalse(wms.is_paused())
        self.assertEqual(self.fake.resume_calls, [])

    def test_pause_resume_cycle_twice(self) -> None:
        for _ in range(2):
            self.assertTrue(wms.pause_weights())
            self.assertTrue(wms.is_paused())
            self.assertTrue(wms.resume_weights())
            self.assertFalse(wms.is_paused())
        self.assertEqual(self.fake.pause_calls, [wms.WEIGHTS_TAG] * 2)
        self.assertEqual(self.fake.resume_calls, [wms.WEIGHTS_TAG] * 2)


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
