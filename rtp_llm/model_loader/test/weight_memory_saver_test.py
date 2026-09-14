"""Unit tests for rtp_llm.model_loader.weight_memory_saver.

GPU-free: torch_memory_saver is faked via sys.modules injection (or forced
to be un-importable with a None sys.modules entry), so the tests validate:
  - default-on behavior (env switch unset -> collective release is enabled)
  - graceful degradation when torch_memory_saver is unavailable
  - region(tag="weights", enable_cpu_backup=True) call forwarding
  - pause/resume forwarding + is_paused state machine (idempotency)
  - region re-entrancy (nested regions enter the real region once)
  - the collective-memory-release switch (env var vs runtime override precedence)
"""

import contextlib
import os
import sys
import threading
import types
import unittest
from typing import Any, Dict, Iterator, List, Optional
from unittest import mock

from rtp_llm.model_loader import weight_memory_saver as wms

_TMS_MODULE = "torch_memory_saver"


class PausableScratchTest(unittest.TestCase):
    def test_online_scratch_allocation_does_not_flush_or_change_allocator(self):
        import torch

        pool = object()
        result = object()
        with (
            mock.patch.object(wms, "is_enabled", return_value=True),
            mock.patch.object(wms, "_get_scratch_pool", return_value=pool),
            mock.patch.object(
                wms,
                "weights_region",
                side_effect=AssertionError("weight region on forward"),
            ),
            mock.patch.object(
                wms,
                "_apply_live_alloc_conf",
                side_effect=AssertionError("runtime allocator changed"),
            ),
            mock.patch.object(
                torch.cuda, "empty_cache", side_effect=AssertionError("cache flushed")
            ),
            mock.patch.object(
                torch.cuda, "use_mem_pool", return_value=contextlib.nullcontext()
            ) as use_pool,
            mock.patch.object(torch, "empty", return_value=result) as empty,
        ):
            for size in (64, 1024, 2048):
                self.assertIs(
                    wms.pausable_empty(
                        (size, 16), device="cuda:0", dtype=torch.bfloat16
                    ),
                    result,
                )
            self.assertEqual(use_pool.call_count, 3)
            use_pool.assert_called_with(pool, device=torch.device("cuda:0"))
            self.assertEqual(empty.call_count, 3)

    def test_disabled_or_cpu_scratch_bypasses_custom_pool(self):
        import torch

        with (
            mock.patch.object(
                wms,
                "_get_scratch_pool",
                side_effect=AssertionError("custom pool not needed"),
            ),
            mock.patch.object(torch, "empty") as empty,
        ):
            with mock.patch.object(wms, "is_enabled", return_value=False):
                wms.pausable_empty((16, 16), device="cuda:0")
            with mock.patch.object(wms, "is_enabled", return_value=True):
                wms.pausable_empty((16, 16), device="cpu")
            self.assertEqual(empty.call_count, 2)

    def test_implicit_cuda_device_uses_the_pausable_pool(self):
        import torch

        pool = object()
        with (
            mock.patch.object(wms, "is_enabled", return_value=True),
            mock.patch.object(
                torch, "get_default_device", return_value=torch.device("cuda:0")
            ),
            mock.patch.object(wms, "_get_scratch_pool", return_value=pool),
            mock.patch.object(
                torch.cuda, "use_mem_pool", return_value=contextlib.nullcontext()
            ) as use_pool,
            mock.patch.object(torch, "empty"),
        ):
            wms.pausable_empty((16, 16))
            use_pool.assert_called_once_with(pool, device=torch.device("cuda:0"))

    def test_pool_is_created_once_and_selects_the_startup_backup_policy(self):
        import torch

        for level, malloc in (
            (1, "rtp_sleep_scratch_malloc_backup"),
            (2, "rtp_sleep_scratch_malloc"),
        ):
            with self.subTest(level=level):
                library = types.SimpleNamespace(
                    __file__="test_compute_ops.so",
                    rtp_llm_ops=types.SimpleNamespace(
                        sleep_memory_allocator_available=lambda: True
                    ),
                )
                pool = object()
                with (
                    mock.patch.dict(sys.modules, {"librtp_compute_ops": library}),
                    mock.patch.object(wms, "_scratch_pools", threading.local()),
                    mock.patch.object(wms, "_scratch_pool_owners", []),
                    mock.patch.object(wms, "sleep_mode_level", return_value=level),
                    mock.patch.object(
                        torch.cuda, "device", return_value=contextlib.nullcontext()
                    ),
                    mock.patch.object(
                        torch.cuda.memory, "CUDAPluggableAllocator"
                    ) as allocator,
                    mock.patch.object(
                        torch.cuda, "MemPool", return_value=pool
                    ) as make_pool,
                ):
                    self.assertIs(wms._get_scratch_pool(0), pool)
                    self.assertIs(wms._get_scratch_pool(0), pool)
                    allocator.assert_called_once_with(
                        "test_compute_ops.so", malloc, "rtp_sleep_scratch_free"
                    )
                    make_pool.assert_called_once()

    def test_threads_and_devices_do_not_enter_the_same_pool(self):
        import torch

        library = types.SimpleNamespace(
            __file__="test_compute_ops.so",
            rtp_llm_ops=types.SimpleNamespace(
                sleep_memory_allocator_available=lambda: True
            ),
        )
        pools = []
        with (
            mock.patch.dict(sys.modules, {"librtp_compute_ops": library}),
            mock.patch.object(wms, "_scratch_pools", threading.local()),
            mock.patch.object(wms, "_scratch_pool_owners", []) as owners,
            mock.patch.object(
                torch.cuda, "device", side_effect=lambda _: contextlib.nullcontext()
            ),
            mock.patch.object(torch.cuda.memory, "CUDAPluggableAllocator"),
            mock.patch.object(torch.cuda, "MemPool", side_effect=lambda **_: object()),
        ):
            first = wms._get_scratch_pool(0)
            second = wms._get_scratch_pool(1)
            worker = threading.Thread(
                target=lambda: pools.append(wms._get_scratch_pool(0))
            )
            worker.start()
            worker.join()
            self.assertIs(wms._get_scratch_pool(0), first)
            self.assertEqual(len({id(first), id(second), id(pools[0])}), 3)
            self.assertEqual(len(owners), 3)


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
        # Every env var this module reads must be saved/restored here: a test that
        # sets one and does not have it restored leaks into every later test in the
        # process (they would silently exercise a different switch state than they
        # believe). ENV_LEVEL / ENV_COLLECTIVE_RELEASE are included for that reason,
        # not because a specific test needs them.
        self._saved_env = {
            wms.ENV_SWITCH: os.environ.get(wms.ENV_SWITCH),
            wms.LEGACY_ENV_SWITCH: os.environ.get(wms.LEGACY_ENV_SWITCH),
            wms.ENV_LEVEL: os.environ.get(wms.ENV_LEVEL),
            wms.ENV_COLLECTIVE_RELEASE: os.environ.get(wms.ENV_COLLECTIVE_RELEASE),
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


class ExpandableCoexistenceTest(WeightMemorySaverTestBase):
    """expandable_segments:True coexists with the torch_memory_saver pool by
    being DEFERRED through init: stripped from the env and forced off at prepare
    (so weights/KV land at low registerable VA), then turned on live only after
    the engine is ready (enable_runtime_expandable), and disabled again around
    each weights-region pool allocation.

    GPU-free: the live allocator setter is patched out so no CUDA driver call is
    made -- the test asserts the env normalization and toggle *sequence*.
    """

    _CONF = "expandable_segments:True,max_split_size_mb:128"

    def setUp(self) -> None:
        super().setUp()
        os.environ[wms.ENV_SWITCH] = "1"
        self._saved_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self._CONF
        self.fake = self._inject_fake_tms()
        # Record enable/disable toggles instead of touching the real driver.
        self.toggles: List[bool] = []
        self._real_setter = wms._set_expandable_segments

        def record_toggle(enabled: bool) -> None:
            self.toggles.append(enabled)
            wms._expandable_live = enabled

        wms._set_expandable_segments = record_toggle

    def tearDown(self) -> None:
        wms._set_expandable_segments = self._real_setter
        if self._saved_conf is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self._saved_conf
        super().tearDown()

    def test_alloc_conf_without_expandable(self) -> None:
        self.assertEqual(
            wms._alloc_conf_without_expandable(self._CONF), "max_split_size_mb:128"
        )
        self.assertEqual(
            wms._alloc_conf_without_expandable("expandable_segments:True"), ""
        )
        self.assertEqual(
            wms._alloc_conf_without_expandable("max_split_size_mb:128"),
            "max_split_size_mb:128",
        )
        self.assertEqual(wms._alloc_conf_without_expandable(""), "")

    def test_prepare_strips_env_and_defers(self) -> None:
        wms._prepare_expandable_coexistence()
        # expandable_segments removed from the env (so TMS init won't refuse),
        # other keys preserved.
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:128")
        self.assertEqual(wms._expandable_base_conf, "max_split_size_mb:128")
        # Requested, but deferred: not yet active, and forced off at init so the
        # whole startup path (weights + KV alloc + KV MR reg) stays low-VA.
        self.assertTrue(wms._expandable_requested)
        self.assertFalse(wms._expandable_active)
        self.assertEqual(self.toggles, [False])

    def test_prepare_runs_once(self) -> None:
        wms._prepare_expandable_coexistence()
        wms._prepare_expandable_coexistence()
        self.assertEqual(self.toggles, [False])

    def test_region_stays_off_before_runtime_enable(self) -> None:
        # Before enable_runtime_expandable(), expandable is not active, so the
        # weights region does not toggle it (the init force-off is the only call).
        with wms.weights_region():
            pass
        self.assertFalse(wms._expandable_active)
        self.assertEqual(self.toggles, [False])
        self.assertEqual(
            self.fake.region_calls,
            [{"tag": wms.WEIGHTS_TAG, "enable_cpu_backup": True}],
        )

    def test_enable_runtime_expandable_turns_on_once(self) -> None:
        wms._prepare_expandable_coexistence()
        wms.enable_runtime_expandable()
        self.assertTrue(wms._expandable_active)
        # init force-off, then runtime turn-on.
        self.assertEqual(self.toggles, [False, True])
        # Idempotent: a second call does not toggle again.
        wms.enable_runtime_expandable()
        self.assertEqual(self.toggles, [False, True])

    def test_enable_runtime_expandable_inert_without_request(self) -> None:
        # No expandable requested -> enable_runtime_expandable is a no-op.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        wms._prepare_expandable_coexistence()
        wms.enable_runtime_expandable()
        self.assertFalse(wms._expandable_active)
        self.assertEqual(self.toggles, [])

    def test_region_toggles_off_then_on_after_runtime_enable(self) -> None:
        wms._prepare_expandable_coexistence()
        wms.enable_runtime_expandable()
        self.assertEqual(self.toggles, [False, True])
        with wms.weights_region():
            pass
        # region enter -> off, region exit -> restore on.
        self.assertEqual(self.toggles, [False, True, False, True])
        self.assertEqual(
            self.fake.region_calls,
            [{"tag": wms.WEIGHTS_TAG, "enable_cpu_backup": True}],
        )

    def test_no_expandable_in_env_is_inert(self) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9"
        wms._prepare_expandable_coexistence()
        self.assertFalse(wms._expandable_requested)
        self.assertFalse(wms._expandable_active)
        self.assertEqual(self.toggles, [])
        # Env left untouched when expandable segments were not requested.
        self.assertEqual(
            os.environ["PYTORCH_CUDA_ALLOC_CONF"], "garbage_collection_threshold:0.9"
        )


class InitSegmentSplitCapTest(WeightMemorySaverTestBase):
    """The init-phase max_split_size_mb cap keeps the loader's ~1 GiB staging
    segments from being split by resident weights (which would strand the segment
    remainder as sleeping-residual GPU memory), and is released once the engine is
    ready so runtime allocations keep torch's default splitting.

    GPU-free: the live allocator setter is patched out, so the tests assert the
    composed config *strings* and the on/off sequence.
    """

    _CONF = "expandable_segments:True,garbage_collection_threshold:0.9"
    _CAP = f"max_split_size_mb:{wms._INIT_SPLIT_CAP_MB}"

    def setUp(self) -> None:
        super().setUp()
        os.environ[wms.ENV_SWITCH] = "1"
        self._saved_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self._CONF
        self._inject_fake_tms()
        # Capture the composed config instead of calling the CUDA driver.
        self.applied: List[str] = []
        # Exercise the actual composition and state transitions, not a test copy
        # of the production config builder (which used to hide composition bugs).
        fake_torch = types.SimpleNamespace(
            _C=types.SimpleNamespace(
                _accelerator_setAllocatorSettings=self.applied.append
            )
        )
        self._torch_patch = mock.patch.dict(sys.modules, {"torch": fake_torch})
        self._torch_patch.start()

    def tearDown(self) -> None:
        self._torch_patch.stop()
        if self._saved_conf is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self._saved_conf
        super().tearDown()

    def test_cap_applied_once_and_released_once(self) -> None:
        wms.limit_init_segment_splitting()
        self.assertTrue(wms._split_cap_live)
        # Idempotent: a second call must not re-push the config.
        wms.limit_init_segment_splitting()
        self.assertEqual(len(self.applied), 1)
        self.assertIn(self._CAP, self.applied[0])

        wms.release_init_segment_splitting()
        self.assertFalse(wms._split_cap_live)
        self.assertNotIn(self._CAP, self.applied[1])
        # Releasing twice is a no-op.
        wms.release_init_segment_splitting()
        self.assertEqual(len(self.applied), 2)

    def test_release_without_cap_is_inert(self) -> None:
        wms.release_init_segment_splitting()
        self.assertEqual(self.applied, [])

    def test_inert_without_sleep_mode(self) -> None:
        # The gain only exists at sleep, so the non-sleep load path must be
        # byte-for-byte unchanged -- no allocator write at all.
        os.environ.pop(wms.ENV_SWITCH, None)
        os.environ.pop(wms.LEGACY_ENV_SWITCH, None)
        wms.limit_init_segment_splitting()
        self.assertFalse(wms._split_cap_live)
        self.assertEqual(self.applied, [])

    def test_env_base_conf_is_replayed(self) -> None:
        # The live setter REPLACES the whole config, so the user's other env keys
        # have to be restated on every write or they revert to torch defaults.
        wms.limit_init_segment_splitting()
        self.assertTrue(self.applied[0].startswith("garbage_collection_threshold:0.9,"))
        # expandable_segments is owned by the deferral logic, never replayed raw.
        self.assertNotIn("expandable_segments:True", wms._expandable_base_conf)

    def test_cap_survives_expandable_toggles(self) -> None:
        # expandable and the cap share one setter; flipping expandable must not
        # silently drop the cap (the whole init phase depends on it staying on).
        wms.limit_init_segment_splitting()
        wms._set_expandable_segments(True)
        self.assertIn(self._CAP, self.applied[-1])
        self.assertIn("expandable_segments:True", self.applied[-1])
        wms._set_expandable_segments(False)
        self.assertIn(self._CAP, self.applied[-1])

    def test_expandable_state_preserved_across_release(self) -> None:
        # Release happens next to enable_runtime_expandable(); it must not undo it.
        wms.limit_init_segment_splitting()
        wms._set_expandable_segments(True)
        wms.release_init_segment_splitting()
        self.assertEqual(
            self.applied[-1],
            "garbage_collection_threshold:0.9,expandable_segments:True",
        )

    def test_large_segment_sizes_deferred_and_restored(self) -> None:
        for size in (12, 16, 20, 64, 128, 256, 512, 1024, 2048, 4096):
            with self.subTest(size=size):
                wms._reset_for_testing()
                self.applied.clear()
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    f"expandable_segments:True,large_segment_size_mb:{size},"
                    "roundup_power2_divisions:[256:1,512:2,>:4]"
                )
                wms.prepare_expandable_coexistence()
                wms.limit_init_segment_splitting()
                self.assertTrue(wms._split_cap_live)
                self.assertTrue(wms._expandable_requested)
                for conf in self.applied:
                    self.assertIn(f"large_segment_size_mb:{min(size, 20)},", conf)
                    self.assertIn("expandable_segments:False", conf)
                    self.assertIn("roundup_power2_divisions:[256:1,512:2,>:4]", conf)
                self.assertIn(self._CAP, self.applied[-1])
                wms.enable_runtime_expandable()
                # BackendManager enables expandable before releasing the split
                # cap; this intermediate config must also remain legal.
                self.assertIn(
                    f"large_segment_size_mb:{min(size, 20)},", self.applied[-1]
                )
                self.assertIn(self._CAP, self.applied[-1])
                wms.release_init_segment_splitting()
                runtime_conf = self.applied[-1]
                self.assertIn(f"large_segment_size_mb:{size},", runtime_conf)
                self.assertNotIn("max_split_size_mb", runtime_conf)
                self.assertTrue(wms._expandable_live)
                # Wake reload and nested weight regions must stay in the loading
                # phase until the outer scope exits, even when an exception occurs.
                with self.assertRaisesRegex(RuntimeError, "reload failed"):
                    with wms.expandable_segments_disabled():
                        loading_conf = self.applied[-1]
                        self.assertIn(
                            f"large_segment_size_mb:{min(size, 20)},", loading_conf
                        )
                        with wms.expandable_segments_disabled():
                            self.assertFalse(wms._expandable_live)
                        self.assertFalse(wms._expandable_live)
                        self.assertEqual(self.applied[-1], loading_conf)
                        raise RuntimeError("reload failed")
                self.assertTrue(wms._expandable_live)
                self.assertEqual(self.applied[-1], runtime_conf)

    def test_user_split_cap_replaced_for_init_and_restored_for_runtime(self) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:2048,expandable_segments:True,large_segment_size_mb:1024"
        )
        wms.prepare_expandable_coexistence()
        wms.limit_init_segment_splitting()
        self.assertEqual(self.applied[-1].count("max_split_size_mb:"), 1)
        self.assertIn(self._CAP, self.applied[-1])
        wms.enable_runtime_expandable()
        wms.release_init_segment_splitting()
        self.assertEqual(self.applied[-1].count("max_split_size_mb:"), 1)
        self.assertIn("max_split_size_mb:2048", self.applied[-1])
        self.assertIn("large_segment_size_mb:1024", self.applied[-1])

    def test_large_segment_without_expandable_restored_after_init_cap(self) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "large_segment_size_mb:512"
        wms.limit_init_segment_splitting()
        self.assertIn("large_segment_size_mb:20,", self.applied[-1])
        self.assertIn(self._CAP, self.applied[-1])
        wms.release_init_segment_splitting()
        self.assertEqual(
            self.applied[-1], "large_segment_size_mb:512,expandable_segments:False"
        )

    def test_large_segment_config_untouched_without_sleep(self) -> None:
        os.environ[wms.ENV_SWITCH] = "0"
        conf = "expandable_segments:True,large_segment_size_mb:1024"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
        wms.prepare_expandable_coexistence()
        wms.limit_init_segment_splitting()
        wms.enable_runtime_expandable()
        wms.release_init_segment_splitting()
        self.assertEqual(self.applied, [])
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], conf)


class CollectiveReleaseSwitchTest(WeightMemorySaverTestBase):
    """``release_collective_memory()`` -- whether sleep also releases the NCCL
    communicator's GPU memory (ncclCommSuspend/Resume).

    Deliberately independent of the sleep level and of the sleep-mode switch: it
    carries costs the level does not (equal pinned host memory, seconds on each of
    sleep/wake, NCCL >= 2.29.7), so it is read on its own. Defaults to off; an
    explicit env/runtime value of true is required to opt in.

    The CLI arg has no C++ RuntimeConfig field, so the env var (mirrored by
    server_args.setup_args) and the explicit override installed by
    configure_from_runtime are the only two inputs -- both covered here.
    """

    def setUp(self) -> None:
        super().setUp()
        os.environ.pop(wms.ENV_COLLECTIVE_RELEASE, None)

    def test_default_is_off(self) -> None:
        self.assertFalse(wms.release_collective_memory())

    def test_env_one_enables(self) -> None:
        os.environ[wms.ENV_COLLECTIVE_RELEASE] = "1"
        self.assertTrue(wms.release_collective_memory())

    def test_env_zero_disables(self) -> None:
        os.environ[wms.ENV_COLLECTIVE_RELEASE] = "0"
        self.assertFalse(wms.release_collective_memory())

    def test_env_garbage_is_off(self) -> None:
        # Anything that is not exactly "1" must read as off: a typo'd value must not
        # silently opt a deployment into the pinned-host / extra-latency costs.
        for value in ("", "true", "True", "yes", "2", "on", " 1"):
            with self.subTest(value=value):
                os.environ[wms.ENV_COLLECTIVE_RELEASE] = value
                self.assertFalse(wms.release_collective_memory())

    def test_runtime_override_enables_over_env_off(self) -> None:
        os.environ[wms.ENV_COLLECTIVE_RELEASE] = "0"
        wms.configure_from_runtime(True, release_collective_memory=True)
        self.assertTrue(wms.release_collective_memory())

    def test_runtime_override_disables_over_env_on(self) -> None:
        os.environ[wms.ENV_COLLECTIVE_RELEASE] = "1"
        wms.configure_from_runtime(True, release_collective_memory=False)
        self.assertFalse(wms.release_collective_memory())

    def test_runtime_none_leaves_env_in_charge(self) -> None:
        os.environ[wms.ENV_COLLECTIVE_RELEASE] = "1"
        wms.configure_from_runtime(True)
        self.assertTrue(wms.release_collective_memory())

        os.environ[wms.ENV_COLLECTIVE_RELEASE] = "0"
        wms.configure_from_runtime(True, sleep_mode_level=2)
        self.assertFalse(wms.release_collective_memory())

    def test_independent_of_sleep_level(self) -> None:
        # The level selects what happens to the weights; this switch selects whether
        # the communicator buffers go too. Level 2 must not imply the release.
        wms.configure_from_runtime(True, sleep_mode_level=2)
        self.assertEqual(wms.sleep_mode_level(), 2)
        self.assertFalse(wms.release_collective_memory())

    def test_reset_for_testing_clears_override(self) -> None:
        wms.configure_from_runtime(True, release_collective_memory=True)
        self.assertTrue(wms.release_collective_memory())
        wms._reset_for_testing()
        self.assertIsNone(wms._collective_release_override)
        # Back to env-driven (env is unset in this class's setUp).
        self.assertFalse(wms.release_collective_memory())


class PausableAllocGuardTest(WeightMemorySaverTestBase):
    """assert_pausable_alloc_safe() refuses a sleep-persistent allocation while
    expandable_segments is live (the silent post-wake corruption guard).

    GPU-free: drives the live-state flag directly rather than through the CUDA
    allocator, so it validates the guard's decision in isolation.
    """

    def test_guard_noop_when_expandable_not_live(self) -> None:
        wms._expandable_live = False
        # Must not raise -- the common, safe case.
        wms.assert_pausable_alloc_safe("unit-test")

    def test_guard_raises_when_expandable_live(self) -> None:
        wms._expandable_live = True
        with self.assertRaises(RuntimeError) as ctx:
            wms.assert_pausable_alloc_safe("unit-test-site")
        msg = str(ctx.exception)
        # Message points at the offending site and explains the corruption.
        self.assertIn("unit-test-site", msg)
        self.assertIn("silently corrupted", msg)
        self.assertIn("expandable_segments_disabled", msg)

    def test_suppressed_region_asserts_when_expandable_live(self) -> None:
        # The level-2 wake reload path (suppress_weights_region) bypasses the
        # expandable_segments_disabled() guard, so weights_region() re-asserts:
        # allocating reload scratch with expandable live would corrupt it across
        # the resume boundary.
        os.environ[wms.ENV_SWITCH] = "1"
        self._inject_fake_tms()
        wms._expandable_live = True
        with self.assertRaises(RuntimeError):
            with wms.suppress_weights_region():
                with wms.weights_region():
                    pass


if __name__ == "__main__":
    unittest.main()
