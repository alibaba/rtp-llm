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

    def is_contiguous(self) -> bool:
        return True

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

    def snapstart_checkpoint(self, **kwargs):
        self.calls.append(("checkpoint", kwargs))
        return 0


class ScrTemplateUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        scr._reset_for_test()

    def tearDown(self) -> None:
        scr._reset_for_test()

    def test_feature_gate_has_one_public_switch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(scr.is_scr_enabled())

        with mock.patch.dict(os.environ, {"RTP_LLM_ENABLE_SCR": "yes"}, clear=True):
            self.assertFalse(scr.is_scr_enabled())

        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "yes"}, clear=True):
            self.assertTrue(scr.is_scr_enabled())

    def test_unified_switch_does_not_choose_controller_phase(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                scr.RTPLLM_ENABLE_SCR_ENV: "1",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_RESTORE,
            },
            clear=True,
        ):
            self.assertTrue(scr.is_scr_enabled())
            self.assertEqual(os.environ[scr.SCR_PHASE_ENV], scr.SCR_PHASE_RESTORE)

    def test_external_shim_phase_is_checked_after_provider_load(self) -> None:
        epsilon = SimpleNamespace(_EXTERNAL_DIR="/custom/scr/epsilon")
        with mock.patch.dict(
            os.environ,
            {
                scr.RTPLLM_ENABLE_SCR_ENV: "1",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_NORMAL,
            },
            clear=True,
        ), mock.patch.object(scr.importlib, "import_module", return_value=epsilon):
            self.assertIs(scr._load_epsilon(), epsilon)

    def test_control_plane_operations_are_not_exposed_by_rtp_llm(self) -> None:
        self.assertFalse(hasattr(scr, "start_scr_checkpoint"))
        self.assertFalse(hasattr(scr, "start_scr_checkpoint_thread"))
        self.assertTrue(hasattr(scr, "ScrParticipantManifest"))
        self.assertTrue(hasattr(scr, "build_scr_participant_manifest"))
        self.assertTrue(hasattr(scr, "arrive_scr_checkpoint_barrier"))

    def test_full_process_manifest_is_stable_and_contiguous(self) -> None:
        manifest = scr.build_scr_participant_manifest(
            [("start_server", "0"), ("backend_manager", "0"),
             ("backend_rank", "0"), ("frontend", "0:0")]
        )
        self.assertEqual(manifest.worker_num, 4)
        self.assertEqual(manifest.worker_id("backend_rank", "0"), 2)
        manifest.validate()

    def test_manifest_rejects_duplicate_participant(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            scr.build_scr_participant_manifest(
                [("backend_rank", "0"), ("backend_rank", "0")]
            )

    def test_rank_arrives_at_epsilon_barrier_with_explicit_mapping(self) -> None:
        epsilon = _FakeEpsilon()
        with mock.patch.dict(
            os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(worker_id=1, worker_num=2),
                0,
            )

        self.assertEqual(
            epsilon.calls,
            [
                (
                    "checkpoint",
                    {
                        "wait_mode": 1,
                        "worker_id": 1,
                        "worker_num": 2,
                        "timeout": 900,
                        "inactivity_timeout": 10,
                    },
                )
            ],
        )

    def test_rank_barrier_rejects_out_of_range_mapping(self) -> None:
        epsilon = _FakeEpsilon()
        with mock.patch.dict(
            os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            self.assertIsNone(
                scr.arrive_scr_checkpoint_barrier(worker_id=2, worker_num=2)
            )
        self.assertEqual(epsilon.calls, [])

    def test_rank_barrier_legacy_epsilon_is_called_once_without_probe_retry(self) -> None:
        epsilon = _FakeEpsilon()
        calls = []

        def legacy_checkpoint(wait_mode, worker_id, worker_num):
            calls.append((wait_mode, worker_id, worker_num))
            return 0

        epsilon.snapstart_checkpoint = legacy_checkpoint
        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1), 0
            )
        self.assertEqual(calls, [(1, 0, 1)])

    def test_internal_type_error_is_not_retried(self) -> None:
        epsilon = _FakeEpsilon()
        calls = []

        def broken_checkpoint(**kwargs):
            calls.append(kwargs)
            raise TypeError("native implementation failed")

        epsilon.snapstart_checkpoint = broken_checkpoint
        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(scr.LOGGER, "exception") as log_exception:
            self.assertIsNone(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)
            )
        self.assertEqual(len(calls), 1)
        log_exception.assert_called_once()

    def test_resolve_worker_mapping_supports_shared_scheduler_scope(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_WORKER_OFFSET_ENV: "4",
                scr.SCR_WORKER_NUM_ENV: "8",
            },
            clear=True,
        ):
            self.assertEqual(
                scr.resolve_scr_worker_mapping(local_rank=2),
                (6, 8),
            )

    def test_resolve_worker_mapping_rejects_invalid_scope(self) -> None:
        with mock.patch.dict(
            os.environ,
            {scr.SCR_WORKER_OFFSET_ENV: "4", scr.SCR_WORKER_NUM_ENV: "4"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "worker_id must be in"):
                scr.resolve_scr_worker_mapping(local_rank=2)

    def test_resolve_worker_mapping_rejects_malformed_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {scr.SCR_WORKER_NUM_ENV: "not-an-int"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                scr.resolve_scr_worker_mapping(local_rank=0)

    def test_backend_mode_reports_provider_without_a_second_gate(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scr.epsilon_backend_mode(), "disabled")
        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True):
            self.assertEqual(scr.epsilon_backend_mode(), "not-loaded")

        native = SimpleNamespace(_EXTERNAL_DIR="")
        external = SimpleNamespace(_EXTERNAL_DIR="/custom/scr/epsilon")
        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True):
            self.assertEqual(scr.epsilon_backend_mode(native), "wheel-native")
            self.assertEqual(scr.epsilon_backend_mode(external), "external-shim")

    def test_registers_native_allocations_without_python_model_views(self) -> None:
        # The allocator includes target/draft slices and all pools in this export.
        base = _FakeTensor(100, nbytes=64)
        indexer = _FakeTensor(200, nbytes=8)
        hbm = _FakeTensor(300, nbytes=32)
        engine = SimpleNamespace(
            gpu_cache_tensors=lambda: [base, indexer, hbm, _FakeTensor(100, nbytes=64)]
        )
        epsilon = _FakeEpsilon()
        with mock.patch.dict(
            os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ), mock.patch.object(
            scr, "_capture_cuda_device", return_value=None
        ), self.assertLogs(
            scr.LOGGER, level="INFO"
        ) as logs:
            self.assertTrue(scr.register_for_scr(engine))
            self.assertTrue(scr.register_for_scr(engine))

        self.assertEqual([call[0] for call in epsilon.calls], ["cache", "before"])
        self.assertEqual(epsilon.calls[0][1], [base, indexer, hbm])
        self.assertEqual(scr._registrations[id(engine)].tensors, (base, indexer, hbm))
        self.assertTrue(any("kv_bytes=104" in line for line in logs.output))

    def test_invalid_native_allocation_rejects_entire_registration(self) -> None:
        cpu = _FakeTensor(200)
        cpu.device = "cpu"
        strided = _FakeTensor(300)
        strided.is_contiguous = lambda: False
        for invalid in (
            cpu,
            strided,
            _FakeTensor(0),
            _FakeTensor(400, numel=0),
            object(),
        ):
            with self.subTest(invalid=invalid):
                scr._reset_for_test()
                engine = SimpleNamespace(
                    gpu_cache_tensors=lambda: [_FakeTensor(100), invalid]
                )
                epsilon = _FakeEpsilon()
                with mock.patch.dict(
                    os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True
                ), mock.patch.object(
                    scr, "_load_epsilon", return_value=epsilon
                ), mock.patch.object(
                    scr,
                    "_is_tensor",
                    side_effect=lambda value: isinstance(value, _FakeTensor),
                ), self.assertLogs(
                    scr.LOGGER, level="ERROR"
                ):
                    self.assertFalse(scr.register_for_scr(engine))
                self.assertEqual(epsilon.calls, [])
                self.assertFalse(scr._registrations[id(engine)].ok)

    def test_missing_native_export_does_not_fall_back_to_model_field_names(
        self,
    ) -> None:
        engine = SimpleNamespace(
            model=SimpleNamespace(
                py_model=SimpleNamespace(
                    kv_cache=SimpleNamespace(kv_cache_base_by_layer=[_FakeTensor(100)])
                )
            )
        )
        epsilon = _FakeEpsilon()
        with mock.patch.dict(
            os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), self.assertLogs(
            scr.LOGGER, level="ERROR"
        ):
            self.assertFalse(scr.register_for_scr(engine))
        self.assertEqual(epsilon.calls, [])

    def test_registration_is_inert_when_epsilon_is_not_active(self) -> None:
        allocations = []
        engine = SimpleNamespace(gpu_cache_tensors=lambda: allocations)
        epsilon = _FakeEpsilon(snap_enabled=False)

        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ):
            self.assertFalse(scr.register_for_scr(engine))
        self.assertEqual(epsilon.calls, [])
        self.assertNotIn(id(engine), scr._registrations)

    def test_registration_failure_can_retry_when_cache_becomes_ready(self) -> None:
        allocations = []
        engine = SimpleNamespace(gpu_cache_tensors=lambda: allocations)
        epsilon = _FakeEpsilon()
        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ):
            self.assertFalse(scr.register_for_scr(engine))
            allocations.append(_FakeTensor(123))
            self.assertTrue(scr.register_for_scr(engine))
        self.assertEqual([call[0] for call in epsilon.calls], ["before", "cache"])

    def test_before_callback_uses_captured_device(self) -> None:
        epsilon = _FakeEpsilon()
        engine = SimpleNamespace(gpu_cache_tensors=lambda: [_FakeTensor(1)])
        with mock.patch.dict(os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True), mock.patch.object(
            scr.importlib, "import_module", return_value=epsilon
        ), mock.patch.object(
            scr, "_is_tensor", side_effect=lambda value: isinstance(value, _FakeTensor)
        ), mock.patch.object(scr, "_capture_cuda_device", return_value=5), mock.patch(
            "torch.cuda.is_available", return_value=True
        ), mock.patch("torch.cuda._initialized", True), mock.patch(
            "torch.cuda.synchronize"
        ) as synchronize:
            self.assertTrue(scr.register_for_scr(engine))
            epsilon.before_callback()
        synchronize.assert_called_once_with(device=5)


if __name__ == "__main__":
    unittest.main()
