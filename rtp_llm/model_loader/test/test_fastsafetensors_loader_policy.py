"""CPU-only tests for RTP's FastSafeTensors integration policy.

Covers:
  - rank-local direct/raw checkpoint key closure
  - stacked-MoE transitional mode parsing and propagation
  - AutoLoader capability gating
  - bounded and full-stacked transient-memory budgets
"""

import logging
import os
import sys
import types
import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
    LoadQuantPerChannelFp8Weight,
    per_channel_cast_to_fp8_expert,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorCollector
from rtp_llm.utils.database import (
    FASTSAFETENSORS_STACKED_MOE_MODE_ENV,
    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    FastSafeTensorsCompatibilityError,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, stack_, stack_moe_w1


class TestFastsafetensorsLoaderPolicy(unittest.TestCase):
    def test_per_expert_copyout_keys_exclude_raw_stacked_keys(self):
        result = ModelLoader._build_fastsafetensors_local_copyout_keys(
            {"direct.weight": object(), "expanded.experts.0": object()},
            {"stacked.raw": "expanded.experts.{expert_id}"},
            FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
        )

        self.assertEqual(
            result,
            frozenset({"direct.weight", "expanded.experts.0"}),
        )

    def test_full_stacked_copyout_keys_include_raw_stacked_keys(self):
        result = ModelLoader._build_fastsafetensors_local_copyout_keys(
            {"direct.weight": object(), "expanded.experts.0": object()},
            {"stacked.raw": "expanded.experts.{expert_id}"},
            FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
        )

        self.assertEqual(
            result,
            frozenset({"direct.weight", "expanded.experts.0", "stacked.raw"}),
        )

    def test_rank_local_copyout_filter_and_mode_are_forwarded(self):
        collector = MagicMock()
        collector.store_tensor.return_value = True
        collector.is_collection_complete.return_value = True
        weight = MagicMock()
        weight.name = "needed-weight"
        weight.load.return_value = {}
        weight_info = ModelLoader.WeightInfo(weight, 7, collector)
        database = MagicMock()

        observed_filter = []
        observed_modes = []

        def iterate(*args, **kwargs):
            predicate = kwargs["local_copyout_filter"]
            observed_modes.append(kwargs["stacked_moe_mode"])
            observed_filter.extend(
                [
                    predicate("needed.tensor"),
                    predicate("stacked.raw"),
                    predicate("unused.tensor"),
                ]
            )
            return iter((("needed.tensor", torch.ones(1)),))

        database.fastsafetensors_weights_iterator.side_effect = iterate

        loader = object.__new__(ModelLoader)
        loader._load_config = types.SimpleNamespace(database=database)
        loader._create_model_weights = MagicMock(return_value=MagicMock())
        loader._generate_weight_info = MagicMock(
            return_value=({"needed.tensor": weight_info}, [weight_info])
        )
        loader._build_stacked_key_config = MagicMock(
            return_value={"stacked.raw": "experts.{expert_id}.weight"}
        )
        loader._is_online_ptpc = MagicMock(return_value=False)

        # Preserve the pre-existing one-argument private-call contract. The
        # transitional default is the bounded per-expert delivery path.
        loader._load_from_fastsafetensor("cuda:0")

        self.assertEqual(observed_filter, [True, False, False])
        self.assertEqual(observed_modes, [FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT])
        loader._build_stacked_key_config.assert_called_once_with(
            [weight_info], database
        )
        self.assertEqual(
            database.fastsafetensors_weights_iterator.call_args.kwargs[
                "stacked_key_config"
            ],
            {"stacked.raw": "experts.{expert_id}.weight"},
        )
        weight.load.assert_called_once()

    def test_stacked_moe_mode_defaults_and_empty_values(self):
        for value in (None, "", "   "):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=False):
                os.environ.pop(FASTSAFETENSORS_STACKED_MOE_MODE_ENV, None)
                if value is not None:
                    os.environ[FASTSAFETENSORS_STACKED_MOE_MODE_ENV] = value
                self.assertEqual(
                    ModelLoader._fastsafetensors_stacked_moe_mode(),
                    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                )

    def test_full_stacked_mode_is_explicit_opt_in(self):
        with patch.dict(
            os.environ,
            {
                FASTSAFETENSORS_STACKED_MOE_MODE_ENV: (
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                )
            },
            clear=False,
        ):
            self.assertEqual(
                ModelLoader._fastsafetensors_stacked_moe_mode(),
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
            )

    def test_unknown_stacked_moe_mode_is_rejected(self):
        with (
            patch.dict(
                os.environ,
                {FASTSAFETENSORS_STACKED_MOE_MODE_ENV: "surprise"},
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "per-expert.*full-stacked"),
        ):
            ModelLoader._fastsafetensors_stacked_moe_mode()

    def test_auto_loader_capabilities_follow_mode(self):
        module = types.ModuleType("fastsafetensors")

        class AutoLoader:
            def __init__(self, pg, files, device, stacked_moe_tensors=None):
                pass

        module.AutoLoader = AutoLoader
        module.SingleGroup = object
        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertIsNone(
                ModelLoader._fastsafetensors_capability_error(
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                )
            )
            self.assertIsNone(
                ModelLoader._fastsafetensors_capability_error(
                    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
                )
            )

    def test_legacy_dim0_split_keyword_remains_supported(self):
        module = types.ModuleType("fastsafetensors")

        class AutoLoader:
            def __init__(self, pg, files, device, dim0_split_templates=None):
                pass

        module.AutoLoader = AutoLoader
        module.SingleGroup = object
        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertIsNone(
                ModelLoader._fastsafetensors_capability_error(
                    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
                )
            )

    def test_kwargs_only_signature_is_not_treated_as_split_capability(self):
        module = types.ModuleType("fastsafetensors")

        class AutoLoader:
            def __init__(self, pg, files, device, **kwargs):
                pass

        module.AutoLoader = AutoLoader
        module.SingleGroup = object
        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertIn(
                "stacked_moe_tensors",
                ModelLoader._fastsafetensors_capability_error(
                    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
                ),
            )

    def test_missing_split_capability_resolves_to_full_stacked(self):
        with patch.object(
            ModelLoader,
            "_fastsafetensors_capability_error",
            side_effect=lambda mode: (
                None
                if mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                else "AutoLoader.__init__ is missing stacked_moe_tensors"
            ),
        ):
            self.assertEqual(
                ModelLoader._resolve_fastsafetensors_mode(
                    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
                ),
                (
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                    "AutoLoader.__init__ is missing stacked_moe_tensors",
                ),
            )

    def test_auto_mode_falls_back_when_wrapper_is_incompatible(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._load_config = types.SimpleNamespace(
            database=types.SimpleNamespace(
                is_safetensor=True,
                get_pretrain_tensor_names=lambda: ["weight"],
            )
        )
        loader._choose_weight_convert_device = MagicMock(return_value="cuda")
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
                "package-not-installed",
            )
        )
        loader._load_from_scratch = MagicMock(return_value="scratch")
        loader._load_from_fastsafetensor = MagicMock()

        self.assertEqual(loader._load_weight("cuda"), "scratch")
        loader._load_from_scratch.assert_called_once_with("cuda")
        loader._load_from_fastsafetensor.assert_not_called()

    def test_env_compat_is_applied_before_capability_import(self):
        observed_config_json = []

        class RecordingModule(types.ModuleType):
            def __getattr__(self, name):
                if name == "AutoLoader":
                    observed_config_json.append(
                        os.environ.get("FASTSAFETENSORS_CONFIG_JSON")
                    )

                    class AutoLoader:
                        def __init__(self, pg, files, device):
                            pass

                    return AutoLoader
                if name == "SingleGroup":
                    return object
                raise AttributeError(name)

        module = RecordingModule("fastsafetensors")
        with (
            patch.dict(sys.modules, {"fastsafetensors": module}),
            patch.dict(
                os.environ,
                {
                    "FASTSAFETENSORS_NOGDS": "1",
                    "FASTSAFETENSORS_CONFIG_JSON": '{"loader":"fuse-shm"}',
                },
                clear=False,
            ),
        ):
            ModelLoader._resolve_fastsafetensors_mode(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
            )

        self.assertTrue(observed_config_json)
        self.assertEqual(
            set(observed_config_json),
            {'{"loader":"base","base":{"copier_type":"nogds"}}'},
        )

    def test_auto_mode_uses_resolved_mode_when_memory_is_enough(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._load_config = types.SimpleNamespace(
            database=types.SimpleNamespace(
                is_safetensor=True,
                get_pretrain_tensor_names=lambda: ["weight"],
            )
        )
        loader._choose_weight_convert_device = MagicMock(return_value="cuda")
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
            )
        )
        loader._is_memory_enough_for_fastsafetensor = MagicMock(return_value=True)
        loader._load_from_fastsafetensor = MagicMock(return_value="fast")

        self.assertEqual(loader._load_weight("cuda"), "fast")
        loader._is_memory_enough_for_fastsafetensor.assert_called_once_with(
            FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
        )
        loader._load_from_fastsafetensor.assert_called_once_with(
            "cuda", FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
        )

    def test_auto_mode_skips_capability_resolution_when_prerequisites_fail(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._load_config = types.SimpleNamespace(
            database=types.SimpleNamespace(
                is_safetensor=False,
                get_pretrain_tensor_names=lambda: ["weight"],
            )
        )
        loader._choose_weight_convert_device = MagicMock(return_value="cuda")
        loader._resolve_and_log_fastsafetensors_mode = MagicMock()
        loader._load_from_scratch = MagicMock(return_value="scratch")

        with self.assertLogs(level="INFO") as logs:
            self.assertEqual(loader._load_weight("cuda"), "scratch")
        loader._resolve_and_log_fastsafetensors_mode.assert_not_called()
        output = "\n".join(logs.output)
        self.assertIn("requested_mode=auto", output)
        self.assertIn("effective_mode=scratch", output)
        self.assertIn("prerequisite-failed", output)
        self.assertIn("falls back to scratch", output)

    def test_auto_mode_uses_scratch_when_memory_is_insufficient(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._load_config = types.SimpleNamespace(
            database=types.SimpleNamespace(
                is_safetensor=True,
                get_pretrain_tensor_names=lambda: ["weight"],
            )
        )
        loader._choose_weight_convert_device = MagicMock(return_value="cuda")
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
            )
        )
        loader._is_memory_enough_for_fastsafetensor = MagicMock(return_value=False)
        loader._load_from_scratch = MagicMock(return_value="scratch")
        loader._load_from_fastsafetensor = MagicMock()

        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(loader._load_weight("cuda"), "scratch")
        self.assertIn("memory-preflight-failed", "\n".join(logs.output))
        self.assertIn("falls back to scratch", "\n".join(logs.output))
        loader._load_from_fastsafetensor.assert_not_called()

    def test_explicit_mode_falls_back_to_scratch_for_incompatible_wrapper(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
                "package-not-installed",
            )
        )
        loader._load_from_scratch = MagicMock(return_value="scratch")

        self.assertEqual(loader._load_weight("cuda"), "scratch")
        loader._load_from_scratch.assert_called_once_with("cuda")

    def test_runtime_compatibility_error_falls_back_to_scratch(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
            )
        )
        loader._load_from_fastsafetensor = MagicMock(
            side_effect=FastSafeTensorsCompatibilityError("native ABI mismatch")
        )
        loader._load_from_scratch = MagicMock(return_value="scratch")

        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(loader._load_weight("cuda"), "scratch")

        loader._load_from_scratch.assert_called_once_with("cuda")
        self.assertIn("runtime-compatibility-failed", "\n".join(logs.output))
        self.assertIn("falls back to scratch", "\n".join(logs.output))

    def test_runtime_fallback_releases_traceback_before_scratch(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
            )
        )
        payload_ref = None

        class Payload:
            pass

        def fail_after_partial_load(*_args):
            nonlocal payload_ref
            payload = Payload()
            payload_ref = weakref.ref(payload)
            raise FastSafeTensorsCompatibilityError("late ABI mismatch")

        loader._load_from_fastsafetensor = fail_after_partial_load
        loader.force_clean_cuda_memory = MagicMock()

        def scratch_after_cleanup(_device):
            self.assertIsNotNone(payload_ref)
            self.assertIsNone(payload_ref())
            return "scratch"

        loader._load_from_scratch = MagicMock(side_effect=scratch_after_cleanup)

        self.assertEqual(loader._load_weight("cuda"), "scratch")
        loader.force_clean_cuda_memory.assert_called_once_with()

    def test_checkpoint_runtime_error_remains_fail_fast(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
            )
        )
        loader._load_from_fastsafetensor = MagicMock(
            side_effect=RuntimeError("checkpoint tensor shape mismatch")
        )
        loader._load_from_scratch = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            loader._load_weight("cuda")

        loader._load_from_scratch.assert_not_called()

    def test_explicit_per_expert_request_skips_memory_preflight(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                None,
            )
        )
        loader._is_memory_enough_for_fastsafetensor = MagicMock()
        loader._load_from_fastsafetensor = MagicMock(return_value="per-expert")

        self.assertEqual(loader._load_weight("cuda"), "per-expert")
        loader._is_memory_enough_for_fastsafetensor.assert_not_called()

    def test_explicit_mode_uses_full_stacked_when_split_is_missing(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                "AutoLoader.__init__ is missing stacked_moe_tensors",
            )
        )
        loader._has_raw_stacked_moe_weights = MagicMock(return_value=True)
        loader._is_memory_enough_for_fastsafetensor = MagicMock(return_value=True)
        loader._load_from_fastsafetensor = MagicMock(return_value="full-stacked")

        self.assertEqual(loader._load_weight("cuda"), "full-stacked")
        loader._is_memory_enough_for_fastsafetensor.assert_called_once_with(
            FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
        )
        loader._load_from_fastsafetensor.assert_called_once_with(
            "cuda", FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
        )

    def test_explicit_mode_uses_scratch_when_full_stacked_budget_is_insufficient(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                "AutoLoader.__init__ is missing stacked_moe_tensors",
            )
        )
        loader._has_raw_stacked_moe_weights = MagicMock(return_value=True)
        loader._is_memory_enough_for_fastsafetensor = MagicMock(return_value=False)
        loader._load_from_scratch = MagicMock(return_value="scratch")
        loader._load_from_fastsafetensor = MagicMock()

        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(loader._load_weight("cuda"), "scratch")
        loader._load_from_scratch.assert_called_once_with("cuda")
        loader._load_from_fastsafetensor.assert_not_called()
        self.assertIn("falls back to scratch", "\n".join(logs.output))

    def test_explicit_full_stacked_request_always_uses_memory_preflight(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                None,
            )
        )
        loader._has_raw_stacked_moe_weights = MagicMock(return_value=True)
        loader._is_memory_enough_for_fastsafetensor = MagicMock(return_value=False)
        loader._load_from_scratch = MagicMock(return_value="scratch")
        loader._load_from_fastsafetensor = MagicMock()

        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(loader._load_weight("cuda"), "scratch")
        loader._is_memory_enough_for_fastsafetensor.assert_called_once_with(
            FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
        )
        loader._load_from_fastsafetensor.assert_not_called()
        self.assertIn("falls back to scratch", "\n".join(logs.output))

    def test_explicit_full_stacked_without_raw_stacked_weights_skips_preflight(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._resolve_and_log_fastsafetensors_mode = MagicMock(
            return_value=(
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                None,
            )
        )
        loader._has_raw_stacked_moe_weights = MagicMock(return_value=False)
        loader._is_memory_enough_for_fastsafetensor = MagicMock()
        loader._load_from_fastsafetensor = MagicMock(return_value="fast")

        self.assertEqual(loader._load_weight("cuda"), "fast")
        loader._is_memory_enough_for_fastsafetensor.assert_not_called()
        loader._load_from_fastsafetensor.assert_called_once_with(
            "cuda", FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
        )

    def test_mode_resolution_logs_stable_structured_fields(self):
        loader = object.__new__(ModelLoader)
        loader._fastsafetensors_stacked_moe_mode = MagicMock(
            return_value=FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
        )
        cases = [
            (
                (None, "package-not-installed: missing"),
                "INFO",
                "effective_mode=scratch",
            ),
            (
                (
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                    "AutoLoader.__init__ is missing stacked_moe_tensors",
                ),
                "WARNING",
                "effective_mode=full-stacked",
            ),
            (
                (FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT, None),
                "INFO",
                "effective_mode=per-expert",
            ),
        ]
        for resolved, level, expected in cases:
            with self.subTest(resolved=resolved):
                loader._resolve_fastsafetensors_mode = MagicMock(return_value=resolved)
                with self.assertLogs(level=level) as logs:
                    loader._resolve_and_log_fastsafetensors_mode("test")
                self.assertEqual([record.levelname for record in logs.records], [level])
                output = "\n".join(logs.output)
                self.assertIn("requested_mode=per-expert", output)
                self.assertIn(expected, output)
                if resolved[1] is not None:
                    self.assertIn("degraded_reason=", output)


class TestMoeAtomicWeightTensorNames(unittest.TestCase):
    @staticmethod
    def _load_config(database, experts=(0, 1)):
        config = MagicMock()
        config.database = database
        config.get_selected_experts.return_value = list(experts)
        return config

    def test_per_expert_template_never_probes_raw_stacked_key(self):
        database = MagicMock()
        weight = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[
                CkptWeightInfo("model.layers.{i}.experts.{expert_id}.gate_proj.weight")
            ],
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )

        names = weight.get_tensor_names(13, self._load_config(database))

        self.assertEqual(
            names,
            {
                "model.layers.13.experts.0.gate_proj.weight",
                "model.layers.13.experts.1.gate_proj.weight",
            },
        )
        database.has_tensor.assert_not_called()

    def test_existing_raw_stacked_tensor_uses_logical_expert_keys(self):
        database = MagicMock()
        database.has_tensor.return_value = True
        weight = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo("model.layers.{i}.moe.w1")],
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )

        names = weight.get_tensor_names(13, self._load_config(database))

        self.assertEqual(
            names,
            {
                f"layers.13.moe.{W.moe_w1}.0.0",
                f"layers.13.moe.{W.moe_w1}.1.0",
            },
        )
        database.has_tensor.assert_called_once_with("model.layers.13.moe.w1")

    def test_mixed_raw_stacked_weights_register_only_existing_atomic_weight(self):
        database = MagicMock()
        database.has_tensor.side_effect = lambda name: name.endswith(".w1")
        config = MoeConfig(expert_num=2)
        w1 = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo("model.layers.{i}.moe.w1")],
            config=config,
            stacked_ckpt_keys=True,
        )
        w2 = MoeAtomicWeight(
            name=W.moe_w2,
            weights=[CkptWeightInfo("model.layers.{i}.moe.w2")],
            config=config,
            stacked_ckpt_keys=True,
        )
        weight_info_list = [
            types.SimpleNamespace(weight=w1, layer_id=13),
            types.SimpleNamespace(weight=w2, layer_id=13),
        ]

        result = ModelLoader._build_stacked_key_config(
            weight_info_list, database=database
        )

        self.assertEqual(
            result,
            {"model.layers.13.moe.w1": (f"layers.13.moe.{W.moe_w1}.{{expert_id}}.0")},
        )

    def test_logical_collector_and_raw_database_loading_are_equivalent(self):
        database = MagicMock()
        raw_key = "model.layers.0.moe.w1"
        raw_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        database.has_tensor.side_effect = lambda name: name == raw_key
        database.load_tensor.side_effect = lambda name, _dtype: (
            [raw_tensor] if name == raw_key else []
        )
        weight = MoeAtomicWeight(
            name="test_moe",
            weights=[CkptWeightInfo("model.layers.{i}.moe.w1")],
            process_fun=stack_,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )
        load_config = self._load_config(database)
        load_config.compute_dtype = torch.float32
        load_config.merge_lora = False
        load_config.moe_pure_tp_preshard = False
        load_config.tp_size = 1
        load_config.dp_size = 1
        load_config.ep_size = 1
        load_config.exported_device.maybe_rewrite_weight_by_key.side_effect = (
            lambda _name, tensor: tensor
        )

        logical_keys = weight.get_tensor_names(0, load_config)
        collector = TensorCollector(logical_keys, database)
        for expert_id in range(2):
            collector.store_tensor(
                f"layers.0.moe.test_moe.{expert_id}.0",
                raw_tensor[expert_id].clone(),
            )

        collector_result = weight.load(collector, 0, "cpu", load_config)["test_moe"]
        database_result = weight.load(
            DatabaseTensorSource(database), 0, "cpu", load_config
        )["test_moe"]

        torch.testing.assert_close(collector_result, database_result)
        torch.testing.assert_close(collector_result, raw_tensor)

    def test_public_moe_layout_resolver_wraps_raw_and_logical_sources(self):
        database = MagicMock()
        raw_key = "model.layers.0.moe.w2"
        database.has_tensor.side_effect = lambda name: name == raw_key
        weight = MoeAtomicWeight(
            name=W.moe_w2,
            weights=[CkptWeightInfo(raw_key)],
            process_fun=stack_,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )
        load_config = self._load_config(database)

        raw_layout = weight.resolve_expert_layout(
            DatabaseTensorSource(database), 0, load_config
        )
        self.assertTrue(raw_layout.uses_stacked_keys)
        self.assertTrue(raw_layout.source_contains_raw_stacked)
        self.assertEqual(raw_layout.selected_experts, (0, 1))
        self.assertEqual(
            [ckpt.name for ckpt in raw_layout.ckpt_weights],
            [f"layers.{{i}}.moe.{W.moe_w2}.{{expert_id}}.0"],
        )

        logical_keys = {
            f"layers.0.moe.{W.moe_w2}.{expert_id}.0" for expert_id in range(2)
        }
        collector = TensorCollector(logical_keys, database)
        for expert_id in range(2):
            collector.store_tensor(
                f"layers.0.moe.{W.moe_w2}.{expert_id}.0",
                torch.ones(1, 2),
            )
        logical_layout = weight.resolve_expert_layout(collector, 0, load_config)
        self.assertTrue(logical_layout.uses_stacked_keys)
        self.assertFalse(logical_layout.source_contains_raw_stacked)
        self.assertIs(logical_layout.tensor_source, collector)

    def test_inline_fp8_uses_logical_keys_from_collector(self):
        database = MagicMock()
        database.has_tensor.return_value = False
        kernel = MoeAtomicWeight(
            name=W.moe_w2,
            weights=[CkptWeightInfo("model.layers.{i}.moe.w2")],
            process_fun=stack_,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )
        load_config = self._load_config(database)
        load_config.compute_dtype = torch.float32
        logical_keys = {
            f"layers.0.moe.{W.moe_w2}.{expert_id}.0" for expert_id in range(2)
        }
        collector = TensorCollector(logical_keys, database)
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_fp8 = []
        expected_scales = []
        for expert_id in range(2):
            key = f"layers.0.moe.{W.moe_w2}.{expert_id}.0"
            fp8_tensor, scale = per_channel_cast_to_fp8_expert(
                source[expert_id].reshape(1, -1)
            )
            collector.store_fp8_quantized(key, fp8_tensor, scale)
            expected_fp8.append(fp8_tensor)
            expected_scales.append(scale)

        quant_weight = object.__new__(LoadQuantPerChannelFp8Weight)
        quant_weight.kernel = kernel
        quant_weight.scale = types.SimpleNamespace(name="test_scale")
        result = quant_weight._load_moe_inline_quant(collector, 0, "cpu", load_config)

        self.assertEqual(result[W.moe_w2].shape, (2, 1, 2))
        self.assertEqual(result["test_scale"].shape, (2, 1, 1))
        torch.testing.assert_close(result[W.moe_w2], torch.stack(expected_fp8))
        torch.testing.assert_close(result["test_scale"], torch.stack(expected_scales))

    def test_inline_fp8_preserves_expert_and_gate_up_order(self):
        database = MagicMock()
        database.has_tensor.return_value = False
        kernel = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[
                CkptWeightInfo("gate.{i}.{expert_id}"),
                CkptWeightInfo("up.{i}.{expert_id}"),
            ],
            process_fun=stack_moe_w1,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )
        load_config = self._load_config(database)
        load_config.compute_dtype = torch.float32
        logical_keys = {
            f"layers.0.moe.{W.moe_w1}.{expert_id}.{weight_idx}"
            for expert_id in range(2)
            for weight_idx in range(2)
        }
        collector = TensorCollector(logical_keys, database)
        expected_fp8 = [[], []]
        expected_scales = [[], []]
        for weight_idx, offset in enumerate((10.0, 100.0)):
            for expert_id in range(2):
                source = torch.tensor(
                    [[offset + expert_id * 10, offset + expert_id * 10 + 1]]
                )
                fp8_tensor, scale = per_channel_cast_to_fp8_expert(source)
                collector.store_fp8_quantized(
                    f"layers.0.moe.{W.moe_w1}.{expert_id}.{weight_idx}",
                    fp8_tensor,
                    scale,
                )
                expected_fp8[expert_id].append(fp8_tensor)
                expected_scales[expert_id].append(scale)

        quant_weight = object.__new__(LoadQuantPerChannelFp8Weight)
        quant_weight.kernel = kernel
        quant_weight.scale = types.SimpleNamespace(name="test_scale")
        result = quant_weight._load_moe_inline_quant(collector, 0, "cpu", load_config)

        expected_weight = torch.stack(
            [torch.cat(parts, dim=0) for parts in expected_fp8]
        )
        expected_scale = torch.stack(
            [torch.cat(parts, dim=0) for parts in expected_scales]
        )
        torch.testing.assert_close(result[W.moe_w1], expected_weight)
        torch.testing.assert_close(result["test_scale"], expected_scale)


class TestFastsafetensorsTransientBudget(unittest.TestCase):
    @staticmethod
    def _module_with_estimate(estimate):
        module = types.ModuleType("fastsafetensors")
        module.load_config = lambda: types.SimpleNamespace(
            estimated_peak_device_bytes=estimate
        )
        return module

    @staticmethod
    def _loader_for_memory_check(free_bytes, max_file_size):
        loader = object.__new__(ModelLoader)
        model_config = MagicMock()
        model_config.eval_model_weight_size.return_value = 0
        loader._weights_info = types.SimpleNamespace(model_config=model_config)
        device_mem_info = (
            None if free_bytes is None else types.SimpleNamespace(free=free_bytes)
        )
        loader._load_config = types.SimpleNamespace(
            exported_device=types.SimpleNamespace(get_mem_info=lambda: device_mem_info),
            database=types.SimpleNamespace(get_max_file_size=lambda: max_file_size),
            ep_size=1,
            tp_size=1,
        )
        loader._is_online_ptpc = MagicMock(return_value=False)
        loader._is_online_quant_without_inline = MagicMock(return_value=False)
        return loader

    def test_uses_positive_configured_bounded_peak(self):
        reserve = 2 * 1024**3
        module = self._module_with_estimate(8 * 1024)
        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertEqual(
                ModelLoader._fastsafetensors_transient_budget_bytes(4096),
                8 * 1024 + reserve,
            )

    def test_invalid_estimates_use_three_max_files(self):
        reserve = 2 * 1024**3
        for estimate in (None, 0, -1, "8192", float("inf"), True):
            with self.subTest(estimate=estimate):
                module = self._module_with_estimate(estimate)
                with patch.dict(sys.modules, {"fastsafetensors": module}):
                    self.assertEqual(
                        ModelLoader._fastsafetensors_transient_budget_bytes(4096),
                        3 * 4096 + reserve,
                    )

    def test_missing_load_config_uses_three_max_files(self):
        reserve = 2 * 1024**3
        module = types.ModuleType("fastsafetensors")
        with (
            patch.dict(sys.modules, {"fastsafetensors": module}),
            self.assertLogs(level="WARNING") as logs,
        ):
            self.assertEqual(
                ModelLoader._fastsafetensors_transient_budget_bytes(4096),
                3 * 4096 + reserve,
            )
        self.assertIn("legacy estimate", "\n".join(logs.output))

    def test_load_config_runtime_and_key_errors_use_three_max_files(self):
        reserve = 2 * 1024**3
        for error in (RuntimeError("bad runtime config"), KeyError("missing field")):
            with self.subTest(error=error):
                module = types.ModuleType("fastsafetensors")
                module.load_config = MagicMock(side_effect=error)
                with patch.dict(sys.modules, {"fastsafetensors": module}):
                    self.assertEqual(
                        ModelLoader._fastsafetensors_transient_budget_bytes(4096),
                        3 * 4096 + reserve,
                    )

    def test_full_stacked_adds_one_max_file_to_positive_estimate(self):
        reserve = 2 * 1024**3
        module = self._module_with_estimate(8 * 1024)
        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertEqual(
                ModelLoader._fastsafetensors_transient_budget_bytes(
                    4096,
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                    has_raw_stacked_moe=True,
                ),
                12 * 1024 + reserve,
            )

    def test_full_stacked_without_raw_stacked_moe_does_not_add_shard(self):
        reserve = 2 * 1024**3
        module = self._module_with_estimate(8 * 1024)
        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertEqual(
                ModelLoader._fastsafetensors_transient_budget_bytes(
                    4096,
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                    has_raw_stacked_moe=False,
                ),
                8 * 1024 + reserve,
            )

    def test_memory_check_returns_false_without_device_info(self):
        loader = self._loader_for_memory_check(None, 1024 * 1024)

        self.assertFalse(
            loader._is_memory_enough_for_fastsafetensor(
                FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
            )
        )

    def test_full_stacked_extra_shard_changes_auto_admission(self):
        gib = 1024**3
        loader = self._loader_for_memory_check(int(4.5 * gib), gib)
        loader._has_raw_stacked_moe_weights = MagicMock(return_value=True)
        module = self._module_with_estimate(2 * gib)

        with patch.dict(sys.modules, {"fastsafetensors": module}):
            self.assertTrue(
                loader._is_memory_enough_for_fastsafetensor(
                    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
                )
            )
            self.assertFalse(
                loader._is_memory_enough_for_fastsafetensor(
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                )
            )


if __name__ == "__main__":
    unittest.main()
