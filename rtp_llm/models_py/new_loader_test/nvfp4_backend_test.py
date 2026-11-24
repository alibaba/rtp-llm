"""CPU layout/admission tests; native CUDA kernels are deliberately not run."""

import types
import unittest
from unittest import mock

import torch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.models_py.modules.factory.fused_moe.utils.nvfp4_weights import (
    Nvfp4WeightViews,
)
from rtp_llm.models_py.new_loader_test import fp4_moe_test as fixture
from rtp_llm.models_py.quant_methods.fp4_moe import _resolve_backend
from rtp_llm.models_py.quant_methods.nvfp4_quantization import quantize_expert_to_nvfp4


class Nvfp4BackendTest(unittest.TestCase):
    def test_auto_strategy_rejects_other_kernels_layout(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
            CudaFp4EpLowLatencyStrategy, CudaFp4EpNormalStrategy, CudaFp4NoDPStrategy,
        )

        model = types.SimpleNamespace(quant_config=ModelOptFp4Config(4, 16, True))
        for backend in ("trtllm", "cutedsl"):
            cfg = types.SimpleNamespace(moe_strategy="auto", model_config=model,
                                        moe_config=types.SimpleNamespace(fp4_moe_op=backend))
            for strategy, required in (
                (CudaFp4NoDPStrategy, "trtllm"),
                (CudaFp4EpNormalStrategy, "trtllm"),
                (CudaFp4EpLowLatencyStrategy, "cutedsl"),
            ):
                checks = []
                strategy.check_conditions(types.SimpleNamespace(check=checks.append), cfg)
                self.assertEqual(all(checks), required == backend)

    def test_marlin_preflight_uses_target_device(self):
        method = fixture._experts(
            online=True,
            backend="marlin",
            moe_strategy="nvfp4_marlin",
            model_type="qwen4_exp",
        ).quant_method
        module = "rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4"
        active = []
        context = mock.MagicMock()
        context.__enter__.side_effect = lambda: active.append("target")
        context.__exit__.side_effect = lambda *args: active.pop() and False
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ), mock.patch("torch.cuda.device", return_value=context) as target, mock.patch(
            module + ".Nvfp4MarlinMoE.is_supported",
            side_effect=lambda: bool(active),
        ), mock.patch(
            module + ".load_library",
            side_effect=lambda: self.assertEqual(active, ["target"]),
        ) as library:
            method.validate_runtime_device("cuda:3")
        target.assert_called_once_with(torch.device("cuda:3"))
        library.assert_called_once_with()
        self.assertEqual(active, [])

    def test_marlin_module_import_and_scale_unswizzle_on_cpu(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.marlin_nvfp4_executor import (
            Qwen4MarlinExecutor,
            _linear_scales,
        )
        from rtp_llm.models_py.utils.quant_layout import swizzle_nvfp4_scales

        original = (torch.arange(2 * 128 * 8).reshape(2, 128, 8) % 7).to(
            torch.float8_e4m3fn
        )
        recovered = _linear_scales(swizzle_nvfp4_scales(original), 2, 128, 8)
        torch.testing.assert_close(
            recovered.view(torch.uint8), original.view(torch.uint8)
        )
        self.assertEqual(Qwen4MarlinExecutor.executor_type().name, "MARLIN_NVFP4")

    def test_backend_resolution_does_not_mutate_frozen_config(self):
        for strategy, required in (
            ("fp4_no_dp", "trtllm"),
            ("fp4_ep_normal", "trtllm"),
            ("fp4_ep_low_latency", "cutedsl"),
        ):
            for requested in ("auto", "trtllm", "cutedsl"):
                cfg = types.SimpleNamespace(moe_strategy=strategy, fp4_moe_op=requested)
                if requested in ("auto", required):
                    self.assertEqual(_resolve_backend(cfg), required)
                else:
                    with self.assertRaisesRegex(ValueError, "requires FP4_MOE_OP"):
                        _resolve_backend(cfg)
        for backend in ("cutlass", "marlin"):
            cfg = types.SimpleNamespace(
                fp4_moe_op="auto", moe_strategy="nvfp4_" + backend
            )
            self.assertEqual(_resolve_backend(cfg), backend)
            self.assertEqual(cfg.fp4_moe_op, "auto")
            with self.assertRaisesRegex(ValueError, "requires"):
                _resolve_backend(
                    types.SimpleNamespace(fp4_moe_op=backend, moe_strategy="auto")
                )
        self.assertEqual(
            _resolve_backend(types.SimpleNamespace(fp4_moe_op="auto")), "trtllm"
        )
        self.assertEqual(
            _resolve_backend(
                types.SimpleNamespace(
                    fp4_moe_op="auto", use_deepep_moe=True, use_deepep_low_latency=True
                )
            ),
            "cutedsl",
        )

    def test_cutlass_checkpoint_preserves_up_gate_order_and_storage(self):
        weights = fixture._weights(True)
        layer = fixture.Fp4MoETest()._load(
            weights,
            online=True,
            backend="auto",
            moe_strategy="nvfp4_cutlass",
            activation_amax=32,
        )
        expected, _, _ = quantize_expert_to_nvfp4(
            torch.cat(
                (
                    weights["experts.0.up_proj.weight"],
                    weights["experts.0.gate_proj.weight"],
                )
            )
        )
        torch.testing.assert_close(layer.w13[0], expected)
        config = types.SimpleNamespace(
            ep_size=1,
            ep_rank=0,
            parallelism_config=types.SimpleNamespace(
                get_ffn_tp_size=lambda: 1, get_ffn_tp_rank=lambda: 0
            ),
        )
        views = Nvfp4WeightViews(config, object(), layer.eplb_runtime_weight_view())
        self.assertEqual(views.w1.data_ptr(), layer.w13.data_ptr())
        self.assertEqual(views.w2.data_ptr(), layer.w2.data_ptr())
        scales = views.cutlass_quant_scales()
        torch.testing.assert_close(scales[0], torch.tensor(6 * 448 / 32.0))
        torch.testing.assert_close(scales[2], layer.w13_input_scale * layer.w13_scale_2)

    def test_qwen4_recipe_defaults_match_main(self):
        cfg = ModelOptFp4Config(
            bits=4, group_size=16, is_quanted=False, model_recipe=QWEN4_FP8_NVFP4_RECIPE
        )
        self.assertEqual(cfg.source_fp8_block_size, 128)
        self.assertEqual(cfg.activation_amax, 32.0)
        self.assertTrue(cfg.require_contiguous_experts)
        self.assertEqual(cfg.supported_model_types, ("qwen4_exp", "qwen4_exp_mtp"))
        self.assertEqual(cfg.dense_storage, "fp8")
        self.assertEqual(cfg.mtp_dense_storage, "inherit")

    def test_backend_scope_matches_main(self):
        with self.assertRaisesRegex(ValueError, "requires online"):
            fixture._experts(backend="cutlass", moe_strategy="nvfp4_cutlass")
        with self.assertRaisesRegex(ValueError, "only for Qwen4"):
            fixture._experts(online=True, backend="marlin", moe_strategy="nvfp4_marlin")
        with self.assertRaisesRegex(ValueError, "redundant"):
            fixture._experts(
                online=True,
                redundant=2,
                backend="cutlass",
                moe_strategy="nvfp4_cutlass",
            )

    def test_cutlass_missing_dependency_fails_in_preflight(self):
        method = fixture._experts(
            online=True, backend="cutlass", moe_strategy="nvfp4_cutlass"
        ).quant_method
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(12, 0)
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp4_moe.importlib.import_module",
            side_effect=ImportError("missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cutlass_fused_moe"):
                method.validate_runtime_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
