"""Runtime dispatch checks for the complete Qwen FP8/NVFP4 recipe."""

import builtins
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, ModelOptFp4Config


class QwenAttentionFp8RuntimeTest(unittest.TestCase):
    def test_native_fp8_does_not_import_nvfp4_shared_overlap(self):
        from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer

        real_import = builtins.__import__

        def reject_nvfp4(name, *args, **kwargs):
            if "cutlass_nvfp4_executor" in name:
                raise AssertionError("native FP8 imported an NVFP4 executor")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", reject_nvfp4):
            Qwen4MoeLayer.prepare_shared_expert_overlap(
                SimpleNamespace(config=SimpleNamespace(_qwen4_full_fp8=True))
            )

    def test_native_sm90_graph_selects_masked_fp8_without_relaxing_hybrid(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_per_block import (
            CudaFp8PerBlockNoDPMaskedStrategy,
            CudaFp8PerBlockNoDPStrategy,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
            StrategyRegistry,
        )

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
                data_type="bf16",
                _qwen4_full_fp8=True,
                _qwen4_full_fp8_nvfp4=False,
            ),
            ep_size=1,
            tp_size=4,
            moe_strategy="auto",
            moe_config=SimpleNamespace(use_all_gather=True),
            moe_quant_method="FP8_PER_BLOCK",
            enable_cuda_graph=True,
        )
        registry = StrategyRegistry()
        registry.register(CudaFp8PerBlockNoDPStrategy())
        registry.register(CudaFp8PerBlockNoDPMaskedStrategy())
        root = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
        with (
            mock.patch("rtp_llm.models_py.utils.arch.get_sm", return_value=(9, 0)),
            mock.patch(root + "deepgemm_hybrid_executor.get_sm", return_value=(9, 0)),
            mock.patch(
                root + "deepgemm_masked_executor_v2.get_sm", return_value=(9, 0)
            ),
            mock.patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm",
                return_value=True,
            ),
        ):
            self.assertIsInstance(
                registry.get_strategy(config), CudaFp8PerBlockNoDPMaskedStrategy
            )
            config.enable_cuda_graph = False
            self.assertIsInstance(
                registry.get_strategy(config), CudaFp8PerBlockNoDPStrategy
            )
            config.enable_cuda_graph = True
            config.model_config._qwen4_full_fp8 = False
            with self.assertRaises(ValueError):
                registry.get_strategy(config)

    def test_runtime_attention_quant_config_does_not_change_moe_config(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextDecoderLayer
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer

        original = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        cfg = SimpleNamespace(quant_config=original, _qwen4_full_fp8=True)
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        self.assertEqual(
            layer._attention_quant_config(cfg, {}).get_method(), "FP8_PER_BLOCK"
        )
        self.assertIs(cfg.quant_config, original)
        cfg._qwen4_full_fp8 = False
        self.assertIs(layer._attention_quant_config(cfg, {}), original)
        cfg._qwen4_nvfp4_source = True
        self.assertIsNone(layer._attention_quant_config(cfg, {}))
        self.assertIs(cfg.quant_config, original)
        cfg._qwen4_nvfp4_source = False
        base = Qwen3NextDecoderLayer.__new__(Qwen3NextDecoderLayer)
        cfg._qwen4_full_fp8 = True
        self.assertIs(base._attention_quant_config(cfg, {}), original)

    def test_offline_nvfp4_dense_projection_keeps_unquantized_consumer(self):
        from rtp_llm.config.quant_config import init_quant_config
        from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import Qwen4LinearFactory

        quant = init_quant_config("QWEN4_NVFP4")
        weights = {"dense": object()}
        with mock.patch.object(LinearFactory, "create_linear_from_weights") as create:
            Qwen4LinearFactory.create_linear_from_weights(
                weights,
                "dense",
                quant_config=quant,
            )
        self.assertIsNone(create.call_args.args[4])
        self.assertTrue(quant.is_quanted())

    def test_full_recipe_quantizes_both_attention_geometries(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
        from rtp_llm.ops import HybridAttentionType

        original = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        cfg = SimpleNamespace(quant_config=original, _qwen4_full_fp8=True)
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        for kind in (HybridAttentionType.LINEAR, HybridAttentionType.NONE):
            layer.layer_type = kind
            self.assertEqual(
                layer._attention_quant_config(cfg, {}).get_method(), "FP8_PER_BLOCK"
            )
            self.assertIs(cfg.quant_config, original)

    def test_missing_optional_fp4_does_not_block_fp8_registration(self):
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.models_py.modules.factory.linear.impl import cuda

        real_import = builtins.__import__

        def import_without_fp4(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fp4_linear" and level == 1:
                raise ModuleNotFoundError("cutlass test sentinel")
            return real_import(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location(
            cuda.__name__ + ".registration_probe", Path(cuda.__file__)
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = cuda.__name__
        with (
            mock.patch.object(builtins, "__import__", import_without_fp4),
            mock.patch.object(LinearFactory, "register") as register,
            mock.patch("rtp_llm.models_py.utils.arch.is_cuda", return_value=True),
            mock.patch("rtp_llm.models_py.utils.arch.get_sm", return_value=(12, 0)),
        ):
            spec.loader.exec_module(module)
        names = [args[0].__name__ for args, _ in register.call_args_list]
        self.assertEqual(
            names, ["CudaF16Linear", "CudaFp8PerTensorLinear", "CudaFp8GEMMLinear"]
        )


if __name__ == "__main__":
    unittest.main()
