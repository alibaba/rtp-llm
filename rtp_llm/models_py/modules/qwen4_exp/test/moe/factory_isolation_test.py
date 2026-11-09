import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FactoryIsolationTest(unittest.TestCase):
    def test_missing_or_uninspectable_histogram_keeps_torch_fallback(self):
        import triton.language as tl

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.decode_capabilities import (
            masked_histogram_available,
        )

        self.assertTrue(masked_histogram_available())
        with patch.object(tl, "histogram", None):
            self.assertFalse(masked_histogram_available())
        with patch("inspect.signature", side_effect=ValueError):
            self.assertFalse(masked_histogram_available())

    def test_legacy_fp8_mlp_keeps_separate_activation_and_projection(self):
        module = DenseMLP.__new__(DenseMLP)
        torch.nn.Module.__init__(module)
        module.is_gated = True
        module.clamp_swiglu = False
        module.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: 1)
        module.act_fn = FusedSiluAndMul()
        torch.manual_seed(1543)
        module.up_proj = Fp8Linear(
            torch.randn(1280, 2560, device="cuda").to(torch.float8_e4m3fn),
            torch.tensor([0.05], device="cuda"),
        )
        module.down_proj = Fp8Linear(
            torch.randn(2560, 640, device="cuda").to(torch.float8_e4m3fn),
            torch.tensor([0.05], device="cuda"),
        )
        for rows in (1, 4, 16, 32):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            expected = module.down_proj(module.act_fn(module.up_proj(x)))
            with patch.object(
                module.down_proj,
                "forward_silu_and_mul",
                side_effect=AssertionError("legacy FP8 model used Qwen4 fusion"),
            ):
                actual = module(x)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_qwen4_construction_preserves_generic_factory_and_weights(self):
        config = ModelConfig()
        config.hidden_size = 2560
        config.expert_num = 512
        config.moe_k = 10
        config.moe_inter_size = 80
        moe = MoeConfig()
        moe.moe_strategy = "no_quant_cpp"
        moe.use_all_gather = True
        weights = {
            W.moe_w1: torch.empty(512, 160, 2560, device="cuda", dtype=torch.bfloat16),
            W.moe_w2: torch.empty(512, 2560, 80, device="cuda", dtype=torch.bfloat16),
        }
        registry = FusedMoeFactory._registry
        for model_type in ("qwen3_next", "qwen4_exp", "qwen3_next"):
            with self.subTest(model_type=model_type):
                config.model_type = model_type
                adapter = MoEConfigAdapter(config, ParallelismConfig(), moe)
                factory = FusedMoeFactory()
                module = factory.create_fused_moe(adapter, weights)
                executor = module.fused_experts
                self.assertIs(type(executor), TritonFusedMoeExecutor)
                self.assertIs(FusedMoeFactory._registry, registry)
                self.assertIs(executor.w1, weights[W.moe_w1])
                self.assertIs(executor.w2, weights[W.moe_w2])
                self.assertEqual(executor._decode_alignment, model_type == "qwen4_exp")
                if model_type != "qwen4_exp":
                    self.assertFalse(executor._decode_tiles)
                    self.assertFalse(executor._small_moe_gemv)


if __name__ == "__main__":
    unittest.main()
