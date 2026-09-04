import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import _resolve_dsv4_moe_strategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_fp4 import (
    CudaGroupedFp4Strategy,
    CudaLocalLoopStrategy,
    CudaMegaMoeSEStrategy,
    CudaMegaMoeStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)


def _config(
    ep_size=1,
    n_shared_experts=0,
    strategy="auto",
    world_size=None,
    has_shared_expert_gate=False,
):
    return Fp8Fp4MoeRuntimeConfig(
        layer_id=0,
        hidden_size=512,
        moe_inter_dim=256,
        expert_num=16,
        moe_k=4,
        n_shared_experts=n_shared_experts,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        world_size=world_size,
        max_tokens_per_rank=128,
        moe_strategy=strategy,
        has_shared_expert_gate=has_shared_expert_gate,
    )


def _registry():
    registry = StrategyRegistry()
    registry.register(CudaMegaMoeSEStrategy())
    registry.register(CudaMegaMoeStrategy())
    registry.register(CudaGroupedFp4Strategy())
    registry.register(CudaLocalLoopStrategy())
    return registry


class Fp8Fp4StrategySelectionTest(unittest.TestCase):
    def test_dsv4_preserves_external_strategy_name(self):
        config = SimpleNamespace(moe_strategy=" external_test_strategy ")
        self.assertEqual(
            _resolve_dsv4_moe_strategy(config),
            "external_test_strategy",
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4._has_fp8_fp4_grouped_kernel",
        return_value=True,
    )
    def test_auto_prefers_grouped_on_single_rank(self, _):
        self.assertIsInstance(
            _registry().get_strategy(_config()), CudaGroupedFp4Strategy
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4._has_fp8_fp4_grouped_kernel",
        return_value=False,
    )
    def test_auto_falls_back_to_local_loop(self, _):
        self.assertIsInstance(
            _registry().get_strategy(_config()), CudaLocalLoopStrategy
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    def test_auto_uses_mega_without_shared_experts(self, _):
        self.assertIsInstance(
            _registry().get_strategy(_config(ep_size=2)), CudaMegaMoeStrategy
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=True,
    )
    def test_auto_prefers_mega_se_for_multiple_shared_experts(self, _, __):
        self.assertIsInstance(
            _registry().get_strategy(_config(ep_size=2, n_shared_experts=3)),
            CudaMegaMoeSEStrategy,
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=False,
    )
    def test_auto_falls_back_to_mega_when_se_is_unavailable(self, _, __):
        self.assertIsInstance(
            _registry().get_strategy(_config(ep_size=2, n_shared_experts=2)),
            CudaMegaMoeStrategy,
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=True,
    )
    def test_auto_keeps_gated_shared_expert_outside_mega_se(self, _, __):
        config = _config(
            ep_size=2,
            n_shared_experts=2,
            has_shared_expert_gate=True,
        )
        self.assertIsInstance(
            _registry().get_strategy(config),
            CudaMegaMoeStrategy,
        )

    def test_grouped_capability_requires_both_deep_gemm_symbols(self):
        available = SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=object(),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": available}), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

        for missing in (
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "get_mk_alignment_for_contiguous_layout",
        ):
            with self.subTest(missing=missing):
                unsupported = SimpleNamespace(
                    **{
                        name: getattr(available, name)
                        for name in vars(available)
                        if name != missing
                    }
                )
                with mock.patch.dict(
                    sys.modules, {"deep_gemm": unsupported}
                ), mock.patch.object(
                    torch.cuda, "is_available", return_value=True
                ), mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 0)
                ):
                    self.assertFalse(_has_fp8_fp4_grouped_kernel())

    def test_grouped_capability_requires_cuda_sm100(self):
        available = SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=object(),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": available}), mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())
        with mock.patch.dict(sys.modules, {"deep_gemm": available}), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=False,
    )
    def test_explicit_mega_se_rejects_unavailable_backend(self, _):
        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='mega_moe_se'"):
            _registry().get_strategy(
                _config(ep_size=2, n_shared_experts=2, strategy="mega_moe_se")
            )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=True,
    )
    def test_mega_strategies_reject_ep_group_different_from_world(self, _, __):
        config = _config(
            ep_size=2, n_shared_experts=2, strategy="mega_moe_se", world_size=4
        )
        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='mega_moe_se'"):
            _registry().get_strategy(config)
        config.moe_strategy = "mega_moe"
        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='mega_moe'"):
            _registry().get_strategy(config)


if __name__ == "__main__":
    unittest.main()
