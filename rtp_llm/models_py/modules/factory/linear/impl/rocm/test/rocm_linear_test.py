import itertools
import os
from types import SimpleNamespace
from typing import Optional
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

import torch
from torch import dtype as _dtype
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeLayer,
)
from rtp_llm.models_py.model_desc.qwen2_mtp import Qwen2MtpModel
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.rocm.f16_linear import (
    RocmF16LinearNoSwizzle,
    RocmF16LinearWithSwizzle,
)
from rtp_llm.ops import ActivationType, HWKernelConfig
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.swizzle_utils import swizzle_tensor


class LinearTorch(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class FakeFusedMoe(nn.Module):
    topk_ids_dtype = torch.int32

    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states)


class FakeSelectTopk(nn.Module):
    def forward(self, router_logits, topk_ids, topk_weights):
        topk_ids.zero_()
        topk_weights.zero_()


class FakeSharedExpert(nn.Module):
    def forward(self, hidden_states, skip_allreduce=False):
        return hidden_states * 2


class FakeSigmoidGateScaleAdd(nn.Module):
    def forward(self, gate_output, shared_expert_output, experts_output):
        experts_output.add_(torch.sigmoid(gate_output) * shared_expert_output)


class LinearTest(TestCase):

    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    # (k, n) pairs: k is input hidden size, n is output size
    K_N_PAIRS = [
        (512, 512),
        (512, 256),
        (512, 1024),
        (768, 768),
        (768, 384),
        (768, 1536),
        (1024, 1024),
        (1024, 512),
        (1024, 2048),
        (2048, 2048),
        (2048, 1024),
        (2048, 4096),
        (4096, 4096),
        (4096, 2048),
        (4096, 8192),
        (8192, 8192),
        (8192, 4096),
        (1280, 3840),  # qkv
        (1280, 1280),  # proj
        (5120, 5120),
        (2048, 5120),
    ]
    HAS_BIAS = [True, False]
    HAS_SWIZZLE = [True, False]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_linear_test(
        self,
        num_tokens: int,
        k: int,
        n: int,
        dtype: _dtype,
        has_bias: bool,
        has_swizzle: bool,
    ):
        torch.manual_seed(0)
        w = torch.randn(k, n, dtype=dtype)
        torch.nn.init.xavier_uniform_(w)
        if has_bias:
            bias = torch.empty(n, dtype=dtype)
            torch.nn.init.normal_(bias, mean=0.0, std=0.01)
        else:
            bias = None

        x = torch.randn(num_tokens, k, dtype=dtype)

        linear_torch = LinearTorch(w, bias)
        torch_output = linear_torch(x)
        hw_kernel_config = HWKernelConfig()
        if has_swizzle:
            # Follow aiter's approach: transpose to (n, k), shuffle, then transpose back to (k, n)
            # This matches the format expected by hipb_mm with bpreshuffle=True
            w_swizzled = swizzle_tensor(w.t(), False, MiM=16).t()  # (n, k) swizzled
            w_dict = {"weight": w_swizzled, "bias": bias}
            hw_kernel_config.use_swizzleA = True
        else:
            w_dict = {"weight": w, "bias": bias}

        linear = LinearFactory.create_linear_from_weights(
            w_dict, "weight", None, "bias", None, hw_kernel_config
        )
        my_output = linear(x)
        self.assertTrue(torch.allclose(torch_output, my_output, atol=1e-2, rtol=1e-2))

    @staticmethod
    def _swizzle_weight(weight: torch.Tensor) -> torch.Tensor:
        return swizzle_tensor(weight.t(), False, MiM=16).t()

    def _assert_swizzled_linear_matches_reference(
        self,
        linear: nn.Module,
        weight: torch.Tensor,
        input: torch.Tensor,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> None:
        self.assertIsInstance(linear, RocmF16LinearWithSwizzle)
        actual = linear(input)
        expected = input @ weight
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol))

    def test_linear(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.K_N_PAIRS,
            self.DTYPES,
            self.HAS_BIAS,
            self.HAS_SWIZZLE,
        ):
            num_tokens, (k, n), dtype, has_bias, has_swizzle = params
            with self.subTest(
                num_tokens=num_tokens,
                k=k,
                n=n,
                dtype=dtype,
                has_bias=has_bias,
                has_swizzle=has_swizzle,
            ):
                self._run_linear_test(num_tokens, k, n, dtype, has_bias, has_swizzle)

    def _run_shared_expert_gate_test(
        self, dtype: _dtype, atol: float, rtol: float
    ) -> None:
        torch.manual_seed(0)
        hidden_size = 32
        num_experts = 16
        num_tokens = 7

        moe_gate_weight = torch.randn(hidden_size, num_experts, dtype=dtype)
        swizzled_moe_gate_weight = swizzle_tensor(
            moe_gate_weight.t(), False, MiM=16
        ).t()
        shared_expert_gate_weight = torch.randn(hidden_size, 1, dtype=dtype)
        weights = {
            W.moe_gate: swizzled_moe_gate_weight,
            W.moe_w1: torch.empty(num_experts, 1, 1, dtype=dtype),
            W.moe_w2: torch.empty(num_experts, 1, 1, dtype=dtype),
            W.shared_expert_gate: shared_expert_gate_weight,
        }

        config = SimpleNamespace(
            hidden_size=hidden_size,
            inter_size=hidden_size * 2,
            expert_num=num_experts,
            moe_k=2,
            quant_config=None,
            activation_type="SiGLU",
            moe_style=2,
            eplb_config=SimpleNamespace(phy_exp_num=lambda count: count),
        )
        parallelism_config = SimpleNamespace(
            ep_size=1,
            dp_rank=0,
            dp_size=1,
            get_ffn_tp_size=lambda: 1,
        )
        moe_config = SimpleNamespace(fake_balance_expert=False)
        hw_kernel_config = HWKernelConfig()
        hw_kernel_config.use_swizzleA = True

        with (
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.SelectTopk",
                return_value=FakeSelectTopk(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.DenseMLP",
                return_value=FakeSharedExpert(),
            ),
            patch("rtp_llm.models_py.model_desc.generic_moe.MoEConfigAdapter"),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.FusedMoeFactory"
            ) as fused_moe_factory,
        ):
            fused_moe_factory.return_value.create_fused_moe.return_value = (
                FakeFusedMoe()
            )
            layer = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                hw_kernel_config=hw_kernel_config,
            )

        self.assertIsInstance(layer.gate, RocmF16LinearWithSwizzle)
        self.assertIsInstance(layer.shared_expert_gate, RocmF16LinearNoSwizzle)

        layer.sigmoid_gate_scale_add = FakeSigmoidGateScaleAdd()
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
        actual_gate = layer.shared_expert_gate(hidden_states)
        expected_gate = hidden_states @ shared_expert_gate_weight
        self.assertTrue(
            torch.allclose(actual_gate, expected_gate, atol=atol, rtol=rtol)
        )

        actual = layer(hidden_states)
        expected = torch.sigmoid(expected_gate) * (hidden_states * 2)
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol))

    def test_shared_expert_gate_falls_back_to_no_swizzle(self):
        dtype_tolerances = (
            (torch.float16, 1e-2, 1e-2),
            (torch.bfloat16, 2e-2, 2e-2),
        )
        for dtype, atol, rtol in dtype_tolerances:
            with self.subTest(dtype=dtype):
                self._run_shared_expert_gate_test(dtype, atol, rtol)

    def test_generic_dense_mlp_uses_swizzle(self):
        torch.manual_seed(0)
        hidden_size = 32
        intermediate_size = 32
        up_weight = torch.randn(hidden_size, intermediate_size * 2, dtype=torch.float16)
        down_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.float16)
        weights = {
            W.ffn_w13: self._swizzle_weight(up_weight),
            W.ffn_w2: self._swizzle_weight(down_weight),
            W.pre_ln_gamma: torch.ones(hidden_size, dtype=torch.float16),
            W.post_ln_gamma: torch.ones(hidden_size, dtype=torch.float16),
        }
        config = SimpleNamespace(
            quant_config=None,
            attn_config=SimpleNamespace(use_mla=False),
            getAttentionConfigs=lambda tp_size: SimpleNamespace(),
            moe_layer_index=[],
            activation_type=ActivationType.Swiglu,
            layernorm_eps=1e-6,
        )
        parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 1,
            get_ffn_tp_size=lambda: 1,
        )
        hw_kernel_config = HWKernelConfig()
        hw_kernel_config.use_swizzleA = True

        with (
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.CausalAttention",
                return_value=nn.Identity(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.RMSResNorm",
                return_value=nn.Identity(),
            ),
        ):
            decoder_layer = GenericMoeDecoderLayer(
                config,
                parallelism_config,
                weights,
                global_weights={},
                layer_idx=0,
                moe_config=SimpleNamespace(),
                hw_kernel_config=hw_kernel_config,
            )

        hidden_states = torch.randn(7, hidden_size, dtype=torch.float16)
        intermediate_states = torch.randn(7, intermediate_size, dtype=torch.float16)
        self._assert_swizzled_linear_matches_reference(
            decoder_layer.mlp.up_proj, up_weight, hidden_states
        )
        self._assert_swizzled_linear_matches_reference(
            decoder_layer.mlp.down_proj, down_weight, intermediate_states
        )

    def test_qwen2_mtp_eh_proj_uses_swizzle(self):
        torch.manual_seed(0)
        hidden_size = 32
        eh_proj_weight = torch.randn(hidden_size * 2, hidden_size, dtype=torch.float16)
        layer_weights = {
            W.multi_tokens_predict_eh_proj: self._swizzle_weight(eh_proj_weight),
            W.multi_tokens_predict_enorm: torch.ones(hidden_size, dtype=torch.float16),
            W.multi_tokens_predict_hnorm: torch.ones(hidden_size, dtype=torch.float16),
            W.multi_tokens_predict_final_ln_gamma: torch.ones(
                hidden_size, dtype=torch.float16
            ),
        }
        weights = SimpleNamespace(
            weights=[layer_weights],
            get_global_weight=lambda name: torch.empty(1, dtype=torch.float16),
        )
        config = SimpleNamespace(
            num_layers=1,
            vocab_size=32,
            layernorm_eps=1e-6,
        )
        parallelism_config = SimpleNamespace()
        hw_kernel_config = HWKernelConfig()
        hw_kernel_config.use_swizzleA = True

        with (
            patch(
                "rtp_llm.models_py.model_desc.qwen2_mtp.Embedding",
                return_value=nn.Identity(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen2_mtp.RMSNorm",
                return_value=nn.Identity(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen2_mtp.Qwen3DecoderLayer",
                return_value=nn.Identity(),
            ),
        ):
            model = Qwen2MtpModel(
                config,
                parallelism_config,
                weights,
                max_generate_batch_size=1,
                py_hw_kernel_config=hw_kernel_config,
            )

        input = torch.randn(7, hidden_size * 2, dtype=torch.float16)
        self._assert_swizzled_linear_matches_reference(
            model.eh_proj, eh_proj_weight, input
        )


if __name__ == "__main__":
    main()
