"""Prepare-once host work tests for the HC/GDN/MoE sections (r2 follow-up).

Covers the low/medium-risk hoists from the hc/gdn/moe per-layer audit:
  * process-cached device capability (no per-layer cudaDeviceGetAttribute)
  * capability-first short-circuit for the SM120-only guard chains (the
    verdict is unchanged: those chains decline on SM90 either way)
  * GenericMoeLayer freezing the group-topk routing helpers at __init__
    instead of rebuilding them every layer
  * the DeepGemmHybridExecutor eager expert-count read over a persistent
    pinned buffer (bit-identical counts)
  * the GDN prefill initial-states buffer reused across a step's layers
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.utils.device_capability import cached_device_capability


class CachedCapabilityTest(unittest.TestCase):
    def test_cpu_devices_decline_without_raising(self):
        self.assertIsNone(cached_device_capability(torch.device("cpu")))
        self.assertIsNone(cached_device_capability("cpu"))
        # The cache serves repeat queries for the same device argument.
        self.assertIsNone(cached_device_capability(torch.device("cpu")))
        info = cached_device_capability.cache_info()
        self.assertGreaterEqual(info.hits, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_capability_matches_torch(self):
        device = torch.device("cuda", torch.cuda.current_device())
        expected = torch.cuda.get_device_capability(device)
        self.assertEqual(cached_device_capability(device), expected)
        self.assertEqual(cached_device_capability(device), expected)
        self.assertGreaterEqual(cached_device_capability.cache_info().hits, 1)


class Sm120OnlyGuardChainTest(unittest.TestCase):
    """The (12, 0)-only chains keep their verdict with the capability first."""

    def _splitk_tensors(self, device):
        normed = torch.zeros(8, 10240, device=device, dtype=torch.bfloat16)
        weight = torch.zeros(4, 10240, device=device, dtype=torch.bfloat16)
        down = torch.zeros(8, 320, device=device, dtype=torch.bfloat16)
        return normed, weight, down

    def _up_mix_tensors(self, device):
        x = torch.zeros(8, 320, device=device, dtype=torch.bfloat16)
        weight = torch.zeros(10240, 320, device=device, dtype=torch.float8_e4m3fn)
        scale = torch.ones((), device=device, dtype=torch.float32)
        normed = torch.zeros(8, 10240, device=device, dtype=torch.bfloat16)
        return x, weight, scale, normed

    def test_chains_decline_on_cpu_tensors(self):
        from rtp_llm.models_py.triton_kernels.common.gated_residual_medium_up_mix_triton import (
            try_fp8_hc_medium_up_mix,
        )
        from rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_up_mix_triton import (
            try_fp8_prefill_up_mix,
        )
        from rtp_llm.models_py.triton_kernels.common.gated_residual_splitk_inject import (
            try_splitk_down_inject,
        )

        normed, weight, down = self._splitk_tensors("cpu")
        self.assertIsNone(try_splitk_down_inject(normed, weight, down))
        x, uweight, scale, unormed = self._up_mix_tensors("cpu")
        self.assertIsNone(try_fp8_hc_medium_up_mix(x, uweight, scale, unormed, 4))
        self.assertIsNone(try_fp8_prefill_up_mix(x, uweight, scale, unormed, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_chains_decline_on_non_sm120_devices(self):
        from rtp_llm.models_py.triton_kernels.common.gated_residual_medium_up_mix_triton import (
            try_fp8_hc_medium_up_mix,
        )
        from rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_up_mix_triton import (
            try_fp8_prefill_up_mix,
        )
        from rtp_llm.models_py.triton_kernels.common.gated_residual_splitk_inject import (
            try_splitk_down_inject,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
            Qwen4PrefillGatedNorm,
        )

        if cached_device_capability(torch.cuda.current_device()) == (12, 0):
            self.skipTest("SM120 device keeps these chains live")

        device = torch.device("cuda", torch.cuda.current_device())
        normed, weight, down = self._splitk_tensors(device)
        self.assertIsNone(try_splitk_down_inject(normed, weight, down))
        x, uweight, scale, unormed = self._up_mix_tensors(device)
        self.assertIsNone(try_fp8_hc_medium_up_mix(x, uweight, scale, unormed, 4))
        self.assertIsNone(try_fp8_prefill_up_mix(x, uweight, scale, unormed, 4))

        # The GDN prefill norm keeps its 24-condition verdict with the
        # capability leading the conjunction.
        module = Qwen4PrefillGatedNorm.__new__(Qwen4PrefillGatedNorm)
        torch.nn.Module.__init__(module)
        module.role = "PREFILL"
        module.native = SimpleNamespace(
            activation="sigmoid",
            group_size=128,
            weight=torch.ones(128, device=device, dtype=torch.bfloat16),
            bias=None,
            eps=1e-6,
        )
        rows = 4096
        x = torch.ones(rows, 3072, device=device, dtype=torch.bfloat16)
        gate = torch.ones_like(x)
        self.assertFalse(module._supports_prefill(x, gate))
        self.assertEqual(module.role, "PREFILL")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ExpertCountsPinnedReadTest(unittest.TestCase):
    def test_counts_match_plain_cpu_read_and_buffer_is_reused(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )

        executor = DeepGemmHybridExecutor.__new__(DeepGemmHybridExecutor)
        executor._expert_counts_pinned = None
        device = torch.device("cuda", torch.cuda.current_device())
        for counts in ([3, 130, 0, 7], [0, 0, 0, 0], [255, 1, 64, 128]):
            source = torch.tensor(counts, dtype=torch.int32, device=device)
            self.assertEqual(
                executor._read_expert_counts_host(source), counts
            )
            # Bit-identical to the legacy read it replaces.
            self.assertEqual(source.cpu().tolist(), counts)
            self.assertTrue(executor._expert_counts_pinned.is_pinned())
        first = executor._expert_counts_pinned
        executor._read_expert_counts_host(
            torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
        )
        self.assertIs(executor._expert_counts_pinned, first)
        # A different expert count reallocates.
        executor._read_expert_counts_host(
            torch.tensor([1, 2], dtype=torch.int32, device=device)
        )
        self.assertIsNot(executor._expert_counts_pinned, first)


class GroupTopKFrozenAtInitTest(unittest.TestCase):
    def _make_layer(self):
        from rtp_llm.models_py.model_desc import generic_moe
        from rtp_llm.utils.model_weight import W

        config = SimpleNamespace(
            hidden_size=8,
            inter_size=16,
            expert_num=4,
            moe_k=2,
            quant_config=None,
            activation_type="SiGLU",
            moe_style=2,
            has_moe_norm=True,
            moe_n_group=2,
            moe_topk_group=1,
            routed_scaling_factor=2.5,
            eplb_config=SimpleNamespace(phy_exp_num=lambda count: count),
        )
        parallelism_config = SimpleNamespace(
            ep_size=1,
            dp_rank=0,
            dp_size=1,
            get_ffn_tp_size=lambda: 1,
            get_attn_tp_size=lambda: 1,
        )
        moe_config = SimpleNamespace(fake_balance_expert=False)
        weights = {
            W.moe_w1: torch.empty(4, 2, 8),
            W.moe_w2: torch.empty(4, 8, 2),
            W.e_score_correction_b: torch.zeros(4),
        }
        fused_moe = SimpleNamespace(
            includes_shared_expert=False,
            topk_ids_dtype=torch.int32,
            router=SimpleNamespace(
                supports_skip_tp_allreduce=True,
                tp_collective_size=1,
            ),
        )
        with (
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.LinearFactory.create_linear_from_weights",
                return_value=Mock(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.SelectTopk",
                return_value=Mock(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.DenseMLP",
                return_value=Mock(),
            ),
            patch("rtp_llm.models_py.model_desc.generic_moe.MoEConfigAdapter"),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe.FusedMoeFactory"
            ) as fused_moe_factory,
        ):
            fused_moe_factory.return_value.create_fused_moe.return_value = fused_moe
            return generic_moe.GenericMoeLayer(
                config, parallelism_config, weights, moe_config
            )

    def test_routing_helpers_frozen_at_init(self):
        from rtp_llm.models_py.modules import GroupTopK

        layer = self._make_layer()
        self.assertIsNotNone(layer.correction_bias)
        # The model-constant routing helpers exist after __init__ and the
        # forward no longer rebuilds them per layer.
        self.assertIsInstance(layer.group_topk, GroupTopK)
        self.assertTrue(layer.renormalize)
        self.assertEqual(layer.num_expert_group, 2)
        self.assertEqual(layer.topk_group, 1)
        self.assertEqual(layer.n_routed_experts, 4)
        self.assertEqual(layer.routed_scaling_factor, 2.5)


class GdnInitialStatesBufferTest(unittest.TestCase):
    def _module(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
        )

        module = Qwen3NextGatedDeltaNetPrefill.__new__(
            Qwen3NextGatedDeltaNetPrefill
        )
        torch.nn.Module.__init__(module)
        module.local_num_v_heads = 4
        module.head_v_dim = 128
        module.head_k_dim = 128
        module.ssm_state_dtype = torch.float32
        module._ssm_initial_states_cache = None
        return module

    def test_buffer_reused_within_a_step_and_reallocated_on_shape_change(self):
        module = self._module()
        first = module._initial_states_buffer(2, torch.device("cpu"))
        self.assertEqual(tuple(first.shape), (2, 4, 128, 128))
        again = module._initial_states_buffer(2, torch.device("cpu"))
        self.assertIs(again, first)
        bigger = module._initial_states_buffer(3, torch.device("cpu"))
        self.assertIsNot(bigger, first)
        self.assertEqual(tuple(bigger.shape), (3, 4, 128, 128))


if __name__ == "__main__":
    unittest.main()
