"""Smoke / sanity tests for ROCm FP8 fused-MoE executors.

PR #882 review (LLLLKKKK) flagged that ``RocmExpertsFp8PerBlock`` and the
refactored ``RocmExpertsFp8PerChannel`` had no end-to-end coverage. This file
runs each executor against a BF16 reference (computed from the dequantized
weights) and asserts shape, finiteness and approximate numerical agreement.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import SkipTest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernel_tuning.aiter import configure_aiter_fmoe_overlays

_AITER_TUNING_STATUS = configure_aiter_fmoe_overlays()

try:
    import aiter
    from aiter.ops.shuffle import shuffle_weight  # noqa: F401

    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.device.device_impl import RocmImpl
    from rtp_llm.models_py.kernel_tuning import ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV
    from rtp_llm.models_py.kernel_tuning.aiter import is_affected_aiter_fmoe_signature
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        ExpertForwardPayload,
        ExpertTokensMetadata,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
        FusedMoEQuantConfig,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
        get_rocm_fp8_dtype,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors import (
        deterministic_fp8_moe,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
        torch_moe_ref,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
        RocmExpertsFp8PerBlock,
        RocmExpertsFp8PerChannel,
        _aiter_fmoe_workload_signature,
        _moe_activation_type,
    )
    from rtp_llm.ops import MoeConfig, ParallelismConfig
    from rtp_llm.utils.model_weight import W

    _IMPORT_ERROR = None
except ImportError as exc:  # stale librtp_compute_ops.so / missing aiter / etc.
    _IMPORT_ERROR = exc

FP8_E4M3FNUZ_MAX = 240.0  # max representable in float8_e4m3fnuz


class DeterministicFp8MoeConfigTest(unittest.TestCase):
    def tearDown(self):
        if _IMPORT_ERROR is None:
            deterministic_fp8_moe._runtime_unsupported_reason.cache_clear()

    def _runtime_reason(
        self,
        *,
        gfx: str = "gfx942",
        cu_count: int = 80,
        version: str | None = None,
        stage1=None,
        stage2=None,
        implementation=None,
    ):
        if version is None:
            version = deterministic_fp8_moe.AITER_FMOE_SUPPORTED_VERSION
        if stage1 is None:
            stage1 = lambda: None
        if stage2 is None:
            stage2 = lambda: None
        if implementation is None:
            implementation = lambda: None
        fused_moe_module = SimpleNamespace(
            ck_moe_stage1=stage1,
            _fused_moe_impl=implementation,
        )
        deterministic_fp8_moe._runtime_unsupported_reason.cache_clear()
        with (
            patch.object(
                deterministic_fp8_moe.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(
                    gcnArchName=f"{gfx}:sramecc+:xnack-",
                    multi_processor_count=cu_count,
                ),
            ),
            patch.object(
                deterministic_fp8_moe.importlib.metadata,
                "version",
                return_value=version,
            ),
            patch.object(
                deterministic_fp8_moe.importlib,
                "import_module",
                return_value=fused_moe_module,
            ),
            patch.object(
                deterministic_fp8_moe.aiter,
                "ck_moe_stage2_fwd",
                stage2,
                create=True,
            ),
        ):
            return deterministic_fp8_moe._runtime_unsupported_reason("cuda:0")

    @unittest.skipIf(
        _IMPORT_ERROR is not None, f"ROCm imports unavailable: {_IMPORT_ERROR}"
    )
    def test_runtime_support_matrix_requires_target_device_version_and_symbols(self):
        self.assertIsNone(self._runtime_reason())
        cases = (
            ({"gfx": "gfx950"}, "GPU architecture"),
            ({"cu_count": 79}, "GPU CU count"),
            ({"version": "future-aiter"}, "AITER version"),
            ({"implementation": False}, "required AITER symbols"),
        )
        for overrides, expected_reason in cases:
            with self.subTest(overrides=overrides):
                self.assertIn(expected_reason, self._runtime_reason(**overrides))

    @unittest.skipIf(
        _IMPORT_ERROR is not None, f"ROCm imports unavailable: {_IMPORT_ERROR}"
    )
    def test_cuda_graph_does_not_block_opt_in_path(self):
        with patch.dict(
            os.environ,
            {
                ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV: "1",
                "ENABLE_CUDA_GRAPH": "1",
                "ENABLE_NATIVE_CUDA_GRAPH": "1",
            },
            clear=True,
        ), patch.object(
            deterministic_fp8_moe,
            "_unsupported_reason",
            return_value="test fallback",
        ):
            self.assertIsNone(
                deterministic_fp8_moe.try_deterministic_fp8_moe(
                    None, None, None, None, None, None, None, None, None
                )
            )


def _per_channel_quant_fp8(w: torch.Tensor, fp8_dtype: torch.dtype):
    """Quantize per output channel. ``w``: [E, OUT, IN]; scale: [E, OUT]."""
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_E4M3FNUZ_MAX).to(torch.float32)
    wq = (w / scale).to(fp8_dtype)
    return wq, scale.squeeze(-1)


def _online_loader_quant_fp8(w: torch.Tensor):
    from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
        per_channel_cast_to_fp8,
    )

    wq, scale = per_channel_cast_to_fp8(w)
    bits = wq.view(torch.int8)
    bits[bits == -128] = 0
    return bits.view(torch.float8_e4m3fnuz), scale * 2.0


def _per_block_quant_fp8(w: torch.Tensor, fp8_dtype: torch.dtype, block: int = 128):
    """Quantize per [block, block] block. ``w``: [E, OUT, IN]; scale: [E, OUT/b, IN/b]."""
    E, OUT, IN = w.shape
    assert OUT % block == 0 and IN % block == 0, "OUT/IN must be divisible by block"
    w_blk = w.reshape(E, OUT // block, block, IN // block, block)
    amax = w_blk.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_E4M3FNUZ_MAX).to(torch.float32)
    wq_blk = (w_blk / scale).to(fp8_dtype)
    wq = wq_blk.reshape(E, OUT, IN)
    return wq, scale.squeeze(2).squeeze(-1)


def _dequant_per_channel(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (wq.to(torch.float32) * scale.unsqueeze(-1)).to(torch.bfloat16)


def _dequant_per_block(
    wq: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    E, OUT, IN = wq.shape
    wq_blk = wq.to(torch.float32).reshape(E, OUT // block, block, IN // block, block)
    deq = wq_blk * scale.unsqueeze(2).unsqueeze(-1)
    return deq.reshape(E, OUT, IN).to(torch.bfloat16)


def _make_parallelism_config(tp_size: int = 1):
    p = ParallelismConfig()
    p.ep_size = 1
    p.ep_rank = 0
    p.tp_size = tp_size
    p.tp_rank = 0
    p.dp_size = 1
    p.dp_rank = 0
    p.world_size = tp_size
    p.world_rank = 0
    p.local_rank = 0
    p.local_world_size = tp_size
    return p


def _make_config_adapter(
    expert_num: int,
    top_k: int,
    inter_dim: int,
    tp_size: int = 1,
    data_type: str | None = None,
    activation_type: str = "silu",
):
    model_config = ModelConfig()
    model_config.attn_config.head_num = 4
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 32000
    model_config.expert_num = expert_num
    model_config.moe_k = top_k
    model_config.inter_size = inter_dim
    model_config.activation_type = activation_type
    if data_type is not None:
        model_config.data_type = data_type

    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=_make_parallelism_config(tp_size),
        moe_config=MoeConfig(),
    )


class _Fp8MoeBaseTest(unittest.TestCase):
    """Shared scaffolding for FP8 MoE executor tests."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            raise SkipTest(
                f"ROCm fused_moe deps unavailable (likely stale librtp_compute_ops.so): {_IMPORT_ERROR}"
            )
        if not torch.cuda.is_available():
            raise SkipTest("CUDA/HIP not available")
        self.device = "cuda"
        torch.set_default_device(self.device)
        torch.manual_seed(42)
        self.fp8_dtype = get_rocm_fp8_dtype()
        self._feature_env = patch.dict(
            os.environ,
            {ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV: "0"},
        )
        self._feature_env.start()
        self.addCleanup(self._feature_env.stop)

    def _build_payload(self, M, K, E, top_k):
        hidden_states = (
            torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.05
        )
        topk_ids = torch.topk(
            torch.rand(M, E, device=self.device), top_k, dim=1
        ).indices.to(torch.int32)
        topk_weights = torch.softmax(
            torch.randn(M, top_k, device=self.device), dim=-1
        ).to(torch.float32)
        return ExpertForwardPayload(
            expert_x=hidden_states,
            expert_x_origin_dtype=hidden_states.dtype,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(None, None, None),
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )


class RocmExpertsFp8PerChannelTest(_Fp8MoeBaseTest):
    """Cover the refactored single-aiter-fused_moe path (review item #7)."""

    M, K, N, E, TOP_K = 16, 256, 256, 4, 2

    def test_activation_type_normalization(self):
        swiglu_config = ModelConfig()
        swiglu_config.activation_type = "SiGLU"
        silu_config = ModelConfig()
        silu_config.activation_type = "silu"

        silu_aliases = (
            "silu",
            "SiGLU",
            "swiglu",
            "gated-silu",
            silu_config.activation_type,
            swiglu_config.activation_type,
        )
        with patch.dict(
            os.environ,
            {ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV: "1"},
        ):
            for activation in silu_aliases:
                with self.subTest(activation=activation):
                    self.assertEqual(
                        _moe_activation_type(activation), aiter.ActivationType.Silu
                    )

            self.assertEqual(_moe_activation_type("gelu"), aiter.ActivationType.Gelu)
            with self.assertRaisesRegex(
                ValueError, "Unsupported AITER FMoE activation"
            ):
                _moe_activation_type("unknown")

    def test_feature_disabled_keeps_legacy_activation_mapping(self):
        self.assertEqual(_moe_activation_type("swiglu"), aiter.ActivationType.Gelu)

    def _run(self, apply_router_weight_on_input: bool):
        payload = self._build_payload(self.M, self.K, self.E, self.TOP_K)
        # apply_router_weight_on_input requires top_k == 1.
        if apply_router_weight_on_input:
            payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
            payload.expert_topk_weights = payload.expert_topk_weights[:, :1]

        # Random BF16 reference weights, then quantize per-channel.
        w1_ref = (
            torch.randn(
                self.E, 2 * self.N, self.K, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )
        w2_ref = (
            torch.randn(
                self.E, self.K, self.N, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )

        w1q, s1 = _per_channel_quant_fp8(w1_ref, self.fp8_dtype)
        w2q, s2 = _per_channel_quant_fp8(w2_ref, self.fp8_dtype)

        # Reference: torch_moe_ref against the dequantized weights so both
        # paths see the same numerical content (modulo fp8 rounding).
        w1_deq = _dequant_per_channel(w1q, s1)
        w2_deq = _dequant_per_channel(w2q, s2)
        ref_out = torch_moe_ref(
            payload=payload,
            activation="silu",
            global_num_experts=self.E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
            w1=w1_deq,
            w2=w2_deq,
        )

        # Build the executor.
        config_adapter = _make_config_adapter(self.E, self.TOP_K, 2 * self.N)
        w1q_runtime = shuffle_weight(w1q, layout=(16, 16))
        w2q_runtime = shuffle_weight(w2q, layout=(16, 16))
        weights = {
            W.moe_w1: w1q_runtime,
            W.moe_w2: w2q_runtime,
            W.moe_s1: s1,
            W.moe_s2: s2,
        }
        executor = RocmExpertsFp8PerChannel(
            config_adapter, FusedMoEQuantConfig(), weights
        )
        self.assertTrue(executor.w1.is_shuffled)
        self.assertTrue(executor.w2.is_shuffled)

        out = executor.execute(
            payload=payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
        ).fused_expert_output

        self.assertEqual(out.shape, (self.M, self.K))
        self.assertTrue(torch.isfinite(out).all().item(), "kernel produced non-finite")
        # FP8 + bf16 accumulation: loose tolerance, mainly a sanity bound.
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)

    def test_basic_forward(self):
        self._run(apply_router_weight_on_input=False)

    def test_apply_router_weight_on_input(self):
        # PR #882 review #7 specifically called out this newly-enabled path.
        self._run(apply_router_weight_on_input=True)

    @patch.dict(
        os.environ,
        {ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV: "1"},
    )
    def test_small_token_tuned_shape_with_loader_postprocess(self):
        self.assertTrue(_AITER_TUNING_STATUS.applied, _AITER_TUNING_STATUS)
        hidden, local_inter, experts, top_k = 2048, 128, 256, 8
        runtime_device = RocmImpl.__new__(RocmImpl)

        # Reproduce the FP8 loader conversion, [gate, up] reorder, and the
        # physical ROCm weight shuffle used by a real checkpoint load.
        w1_checkpoint = (
            torch.randn(
                experts,
                2 * local_inter,
                hidden,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.02
        )
        w2_checkpoint = (
            torch.randn(
                experts,
                hidden,
                local_inter,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.02
        )
        w1q_checkpoint, s1_checkpoint = _online_loader_quant_fp8(w1_checkpoint)
        w2q, s2 = _online_loader_quant_fp8(w2_checkpoint)
        w1q_loader = runtime_device.cat_0(
            [
                w1q_checkpoint[:, local_inter:, :],
                w1q_checkpoint[:, :local_inter, :],
            ],
            dim=1,
        )
        s1_loader = torch.cat(
            [
                s1_checkpoint[:, local_inter:, :],
                s1_checkpoint[:, :local_inter, :],
            ],
            dim=1,
        )
        weights = {
            W.moe_w1: runtime_device.shuffle_moe_weight(
                w1q_loader, w1q_loader.dtype, W.moe_w1
            ),
            W.moe_w2: runtime_device.shuffle_moe_weight(w2q, w2q.dtype, W.moe_w2),
            W.moe_s1: runtime_device.shuffle_moe_weight(
                s1_loader, s1_loader.dtype, W.moe_s1
            ),
            W.moe_s2: s2,
        }
        w1_dequant = (w1q_checkpoint.float() * s1_checkpoint).to(torch.bfloat16)
        w2_dequant = (w2q.float() * s2).to(torch.bfloat16)

        # TP is deliberately varied while the local AITER dispatch shape stays
        # fixed. The tuning decision must therefore be identical for both.
        for tp_size in (2, 4):
            config = _make_config_adapter(
                experts,
                top_k,
                local_inter * tp_size,
                tp_size=tp_size,
                data_type="bf16",
                activation_type="SiGLU",
            )
            signature = _aiter_fmoe_workload_signature(
                weights[W.moe_w1],
                weights[W.moe_w2],
                config.moe_k,
                config.activation_type,
                config.model_config.compute_dtype,
            )
            self.assertTrue(is_affected_aiter_fmoe_signature(signature))

            executor = RocmExpertsFp8PerChannel(
                config,
                FusedMoEQuantConfig(),
                weights,
            )
            for token_count in (1, 2, 4, 8, 16):
                with self.subTest(tp_size=tp_size, token_count=token_count):
                    payload = self._build_payload(token_count, hidden, experts, top_k)
                    reference = torch_moe_ref(
                        payload=payload,
                        activation="silu",
                        global_num_experts=experts,
                        expert_map=None,
                        a2_scale=None,
                        apply_router_weight_on_input=False,
                        extra_expert_args=None,
                        w1=w1_dequant,
                        w2=w2_dequant,
                    )
                    output = executor.execute(
                        payload=payload,
                        activation="SiGLU",
                        expert_map=None,
                        a2_scale=None,
                        apply_router_weight_on_input=False,
                        extra_expert_args=None,
                    ).fused_expert_output
                    cosine = torch.nn.functional.cosine_similarity(
                        output.float(), reference.float(), dim=-1
                    ).mean()

                    self.assertEqual(output.shape, (token_count, hidden))
                    self.assertTrue(torch.isfinite(output).all().item())
                    self.assertGreater(cosine.item(), 0.99)


class RocmExpertsFp8PerBlockTest(_Fp8MoeBaseTest):
    """Smoke / sanity for the brand-new PerBlock executor (review item #4)."""

    # Sizes must be divisible by 128 for per-128x128 quant.
    M, K, N, E, TOP_K = 16, 256, 256, 4, 2

    def _run(self, apply_router_weight_on_input: bool):
        payload = self._build_payload(self.M, self.K, self.E, self.TOP_K)
        if apply_router_weight_on_input:
            payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
            payload.expert_topk_weights = payload.expert_topk_weights[:, :1]

        w1_ref = (
            torch.randn(
                self.E, 2 * self.N, self.K, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )
        w2_ref = (
            torch.randn(
                self.E, self.K, self.N, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )

        w1q, s1 = _per_block_quant_fp8(w1_ref, self.fp8_dtype)
        w2q, s2 = _per_block_quant_fp8(w2_ref, self.fp8_dtype)

        w1_deq = _dequant_per_block(w1q, s1)
        w2_deq = _dequant_per_block(w2q, s2)
        ref_out = torch_moe_ref(
            payload=payload,
            activation="silu",
            global_num_experts=self.E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
            w1=w1_deq,
            w2=w2_deq,
        )

        config_adapter = _make_config_adapter(self.E, self.TOP_K, 2 * self.N)
        weights = {
            W.moe_w1: w1q,
            W.moe_w2: w2q,
            W.moe_s1: s1,
            W.moe_s2: s2,
        }
        executor = RocmExpertsFp8PerBlock(
            config_adapter, FusedMoEQuantConfig(), weights
        )

        out = executor.execute(
            payload=payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
        ).fused_expert_output

        self.assertEqual(out.shape, (self.M, self.K))
        self.assertTrue(torch.isfinite(out).all().item(), "kernel produced non-finite")
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)

    def test_basic_forward(self):
        self._run(apply_router_weight_on_input=False)

    def test_apply_router_weight_on_input(self):
        self._run(apply_router_weight_on_input=True)


if __name__ == "__main__":
    unittest.main()
