import os
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeLayer,
)
from rtp_llm.models_py.modules.hybrid import hy4_cmp as bridge
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention


class _Callable(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _config(model_type: str = "hy_v4") -> SimpleNamespace:
    return SimpleNamespace(model_type=model_type)


def _parallelism(tp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(tp_size=tp_size, get_attn_tp_size=lambda: tp_size)


def _attention(*, indexer=None) -> SimpleNamespace:
    return SimpleNamespace(
        q_lora_rank=2048,
        gate_proj=Mock(),
        gating_type="elementwise",
        indexer=indexer,
        reuse_topk_indices=False,
    )


class Hy4SmallTHeadGateTest(unittest.TestCase):
    def _make_indexer(self, *, enabled=True, hadamard=False, device="cpu", heads=32):
        namespace = Indexer.__init__.__globals__
        keys = namespace["W"]
        # Match the transposed FP32 Linear weight layout used by HY4.
        weight = torch.randn(6144, heads, device=device).t() * 0.02
        if hadamard:
            # nn.Linear registers a Parameter, unlike the production factory's
            # plain weight attribute. Its legacy path needs no layout change.
            weight = weight.contiguous()
        linear = nn.Linear(6144, heads, bias=False, device=device)
        linear.weight = nn.Parameter(weight, requires_grad=False)
        original = linear.weight
        factory = SimpleNamespace(
            create_linear_from_weights=Mock(
                side_effect=[nn.Identity(), nn.Identity(), linear]
            )
        )
        config = SimpleNamespace(
            indexer_head_num=heads,
            indexer_head_dim=128,
            indexer_topk=2048,
            rope_head_dim=64,
            kernel_tokens_per_block=64,
            rope_config=SimpleNamespace(indexer_is_neox_style=False),
        )
        with patch(
            "rtp_llm.models_py.utils.fuse_config.fuse_kernels_enabled",
            return_value=enabled,
        ), patch.dict(
            namespace,
            LinearFactory=factory,
            LayerNorm=Mock(return_value=nn.Identity()),
            IndexerOp=Mock(),
        ):
            obj = Indexer(
                config,
                {keys.mla_indexer_k_norm_w: None, keys.mla_indexer_k_norm_b: None},
                {keys.rope_cos_sin_cache: None},
                0,
                1e-5,
                None,
                use_hadamard=hadamard,
            )
        return obj, original

    def test_init_preserves_large_gemm_weight_and_checkpoint(self):
        obj, original = self._make_indexer()
        self.assertIs(obj.weights_proj.weight, original)
        self.assertFalse(original.is_contiguous())
        small = obj._hy4_small_t_head_gate_weight
        self.assertTrue(small.is_contiguous())
        self.assertNotEqual(small.data_ptr(), original.data_ptr())
        torch.testing.assert_close(small, original, rtol=0, atol=0)
        self.assertNotIn("_hy4_small_t_head_gate_weight", obj.state_dict())

    def test_dispatch_small_rows_and_large_boundary(self):
        obj, original = self._make_indexer()
        namespace = Indexer._get_logits_head_gate.__globals__
        for rows in (0, 1, 8, 32, 33, 256):
            for ndim in (2, 3):
                with self.subTest(rows=rows, ndim=ndim):
                    x = torch.randn(rows, 6144, dtype=torch.bfloat16)
                    qs = torch.rand(rows, 32)
                    if ndim == 3:
                        qs = qs.unsqueeze(-1)
                    producer = x.float()
                    fast, fallback = Mock(), Mock()
                    with patch.dict(
                        namespace,
                        fused_logits_head_gate=fast,
                        fp32_linear_logits_head_gate=fallback,
                    ):
                        out = obj._get_logits_head_gate(x, qs, x_fp32=producer)
                    if 1 <= rows <= 32:
                        fast.assert_called_once()
                        fallback.assert_not_called()
                        self.assertIs(out, fast.return_value)
                        self.assertIs(
                            fast.call_args.args[2], obj._hy4_small_t_head_gate_weight
                        )
                    else:
                        fast.assert_not_called()
                        fallback.assert_called_once()
                        self.assertIs(out, fallback.return_value)
                        self.assertIs(fallback.call_args.kwargs["x_fp32"], producer)
                    self.assertIs(obj.weights_proj.weight, original)

    def test_fusion_off_other_shapes_and_unsupported_inputs_fall_back(self):
        namespace = Indexer._get_logits_head_gate.__globals__
        for enabled, heads in ((False, 32), (True, 64)):
            obj, _ = self._make_indexer(enabled=enabled, heads=heads)
            self.assertIsNone(obj._hy4_small_t_head_gate_weight)
        obj, _ = self._make_indexer()
        for x, qs in (
            (torch.randn(8, 6144), torch.rand(8, 32)),
            (torch.randn(6144, 8, dtype=torch.bfloat16).t(), torch.rand(8, 32)),
            (torch.randn(8, 6144, dtype=torch.bfloat16), torch.rand(8, 64)[:, ::2]),
        ):
            fast, fallback = Mock(), Mock()
            with patch.dict(
                namespace,
                fused_logits_head_gate=fast,
                fp32_linear_logits_head_gate=fallback,
            ):
                obj._get_logits_head_gate(x, qs)
            fast.assert_not_called()
            fallback.assert_called_once()

    def test_hadamard_dispatch_is_unchanged(self):
        obj, _ = self._make_indexer(hadamard=True)
        self.assertTrue(obj._fuse_logits_head_gate)
        self.assertIsNone(obj._hy4_small_t_head_gate_weight)
        fast, fallback = Mock(), Mock()
        with patch.dict(
            Indexer._get_logits_head_gate.__globals__,
            fused_logits_head_gate=fast,
            fp32_linear_logits_head_gate=fallback,
        ):
            obj._get_logits_head_gate(
                torch.randn(256, 6144, dtype=torch.bfloat16), torch.rand(256, 32)
            )
        fast.assert_called_once()
        self.assertFalse(fast.call_args.kwargs["high_precision"])
        fallback.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_precision_and_graph_replay(self):
        torch.manual_seed(20260908)
        obj, original = self._make_indexer(device="cuda")
        weight_ptr, weight_stride = original.data_ptr(), original.stride()
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for rows in (1, 8, 32, 33, 256):
                with self.subTest(rows=rows), torch.inference_mode():
                    x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
                    qs = torch.rand(rows, 32, 1, device="cuda")
                    x_fp32 = x.float()
                    for _ in range(3):
                        obj._get_logits_head_gate(x, qs, x_fp32=x_fp32)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = obj._get_logits_head_gate(x, qs, x_fp32=x_fp32)
                    for _ in range(3):
                        x.copy_(torch.randn_like(x))
                        x_fp32.copy_(x.float())
                        qs.copy_(torch.rand_like(qs))
                        graph.replay()
                        ref = (
                            obj.weights_proj(x.float()).unsqueeze(-1)
                            * qs
                            * (128**-0.5 * 32**-0.5)
                        )
                        torch.testing.assert_close(out, ref, atol=2e-6, rtol=2e-5)
                    self.assertEqual(original.data_ptr(), weight_ptr)
                    self.assertEqual(original.stride(), weight_stride)
                    if rows in (8, 32):
                        # Isolate head-weight error at a synthetic Top-2048
                        # boundary; this is not a full-model acceptance test.
                        positive_qk = torch.randn(rows, 32, 2560, device="cuda").relu_()
                        actual_scores = (positive_qk * out).sum(dim=1)
                        reference_scores = (positive_qk * ref).sum(dim=1)
                        actual_topk = actual_scores.topk(2048).indices.sort().values
                        reference_topk = (
                            reference_scores.topk(2048).indices.sort().values
                        )
                        self.assertTrue(torch.equal(actual_topk, reference_topk))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


class Hy4CmpSchedulingControlsTest(unittest.TestCase):

    def test_removed_overrides_cannot_reenable_experimental_schedules(self):
        env = {
            "RTP_LLM_HY4_CMP_EARLY_HEAD_GATE": "1",
            "RTP_LLM_HY4_CMP_OUTPUT_GATE_STAGE": "before_attention",
            "RTP_LLM_HY4_CMP_GATE_OVERLAP": "1",
            "RTP_LLM_HY4_CMP_MTP_MULTISTREAM": "1",
        }
        for model_type in ("hy_v4", "hy_v4_mtp"):
            with self.subTest(model_type=model_type), patch.dict(os.environ, env):
                cmp = bridge.Hy4Cmp(
                    config=_config(model_type),
                    parallelism_config=_parallelism(),
                    self_attn=_attention(),
                )
                self.assertNotIn("_early_head_gate", vars(cmp))
                self.assertNotIn("_output_gate_stage", vars(cmp))
                self.assertNotIn("_gate_overlap", vars(cmp))
                self.assertNotIn("_mtp_multistream", vars(cmp))
                self.assertFalse(hasattr(Indexer, "project_head_gate"))

    def test_graph_clone_keeps_events_graph_local(self):
        cmp = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(),
            self_attn=_attention(),
        )
        cmp._events = object()
        attention = _attention()
        clone = cmp.clone_for_cuda_graph(self_attn=attention)
        self.assertIsNone(clone._events)
        self.assertIs(clone.self_attn, attention)

    def test_target_indexer_switch_survives_schedule_cleanup(self):
        with patch.dict(
            os.environ,
            {"RTP_LLM_HY4_CMP_INDEXER_FRONTEND": "0"},
        ):
            cmp = bridge.Hy4Cmp(
                config=_config(),
                parallelism_config=_parallelism(),
                self_attn=_attention(),
            )
        clone = cmp.clone_for_cuda_graph(self_attn=_attention())
        self.assertFalse(clone._indexer_frontend_parallel)
        self.assertNotIn("_mtp_multistream", vars(clone))

    def test_late_head_weights_follow_q_quantization_without_duplicate_gemm(self):
        indexer = object.__new__(Indexer)
        nn.Module.__init__(indexer)
        indexer.use_hadamard = False
        indexer._fuse_logits_head_gate = False
        indexer._is_sparse_prefill_cp = lambda _: False
        indexer.softmax_scale, indexer.weights_scale = 0.125, 0.25
        x = torch.randn(4, 6, dtype=torch.bfloat16)
        q, key = torch.randn(4, 3), torch.randn(4, 6)
        q_scale = torch.rand(4, 3, 1)
        weight = torch.randn(3, 6)
        order = []
        indexer._get_q_k_bf16 = Mock(
            side_effect=lambda *a: (order.append("qk"), (q, key))[1]
        )
        indexer._quantize_q_k = Mock(
            side_effect=lambda *a: (order.append("quant"), (q, q_scale))[1]
        )
        indexer.weights_proj = _Callable(
            Mock(
                side_effect=lambda value: (order.append("head_gate"), value @ weight.T)[
                    1
                ]
            )
        )
        indexer._compute_topk = Mock(
            side_effect=lambda *a: (order.append("topk"), "indices")[1]
        )
        result = indexer(
            x,
            q,
            None,
            None,
            SimpleNamespace(is_prefill=False),
            use_fast_path=False,
            x_fp32=x.float(),
        )
        self.assertEqual(result, "indices")
        self.assertEqual(order, ["qk", "quant", "head_gate", "topk"])
        indexer.weights_proj.fn.assert_called_once()
        torch.testing.assert_close(
            indexer._compute_topk.call_args.args[1],
            ((x.float() @ weight.T).unsqueeze(-1) * q_scale) * 0.03125,
            rtol=0,
            atol=0,
        )

    def test_boolean_aliases_and_invalid_values(self):
        key = "RTP_LLM_HY4_CMP"
        for value in ("1", "true", "yes", "on", " TRUE ", "\tOn\n"):
            with patch.dict(os.environ, {key: value}):
                self.assertTrue(bridge.resolve_hy4_cmp_enabled())
        for value in ("0", "false", "no", "off", "", "  ", " OFF "):
            with patch.dict(os.environ, {key: value}):
                self.assertFalse(bridge.resolve_hy4_cmp_enabled())
        with patch.dict(os.environ, {key: "invalid"}):
            with self.assertRaises(ValueError):
                bridge.resolve_hy4_cmp_enabled()


class Hy4CmpTest(unittest.TestCase):
    def test_switch_defaults_on(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(bridge.resolve_hy4_cmp_enabled())
        with patch.dict(os.environ, {"RTP_LLM_HY4_CMP": "0"}):
            self.assertFalse(bridge.resolve_hy4_cmp_enabled())
        with patch.dict(os.environ, {"RTP_LLM_HY4_CMP": "1"}):
            self.assertTrue(bridge.resolve_hy4_cmp_enabled())
        with patch.dict(os.environ, {"RTP_LLM_HY4_CMP": "maybe"}):
            with self.assertRaisesRegex(ValueError, "invalid RTP_LLM_HY4_CMP"):
                bridge.resolve_hy4_cmp_enabled()

    def test_static_contract_is_hy4_tp1_gated_mla(self) -> None:
        cmp = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(),
            self_attn=_attention(),
        )
        self.assertIsNone(cmp._disabled_reason)

        cmp = bridge.Hy4Cmp(
            config=_config("glm_5"),
            parallelism_config=_parallelism(),
            self_attn=_attention(),
        )
        self.assertEqual(cmp._disabled_reason, "unsupported model type")

        cmp = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(8),
            self_attn=_attention(),
        )
        self.assertEqual(cmp._disabled_reason, "HY4 CMP requires TP=1")

    def test_model_call_selects_cmp_for_all_layers_or_none(self) -> None:
        first = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(),
            self_attn=_attention(indexer=SimpleNamespace(use_hadamard=False)),
        )
        second = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(),
            self_attn=_attention(indexer=None),
        )
        layers = [SimpleNamespace(hy4_cmp=first), SimpleNamespace(hy4_cmp=second)]
        cache = SimpleNamespace(get_layer_cache=Mock(return_value=object()))
        hidden = object()

        with patch.object(first, "_dynamic_disabled_reason", return_value=None):
            self.assertTrue(
                bridge.should_enable_hy4_cmp(layers, 2, hidden, object(), cache)
            )

        second._disabled_reason = "unsupported"
        with patch.object(first, "_dynamic_disabled_reason", return_value=None):
            self.assertFalse(
                bridge.should_enable_hy4_cmp(layers, 2, hidden, object(), cache)
            )

    def test_model_call_requires_first_indexer_and_dynamic_contract(self) -> None:
        cmp = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(),
            self_attn=_attention(indexer=None),
        )
        layers = [SimpleNamespace(hy4_cmp=cmp)]
        cache = SimpleNamespace(get_layer_cache=Mock(return_value=object()))

        with patch.object(cmp, "_dynamic_disabled_reason", return_value=None):
            self.assertFalse(
                bridge.should_enable_hy4_cmp(layers, 1, object(), object(), cache)
            )
        cmp.self_attn.indexer = object()
        with patch.object(
            cmp, "_dynamic_disabled_reason", return_value="ordinary prefill"
        ):
            self.assertFalse(
                bridge.should_enable_hy4_cmp(layers, 1, object(), object(), cache)
            )

    def test_long_kv_threshold_matches_glm5_cmp(self) -> None:
        cmp = bridge.Hy4Cmp(
            config=_config(),
            parallelism_config=_parallelism(),
            self_attn=_attention(indexer=SimpleNamespace(use_hadamard=False)),
        )
        below = SimpleNamespace(
            attn_inputs=SimpleNamespace(kv_cache_block_id_device=torch.empty(1, 8191))
        )
        threshold = SimpleNamespace(
            attn_inputs=SimpleNamespace(kv_cache_block_id_device=torch.empty(1, 8192))
        )

        self.assertFalse(cmp._serialize_score_after_q_path(below))
        self.assertTrue(cmp._serialize_score_after_q_path(threshold))

    def test_hy4_mega_moe_prepack_views_require_exact_abi(self) -> None:
        layer = object.__new__(GenericMoeLayer)
        nn.Module.__init__(layer)
        layer._hy4_mega_moe_prepack = True
        layer.hidden_dim = 128
        layer.top_k = 2
        activation = torch.empty(4, 128, dtype=torch.float8_e4m3fn)
        scale = torch.empty(4, 1, dtype=torch.int32)
        indices = torch.empty(4, 2, dtype=torch.int64)
        weights = torch.empty(4, 2, dtype=torch.float32)
        layer.fused_moe = SimpleNamespace(
            topk_ids_dtype=torch.int64,
            prepacked_input_views=Mock(
                return_value=(activation, scale, indices, weights)
            ),
        )

        valid_views = GenericMoeLayer.hy4_prepacked_input_views(layer, 4)
        self.assertTrue(
            all(
                actual is expected
                for actual, expected in zip(
                    valid_views, (activation, scale, indices, weights)
                )
            )
        )

        layer.fused_moe.prepacked_input_views.return_value = (
            activation,
            scale,
            indices.to(torch.int32),
            weights,
        )
        self.assertEqual(
            GenericMoeLayer.hy4_prepacked_input_views(layer, 4),
            (None, None, None, None),
        )

    def test_forward_prepacked_reuses_dense_mxfp8_for_shared_expert(self) -> None:
        layer = object.__new__(GenericMoeLayer)
        nn.Module.__init__(layer)
        experts = torch.randn(2, 4)
        shared = torch.randn(2, 4)
        layer.fake_balance_expert = None
        layer._use_mega_moe_fused_shared = False
        layer.fused_moe = SimpleNamespace(forward_prepacked=Mock(return_value=experts))
        calls = {}

        def run_shared(hidden_states, x_fp8=None, x_scale=None):
            calls.update(hidden=hidden_states, fp8=x_fp8, scale=x_scale)
            return shared

        layer.shared_expert = _Callable(run_shared)
        hidden = torch.randn(2, 4)
        fp8 = object()
        scale = object()

        output = GenericMoeLayer.forward_prepacked(
            layer,
            hidden,
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(2, 2),
            x_fp8=fp8,
            x_scale=scale,
        )

        torch.testing.assert_close(output, experts + shared)
        self.assertIs(calls["hidden"], hidden)
        self.assertIs(calls["fp8"], fp8)
        self.assertIs(calls["scale"], scale)


class Hy4CmpIndexerFusionTest(unittest.TestCase):
    @staticmethod
    def _cos_sin_cache(max_position: int = 4096) -> torch.Tensor:
        rope_dim = 64
        inv_freq = 1.0 / (
            10_000_000.0
            ** (
                torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32)
                / rope_dim
            )
        )
        positions = torch.arange(max_position, device="cuda", dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def _run_precision_case(self, rows: int, seed: int) -> None:
        import flashinfer.rope as fi_rope

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.modules.base.cuda.indexer_op import _unpack_ue8m0_scale
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_hy4_indexer_rope_quant import (
            fused_hy4_indexer_rope_quant_cache,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        torch.manual_seed(seed)
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16).contiguous()
        k = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.randint(0, 4096, (rows,), device="cuda", dtype=torch.int32)
        slots = torch.randperm(4 * 64, device="cuda")[:rows].to(torch.int64)
        slots[::5] = -1
        cos_sin = self._cos_sin_cache()
        cache_ref = torch.full((4, 64, 132), 0xA5, device="cuda", dtype=torch.uint8)
        cache_out = cache_ref.clone()

        q_ref = q.clone()
        k_ref = k.clone()
        fi_rope._apply_rope_pos_ids_cos_sin_cache(
            q=q_ref[:, :, :64],
            k=k_ref[:, :64].unsqueeze(1),
            q_rope=q_ref[:, :, :64],
            k_rope=k_ref[:, :64].unsqueeze(1),
            cos_sin_cache=cos_sin,
            pos_ids=positions,
            interleave=True,
        )
        q_ref_fp8, q_ref_scale = sgl_per_token_group_quant_fp8(
            q_ref.view(-1, 128),
            group_size=128,
            eps=1.0e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        q_ref_fp8 = q_ref_fp8.view(rows, 32, 128)
        q_ref_scale = _unpack_ue8m0_scale(q_ref_scale).view(rows, 32, 1)
        rtp_llm_ops.indexer_k_quant_and_cache(k_ref, cache_ref, slots, 128, "ue8m0")

        fused = fused_hy4_indexer_rope_quant_cache(
            q,
            k,
            positions,
            cos_sin,
            slots,
            cache_out,
            is_neox_style=False,
        )
        self.assertIsNotNone(fused)
        q_out_fp8, q_out_scale = fused
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(q_ref_fp8.view(torch.uint8), q_out_fp8.view(torch.uint8))
        )
        self.assertTrue(torch.equal(q_ref_scale, q_out_scale))
        self.assertTrue(torch.equal(cache_ref, cache_out))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_matches_existing_hy4_operator_chain_byte_exact(self) -> None:
        for seed in (0, 7, 1234):
            for rows in (1, 4, 8, 12, 24, 64, 256):
                with self.subTest(seed=seed, rows=rows):
                    self._run_precision_case(rows, seed)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_graph_replay(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_hy4_indexer_rope_quant import (
            fused_hy4_indexer_rope_quant_cache,
        )

        q = torch.randn(24, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(24, device="cuda", dtype=torch.int32)
        slots = torch.arange(24, device="cuda", dtype=torch.int64)
        cos_sin = self._cos_sin_cache()
        cache = torch.zeros(1, 64, 132, device="cuda", dtype=torch.uint8)

        for _ in range(3):
            fused_hy4_indexer_rope_quant_cache(
                q,
                k,
                positions,
                cos_sin,
                slots,
                cache,
                is_neox_style=False,
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_hy4_indexer_rope_quant_cache(
                q,
                k,
                positions,
                cos_sin,
                slots,
                cache,
                is_neox_style=False,
            )
        self.assertIsNotNone(captured)
        graph.replay()
        torch.cuda.synchronize()

    def test_cmp_indexer_post_selects_existing_fusion(self) -> None:
        q_fp8 = object()
        q_scale = object()
        topk = object()
        q_projection = torch.randn(2, 8)
        precomputed_k = torch.randn(2, 4)
        indexer = SimpleNamespace(
            use_hadamard=False,
            _is_sparse_prefill_cp=Mock(return_value=False),
            _is_multi_token_decode=Mock(return_value=False),
            wq_b=Mock(return_value=q_projection),
            index_n_heads=2,
            index_head_dim=4,
            indexer_op=SimpleNamespace(
                cos_sin_cache=object(),
                is_neox_style=False,
                _kv_cache_blocks=Mock(return_value=object()),
            ),
            _get_q_k_bf16=Mock(
                side_effect=AssertionError("unfused Q/K path must not run")
            ),
            _quantize_q_k=Mock(
                side_effect=AssertionError("unfused quant path must not run")
            ),
            _get_logits_head_gate=Mock(return_value=object()),
            _compute_topk=Mock(return_value=topk),
        )
        fmha_params = SimpleNamespace(
            positions_d=torch.arange(2), slot_mapping=torch.arange(2)
        )

        with patch(
            "rtp_llm.models_py.modules.hybrid.indexer.fused_hy4_indexer_rope_quant_cache",
            return_value=(q_fp8, q_scale),
        ) as fused_kernel:
            cmp = object.__new__(bridge.Hy4Cmp)
            cmp.self_attn = SimpleNamespace(indexer=indexer)
            result = cmp._indexer_post(
                torch.randn(2, 16),
                q_projection.view(2, 2, 4),
                precomputed_k,
                SimpleNamespace(
                    fmha_params=fmha_params,
                    attn_inputs=SimpleNamespace(is_prefill=False),
                ),
                object(),
                None,
            )

        self.assertIs(result[0], q_fp8)
        fused_call = fused_kernel.call_args.args
        self.assertEqual(tuple(fused_call[0].shape), (2, 2, 4))
        self.assertIs(fused_call[1], precomputed_k)
        indexer._get_logits_head_gate.assert_called_once()
        indexer._compute_topk.assert_not_called()

    def test_cmp_k_preserves_projection_then_norm_formula(self) -> None:
        torch.manual_seed(7)
        indexer = SimpleNamespace(
            wk=nn.Linear(4, 2, bias=False),
            k_norm=nn.LayerNorm(2),
        )
        hidden = torch.randn(3, 4)

        expected = indexer.k_norm(indexer.wk(hidden))
        cmp = object.__new__(bridge.Hy4Cmp)
        cmp.self_attn = SimpleNamespace(indexer=indexer)
        actual = cmp._indexer_k(hidden, None, None)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class Hy4IndependentCmpTest(unittest.TestCase):
    """Compare the dedicated path with ordinary MLA; inspect actual DAG edges."""

    def _case(self, model_type="hy_v4", *, frontend=True, device="cpu"):
        torch.manual_seed(37)
        attn = object.__new__(MlaAttention)
        nn.Module.__init__(attn)
        attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim = 2, 1, 1
        attn.num_heads, attn.q_head_dim, attn.v_head_dim = 1, 2, 2
        attn.layer_idx, attn.token_per_block = 0, 64
        attn.gating_type = "elementwise"
        attn.reuse_topk_indices = False
        attn._reuse_mxfp8_hidden_quant = False
        attn._fuse_q_a_norm_mode = "off"
        attn._fuse_kv_a_norm = False
        attn._fuse_gated_mla_quant = False
        attn.attn_sink = torch.tensor([0.3], device=device)
        attn.parallelism_config = _parallelism()
        log = []
        current = ["caller"]

        def linear(name, inp, out):
            weight = torch.randn(inp, out, device=device) * 0.2

            def run(x, input_scales=None):
                log.append((name, current[0]))
                if input_scales is not None:
                    x = x * input_scales
                return x @ weight

            return _Callable(run)

        def norm(fn, width):
            obj = _Callable(fn)
            obj.weight = nn.Parameter(torch.ones(width, device=device))
            obj.variance_epsilon = 1e-5
            return obj

        attn.fused_qkv_a_proj = linear("qkv", 4, 4)
        attn.q_a_layernorm = norm(lambda x: x * 0.5, 2)
        attn.q_b_proj = linear("qb", 2, 2)
        attn.q_b_proj.scale_ue8m0 = False
        attn.kv_a_layernorm = norm(lambda x: x + 0.1, 1)
        attn.gate_proj = linear("gate", 4, 2)
        attn.o_proj = linear("output", 2, 4)
        indexer = object.__new__(Indexer)
        nn.Module.__init__(indexer)
        indexer.use_hadamard = False
        indexer._fuse_logits_head_gate = False
        indexer.softmax_scale, indexer.weights_scale = 0.5, 1.0
        indexer.index_n_heads, indexer.index_head_dim = 1, 2
        indexer.wk = linear("index_k", 4, 2)
        indexer.k_norm = norm(lambda x: x + 0.2, 2)
        indexer.wq_b = linear("index_q", 2, 2)
        indexer.weights_proj = linear("head", 4, 1)
        cache = torch.zeros(2, 2, device=device)

        def quant(q, k, cache, slots):
            log.append(("post", current[0]))
            cache.copy_(k)
            return q, torch.ones(q.shape[:-1] + (1,), device=q.device)

        def score(q, weights, cache, params, inputs):
            log.append(("score", current[0]))
            logits = (q.squeeze(1) @ cache.T) * weights.reshape(-1, 1)
            return logits.topk(1, dim=-1).indices.to(torch.int32)

        indexer.indexer_op = SimpleNamespace(
            apply_rope_and_rotate_q_k=lambda q, k, pos: (q + 0.01, k + 0.01),
            quant_q_k=quant,
            _get_topk_paged=score,
            cos_sin_cache=torch.empty(0, device=device),
            is_neox_style=False,
            _kv_cache_blocks=lambda cache: cache,
        )
        attn.indexer = indexer

        def prepare(q, kv, k_pe, cache, layer, sink, **kwargs):
            log.append(("main_ready", current[0]))
            if kwargs:
                self.assertIn("kv_norm_weight", kwargs)
                kv = kv + 0.1
            return q.squeeze(1) + kv + k_pe + sink

        def finish(prepared, topk):
            log.append(("mla", current[0]))
            return prepared + topk

        fmha = SimpleNamespace(
            is_sparse=lambda: True,
            supports_topk_late_binding=True,
            cp_params=None,
            attn_inputs=SimpleNamespace(is_prefill=False),
            fmha_params=SimpleNamespace(
                positions_d=torch.arange(2, device=device),
                slot_mapping=torch.arange(2, device=device),
            ),
            can_fuse_kv_norm_cache=lambda *a: False,
            prepare_topk_independent_forward=prepare,
            finish_topk_dependent_forward=finish,
        )
        fmha.forward = lambda q, kv, kp, cache, layer, topk, **kw: finish(
            prepare(q, kv, kp, cache, layer, kw.get("attn_sink")), topk
        )
        cmp = bridge.Hy4Cmp(
            config=_config(model_type),
            parallelism_config=_parallelism(),
            self_attn=attn,
        )
        cmp._indexer_frontend_parallel = frontend
        return SimpleNamespace(
            attn=attn,
            cmp=cmp,
            hidden=torch.randn(2, 4, device=device),
            cache=cache,
            fmha=fmha,
            log=log,
            current=current,
        )

    @contextmanager
    def _queues(self, case):
        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                case.log.append(("wait", self.name, event.name))

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self):
                case.log.append(("record", case.current[0], self.name))

        caller, k, q = (Stream(n) for n in ("caller", "index", "index_q"))
        events = bridge._Events(
            *(Event(n) for n in ("input", "qc", "q_ready", "main", "done"))
        )

        @contextmanager
        def stream(s):
            old = case.current[0]
            case.current[0] = s.name
            try:
                yield
            finally:
                case.current[0] = old

        with patch.object(
            case.cmp, "_dynamic_disabled_reason", return_value=None
        ), patch.object(
            case.cmp, "_side_streams", return_value=(k, q)
        ) as streams, patch.object(
            case.cmp, "_new_events", return_value=events
        ) as create_events, patch.object(
            bridge.torch.cuda, "stream", side_effect=stream
        ), patch.object(
            bridge.torch.cuda, "current_stream", return_value=caller
        ), patch.object(
            bridge, "_record_stream"
        ) as lifetime:
            yield SimpleNamespace(
                streams=streams, events=create_events, lifetime=lifetime
            )

    def test_dedicated_path_matches_plain_attention_and_does_not_call_old_forwards(
        self,
    ):
        for model in ("hy_v4", "hy_v4_mtp"):
            for frontend in (False, True):
                for long_kv in (False, True):
                    with self.subTest(model=model, frontend=frontend, long_kv=long_kv):
                        case = self._case(model, frontend=frontend)
                        expected, indices = case.attn(
                            case.hidden, case.fmha, case.cache, return_topk=True
                        )
                        expected_cache = case.cache.clone()
                        case.cache.zero_()
                        case.log.clear()
                        with self._queues(case), patch.object(
                            case.cmp,
                            "_serialize_score_after_q_path",
                            return_value=long_kv,
                        ), patch.object(
                            case.attn,
                            "forward",
                            side_effect=AssertionError("ordinary MLA called"),
                        ), patch.object(
                            case.attn.indexer,
                            "forward",
                            side_effect=AssertionError("ordinary Indexer called"),
                        ), patch.object(
                            case.fmha,
                            "forward",
                            side_effect=AssertionError("ordinary backend called"),
                        ):
                            output, topk = case.cmp.forward_attention(
                                case.hidden, case.fmha, case.cache, return_topk=True
                            )
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(topk, indices))
                        self.assertTrue(torch.equal(case.cache, expected_cache))
                        for name in (
                            "qkv",
                            "qb",
                            "index_k",
                            "index_q",
                            "head",
                            "gate",
                            "score",
                            "mla",
                            "output",
                        ):
                            self.assertEqual(
                                sum(x[0] == name for x in case.log), 1, name
                            )

    def test_q_projection_waits_only_for_qc_and_fused_post_joins_both_branches(self):
        case = self._case()
        with self._queues(case):
            case.cmp.forward_attention(case.hidden, case.fmha, case.cache)
        self.assertIn(("index_q", "index_q"), case.log)
        self.assertIn(("index_k", "index"), case.log)
        self.assertIn(("post", "index"), case.log)
        self.assertEqual(
            [x for x in case.log if x[:2] == ("wait", "index_q")],
            [("wait", "index_q", "qc")],
        )
        self.assertLess(
            case.log.index(("record", "index_q", "q_ready")),
            case.log.index(("wait", "index", "q_ready")),
        )
        self.assertLess(
            case.log.index(("wait", "index", "q_ready")),
            case.log.index(("post", "index")),
        )
        self.assertLess(
            case.log.index(("wait", "caller", "done")),
            case.log.index(("mla", "caller")),
        )
        self.assertLess(
            case.log.index(("qb", "caller")), case.log.index(("gate", "caller"))
        )

    def test_long_kv_delays_score_until_complete_main_query(self):
        case = self._case()
        with self._queues(case), patch.object(
            case.cmp, "_serialize_score_after_q_path", return_value=True
        ):
            case.cmp.forward_attention(case.hidden, case.fmha, case.cache)
        self.assertLess(
            case.log.index(("main_ready", "caller")),
            case.log.index(("record", "caller", "main")),
        )
        self.assertLess(
            case.log.index(("wait", "index", "main")),
            case.log.index(("score", "index")),
        )

    def test_frontend_off_keeps_score_on_side_stream(self):
        case = self._case(frontend=False)
        with self._queues(case):
            case.cmp.forward_attention(case.hidden, case.fmha, case.cache)
        for name in ("index_k", "index_q", "post", "head"):
            self.assertIn((name, "caller"), case.log)
        self.assertIn(("score", "index"), case.log)
        self.assertFalse(any(x[1] == "index_q" for x in case.log))

    def test_mtp_and_topk_reuse_do_not_create_streams_or_events(self):
        for model, reuse in (
            ("hy_v4_mtp", False),
            ("hy_v4", True),
            ("hy_v4_mtp", True),
        ):
            case = self._case(model)
            previous = torch.zeros(2, 1, dtype=torch.int32)
            with self._queues(case) as queues:
                _, topk = case.cmp.forward_attention(
                    case.hidden,
                    case.fmha,
                    case.cache,
                    force_reuse_topk_indices=reuse,
                    prev_topk_indices=previous,
                    return_topk=True,
                )
            queues.streams.assert_not_called()
            queues.events.assert_not_called()
            queues.lifetime.assert_not_called()
            if reuse:
                self.assertIs(topk, previous)
                self.assertFalse(
                    any(
                        x[0] in ("index_k", "index_q", "post", "score")
                        for x in case.log
                    )
                )

    def test_shared_quantization_and_q_norm_inputs_are_not_recomputed(self):
        case = self._case()
        case.attn._reuse_mxfp8_hidden_quant = True
        case.attn._fuse_q_a_norm_mode = "mxfp8"
        x8, xs = case.hidden.clone(), torch.ones(2, 1)
        q8, qs = torch.randn(2, 2), torch.ones(2, 1)
        with self._queues(case), patch.object(
            bridge, "_mxfp8_quantize_hidden", return_value=(x8, xs)
        ) as quant, patch(
            "rtp_llm.models_py.modules.hybrid.mla_attention.fused_strided_rmsnorm_per_token_fp8_quant",
            return_value=(q8, qs),
            create=True,
        ) as qnorm, patch.object(
            case.attn.q_b_proj, "forward", wraps=case.attn.q_b_proj.forward
        ) as main_q, patch.object(
            case.attn.indexer.wq_b, "forward", wraps=case.attn.indexer.wq_b.forward
        ) as index_q, patch.object(
            case.attn.indexer.wk, "forward", wraps=case.attn.indexer.wk.forward
        ) as index_k:
            case.cmp.forward_attention(case.hidden, case.fmha, case.cache)
        quant.assert_called_once_with(case.hidden)
        qnorm.assert_called_once()
        self.assertTrue(qnorm.call_args.kwargs["mxfp8_semantics"])
        for call in (main_q.call_args, index_q.call_args):
            self.assertIs(call.args[0], q8)
            self.assertIs(call.kwargs["input_scales"], qs)
        self.assertIs(index_k.call_args.args[0], x8)
        self.assertIs(index_k.call_args.kwargs["input_scales"], xs)

    def test_kv_norm_and_output_gate_fusions_keep_original_contracts(self):
        case = self._case("hy_v4_mtp")
        case.attn._fuse_kv_a_norm = True
        case.fmha.can_fuse_kv_norm_cache = lambda *a: True
        case.attn._fuse_gated_mla_quant = True
        case.attn._gated_mla_quant_group_size = 32
        case.attn._gated_mla_scale_ue8m0 = True
        case.attn._gated_mla_round_scale_to_pow2 = True
        scale = torch.ones(2, 1)
        with self._queues(case), patch(
            "rtp_llm.models_py.modules.hybrid.mla_attention.sigmoid_mul_fp8_quant_fwd",
            side_effect=lambda x, gate, **kw: (x * torch.sigmoid(gate), scale),
            create=True,
        ) as fused, patch.object(
            case.attn.o_proj, "forward", wraps=case.attn.o_proj.forward
        ) as output:
            case.cmp.forward_attention(case.hidden, case.fmha, case.cache)
        self.assertEqual(
            fused.call_args.kwargs,
            dict(
                quant_group_size=32,
                scale_ue8m0=True,
                round_scale_to_pow2=True,
                column_major_scales=True,
            ),
        )
        self.assertIs(output.call_args.kwargs["input_scales"], scale)

    def test_capture_requires_warmup_and_clone_gets_fresh_events(self):
        case = self._case()
        with patch.object(bridge.Hy4Cmp, "_streams_by_device", {}), patch.object(
            bridge, "_is_capturing", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                case.cmp._side_streams(torch.device("cuda:0"))
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                case.cmp._new_events(torch.device("cuda:0"))
        case.cmp._events = object()
        clone = case.cmp.clone_for_cuda_graph(self_attn=case.attn)
        self.assertIsNone(clone._events)
        self.assertIs(clone.self_attn, case.attn)

    def test_missing_reuse_topk_fails_before_gpu_submission(self):
        case = self._case()
        with self._queues(case), patch.object(case.cmp, "mla_prologue") as prologue:
            with self.assertRaisesRegex(RuntimeError, "previous TopK"):
                case.cmp.forward_attention(
                    case.hidden, case.fmha, case.cache, force_reuse_topk_indices=True
                )
        prologue.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_cuda_streams_and_graph_replay_match_serial_attention(self):
        # Real CUDA queues/events and math, with a small deterministic backend.
        # This validates the coordinator, not full-model numerical accuracy.
        for model in ("hy_v4", "hy_v4_mtp"):
            for defer in (False, True):
                case = self._case(model, device="cuda")
                with torch.inference_mode(), patch.object(
                    case.cmp, "_serialize_score_after_q_path", return_value=defer
                ):
                    for _ in range(3):
                        case.cmp.forward_attention(case.hidden, case.fmha, case.cache)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual, indices = case.cmp.forward_attention(
                            case.hidden, case.fmha, case.cache, return_topk=True
                        )
                    for value in (0.2, -0.7, 1.4):
                        case.hidden.fill_(value)
                        graph.replay()
                        torch.cuda.synchronize()
                        actual_copy, indices_copy, cache_copy = (
                            actual.clone(),
                            indices.clone(),
                            case.cache.clone(),
                        )
                        expected, topk = case.attn(
                            case.hidden, case.fmha, case.cache, return_topk=True
                        )
                        torch.testing.assert_close(
                            actual_copy, expected, rtol=0, atol=0
                        )
                        self.assertTrue(torch.equal(indices_copy, topk))
                        self.assertTrue(torch.equal(cache_copy, case.cache))


if __name__ == "__main__":
    unittest.main()
