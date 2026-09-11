import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.hybrid.indexer import Indexer


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

        # K-only must not touch Q outputs; Q-only must not touch the cache.
        split_cache = torch.full_like(cache_out, 0xA5)
        split_q = torch.zeros_like(q_out_fp8)
        split_scale = torch.full_like(q_out_scale, -1)
        fused_hy4_indexer_rope_quant_cache(
            q,
            k,
            positions,
            cos_sin,
            slots,
            split_cache,
            is_neox_style=False,
            branch="k",
            out=(split_q, split_scale),
        )
        self.assertTrue(
            torch.equal(
                split_q.view(torch.uint8), torch.zeros_like(split_q).view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(split_scale, torch.full_like(split_scale, -1)))
        self.assertTrue(torch.equal(split_cache, cache_ref))
        raw_gate = torch.randn(rows, 32, device="cuda", dtype=torch.float32)
        head_weights = torch.empty_like(raw_gate)
        fused_hy4_indexer_rope_quant_cache(
            q,
            k,
            positions,
            cos_sin,
            slots,
            split_cache,
            is_neox_style=False,
            branch="q",
            out=(split_q, split_scale),
            raw_head_gate=raw_gate,
            head_weights=head_weights,
            head_scale=1 / 64,
        )
        self.assertTrue(
            torch.equal(split_q.view(torch.uint8), q_out_fp8.view(torch.uint8))
        )
        self.assertTrue(torch.equal(split_scale, q_out_scale))
        self.assertTrue(torch.equal(split_cache, cache_ref))
        self.assertTrue(
            torch.equal(head_weights, (raw_gate * q_out_scale.squeeze(-1)) * (1 / 64))
        )

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

    @unittest.skipUnless(
        os.environ.get("RUN_HY4_CMP_INDEXER_BENCH") == "1",
        "manual HY4 CMP Indexer benchmark",
    )
    def test_benchmark_operator_chain(self) -> None:
        import flashinfer.rope as fi_rope

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.modules.base.cuda.indexer_op import _unpack_ue8m0_scale
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_hy4_indexer_rope_quant import (
            fused_hy4_indexer_rope_quant_cache,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        def elapsed_us(fn, warmup: int = 50, iterations: int = 500) -> float:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000.0 / iterations

        cos_sin = self._cos_sin_cache()
        for rows in (6, 8, 12, 24):
            q_ref = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
            k_ref = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
            q_fused, k_fused = q_ref.clone(), k_ref.clone()
            positions = torch.arange(rows, device="cuda", dtype=torch.int32)
            slots = torch.arange(rows, device="cuda", dtype=torch.int64)
            cache_ref = torch.zeros(1, 64, 132, device="cuda", dtype=torch.uint8)
            cache_fused = cache_ref.clone()

            def baseline() -> None:
                fi_rope._apply_rope_pos_ids_cos_sin_cache(
                    q=q_ref[:, :, :64],
                    k=k_ref[:, :64].unsqueeze(1),
                    q_rope=q_ref[:, :, :64],
                    k_rope=k_ref[:, :64].unsqueeze(1),
                    cos_sin_cache=cos_sin,
                    pos_ids=positions,
                    interleave=True,
                )
                _, scale = sgl_per_token_group_quant_fp8(
                    q_ref.view(-1, 128),
                    group_size=128,
                    eps=1.0e-10,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                _unpack_ue8m0_scale(scale)
                rtp_llm_ops.indexer_k_quant_and_cache(
                    k_ref, cache_ref, slots, 128, "ue8m0"
                )

            def fused() -> None:
                fused_hy4_indexer_rope_quant_cache(
                    q_fused,
                    k_fused,
                    positions,
                    cos_sin,
                    slots,
                    cache_fused,
                    is_neox_style=False,
                )

            baseline_us = elapsed_us(baseline)
            fused_us = elapsed_us(fused)
            torch.cuda.synchronize()
            baseline_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(baseline_graph):
                baseline()
            fused_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fused_graph):
                fused()
            baseline_graph_us = elapsed_us(baseline_graph.replay)
            fused_graph_us = elapsed_us(fused_graph.replay)
            print(
                f"HY4 CMP Indexer rows={rows}: eager baseline={baseline_us:.3f} us, "
                f"fused={fused_us:.3f} us; graph baseline={baseline_graph_us:.3f} us, "
                f"fused={fused_graph_us:.3f} us, "
                f"graph speedup={baseline_graph_us / fused_graph_us:.2f}x"
            )



if __name__ == "__main__":
    unittest.main()
