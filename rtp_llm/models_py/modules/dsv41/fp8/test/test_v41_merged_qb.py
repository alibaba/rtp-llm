"""Shared Q-B weights, reload, strided consumers and CUDA Graph replay."""

import gc
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention as mega
from rtp_llm.models_py.modules.dsv41.fp8._v41_merged_qb import MergedQB
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear
from rtp_llm.utils.model_weight import W


def linear(rows):
    weight = torch.randint(-8, 9, (rows, 1280), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones(rows // 32, 40, device="cuda").to(torch.float8_e8m0fnu)
    return V41MXFP8Linear(weight, scale), scale


def fixture():
    main, _ = linear(32768)
    index, raw_scale = linear(4096)
    backend = mega._backend()
    if backend is None:
        raise unittest.SkipTest("FlashMLA mega attention backend unavailable")
    attn = SimpleNamespace(
        wq_b=main,
        index_wq=index,
        n_heads=64,
        n_groups=8,
        is_index_source=True,
        attn_sink=torch.zeros(64, device="cuda"),
        freqs_cis=torch.ones(256, 32, dtype=torch.complex64, device="cuda"),
        _wo_a_stk_w=torch.zeros(8, 128, 4096, dtype=torch.float8_e4m3fn, device="cuda"),
        _wo_a_stk_s=torch.full(
            (8, 32, 128), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
        ).transpose(1, 2),
    )
    qw, qs, ow, os = mega._permuted_weights(attn, backend)
    state = mega.MegaAttention(
        backend,
        qw,
        qs,
        ow,
        os,
        torch.ones(256, 64, device="cuda"),
        attn.attn_sink.clone(),
        512**-0.5,
        64,
        8,
    )
    owner = {W.v4_indexer_wq_b_w: index.weight, W.v4_indexer_wq_b_s: raw_scale}
    return attn, state, owner


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MergedQBCUDA(unittest.TestCase):
    @torch.inference_mode()
    def test_decode_wiring(self):
        from rtp_llm.models_py.modules.dsv41 import _v41_query_quant
        from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )

        attn, state, owner = fixture()
        state.merged_qb = MergedQB.create(attn, state, owner)
        attn.rope_head_dim, attn.head_dim = 64, 512
        attn.skip_post_q_norm, attn.eps = True, 1e-6
        attn.kv_norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(2, 3, 1280, dtype=torch.bfloat16, device="cuda")
        quantized = attn.wq_b._quantize_input(x)
        kv = torch.randn(2, 3, 512, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(6, device="cuda")
        pool = torch.zeros(6, 32, 528, dtype=torch.uint8, device="cuda")
        with patch.object(
            _v41_query_quant,
            "try_project_quantized_qkv",
            return_value=(x, kv, quantized),
        ):

            def call():
                return decode_compute_qkv(
                    attn,
                    x,
                    positions,
                    swa_pool=pool,
                    swa_slots=positions * 32,
                    mega_attention=state,
                )

            actual = call()
            expected_q = state.project_q(attn.wq_b, x, quantized).view(2, 3, 64, 512)
            expected_index = attn.index_wq.forward_quantized(*quantized)
            torch.testing.assert_close(actual.q, expected_q, atol=0, rtol=0)
            torch.testing.assert_close(actual.index_q, expected_index, atol=0, rtol=0)
            self.assertTrue(actual.swa_written)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured.q, expected_q, atol=0, rtol=0)
            torch.testing.assert_close(captured.index_q, expected_index, atol=0, rtol=0)

    @torch.inference_mode()
    def test_storage_and_reload_graph(self):
        torch.manual_seed(41)
        attn, state, owner = fixture()
        original_main = attn.wq_b.weight
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["requested_bytes.all.current"]
        state.merged_qb = MergedQB.create(attn, state, owner)
        merged = state.merged_qb
        self.assertIsNotNone(merged)
        gc.collect()
        torch.cuda.synchronize()
        increase = torch.cuda.memory_stats()["requested_bytes.all.current"] - before
        # FP8 replaces the two allocations. Only fused packed scales are new.
        self.assertEqual(increase, merged.weight_scales.numel() * 4)
        self.assertIs(attn.wq_b.weight, original_main)
        self.assertIs(owner[W.v4_indexer_wq_b_w], attn.index_wq.weight)
        self.assertEqual(state.query_weight.data_ptr(), merged.weight.data_ptr())
        self.assertEqual(
            attn.index_wq.weight.data_ptr(), merged.weight.data_ptr() + 32768 * 1280
        )
        tensors = [
            merged.weight,
            merged.weight_scales,
            state.query_weight,
            state.query_scale,
            attn.index_wq.weight,
            attn.index_wq.weight_scales,
        ]
        pointers = [t.data_ptr() for t in tensors]
        for batch, span in ((1, 1), (2, 3), (16, 1), (64, 1), (64, 6)):
            with self.subTest(batch=batch, span=span):
                x = torch.randn(batch, span, 1280, dtype=torch.bfloat16, device="cuda")
                quantized = attn.wq_b._quantize_input(x)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        merged.forward_quantized(*quantized)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = merged.forward_quantized(*quantized)
                for phase in range(3):
                    new_main, main_scale = linear(32768)
                    new_index, new_scale = linear(4096)
                    main_scale.copy_(
                        torch.full_like(main_scale.float(), 2**phase).to(
                            main_scale.dtype
                        )
                    )
                    new_main.weight_scales.copy_(new_main.pack_scales(main_scale))
                    new_scale.copy_(
                        torch.full_like(new_scale.float(), 2**phase).to(new_scale.dtype)
                    )
                    new_index.weight_scales.copy_(new_index.pack_scales(new_scale))
                    # Change both weights and scales; never replace live pointers.
                    attn.wq_b.weight.copy_(new_main.weight)
                    attn.wq_b.weight_scales.copy_(new_main.weight_scales)
                    owner[W.v4_indexer_wq_b_w].copy_(new_index.weight)
                    owner[W.v4_indexer_wq_b_s].copy_(new_scale)
                    merged.weight_scales.zero_()
                    attn.index_wq.weight_scales.zero_()
                    state.reload_weights(attn)
                    graph.replay()
                    torch.cuda.synchronize()
                    reference_q = new_main.forward_quantized(*quantized)
                    reference_q = (
                        reference_q.view(batch, span, 64, 32, 16)
                        .transpose(-3, -2)
                        .contiguous()
                        .flatten(-3)
                    )
                    reference_index = new_index.forward_quantized(*quantized)
                    for actual, expected in zip(
                        captured, (reference_q, reference_index)
                    ):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(
                        attn.index_wq.forward_quantized(*quantized),
                        reference_index,
                        atol=0,
                        rtol=0,
                    )
                    self.assertEqual(pointers, [t.data_ptr() for t in tensors])
                    self.assertEqual(state.query_weight.data_ptr(), pointers[2])
                    self.assertEqual(attn.index_wq.weight.data_ptr(), pointers[4])
                print(
                    f"PASS B={batch} S={span}: three reloads, exact GEMM/graph, stable pointers",
                    flush=True,
                )
        print(
            f"PASS memory: extra FP8=0, extra scales={increase} bytes/layer", flush=True
        )

    @torch.inference_mode()
    def test_strided_consumers(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_indexer_q_triton as iq
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qkv_rope_cache_triton as kv

        _, state, _ = fixture()
        for batch, span in ((1, 1), (2, 3), (64, 1), (64, 6)):
            output = torch.randn(
                batch, span, 36864, dtype=torch.bfloat16, device="cuda"
            )
            main, index = output.split((32768, 4096), dim=-1)
            index = index.unflatten(-1, (32, 128))
            positions = torch.arange(batch * span, device="cuda")
            angles = torch.randn(batch * span + 1, 32, device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            weights = torch.randn(batch, span, 32, device="cuda", dtype=torch.bfloat16)
            for dtype in (torch.float32, torch.bfloat16):
                actual = iq.try_fused_indexer_q(
                    index, weights, freqs, positions, weights_out_dtype=dtype
                )
                expected = iq.try_fused_indexer_q(
                    index.contiguous(),
                    weights,
                    freqs,
                    positions,
                    weights_out_dtype=dtype,
                )
                self.assertIsNotNone(actual)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(
                        value.view(torch.uint8),
                        reference.view(torch.uint8),
                        atol=0,
                        rtol=0,
                    )
            q = main.unflatten(-1, (64, 512))
            raw_kv = torch.randn(batch, span, 512, device="cuda", dtype=torch.bfloat16)
            norm = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            pool = torch.zeros(batch * span, 32, 528, device="cuda", dtype=torch.uint8)
            slots = positions * 32
            actual = kv.try_fused_qkv_rope_cache(
                q,
                raw_kv,
                norm,
                positions,
                freqs,
                pool,
                slots,
                eps=1e-6,
                apply_q_rope=False,
                pad_query_heads=64,
            )
            self.assertIsNotNone(actual)
            self.assertIs(actual.q, q)
            expected_pool = torch.zeros_like(pool)
            expected = kv.try_fused_qkv_rope_cache(
                q.contiguous(),
                raw_kv,
                norm,
                positions,
                freqs,
                expected_pool,
                slots,
                eps=1e-6,
                apply_q_rope=False,
                pad_query_heads=64,
            )
            torch.testing.assert_close(pool, expected_pool, atol=0, rtol=0)
            torch.testing.assert_close(actual.kv, expected.kv, atol=0, rtol=0)
            state.cos_sin = torch.cat((freqs.real, freqs.imag), dim=-1)
            ids = slots.int().unsqueeze(1).expand(-1, 128).contiguous()
            native = state.decode(q, pool, ids, positions.int())
            contiguous = state.decode(q.contiguous(), pool, ids, positions.int())
            for value, reference in zip(native, contiguous):
                torch.testing.assert_close(
                    value.contiguous().view(torch.uint8),
                    reference.contiguous().view(torch.uint8),
                    atol=0,
                    rtol=0,
                )
            if batch * span > 1:
                # Padding and Q RoPE kernels still require contiguous Q.
                for kwargs in (
                    dict(apply_q_rope=True),
                    dict(apply_q_rope=False, pad_query_heads=128),
                ):
                    self.assertIsNone(
                        kv.try_fused_qkv_rope_cache(
                            q,
                            raw_kv,
                            norm,
                            positions,
                            freqs,
                            pool,
                            slots,
                            eps=1e-6,
                            **kwargs,
                        )
                    )

    def test_unsupported_geometry_preserves_original_path(self):
        attn, state, owner = fixture()
        original = attn.index_wq.weight
        for heads, is_source in ((32, True), (64, False)):
            attn.n_heads, attn.is_index_source = heads, is_source
            self.assertIsNone(MergedQB.create(attn, state, owner))
            self.assertIs(attn.index_wq.weight, original)
            self.assertIs(owner[W.v4_indexer_wq_b_w], original)


if __name__ == "__main__":
    unittest.main()
