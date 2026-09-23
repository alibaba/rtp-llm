"""Production decode projection, quantized indexer input, and SWA write wiring."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4 import _v41_query_quant
from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import try_project_qr_kv
from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_qkv_rope_cache_triton as rope_cache
from rtp_llm.models_py.modules.dsv4.fp8.decode import compute_qkv
from rtp_llm.models_py.modules.dsv4.fp8.decode.rope_metadata import decode_rope_metadata
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


def flags(enabled):
    return {
        "DSV41_FUSED_QUERY_QUANT": str(int(enabled)),
        "DSV41_REUSE_DECODE_ROPE": str(int(enabled)),
        "DSV41_FUSED_QKV_ROPE_CACHE": str(int(enabled)),
        "DSV41_FUSED_QKV": "1",
        "DSV41_FUSED_DECODE_Q_ROPE": "1",
        "DSV4_FP8_QUANT_KERNEL": "auto",
    }


class DecodeQKVIntegrationCPU(unittest.TestCase):
    def test_quantized_projection_and_completed_write_are_forwarded(self):
        qr = torch.randn(1, 2, 128, dtype=torch.bfloat16)
        kv = torch.randn(1, 2, 512, dtype=torch.bfloat16)
        quantized = (torch.zeros_like(qr), torch.ones(2, 1, dtype=torch.int32))
        q = torch.randn(1, 2, 512, dtype=torch.bfloat16)
        positions = torch.tensor([1, 3], dtype=torch.int32)
        table = torch.ones(8, 32, dtype=torch.complex64)
        freqs = table[positions.long()]
        consumer = SimpleNamespace(forward_quantized=Mock(return_value=q))
        attn = SimpleNamespace(
            rope_head_dim=64,
            head_dim=512,
            n_heads=1,
            skip_post_q_norm=True,
            freqs_cis=table,
            wq_b=consumer,
            kv_norm=torch.ones(512, dtype=torch.bfloat16),
            eps=1e-6,
            _try_fused_qr_kv=Mock(side_effect=AssertionError("reprojected Q/KV")),
            _lin=Mock(side_effect=AssertionError("requantized Q")),
        )
        pool = torch.zeros(1, 8, 528, dtype=torch.uint8)
        slots = torch.tensor([2, 5])
        completed = rope_cache.FusedQKVRopeCache(q.view(1, 2, 1, 512), kv, freqs)
        with patch.object(
            _v41_query_quant,
            "try_project_quantized_qkv",
            return_value=(qr, kv, quantized),
        ), patch.object(
            rope_cache, "try_fused_qkv_rope_cache", return_value=completed
        ) as fused, patch.object(
            compute_qkv, "fused_rmsnorm_rope"
        ) as old_norm:
            result = compute_qkv.decode_compute_qkv(
                attn, qr, positions, freqs_cis=freqs, swa_pool=pool, swa_slots=slots
            )
        self.assertTrue(result.swa_written)
        self.assertIs(result.qr_quantized, quantized)
        self.assertIs(result.qr, qr)
        self.assertIs(result.kv, kv)
        self.assertIs(result.freqs_cis, freqs)
        consumer.forward_quantized.assert_called_once_with(*quantized)
        self.assertIs(fused.call_args.args[1], kv)
        self.assertIs(fused.call_args.args[5], pool)
        self.assertIs(fused.call_args.args[6], slots)
        old_norm.assert_not_called()

    def test_unsupported_cache_retains_old_norm_and_requests_write(self):
        qr = torch.randn(1, 2, 128, dtype=torch.bfloat16)
        kv = torch.randn(1, 2, 512, dtype=torch.bfloat16)
        q = torch.randn(1, 2, 512, dtype=torch.bfloat16)
        positions = torch.tensor([1, 3], dtype=torch.int32)
        table = torch.ones(8, 32, dtype=torch.complex64)
        attn = SimpleNamespace(
            rope_head_dim=64,
            head_dim=512,
            n_heads=1,
            skip_post_q_norm=True,
            freqs_cis=table,
            wq_b=object(),
            kv_norm=torch.ones(512, dtype=torch.bfloat16),
            eps=1e-6,
            _try_fused_qr_kv=lambda _x: (qr, kv),
            _lin=lambda _layer, _x: q,
        )
        pool = torch.full((1, 8, 584), 173, dtype=torch.uint8)
        normalized = kv + 1
        with patch.object(
            _v41_query_quant, "try_project_quantized_qkv", return_value=None
        ), patch.object(compute_qkv, "_apply_v41_q_rope") as old_rope, patch.object(
            compute_qkv, "fused_rmsnorm_rope", return_value=normalized
        ) as old_norm:
            result = compute_qkv.decode_compute_qkv(
                attn, qr, positions, swa_pool=pool, swa_slots=torch.tensor([2, 5])
            )
        self.assertFalse(result.swa_written)
        self.assertIsNone(result.qr_quantized)
        self.assertIs(result.kv, normalized)
        old_rope.assert_called_once()
        old_norm.assert_called_once()
        self.assertIs(old_norm.call_args.args[0], kv)
        torch.testing.assert_close(result.freqs_cis, table[positions.long()])
        self.assertTrue(torch.all(pool == 173).item())


def native_attention(batch, span):
    from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8

    def linear(output, inputs):
        weight = (torch.randn(output, inputs, device="cuda") * 0.02).to(
            torch.float8_e4m3fn
        )
        scales = torch.ones(output // 32, inputs // 32, device="cuda").to(
            torch.float8_e8m0fnu
        )
        return V41MXFP8Linear(weight, scales)

    angles = torch.randn(2048, 32, device="cuda")
    attn = SimpleNamespace(
        dim=512,
        q_lora_rank=1280,
        head_dim=512,
        rope_head_dim=64,
        n_heads=8,
        skip_post_q_norm=True,
        eps=1e-6,
        wq_a_wkv=linear(1792, 512),
        wq_b=linear(8 * 512, 1280),
        q_norm=(1 + torch.randn(1280, device="cuda") * 0.1).bfloat16(),
        kv_norm=(1 + torch.randn(512, device="cuda") * 0.1).bfloat16(),
        freqs_cis=torch.polar(torch.ones_like(angles), angles),
        index_wq=linear(8 * 128, 1280),
        index_n_heads=8,
        index_head_dim=128,
        index_topk=4,
        index_weights=torch.randn(8, 512, device="cuda", dtype=torch.bfloat16),
        is_index_source=True,
        kv_source_layer_id=0,
        layer_id=0,
        compress_ratio=2,
        v41_config={},
        _lin=lambda layer, x: layer(x),
    )
    attn._try_fused_qr_kv = lambda x: try_project_qr_kv(
        attn.wq_a_wkv, x, attn.q_norm, attn.q_lora_rank, attn.eps
    )
    attn._shared_attention = {
        "layers": {0: attn},
        "global": {0: torch.randn(batch, 11, 128, device="cuda")},
    }
    attn._begin_forward = lambda: AttentionV41FP8._begin_forward(attn)
    attn._decode_write_swa_fp8 = lambda *args: AttentionV41FP8._decode_write_swa_fp8(
        attn, *args
    )
    x = torch.randn(batch, span, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(batch * span, device="cuda", dtype=torch.int32) + 32
    slots = torch.arange(batch * span, device="cuda", dtype=torch.int64) + 2
    slots[0] = -1
    return attn, x, positions, slots


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeQKVIntegrationCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("native group-32 DeepGEMM requires Blackwell")

    def setUp(self):
        torch.manual_seed(418)

    def run_path(self, attn, x, positions, slots, pool, enabled):
        with patch.dict(os.environ, flags(enabled)):
            converted, freqs = decode_rope_metadata({}, attn.freqs_cis, positions)
            result = compute_qkv.decode_compute_qkv(
                attn, x, converted, freqs_cis=freqs, swa_pool=pool, swa_slots=slots
            )
            if not result.swa_written:
                from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import (
                    quantize_and_insert_swa_k_cache,
                )

                quantize_and_insert_swa_k_cache(result.kv.reshape(-1, 512), pool, slots)
            return result

    def assert_qkv_equal(self, actual, expected):
        for name in ("qr", "q", "kv", "freqs_cis"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0
            )

    @torch.no_grad()
    def test_native_projection_cache_and_indexer_match_old_path(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8

        for batch, span in ((1, 1), (4, 6), (8, 6)):
            with self.subTest(batch=batch, span=span):
                attn, x, positions, slots = native_attention(batch, span)
                pool = torch.full((2, 134, 528), 173, device="cuda", dtype=torch.uint8)
                reference_pool = pool.clone()
                expected = self.run_path(
                    attn, x, positions, slots, reference_pool, False
                )
                actual = self.run_path(attn, x, positions, slots, pool, True)
                self.assertFalse(expected.swa_written)
                self.assertTrue(actual.swa_written)
                self.assertIsNone(expected.qr_quantized)
                self.assertIsNotNone(actual.qr_quantized)
                self.assert_qkv_equal(actual, expected)
                torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
                old_index_q = attn.index_wq(expected.qr)
                new_index_q = attn.index_wq.forward_quantized(*actual.qr_quantized)
                torch.testing.assert_close(new_index_q, old_index_q, rtol=0, atol=0)
                expected_ids = AttentionV41FP8._select_indices_decode(
                    attn, x, expected.qr, positions.long()
                )
                with patch.object(attn.index_wq, "_quantize_input") as requantize:
                    actual_ids = AttentionV41FP8._select_indices_decode(
                        attn, x, actual.qr, positions.long(), actual.qr_quantized
                    )
                requantize.assert_not_called()
                torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)

    @torch.no_grad()
    def test_nondefault_stream_graph_updates_projection_positions_and_cache(self):
        attn, x, positions, slots = native_attention(4, 6)
        pool = torch.full((2, 134, 528), 173, device="cuda", dtype=torch.uint8)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.run_path(attn, x, positions, slots, pool, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = self.run_path(attn, x, positions, slots, pool, True)
            index_q = attn.index_wq.forward_quantized(*actual.qr_quantized)
        for scale in (0.0, 1.0, -2.0):
            x.copy_(torch.randn_like(x) * scale)
            positions.add_(7)
            slots[1:].add_(3)
            pool.fill_(173)
            reference_pool = pool.clone()
            expected = self.run_path(attn, x, positions, slots, reference_pool, False)
            expected_index_q = attn.index_wq(expected.qr)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            self.assert_qkv_equal(actual, expected)
            torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
            torch.testing.assert_close(index_q, expected_index_q, rtol=0, atol=0)

    @torch.no_grad()
    def test_attention_body_falls_back_for_strided_slots_without_double_write(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8

        attn, x, positions, slots = native_attention(4, 6)
        attn.compress_ratio = 0
        attn._forward_decode_swa_only = lambda q, *_args: q
        attn._project_output = lambda q, _freqs: q[:, :, 0, :].contiguous()
        attn._prefill_output_all_reduce = lambda _out: None
        original_compute = compute_qkv.decode_compute_qkv
        for strided in (False, True):
            if strided:
                backing = torch.empty(
                    slots.numel() * 2, device="cuda", dtype=slots.dtype
                )
                current_slots = backing[::2]
                current_slots.copy_(slots)
            else:
                current_slots = slots
            pool = torch.full((2, 134, 528), 173, device="cuda", dtype=torch.uint8)
            reference_pool = pool.clone()
            expected = self.run_path(attn, x, positions, slots, reference_pool, False)
            attn._pool_view_3d_fp8 = lambda _region: pool
            metadata = SimpleNamespace(
                position_ids=positions,
                pool_write_slot_mappings={SWA_KV: current_slots},
            )
            computed = []

            def capture_qkv(*args, **kwargs):
                result = original_compute(*args, **kwargs)
                computed.append(result)
                return result

            with patch.dict(os.environ, flags(True)), patch.object(
                compute_qkv, "decode_compute_qkv", side_effect=capture_qkv
            ), patch.object(
                attn, "_decode_write_swa_fp8", wraps=attn._decode_write_swa_fp8
            ) as old_write:
                output = AttentionV41FP8._forward_decode_body(attn, x, metadata)
            self.assertEqual(old_write.call_count, int(strided))
            self.assertEqual(computed[0].swa_written, not strided)
            self.assert_qkv_equal(computed[0], expected)
            torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
            torch.testing.assert_close(output, expected.q[:, :, 0, :], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
