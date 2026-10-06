"""Real conversion and packed-KV4 wrapper checks; no model/service traffic."""

import json
import os
import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    quantize_main_index_rows_to_planes,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import topk_bt_fused as op
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.nvfp4_q8_index_score import (
    q8kv4_prefill_index_score,
)


@triton.jit
def _training_scale_one(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Exact scale-one arithmetic from the checked-in training quantizer:
    # load BF16 -> FP32 / scale -> BF16 -> E4M3.
    value = tl.load(src + offsets, offsets < N, other=0.0).to(tl.float32)
    value = (value / 1.0).to(tl.bfloat16)
    tl.store(dst + offsets, value.to(tl.float8e4nv), offsets < N)


def delta(a, b):
    a, b = a.float(), b.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm()).item(),
        "unequal": (a != b).sum().item(),
    }


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Q8KV4PrefillContractTest(unittest.TestCase):
    report = {}

    def test_actual_cast_matches_training_bytes_and_chunks(self):
        torch.manual_seed(1001)
        for rows in (0, 1, 17, 128, 129):
            with self.subTest(rows=rows):
                source = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
                special = torch.tensor(
                    [
                        0.0,
                        -0.0,
                        1 / 512,
                        -1 / 512,
                        448.0,
                        -448.0,
                        480.0,
                        -480.0,
                        512.0,
                        -512.0,
                        3000.0,
                        -3000.0,
                        0.0625,
                        -0.0625,
                    ],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                if rows:
                    source.reshape(-1)[: len(special)] = special
                before = source.view(torch.uint8).clone()
                full = torch.empty_like(source, dtype=torch.float8_e4m3fn)
                expected = torch.empty_like(full)
                address = full.data_ptr()
                self.assertIs(op.scale1_query_fp8(source, full), full)
                if rows:
                    _training_scale_one[(triton.cdiv(source.numel(), 1024),)](
                        source, expected, source.numel(), 1024
                    )
                    torch.testing.assert_close(
                        full.view(torch.uint8),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    saturated = special.clamp(-448, 448).to(torch.float8_e4m3fn)
                    torch.testing.assert_close(
                        full.reshape(-1)[: len(special)].view(torch.uint8),
                        saturated.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(full.reshape(-1).view(torch.uint8)[1].item(), 128)
                for chunk in (7, 128):
                    tiled = torch.empty_like(full)
                    for start in range(0, rows, chunk):
                        op.scale1_query_fp8(
                            source[start : start + chunk], tiled[start : start + chunk]
                        )
                    torch.testing.assert_close(
                        tiled.view(torch.uint8), full.view(torch.uint8), rtol=0, atol=0
                    )
                self.assertEqual(full.data_ptr(), address)
                torch.testing.assert_close(
                    source.view(torch.uint8), before, rtol=0, atol=0
                )

    def test_cast_input_contract(self):
        source = torch.zeros(1, 4, 128, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            op.scale1_query_fp8(source.float(), out)
        with self.assertRaises(ValueError):
            op.scale1_query_fp8(source.transpose(1, 2), out.transpose(1, 2))
        with self.assertRaises(ValueError):
            op.scale1_query_fp8(source, out.bfloat16())

    def test_optional_output_abi_cached_per_callable(self):
        def legacy(q):
            return q

        def caller_owned(q, *, out=None):
            return out

        op._nvfp4_attention_supports_out.cache_clear()
        with patch.object(
            op.inspect, "signature", wraps=op.inspect.signature
        ) as signature:
            self.assertFalse(op._nvfp4_attention_supports_out(legacy))
            self.assertTrue(op._nvfp4_attention_supports_out(caller_owned))
            self.assertFalse(op._nvfp4_attention_supports_out(legacy))
            self.assertTrue(op._nvfp4_attention_supports_out(caller_owned))
            self.assertEqual(signature.call_count, 2)
        op._nvfp4_attention_supports_out.cache_clear()

    def test_optional_flat_combine_abi_and_shape_contract(self):
        def legacy(q, *, out=None):
            return out

        def flat(q, *, out=None, combine_cu_seqlens_q=None):
            return out

        op._nvfp4_attention_supports_flat_combine.cache_clear()
        with patch.object(
            op.inspect, "signature", wraps=op.inspect.signature
        ) as signature:
            for _ in range(2):
                self.assertFalse(op._nvfp4_attention_supports_flat_combine(legacy))
                self.assertTrue(op._nvfp4_attention_supports_flat_combine(flat))
            self.assertEqual(signature.call_count, 2)
        op._nvfp4_attention_supports_flat_combine.cache_clear()
        # Old optional ABI must not be mistaken for the new TopK32 contract.
        self.assertFalse(op._nvfp4_attention_supports_flat_combine(flat, 32))

        def flat32(q, *, out=None, combine_cu_seqlens_q=None):
            return out

        flat32.supported_flat_combine_topks = (16, 32)
        self.assertTrue(op._nvfp4_attention_supports_flat_combine(flat32, 32))
        self.assertTrue(op._nvfp4_attention_supports_flat_combine(flat32, 16))
        self.assertFalse(op._nvfp4_attention_supports_flat_combine(flat32, 8))
        self.assertTrue(op._flat_combine_shape_supported(64, 128, 16, torch.bfloat16))
        self.assertTrue(op._flat_combine_shape_supported(64, 128, 32, torch.bfloat16))
        for heads, dim, topk, dtype in (
            (32, 128, 16, torch.bfloat16),
            (64, 64, 16, torch.bfloat16),
            (64, 128, 4, torch.bfloat16),
            (64, 128, 16, torch.float32),
        ):
            self.assertFalse(op._flat_combine_shape_supported(heads, dim, topk, dtype))

    def test_real_writer_q8k4_index_and_native_prefill_wrapper(self):
        torch.manual_seed(1001)
        rows, tokens, pages, heads, query_heads = 17, 512, 4, 4, 64
        k = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        index_k = torch.randn(tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(rows, query_heads, 128, device="cuda", dtype=torch.bfloat16)
        iq = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
        permutation = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
        positions = torch.arange(tokens, device="cuda")
        slots = (permutation[positions // 128].long() * 128 + positions % 128).long()
        kp = torch.zeros(pages, heads, 128, 64, device="cuda", dtype=torch.uint8)
        vp = torch.zeros_like(kp)
        ks = torch.zeros(pages, heads, 128, 8, device="cuda", dtype=torch.float8_e4m3fn)
        vs = torch.zeros_like(ks)
        ip = torch.zeros(pages, 1, 128, 64, device="cuda", dtype=torch.uint8)
        isc = torch.zeros(pages, 1, 128, 8, device="cuda", dtype=torch.float8_e4m3fn)
        quantize_main_index_rows_to_planes(
            k, v, index_k, slots, kp, ks, vp, vs, ip, isc
        )
        cuq = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([tokens], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([tokens - rows], device="cuda", dtype=torch.int32)
        iq8 = torch.empty_like(iq, dtype=torch.float8_e4m3fn)
        op.scale1_query_fp8(iq, iq8)
        idx_scale_mma = isc.view(pages, 1, 2, 32, 4, 4)
        cu_page_offsets = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
        rawscore = torch.empty(heads, rows, pages, device="cuda", dtype=torch.float32)
        q8kv4_prefill_index_score(
            iq8,
            ip,
            idx_scale_mma,
            cuq,
            lengths,
            prefixes,
            cu_page_offsets,
            permutation,
            rawscore,
            max_seqlen_q=rows,
        )
        # Independent nibble/MMA-scale decode; do not use the production reader
        # as its own oracle.
        codes = torch.empty(pages, 1, 128, 128, device="cuda", dtype=torch.uint8)
        codes[..., ::2], codes[..., 1::2] = ip & 15, ip >> 4
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
        value = lut[(codes & 7).long()]
        value = torch.where(codes & 8 != 0, -value, value)
        row = torch.arange(128, device="cuda")[:, None]
        group = torch.arange(8, device="cuda")[None, :]
        offsets = group // 4 * 512 + row % 32 * 16 + row // 32 * 4 + group % 4
        scale = isc.float().reshape(pages, 1, 1024)[..., offsets]
        decoded = (
            (value * scale.repeat_interleave(16, -1))
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        logical_k = decoded[permutation.long(), 0].reshape(tokens, 128).float()
        oracle = torch.einsum("thd,kd->htk", iq8.float(), logical_k)
        valid = torch.arange(tokens, device="cuda")[None, :] <= (
            torch.arange(rows, device="cuda")[:, None] + tokens - rows
        )
        oracle.masked_fill_(~valid[None], float("-inf"))
        oracle = oracle.reshape(heads, rows, pages, 128).amax(-1)
        torch.testing.assert_close(rawscore, oracle, rtol=1e-5, atol=1e-5)
        # Both producer contracts must reach the same real packed-K4 score and
        # production TopK; pre-cast Q must not allocate or invoke the cast path.
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
            PrefillScoreHostMetadata,
        )

        plans = [
            {
                "_fp4_host_metadata": PrefillScoreHostMetadata(
                    (rows,), (tokens,), (tokens - rows,), (0,)
                )
            }
            for _ in range(2)
        ]

        def index(value, plan):
            return op.flash_prefill_topk_to_block_tables_fp4(
                value,
                ip,
                idx_scale_mma,
                cuq,
                lengths,
                prefixes,
                rows,
                tokens,
                128,
                4,
                pages,
                index_score_plan=plan,
                kv_indices=permutation,
                emit_block_table=True,
            )

        projected_index = index(iq, plans[0])
        with patch.object(
            op, "scale1_query_fp8", side_effect=AssertionError("Q8 must skip cast")
        ):
            precast_index = index(iq8, plans[1])
        for a, b in zip(projected_index, precast_index):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertIn("_q8_index_query_buffer", plans[0])
        self.assertNotIn("_q8_index_query_buffer", plans[1])
        self.report["index_bf16_projection_cast_vs_precast_q8"] = (
            "exact TopK/block tables/seqlens"
        )
        topk = rawscore.topk(4, dim=-1).indices.int().contiguous()
        q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        op.scale1_query_fp8(q, q8)
        originals = [x.view(torch.uint8).clone() for x in (q, kp, vp, ks, vs)]

        def main(value, chunk):
            plan = op.build_sparse_attn_plan(
                cuq, lengths, prefixes, query_heads, heads, 128, 4
            )
            with patch.dict(
                os.environ,
                {
                    "M3_SPARSE_ATTN_CHUNK_SIZE": str(chunk),
                    "M3_SPARSE_ATTN_CHUNK_ENABLE": "1",
                },
            ):
                result = op.sparse_prefill_from_topk_fp4(
                    value,
                    kp,
                    vp,
                    ks.view(torch.uint8).view(-1, 8),
                    vs.view(torch.uint8).view(-1, 8),
                    topk,
                    permutation,
                    plan,
                    4,
                    128,
                    128**-0.5,
                )
            return result, plan

        full, _ = main(q8, rows)
        projected, plan = main(q, 7)
        precast, _ = main(q8, 7)
        self.assertTrue(torch.isfinite(projected).all().item())
        torch.testing.assert_close(projected, precast, rtol=0, atol=0)
        torch.testing.assert_close(projected, full, rtol=0.015625, atol=0.0009765625)
        for x, before in zip((q, kp, vp, ks, vs), originals):
            torch.testing.assert_close(x.view(torch.uint8), before, rtol=0, atol=0)
        # Call the actual leaf wrapper with the real CSR builder. It must
        # reject BF16 before any packed-KV4 attention calculation.
        meta = plan["_chunk_meta_fp4"]
        c = meta["chunks"][0]
        words = meta["ws_csr_words"]
        ws = torch.empty(words, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "requires Q8"):
            op.run_sparse_attn_chunk(
                q[c["g0"] : c["g1"]],
                kp,
                vp,
                topk[:, c["g0"] : c["g1"]].contiguous(),
                c,
                c["pt"],
                builder=meta["builder"],
                topk=4,
                block_size_k=128,
                sm_scale=128**-0.5,
                causal=True,
                partial_dtype=torch.bfloat16,
                usable_sm=-1,
                ws_csr=ws,
                ws_fwd=ws[:0].view(torch.uint8),
                out=projected[: c["csz"]],
                k_scale_128x4=ks.view(torch.uint8).view(-1, 8),
                v_scale_128x4=vs.view(torch.uint8).view(-1, 8),
            )
        self.report["index_q8k4_direct_vs_oracle"] = delta(rawscore, oracle)
        self.report["main_bf16_projection_cast_vs_precast_q8"] = delta(
            projected, precast
        )
        self.report["main_chunk7_vs_full17"] = delta(projected, full)
        self.report["boundary"] = (
            "real writer, direct packed-K4 Q8 IndexScore, actual native packed-KV4 attention wrapper; not model E2E"
        )


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if torch.cuda.is_available():
        Q8KV4PrefillContractTest.report.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        print(json.dumps(Q8KV4PrefillContractTest.report, indent=2))
    raise SystemExit(not result.wasSuccessful())
