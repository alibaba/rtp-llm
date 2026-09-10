"""K3 ordinary FP8 MLA: cache, ragged Prefill, Decode and Verify contracts."""

import os
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    gather_fp8_prefix,
    quantize_fp8,
)

_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
if _TEST_TMPDIR:
    os.environ.setdefault("DG_JIT_CACHE_DIR", os.path.join(_TEST_TMPDIR, "deep_gemm"))


class MlaFp8Test(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires CUDA; no passing by skip")
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        torch.manual_seed(101)

    def test_cp_prefix_pack_padded_cache(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
            _pack_cp_prefix,
        )

        logical = torch.randn(6, 128, 576, device="cuda").bfloat16()
        request = torch.tensor([0, 1, 1, 0], dtype=torch.int64, device="cuda")
        column = torch.tensor([1, 0, 2, 3], dtype=torch.int64, device="cuda")
        offset = torch.tensor([127, 0, 67, 63], dtype=torch.int64, device="cuda")
        for dtype, scale in ((torch.bfloat16, 1.0), (torch.float8_e4m3fn, 16.0)):
            storage = torch.full((7, 128 * 576 + 64), float("nan"), dtype=dtype, device="cuda")
            cache = storage[1:, :128 * 576].view(6, 128, 576)
            cache.copy_((logical.float() / scale).clamp(-448, 448).to(dtype))
            table_storage = torch.full((2, 7), -1, dtype=torch.int32, device="cuda")
            table = table_storage[:, :4]
            output = torch.full((7, 576), -123.0, dtype=torch.bfloat16, device="cuda")
            for pages in ([[3, 1, 5, 2], [4, 2, 1, 5]], [[2, 5, 1, 3], [5, 3, 4, 2]]):
                table.copy_(torch.tensor(pages, dtype=torch.int32, device="cuda"))
                _pack_cp_prefix[(4,)](
                    cache, table, request, column, offset, output,
                    table.stride(0), table.stride(1), cache.stride(0), cache.stride(1),
                    576, scale, 1024,
                )
                expected = (cache.float()[table[request, column].long(), offset] * scale).bfloat16()
                torch.testing.assert_close(output[:4], expected, atol=0, rtol=0)
                self.assertTrue(torch.all(output[4:] == -123))
                self.assertTrue(torch.all(table_storage[:, 4:] == -1))
                self.assertTrue(torch.isnan(storage[:, 128 * 576:].float()).all())

    def test_quantizer_saturation_noncontiguous_and_graph(self):
        x = (
            torch.linspace(-1000, 1000, 4096, device="cuda")
            .reshape(32, 128)
            .bfloat16()
            .t()
        )
        expected = (x.float() / 2).clamp(-448, 448).to(torch.float8_e4m3fn)
        actual = quantize_fp8(x, 2)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
        source = x.contiguous()
        for _ in range(3):
            quantize_fp8(source, 2, actual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantize_fp8(source, 2, actual)
        for value in (0.25, -17.0, 999.0):
            source.fill_(value)
            graph.replay()
            ref = (source.float() / 2).clamp(-448, 448).to(torch.float8_e4m3fn)
            torch.testing.assert_close(actual.float(), ref.float(), atol=0, rtol=0)

    def test_quantizer_preserves_nan_and_saturates_infinity(self):
        x = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0], device="cuda"
        )
        actual = quantize_fp8(x).float()
        self.assertTrue(torch.isnan(actual[0]).item())
        torch.testing.assert_close(
            actual[1:], torch.tensor([448.0, -448.0, 0.0], device="cuda")
        )

    def test_paged_prefix_and_suffix_nonunit_scale(self):
        page_size, latent, rope_dim = 128, 512, 64
        # Noncontiguous page strides, shuffled pages and partially filled pages.
        storage = quantize_fp8(
            torch.randn(8, page_size + 1, 576, device="cuda").bfloat16()
        )
        cache = storage[:, :page_size]
        pages = torch.tensor([5, 1, 7], device="cuda", dtype=torch.int32)
        info = torch.tensor(
            [[0, 133, 0, 2], [1, 17, 2, 1]], device="cuda", dtype=torch.int32
        )
        q_ind = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        new = torch.randn(5, 576, device="cuda").bfloat16()
        c, r = new[:, :latent], new[:, latent:]
        out_c = torch.empty(155, latent, device="cuda", dtype=torch.bfloat16)
        out_r = torch.empty(155, rope_dim, device="cuda", dtype=torch.bfloat16)
        gather_fp8_prefix(
            out_c, out_r, c, r, cache, pages, info, q_ind, page_size, scale=0.5
        )
        expected = torch.cat(
            [
                (cache[5].float() * 0.5).bfloat16(),
                (cache[1, :5].float() * 0.5).bfloat16(),
                new[:3],
                (cache[7, :17].float() * 0.5).bfloat16(),
                new[3:],
            ]
        )
        torch.testing.assert_close(out_c, expected[:, :latent], atol=0, rtol=0)
        torch.testing.assert_close(out_r, expected[:, latent:], atol=0, rtol=0)

    def test_full_and_bounded_prefix_routes_agree(self):
        from unittest import mock

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
            MlaFlashMLAPrefillOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_dense_prefill_packed_kv_test import (
            FlashMLADensePrefillPackedKVTest,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_forward_test_utils import (
            EXPANDED_KV_BYTES_PER_TOKEN,
            DeterministicPackedProjection,
            output_and_lse,
        )
        from rtp_llm.ops import KvCacheDataType

        # Nonzero page offsets, partial pages, ragged Q and multiple launches.
        for q_lens, prefix_lens in (((2, 3), (0, 300)), ((1, 1), (127, 640))):
            inputs = FlashMLADensePrefillPackedKVTest._make_inputs(q_lens, prefix_lens)
            inputs.kv_cache.kv_cache_base = quantize_fp8(
                inputs.kv_cache.kv_cache_base, 0.5
            )
            results = []
            for capacity in (0, 256):
                op = MlaFlashMLAPrefillOp(
                    num_heads=12,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=128,
                    v_head_dim=128,
                    page_size=128,
                    softmax_extra_scale=1.0,
                    use_mla=True,
                    weights=[{}],
                    kv_cache_dtype=KvCacheDataType.FP8,
                    fp8_compute=True,
                    q_scale=0.5,
                    kv_scale=0.5,
                    expanded_kv_budget_bytes=capacity * EXPANDED_KV_BYTES_PER_TOKEN,
                )
                op.plan(inputs.params)
                with mock.patch.object(
                    op,
                    "_create_kv_b_proj",
                    return_value=DeterministicPackedProjection(),
                ):
                    output, lse = output_and_lse(op, inputs)
                    repeated, repeated_lse = output_and_lse(op, inputs)
                torch.testing.assert_close(output, repeated, atol=0, rtol=0)
                torch.testing.assert_close(lse, repeated_lse, atol=0, rtol=0)
                results.append((output, lse))
                if capacity:
                    self.assertLessEqual(op._fp8_prefix_rope.shape[0], capacity)
            torch.testing.assert_close(
                results[0][0].float(), results[1][0].float(), atol=2e-3, rtol=3e-2
            )
            torch.testing.assert_close(
                results[0][1], results[1][1], atol=2e-3, rtol=2e-3
            )

    def test_quantized_producer_full_and_bounded_prefix(self):
        from dataclasses import replace
        from unittest import mock

        from rtp_llm.config.quant_config import init_quant_config
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
            MlaFlashMLAPrefillOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_dense_prefill_packed_kv_test import (
            FlashMLADensePrefillPackedKVTest,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_forward_test_utils import (
            EXPANDED_KV_BYTES_PER_TOKEN,
            output_and_lse,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import rmsnorm_fp8
        from rtp_llm.ops import KvCacheDataType

        config = init_quant_config("FP8_PER_BLOCK")
        projection = CudaFp8DeepGEMMLinear(
            weight=(torch.randn(3072, 512, device="cuda") * 0.03).to(
                torch.float8_e4m3fn
            ),
            weight_scales=torch.full(
                (1, 3072), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).T,
            input_scales=None,
            bias=None,
            quant_config=config,
        )
        for fp8_cache in (False, True):
            for prefix_lens in ((0, 0), (0, 300), (127, 640)):
                inputs = FlashMLADensePrefillPackedKVTest._make_inputs(
                    (2, 3), prefix_lens
                )
                payload = rmsnorm_fp8(
                    inputs.compressed_kv,
                    torch.ones(512, device="cuda", dtype=torch.bfloat16),
                    1.0e-6,
                    retain_bf16=True,
                )
                if fp8_cache and inputs.kv_cache is not None:
                    inputs.kv_cache.kv_cache_base = quantize_fp8(
                        inputs.kv_cache.kv_cache_base, 0.5
                    )
                for capacity in (0, 256):
                    results = []
                    for fused in (False, True):
                        op = MlaFlashMLAPrefillOp(
                            num_heads=12,
                            kv_lora_rank=512,
                            qk_rope_head_dim=64,
                            qk_nope_head_dim=128,
                            v_head_dim=128,
                            page_size=128,
                            softmax_extra_scale=1.0,
                            use_mla=True,
                            weights=[{}],
                            quant_config=config if fused else None,
                            kv_cache_dtype=(
                                KvCacheDataType.FP8
                                if fp8_cache
                                else KvCacheDataType.BASE
                            ),
                            fp8_compute=fp8_cache,
                            q_scale=0.5,
                            kv_scale=0.5,
                            expanded_kv_budget_bytes=capacity
                            * EXPANDED_KV_BYTES_PER_TOKEN,
                        )
                        op.plan(inputs.params)
                        with mock.patch.object(
                            op, "_create_kv_b_proj", return_value=projection
                        ):
                            results.append(
                                output_and_lse(
                                    op,
                                    replace(
                                        inputs,
                                        compressed_kv=(
                                            payload if fused else payload.bf16
                                        ),
                                    ),
                                )
                            )
                    with self.subTest(
                        fp8_cache=fp8_cache, prefix=prefix_lens, capacity=capacity
                    ):
                        torch.testing.assert_close(
                            results[0][0], results[1][0], atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            results[0][1], results[1][1], atol=0, rtol=0
                        )

    def test_cache_writer_scale_page_boundary_and_skipped_slot(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import KvCacheDataType

        op = MlaKVCacheWriteOp(KvCacheDataType.FP8, True, True, 0.5)
        # One new page, one existing page, and a graph padding token.
        cache = torch.full((3, 128, 576), 7.0, device="cuda").to(torch.float8_e4m3fn)
        x = torch.randn(3, 576, device="cuda").bfloat16()
        slots = torch.tensor([128, 5, -1], device="cuda", dtype=torch.int64)
        op.forward(
            x[:, :512],
            x[:, 512:],
            SimpleNamespace(kv_cache_base=cache),
            SimpleNamespace(slot_mapping=slots),
        )
        expected = quantize_fp8(x, 0.5).float()
        torch.testing.assert_close(cache[1, 0].float(), expected[0], atol=0, rtol=0)
        torch.testing.assert_close(cache[0, 5].float(), expected[1], atol=0, rtol=0)
        self.assertEqual(cache[1, 1:].float().count_nonzero().item(), 0)
        self.assertTrue((cache[2].float() == 7).all().item())
        self.assertTrue((cache[0, 6:].float() == 7).all().item())

    def test_prefill_causal_and_noncausal_fp8_compute(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
            MlaFlashMLAPrefillOp,
        )
        from rtp_llm.ops import KvCacheDataType

        op = MlaFlashMLAPrefillOp(
            12,
            512,
            64,
            128,
            128,
            128,
            1.0,
            True,
            [{}],
            kv_cache_dtype=KvCacheDataType.FP8,
            fp8_compute=True,
            q_scale=0.5,
            kv_scale=0.25,
        )
        for causal in (False, True):
            q = torch.randn(11, 12, 192, device="cuda").bfloat16() * 0.1
            k = torch.randn(29, 12, 192, device="cuda").bfloat16() * 0.1
            v = torch.randn(29, 12, 128, device="cuda").bfloat16() * 0.1
            qi = torch.tensor([0, 4, 11], device="cuda", dtype=torch.int32)
            ki = torch.tensor([0, 12, 29], device="cuda", dtype=torch.int32)
            out, lse = op._run_dense_attention(
                q,
                k,
                v,
                qo_indptr=qi,
                kv_indptr=ki,
                max_q_len=7,
                max_kv_len=17,
                causal=causal,
            )
            qf = quantize_fp8(q, 0.5).float() * 0.5
            kf, vf = quantize_fp8(k).float(), quantize_fp8(v).float()
            for qs, qe, ks, ke in ((0, 4, 0, 12), (4, 11, 12, 29)):
                score = torch.einsum("qhd,khd->hqk", qf[qs:qe], kf[ks:ke]) * op.scale
                if causal:
                    allowed = (
                        torch.arange(ke - ks, device="cuda")[None, :]
                        <= (ke - ks - (qe - qs) + torch.arange(qe - qs, device="cuda"))[
                            :, None
                        ]
                    )
                    score.masked_fill_(~allowed, float("-inf"))
                # This small reference fits one 128-token KV tile. P@V is
                # also FP8: round unnormalized exponentials, retain the FP32
                # denominator, then perform PV in FP32 and compare BF16 output.
                p = (score - score.max(-1, keepdim=True).values).exp()
                p8 = p.to(torch.float8_e4m3fn).float()
                ref = (
                    torch.einsum("hqk,khd->qhd", p8, vf[ks:ke])
                    / p.sum(-1).t()[..., None]
                )
                torch.testing.assert_close(
                    out[qs:qe].float(), ref, atol=0.0005, rtol=0.01
                )
                torch.testing.assert_close(
                    lse[qs:qe], score.logsumexp(-1).t(), atol=0.003, rtol=0.002
                )

    def test_diagnostic_ranges_and_graph_exclusion(self):
        import json
        from unittest.mock import patch

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            mla_fp8_kernels as kernels,
        )

        x = torch.tensor(
            [-1000.0, 200.0, float("nan"), float("inf"), 0.0], device="cuda"
        )
        with patch.object(kernels, "_FP8_DIAGNOSTICS", True):
            with self.assertLogs(level="INFO") as logs:
                kernels.quantize_fp8(x, 2.0, name="diagnostic_test")
            row = json.loads(logs.output[-1].split("K3_MLA_FP8_RANGE ", 1)[1])
            self.assertEqual(row["nonfinite"], 2)
            self.assertEqual(row["clipped"], 1)
            self.assertEqual(row["finite_absmax"], 1000.0)
            self.assertEqual(row["clipped_fraction"], 0.2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = kernels.quantize_fp8(x, 2.0, name="diagnostic_capture")
            graph.replay()
            self.assertEqual(result[0].float().item(), -448.0)

    def test_decode_verify_fp8_kernel_page128(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
            _load_tokenspeed_mla,
        )

        self.assertTrue(_load_tokenspeed_mla())
        from tokenspeed_mla.mla_decode import tokenspeed_mla_decode

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
            _get_tokenspeed_workspace,
        )

        for q_len in (1, 2, 3, 4):
            query = quantize_fp8(
                torch.randn(1, q_len, 12, 576, device="cuda").bfloat16() * 0.1, 0.5
            )
            cache = quantize_fp8(
                torch.randn(2, 128, 576, device="cuda").bfloat16() * 0.1, 0.25
            )
            pages = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
            lens = torch.tensor([193], device="cuda", dtype=torch.int32)
            out = torch.empty(1, q_len, 12, 512, device="cuda", dtype=torch.bfloat16)
            workspace = _get_tokenspeed_workspace(
                torch.device("cuda", 0), 12, 512, q_len
            )
            tokenspeed_mla_decode(
                query=query,
                kv_cache=cache,
                workspace_buffer=workspace,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=pages,
                seq_lens=lens,
                max_seq_len=256,
                softmax_scale=192**-0.5 * 0.5 * 0.25,
                output_scale=0.25,
                out=out,
                is_var_seq=True,
                causal_mask=True,
                enable_pdl=False,
            )
            real_cache = torch.cat([cache[1].float(), cache[0].float()])[:193] * 0.25
            score = (
                torch.einsum("qhd,kd->hqk", query[0].float() * 0.5, real_cache)
                * 192**-0.5
            )
            allowed = (
                torch.arange(193, device="cuda")[None, :]
                <= (193 - q_len + torch.arange(q_len, device="cuda"))[:, None]
            )
            score.masked_fill_(~allowed, float("-inf"))
            ref = torch.einsum("hqk,kd->qhd", score.softmax(-1), real_cache[:, :512])
            torch.testing.assert_close(out[0].float(), ref, atol=0.002, rtol=0.02)

    def test_decode_adapter_graph_reuses_fp8_query_buffer(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
            TokenSpeedMlaDecodeOp,
        )
        from rtp_llm.utils.model_weight import W

        weights = [
            {
                W.mla_kc: torch.randn(12, 128, 512, device="cuda").bfloat16() * 0.01,
                W.mla_vc: torch.randn(12, 512, 128, device="cuda").bfloat16() * 0.01,
            }
        ]
        kwargs = dict(
            num_heads=12,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            token_per_block=128,
            softmax_extra_scale=1.0,
            weights=weights,
            max_bs=2,
            max_q_len=4,
            max_context_len=256,
            fp8_compute=True,
            q_scale=0.5,
            kv_scale=0.25,
        )
        eager = TokenSpeedMlaDecodeOp(**kwargs)
        captured = TokenSpeedMlaDecodeOp(**kwargs, is_cuda_graph=True)
        params = SimpleNamespace(
            qo_indptr_h=torch.tensor([0, 4, 8], dtype=torch.int32),
            kvlen_h=torch.tensor([193, 131], dtype=torch.int32),
            kvlen_d=torch.tensor([193, 131], dtype=torch.int32, device="cuda"),
            decode_page_indptr_h=torch.tensor([0, 2, 4], dtype=torch.int32),
            decode_page_indptr_d=torch.tensor(
                [0, 2, 4], dtype=torch.int32, device="cuda"
            ),
            page_indice_d=torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda"),
        )
        eager.plan(params)
        captured.plan(params)
        cache = SimpleNamespace(
            kv_cache_base=quantize_fp8(
                torch.randn(4, 128, 576, device="cuda").bfloat16() * 0.1, 0.25
            )
        )
        q = torch.randn(8, 12, 128, device="cuda").bfloat16() * 0.1
        rope = torch.randn(8, 12, 64, device="cuda").bfloat16() * 0.1
        for _ in range(3):
            captured.forward(q, rope, cache, 0)
        torch.cuda.synchronize()
        pointer = captured._q_fp8.data_ptr()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = captured.forward(q, rope, cache, 0)
        for _ in range(5):
            q.normal_(std=0.1)
            rope.normal_(std=0.1)
            graph.replay()
            reference = eager.forward(q, rope, cache, 0)
            self.assertEqual(captured._q_fp8.data_ptr(), pointer)
            self.assertEqual(captured._q_fp8.dtype, torch.float8_e4m3fn)
            self.assertEqual(output.dtype, torch.bfloat16)
            torch.testing.assert_close(output, reference, atol=0.0001, rtol=0.001)


if __name__ == "__main__":
    unittest.main()
