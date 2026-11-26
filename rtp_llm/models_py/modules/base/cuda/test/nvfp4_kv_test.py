import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.nvfp4_kv import (
    INDEXER_ROW_BYTES,
    MLA_ROW_BYTES,
    gather_indexer_fp8,
    indexer_paged_fp8,
    nvfp4_mla_rows_to_fp8,
    pack_nvfp4_rows,
    unpack_nvfp4_rope,
    unpack_nvfp4_rows,
    write_nvfp4_rows,
)


def decode_fp8(data, scale):
    return data.view(torch.float8_e4m3fn).float() * scale


class Nvfp4KVTest(unittest.TestCase):
    def test_e2m1_layout(self):
        values = (
            torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0],
                dtype=torch.bfloat16,
            )
            .repeat(8)
            .view(1, 128)
        )
        packed = pack_nvfp4_rows(values)
        self.assertEqual(packed.shape, (1, INDEXER_ROW_BYTES))
        self.assertEqual(
            packed[0, :8].tolist(), [0x10, 0x32, 0x54, 0x76, 0xA9, 0xCB, 0xED, 0x0F]
        )
        torch.testing.assert_close(unpack_nvfp4_rows(packed, 128), values.float())

    def test_mla_fp8_staging_preserves_rope_and_all_groups(self):
        gen = torch.Generator().manual_seed(234)
        values = torch.randn(5, 512, generator=gen, dtype=torch.bfloat16)
        values[3].zero_()
        values[4, 0:16] *= 1000
        values[4, 16:32] *= 0.01
        rope = torch.randn(5, 64, generator=gen, dtype=torch.bfloat16)
        rope[3].zero_()
        packed = pack_nvfp4_rows(values, rope)
        self.assertEqual(packed.shape, (5, MLA_ROW_BYTES))
        self.assertEqual(MLA_ROW_BYTES, 336)
        self.assertTrue((packed[:, 332:] == 0).all())
        unpacked_rope = unpack_nvfp4_rope(packed)
        self.assertEqual(unpacked_rope.dtype, torch.bfloat16)
        self.assertEqual(unpacked_rope.shape, (5, 64))
        self.assertTrue((unpacked_rope[3] == 0).all())
        cos = torch.nn.functional.cosine_similarity(
            unpacked_rope[:3].float(), rope[:3].float(), dim=-1
        )
        self.assertTrue((cos > 0.985).all(), f"RoPE NVFP4 cosine too low: {cos}")
        # Verify de-interleave block grouping: even channels [0::2] and odd channels [1::2]
        # occupy independent 16-element NVFP4 blocks (0..1 vs 2..3).
        asym_rope = torch.zeros(1, 64, dtype=torch.bfloat16)
        asym_rope[0, 0::2] = 0.25
        asym_rope[0, 1::2] = 256.0
        asym_unpacked = unpack_nvfp4_rope(pack_nvfp4_rows(values[:1], asym_rope))
        torch.testing.assert_close(
            asym_unpacked[0, 0::2], asym_rope[0, 0::2], rtol=0.05, atol=0.01
        )
        # Use the actual NVFP4 quantized values as the transcoding reference.
        rows = nvfp4_mla_rows_to_fp8(packed[[4, 0, 3, 4]])
        scales = rows[:, 512:528].view(torch.float32)
        # UE8M0 scales must be exact powers of two (mantissa == 0).
        self.assertTrue(((scales.view(torch.int32) & 0x007FFFFF) == 0).all())
        decoded = decode_fp8(
            rows[:, :512].reshape(-1, 4, 128),
            scales.unsqueeze(-1),
        ).flatten(1)
        expected = unpack_nvfp4_rows(packed, 512)[[4, 0, 3, 4]]
        torch.testing.assert_close(decoded, expected, rtol=0.08, atol=0.01)
        self.assertTrue(torch.isfinite(decoded).all())
        torch.testing.assert_close(
            rows[:, 528:].view(torch.bfloat16),
            unpacked_rope[[4, 0, 3, 4]],
            rtol=0,
            atol=0,
        )

    def _indexer_cache(self, device):
        values = torch.ones((8, 128), dtype=torch.bfloat16, device=device)
        values *= torch.arange(1, 9, device=device)[:, None]
        cache = torch.full(
            (3, 4, INDEXER_ROW_BYTES), 190, dtype=torch.uint8, device=device
        )
        slots = torch.tensor([2, 3, 4, 5, 6, 7, 8, -1], device=device)
        write_nvfp4_rows(cache, pack_nvfp4_rows(values), slots)
        return cache

    def _check_indexer_layouts(self, device):
        cache = self._indexer_cache(device)
        self.assertTrue((cache.view(-1, INDEXER_ROW_BYTES)[0] == 190).all())
        block_table = torch.tensor([[1, 0], [2, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([[2, 3], [0, 1]], dtype=torch.int32, device=device)
        paged, remap = indexer_paged_fp8(cache, block_table, context_lens, 4, True)
        self.assertEqual(remap.tolist(), [[0, 1], [2, 3]])
        self.assertEqual(paged.shape, (4, 4, 1, 132))
        pages = paged.view(4, -1)
        data = pages[:, :512].view(4, 4, 128)
        scale = pages[:, 512:].view(torch.float32).unsqueeze(-1)
        actual = decode_fp8(data, scale)
        expected = (
            torch.tensor(
                [[3, 4, 5, 0], [0, 0, 0, 0], [7, 0, 0, 0], [0, 0, 0, 0]],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(-1)
            .expand_as(actual)
        )
        torch.testing.assert_close(actual, expected, rtol=0.065, atol=0.01)

        data = torch.empty((4, 128), dtype=torch.float8_e4m3fn, device=device)
        scale_bytes = torch.empty((4, 4), dtype=torch.uint8, device=device)
        cu_lens = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
        gather_indexer_fp8(cache, data, scale_bytes, block_table, cu_lens, 4, False)
        actual = decode_fp8(data, scale_bytes.view(torch.float32))
        expected = torch.tensor([3, 4, 5, 7], device=device, dtype=torch.float32)
        torch.testing.assert_close(
            actual, expected[:, None].expand_as(actual), rtol=0.065, atol=0.01
        )

    def test_indexer_paged_and_ragged_cpu(self):
        self._check_indexer_layouts("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_indexer_paged_and_ragged_cuda(self):
        self._check_indexer_layouts("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_indexer_cuda_graph_live_slots_and_lengths(self):
        cache = self._indexer_cache("cuda")
        keys = torch.full((2, 128), 1.5, dtype=torch.bfloat16, device="cuda")
        slots = torch.tensor([4, -1], device="cuda")
        block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
        context_lens = torch.tensor([[1]], dtype=torch.int32, device="cuda")

        def stage():
            write_nvfp4_rows(cache, pack_nvfp4_rows(keys), slots)
            return indexer_paged_fp8(cache, block_table, context_lens, 4, True)

        stage()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged, remap = stage()

        keys.fill_(6)
        slots[0] = 5
        context_lens.fill_(2)
        graph.replay()
        page = paged[0].view(-1)
        decoded = decode_fp8(
            page[:512].view(4, 128), page[512:].view(torch.float32).unsqueeze(-1)
        )
        expected = torch.tensor([1.5, 6, 0, 0], dtype=torch.float32, device="cuda")
        torch.testing.assert_close(
            decoded, expected[:, None].expand_as(decoded), rtol=0.065, atol=0.01
        )
        self.assertEqual(remap.tolist(), [[0, 1]])

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_indexer_forward_deepgemm_and_mtp_graph(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("Requires a Blackwell GPU")
        from rtp_llm.models_py.modules.base.cuda import indexer_op

        if indexer_op.deep_gemm is None:
            self.skipTest("Requires DeepGEMM")
        # The winning 2048 keys have strictly higher scores than the other
        # 128 keys, so this exercises actual TopK and its local index domain.
        seq_len = 2176
        topk = 2048
        pages = seq_len // 64
        lengths = torch.full((2,), seq_len, device="cuda", dtype=torch.int32)
        block_table = torch.arange(1, 2 * pages + 1, device="cuda", dtype=torch.int32)
        block_table = block_table.view(2, pages)
        q = torch.ones((4, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        weights = torch.ones((4, 32), device="cuda")
        slots = torch.arange(64, 64 + 2 * seq_len, dtype=torch.int64, device="cuda")
        values = torch.full((2, seq_len, 128), 4.0, device="cuda", dtype=torch.bfloat16)
        fmha = SimpleNamespace(
            kvlen_d=lengths,
            expanded_seq_lens=lengths.repeat_interleave(2),
        )
        attn = SimpleNamespace(
            kv_cache_kernel_block_id_device=block_table, is_target_verify=True
        )
        ragged = SimpleNamespace(
            kv_cache_kernel_block_id_device=block_table,
            cu_kv_seqlens=torch.tensor(
                [0, seq_len, 2 * seq_len], device="cuda", dtype=torch.int32
            ),
        )
        prefill = SimpleNamespace(
            prefill_total_kv_tokens=2 * seq_len,
            prefill_max_kv_len=seq_len,
            ks=ragged.cu_kv_seqlens[:2],
            ke=ragged.cu_kv_seqlens[1:],
            topk_indices_offset=torch.zeros((2,), device="cuda", dtype=torch.int32),
        )

        for nvfp4 in (False, True):
            with self.subTest(nvfp4=nvfp4), patch.object(
                indexer_op, "_INDEXER_FP8_STAGING_BYTES", seq_len * 132
            ), patch.dict("os.environ", {"RTP_LLM_ADAPTIVE_VERIFICATION": "0"}):
                op = indexer_op.IndexerOp(32, 128, topk, 64, nvfp4_kv_cache=nvfp4)
                cache = SimpleNamespace(
                    kv_scale_base=torch.empty(
                        (2 * pages + 1, 64, INDEXER_ROW_BYTES if nvfp4 else 132),
                        device="cuda",
                        dtype=torch.uint8,
                    )
                )
                values.fill_(4)
                values[:, : seq_len - topk].fill_(1)
                op.quant_k_only(values.flatten(0, 1), cache, slots)
                expected = torch.arange(
                    seq_len - topk, seq_len, device="cuda", dtype=torch.int32
                )
                result = op._get_topk_ragged(q[:2], weights[:2], cache, prefill, ragged)
                torch.testing.assert_close(
                    result.sort(1).values, expected.expand(2, -1)
                )
                result = op._get_topk_paged(q, weights, cache, fmha, attn)
                torch.testing.assert_close(
                    result.sort(1).values, expected.expand(4, -1)
                )

                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = op._get_topk_paged(q, weights, cache, fmha, attn)
                values.fill_(4)
                values[:, topk:].fill_(1)
                op.quant_k_only(values.flatten(0, 1), cache, slots)
                graph.replay()
                expected = torch.arange(topk, device="cuda", dtype=torch.int32)
                torch.testing.assert_close(
                    result.sort(1).values, expected.expand(4, -1)
                )


if __name__ == "__main__":
    unittest.main()
