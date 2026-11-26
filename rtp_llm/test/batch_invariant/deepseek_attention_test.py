import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    roundtrip_swa_kv,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    compute_window_topk_and_length_varlen,
)
from rtp_llm.models_py.modules.dsv4.fp8.batch_invariant_attention import (
    indexed_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
    translate_local_to_global_slots,
)
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_triton, _v41_swa_triton
from rtp_llm.models_py.modules.dsv41.fp8._v41_batch_invariant_indexer import score_paged
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
    INDEXER_KV,
    AttentionV41FP8,
)
from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
    packed_fp8_paged_mqa_topk,
)

_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (8, 9)
)


@unittest.skipUnless(_CUDA, "requires NVIDIA SM89+")
class BatchInvariantIndexedAttentionTest(unittest.TestCase):
    def test_prefill_and_dual_pool_decode(self):
        self._run_dual_pool(v41=False)

    def test_v41_prefill_and_dual_pool_decode(self):
        self._run_dual_pool(v41=True)

    def _run_dual_pool(self, *, v41):
        torch.manual_seed(314159)
        device, dtype = "cuda", torch.bfloat16
        heads = 18
        requests = 3
        dims = 512
        swa = torch.randn((64, dims), device=device, dtype=dtype) * 0.25
        cmp = torch.randn((64, dims), device=device, dtype=dtype) * 0.25
        query = torch.randn((requests, heads, dims), device=device, dtype=dtype)
        sink = torch.randn(heads, device=device)

        slots = torch.arange(swa.size(0), device=device, dtype=torch.int64)
        if v41:
            swa_pool = torch.empty((2, 32, 528), dtype=torch.uint8, device=device)
            cmp_pool = torch.empty((4, 16, 288), dtype=torch.uint8, device=device)
            _v41_swa_triton.quantize_and_insert_swa_k_cache(swa, swa_pool, slots)
            _v41_fp4_triton.quantize_and_insert_k_cache_fp4(cmp, cmp_pool, slots)
            swa = _v41_swa_triton.roundtrip_swa_kv(swa)
            cmp = _v41_fp4_triton.dequantize_k_cache_slots_fp4(cmp_pool, slots)
        else:
            swa_pool = torch.empty((2, 32, 584), dtype=torch.uint8, device=device)
            cmp_pool = torch.empty((4, 16, 584), dtype=torch.uint8, device=device)
            quantize_and_insert_k_cache(swa, swa_pool, slots)
            quantize_and_insert_k_cache(cmp, cmp_pool, slots)
            swa, cmp = roundtrip_swa_kv(swa), roundtrip_swa_kv(cmp)

        swa_indices = torch.full((requests, 40), -1, device=device, dtype=torch.int32)
        cmp_indices = torch.full((requests, 36), -1, device=device, dtype=torch.int32)
        swa_counts = [39, 5, 0]
        cmp_counts = [11, 34, 0]
        prefill_indices = torch.full(
            (requests, 76), -1, device=device, dtype=torch.int32
        )
        for row, (swa_count, cmp_count) in enumerate(zip(swa_counts, cmp_counts)):
            swa_indices[row, :swa_count] = slots[row : row + swa_count]
            cmp_indices[row, :cmp_count] = slots[row : row + cmp_count]
            prefill_indices[row, :cmp_count] = cmp_indices[row, :cmp_count] + swa.size(
                0
            )
            prefill_indices[row, cmp_count : cmp_count + swa_count] = swa_indices[
                row, :swa_count
            ]
        swa_sizes = torch.tensor(swa_counts, device=device, dtype=torch.int32)
        cmp_sizes = torch.tensor(cmp_counts, device=device, dtype=torch.int32)
        prefill_kv = torch.cat((swa, cmp))
        scale = dims**-0.5
        prefill = indexed_attention(
            query,
            prefill_kv[:, None],
            prefill_indices,
            swa_sizes + cmp_sizes,
            sink,
            scale,
        )

        op = SparseAttnV4DecodeFp8Op(heads, dims, scale, batch_invariant=True)

        def decode(q, sw, cp, sw_len, cp_len):
            return op.forward(
                q[:, None],
                swa_pool,
                sink,
                sw[:, None],
                None,
                topk_length=sw_len,
                extra_k_cache=cmp_pool,
                extra_topk_idxs=cp[:, None],
                extra_topk_length=cp_len,
            )[:, 0]

        mixed = decode(query, swa_indices, cmp_indices, swa_sizes, cmp_sizes)
        self.assertTrue(torch.equal(prefill.view(torch.uint8), mixed.view(torch.uint8)))
        self.assertEqual(torch.count_nonzero(mixed[-1]).item(), 0)
        one = decode(
            *[x[:1] for x in (query, swa_indices, cmp_indices, swa_sizes, cmp_sizes)]
        )
        self.assertTrue(
            torch.equal(prefill[:1].view(torch.uint8), one.view(torch.uint8))
        )

        first_keys = prefill_kv[prefill_indices[0, : swa_counts[0] + cmp_counts[0]]]
        scores = query[0].float() @ first_keys.float().T * scale
        prob = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=1)[:, :-1]
        expected = (prob @ first_keys.float()).to(dtype)
        torch.testing.assert_close(prefill[0], expected, atol=0.002, rtol=0.02)

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = decode(query, swa_indices, cmp_indices, swa_sizes, cmp_sizes)
        for x in query, swa_indices, cmp_indices, swa_sizes, cmp_sizes:
            x.copy_(x.flip(0))
        graph.replay()
        self.assertTrue(
            torch.equal(captured.view(torch.uint8), prefill.flip(0).view(torch.uint8))
        )

    def test_full_windows_match_prefill_at_logical_page_boundaries(self):
        torch.manual_seed(871)
        device, dtype = "cuda", torch.bfloat16
        dims, heads, total, window, logical_page = 512, 32, 518, 128, 256
        int_options = {"device": device, "dtype": torch.int32}
        query = torch.randn((total, heads, dims), device=device, dtype=dtype) * 0.2
        swa = torch.randn((total, dims), device=device, dtype=dtype) * 0.2
        swa[:15] *= 1e-7
        sink = torch.randn(heads, device=device)
        positions = torch.arange(total, **int_options)
        indices, lengths = compute_window_topk_and_length_varlen(
            window,
            torch.tensor([0, total], **int_options),
            positions,
            torch.zeros(1, **int_options),
            torch.zeros(total, **int_options),
        )
        rows = torch.tensor(
            [1, 35, 126, 127, 128, 161, 163, 254, 255, 256, 386, 513, 517],
            **int_options,
        ).long()
        requests = rows.numel()
        table = torch.arange(1, 1 + requests * 4, **int_options).reshape(requests, 4)
        swa_indices = indices[rows]
        swa_lengths = lengths[rows]
        q = query[rows]
        slots = translate_local_to_global_slots(
            torch.arange(requests, **int_options),
            table,
            swa_indices,
            entries_per_block=window,
            tokens_per_block_for_block_table=logical_page,
        )
        valid = slots >= 0

        cmp = torch.randn((96, dims), device=device, dtype=dtype) * 0.2
        cmp_slots = torch.arange(cmp.size(0), device=device, dtype=torch.int64)
        cmp_lengths = ((rows + 1) // 4).clamp_max(cmp.size(0)).int()
        cmp_indices = cmp_slots.int().expand(requests, -1)
        cmp_indices = cmp_indices.masked_fill(cmp_indices >= cmp_lengths[:, None], -1)
        prefill_cmp_indices = torch.where(cmp_indices >= 0, cmp_indices + total, -1)
        prefill_indices = torch.cat((prefill_cmp_indices, swa_indices), dim=-1)
        prefill_indices = prefill_indices.gather(
            -1, (prefill_indices < 0).int().argsort(dim=-1, stable=True)
        )
        op = SparseAttnV4DecodeFp8Op(heads, dims, dims**-0.5, batch_invariant=True)

        for v41 in (False, True):
            with self.subTest(v41=v41):
                swa_pool = torch.zeros(
                    (requests * 4 + 1, window, 528 if v41 else 584),
                    device=device,
                    dtype=torch.uint8,
                )
                cmp_pool = torch.zeros(
                    (6, 16, 288 if v41 else 584), device=device, dtype=torch.uint8
                )
                if v41:
                    _v41_swa_triton.quantize_and_insert_swa_k_cache(
                        swa[swa_indices[valid]], swa_pool, slots[valid].long()
                    )
                    dense_swa = _v41_swa_triton.roundtrip_swa_kv(swa)
                    _v41_fp4_triton.quantize_and_insert_k_cache_fp4(
                        cmp, cmp_pool, cmp_slots
                    )
                    dense_cmp = _v41_fp4_triton.dequantize_k_cache_slots_fp4(
                        cmp_pool, cmp_slots
                    )
                else:
                    quantize_and_insert_k_cache(
                        swa[swa_indices[valid]], swa_pool, slots[valid].long()
                    )
                    dense_swa = roundtrip_swa_kv(swa)
                    quantize_and_insert_k_cache(cmp, cmp_pool, cmp_slots)
                    dense_cmp = roundtrip_swa_kv(cmp)
                prefill = indexed_attention(
                    query, dense_swa[:, None], indices, lengths, sink, dims**-0.5
                )[rows]
                decode = op.forward(
                    q[:, None],
                    swa_pool,
                    sink,
                    slots[:, None],
                    None,
                    topk_length=swa_lengths,
                )[:, 0]
                self.assertTrue(
                    torch.equal(prefill.view(torch.uint8), decode.view(torch.uint8))
                )

                prefill = indexed_attention(
                    q,
                    torch.cat((dense_swa, dense_cmp))[:, None],
                    prefill_indices,
                    swa_lengths + cmp_lengths,
                    sink,
                    dims**-0.5,
                )
                decode = op.forward(
                    q.flip(0)[:, None],
                    swa_pool,
                    sink,
                    slots.flip(0)[:, None],
                    None,
                    topk_length=swa_lengths.flip(0),
                    extra_k_cache=cmp_pool,
                    extra_topk_idxs=cmp_indices.flip(0)[:, None],
                    extra_topk_length=cmp_lengths.flip(0),
                )[:, 0]
                self.assertTrue(
                    torch.equal(
                        prefill.flip(0).view(torch.uint8), decode.view(torch.uint8)
                    )
                )

    def test_v41_fp4_indexer_half_pages(self):
        torch.manual_seed(13)
        device, dtype = "cuda", torch.bfloat16
        heads, rows, logical, physical, head_dim = 32, 9, 16, 32, 128
        key0 = torch.randn((logical * 2, head_dim), dtype=dtype, device=device)
        key1 = torch.randn_like(key0)
        query = torch.randn((rows, heads, head_dim), dtype=dtype, device=device)
        weights = torch.randn((rows, heads), device=device)
        pages = torch.tensor([[4, 1], [3, 2]], dtype=torch.int32, device=device)
        pool = torch.empty((5, physical, 68), dtype=torch.uint8, device=device)
        offsets = torch.arange(logical, device=device)
        for keys, table in zip((key0, key1), pages):
            slots = (table[:, None] * physical + offsets).reshape(-1)
            _v41_fp4_triton.quantize_indexer_k_fp4(keys, slots, pool)
        q, q_scale = _v41_fp4_triton.quantize_rows_fp4(query)
        lengths = torch.tensor([31, 17, 2, 6, 5, 16, 28, 9, 8], device=device)
        requests = torch.arange(rows, device=device) % 2
        score = score_paged(
            q, q_scale, weights, pool, pages, lengths, requests, logical
        )
        order = torch.arange(rows - 1, -1, -1, device=device)
        mixed = score_paged(
            q[order],
            q_scale[order],
            weights[order],
            pool,
            pages.flip(0),
            lengths[order],
            1 - requests[order],
            logical,
        )
        self.assertTrue(
            torch.equal(score[order].view(torch.uint8), mixed.view(torch.uint8))
        )
        for row in range(rows):
            single = score_paged(
                q[row : row + 1],
                q_scale[row : row + 1],
                weights[row : row + 1],
                pool,
                pages,
                lengths[row : row + 1],
                requests[row : row + 1],
                logical,
            )
            self.assertTrue(
                torch.equal(
                    score[row : row + 1].view(torch.uint8), single.view(torch.uint8)
                )
            )
            self.assertTrue(torch.isneginf(score[row, lengths[row] :]).all().item())

    def test_v4_fp8_indexer_ties_and_padding(self):
        device = "cuda"
        page_size, heads, dim = 8, 16, 128
        # The allocator may pad each physical page after its FP32 scale plane.
        pool = torch.zeros((3, page_size * 132 + 128), dtype=torch.uint8, device=device)
        pages = torch.tensor(
            [[1, 2], [2, -1], [0, 1]], dtype=torch.int32, device=device
        )
        ascending = torch.arange(1, page_size + 1, dtype=torch.float32, device=device)
        for page, values in ((1, ascending), (2, ascending.flip(0))):
            keys = values[:, None].expand(-1, dim).to(torch.float8_e4m3fn)
            pool[page, : page_size * dim] = (
                keys.contiguous().view(torch.uint8).flatten()
            )
            pool[page, page_size * dim : page_size * 132] = torch.ones_like(
                values
            ).view(torch.uint8)
        pool = pool[:, : page_size * 132].view(3, page_size, 132)
        q = torch.ones(5, heads, dim, device=device).to(torch.float8_e4m3fn)
        weights = torch.ones(5, heads, device=device)
        weights[[0, 4]] = 0
        requests = torch.tensor([0, 1, 0, -1, 2], dtype=torch.int32, device=device)
        lengths = torch.tensor([7, 10, 7, 8, 10], dtype=torch.int32, device=device)
        expected = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [3, 4, 5, 6],
                [-1, -1, -1, -1],
                [8, 9, -1, -1],
            ],
            dtype=torch.int32,
            device=device,
        )

        def select(rows):
            return packed_fp8_paged_mqa_topk(
                q[rows], pool, weights[rows], requests[rows], lengths[rows], pages, 4
            )

        self.assertTrue(torch.equal(select(slice(None)), expected))
        for row in range(q.size(0)):
            rows = slice(row, row + 1)
            self.assertTrue(torch.equal(select(rows), expected[rows]))

    def test_v41_index_source_and_consumer_skip_unallocated_pages(self):
        device, dtype = "cuda", torch.bfloat16
        heads, dim = 16, 128
        pool = torch.zeros(3, 4, 68, dtype=torch.uint8, device=device)
        pages = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int32, device=device)
        positions = torch.tensor([5, 11], device=device)
        requests = torch.arange(2, device=device)
        query = torch.zeros(2, heads, dim, dtype=dtype, device=device)
        x = torch.zeros(2, dim, dtype=dtype, device=device)
        shared = {}
        source = SimpleNamespace(
            _source_pool=lambda region: pool,
            _block_tables_by_type={INDEXER_KV: pages},
            _kv_cache=SimpleNamespace(get_kernel_seq_size_per_block=lambda region: 4),
            compress_ratio=2,
            _lin=lambda weight, qr: qr.clone(),
            index_wq=None,
            index_n_heads=heads,
            index_head_dim=dim,
            rope_head_dim=64,
            freqs_cis=torch.ones(12, 32, dtype=torch.complex64, device=device),
            index_weights=torch.zeros(heads, dim, dtype=dtype, device=device),
            _shared_attention=shared,
            v41_config={
                "candidate_source_layer_id": 0,
                "candidate_topk_blocks": 1,
                "candidate_block_size": 2,
            },
            layer_id=0,
            index_topk=4,
        )
        select = AttentionV41FP8._select_indices_batch_invariant
        self.assertEqual(
            select(source, x, query, positions, requests).tolist(),
            [[2, -1, -1, -1], [0, 1, 2, 3]],
        )
        self.assertEqual(shared["candidates"].tolist(), [[1], [1]])
        source.layer_id = 1
        # Only the short row ignores the candidate mask.
        shared["candidates"][0, 0] = 0
        self.assertEqual(
            select(source, x, query, positions, requests).tolist(),
            [[2, -1, -1, -1], [2, 3, -1, -1]],
        )


if __name__ == "__main__":
    unittest.main()
