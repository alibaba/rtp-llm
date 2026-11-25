"""Unit & CUDA tests for SM90 batched prefill pool gather and FP8 indexer select."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.attn_type import INDEXER_KV
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_batched_prefill_select as batched_select
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_grouped_prefill_score as grouped_score
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_pools as pools
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_fp8_indexer_cache as fp8_cache
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_fp8_prefill as sm90_fp8
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_v4_cache as sm90_v4


class Sm90PrefillPoolsHostTest(unittest.TestCase):
    def test_sm90_rejects_non_fp8_or_non_v4_without_device_work(self):
        attn = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=8, kv_cache_sharded=True),
            compress_ratio=2,
            _indexer_mode="fp4",
            _use_v4_kv_cache=True,
            _swa_codec=sm90_v4,
        )
        main = SimpleNamespace(
            device=torch.device("cuda:0"), dtype=torch.uint8, ndim=3, shape=(4, 64, 584)
        )
        index = SimpleNamespace(
            device=torch.device("cuda:0"),
            dtype=torch.uint8,
            ndim=3,
            shape=(4, 64, 132),
            is_contiguous=lambda: True,
        )
        seq_ends = Mock(spec=torch.Tensor)
        seq_ends.device = torch.device("cuda:0")
        seq_ends.dtype = torch.int64
        seq_ends.shape = (2,)
        seq_ends.stride.return_value = 1
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), patch.object(torch, "arange", side_effect=AssertionError("device work")):
            self.assertIsNone(
                pools.try_gather_prefill_pools(attn, main, index, [4, 8], seq_ends)
            )
            attn._indexer_mode = "fp8_dg"
            attn._use_v4_kv_cache = False
            self.assertIsNone(
                pools.try_gather_prefill_pools(attn, main, index, [4, 8], seq_ends)
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Sm90BatchedPrefillCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("SM90 CUDA contract")

    def test_sm90_batched_pool_gather_matches_per_request_bit_exact(self):
        torch.manual_seed(7)
        for ratio in (2, 1):
            counts = [0, 37, 256, 257, 600]
            ends = [c * ratio for c in counts]
            total_keys = sum(counts)
            entries_per_block = 128 // ratio
            num_blocks = (total_keys + entries_per_block - 1) // entries_per_block + 2

            main_pool = torch.zeros(
                (num_blocks, entries_per_block, 584), dtype=torch.uint8, device="cuda"
            )
            index_pool = torch.zeros(
                (num_blocks, entries_per_block, 132), dtype=torch.uint8, device="cuda"
            )
            # Fill valid blocks 1..num_blocks-1 with valid FP8/V4 data
            all_k = torch.randn((total_keys, 512), dtype=torch.bfloat16, device="cuda")
            all_idx_k = torch.randn((total_keys, 128), dtype=torch.bfloat16, device="cuda")
            flat_slots = torch.arange(
                entries_per_block,
                entries_per_block + total_keys,
                dtype=torch.int64,
                device="cuda",
            )
            sm90_v4.quantize_and_insert_k_cache(all_k, main_pool, flat_slots)
            # Write FP8 indexer keys into pages (planar [P, 64*132])
            pages = fp8_cache.dg_pages(index_pool)
            pages_flat = pages.view(-1, 64 * 132)
            q_idx = sm90_fp8.quantize_contiguous_fp8_indexer(all_idx_k)
            for i, s in enumerate(flat_slots.tolist()):
                p, o = s // 64, s % 64
                pages_flat[p, o * 128 : (o + 1) * 128].copy_(
                    q_idx.quant[i].view(torch.uint8)
                )
                pages_flat[p, 64 * 128 + o * 4 : 64 * 128 + (o + 1) * 4].copy_(
                    q_idx.scale[i : i + 1].view(torch.uint8)
                )

            offsets = []
            cur = entries_per_block
            for c in counts:
                offsets.append(cur)
                cur += c
            offsets_t = torch.tensor(offsets, dtype=torch.int64, device="cuda")
            counts_t = torch.tensor(counts, dtype=torch.int64, device="cuda")

            def _slots(region, pos, req):
                idx = (pos + 1) // ratio - 1
                valid = (pos >= 0) & (idx >= 0) & (idx < counts_t[req])
                return torch.where(valid, offsets_t[req] + idx, -1)

            gather_calls = []

            def _gather_shards(val):
                gather_calls.append(tuple(val.shape))
                return val

            attn = SimpleNamespace(
                _cp_ctx=SimpleNamespace(cp_size=8, kv_cache_sharded=True),
                compress_ratio=ratio,
                _indexer_mode="fp8_dg",
                _use_v4_kv_cache=True,
                _swa_codec=sm90_v4,
                _slots=_slots,
                _global_region=lambda: "GLOBAL_KV",
                _gather_shards=_gather_shards,
            )
            seq_ends = torch.tensor(ends, dtype=torch.int64, device="cuda")
            batched = pools.try_gather_prefill_pools(
                attn, main_pool, index_pool, ends, seq_ends
            )
            self.assertIsNotNone(batched)
            # Should do exactly 2 all-reduces total (1 index + 1 global), NOT 2 * len(ends)!
            self.assertEqual(len(gather_calls), 2)
            self.assertEqual(len(batched), len(ends))

            # Compare bit-for-bit against legacy per-request path
            for b, end in enumerate(ends):
                count = end // ratio
                idx = torch.arange(count, device="cuda", dtype=torch.long)
                pos = (idx + 1) * ratio - 1
                req = torch.full_like(pos, b)
                raw_idx = sm90_fp8.gather_raw_fp8_indexer(
                    pages, _slots(INDEXER_KV, pos, req)
                )
                ref_k = sm90_fp8.unpack_raw_fp8_indexer(raw_idx)
                raw_g = sm90_v4.gather_k_cache_bytes(
                    main_pool, _slots("GLOBAL_KV", pos, req)
                )
                ref_g = sm90_v4.dequantize_k_cache_bytes(raw_g)
                act_g, act_k = batched[b]
                torch.testing.assert_close(act_g, ref_g, rtol=0, atol=0)
                torch.testing.assert_close(
                    act_k.quant.view(torch.uint8),
                    ref_k.quant.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(act_k.scale, ref_k.scale, rtol=0, atol=0)

    def test_sm90_batched_fp8_select_matches_per_request_bit_exact(self):
        torch.manual_seed(11)
        counts = [300, 512, 700, 620]
        row_counts = [64, 128, 96, 80]
        ratio = 2
        padded = [(c + 255) // 256 * 256 for c in counts]
        total_k = sum(padded)
        k_quant_slab = torch.randn((total_k, 128), device="cuda").to(torch.float8_e4m3fn)
        k_scale_slab = torch.rand((total_k,), dtype=torch.float32, device="cuda") + 0.5

        globals_by_req = []
        cursor = 0
        for c, p in zip(counts, padded):
            kq = k_quant_slab[cursor : cursor + c]
            ks = k_scale_slab[cursor : cursor + c]
            globals_by_req.append((None, sm90_fp8.PrefillFP8IndexerKeys(kq, ks)))
            cursor += p

        total_rows = sum(row_counts)
        q_fp8 = torch.randn((total_rows, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        weights = torch.randn((total_rows, 32), dtype=torch.float32, device="cuda")
        slices = []
        pos_list = []
        req_list = []
        r0 = 0
        for b, (rc, kc) in enumerate(zip(row_counts, counts)):
            slices.append(slice(r0, r0 + rc))
            # Causal positions up to kc * ratio - 1
            pos_list.append(
                torch.linspace(0, kc * ratio - 1, rc, device="cuda").to(torch.int64)
            )
            req_list.append(torch.full((rc,), b, dtype=torch.int64, device="cuda"))
            r0 += rc
        positions = torch.cat(pos_list)
        req_ids = torch.cat(req_list)
        prefix_lens = torch.zeros(len(counts), dtype=torch.int64, device="cuda")
        input_lens = torch.tensor(
            [c * ratio for c in counts], dtype=torch.int64, device="cuda"
        )

        topk = 512
        out_batched = torch.full((total_rows, topk), -1, dtype=torch.int32, device="cuda")
        attn = SimpleNamespace(
            _shared_attention={},
            _cp_ctx=SimpleNamespace(
                prefix_lengths=prefix_lens,
                input_lengths_global=input_lens,
            ),
            compress_ratio=ratio,
            layer_id=2,
            index_topk=topk,
        )
        ok = batched_select.try_select_batched(
            attn,
            q_fp8,
            None,
            weights,
            globals_by_req,
            slices,
            positions,
            out_batched,
            candidate_source=-1,
            publish_candidates=False,
            candidate_size=0,
            candidate_blocks=0,
            req_ids=req_ids,
        )
        self.assertTrue(ok)

        # Reference: per-request score_fp8_prefill_chunk + _v41_prefill_topk
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as prefill_topk

        out_ref = torch.full((total_rows, topk), -1, dtype=torch.int32, device="cuda")
        for b, ((_, keys), sl) in enumerate(zip(globals_by_req, slices)):
            vis = (positions[sl] + 1) // ratio
            logits = sm90_fp8.score_fp8_prefill_chunk(
                q_fp8[sl], weights[sl], keys, vis
            )
            k = min(topk, len(keys))
            sel = prefill_topk.try_select_tokens(logits, vis, k, out=out_ref[sl, :k])
            if sel is None:
                grouped_score._mask_tail_kernel[(logits.shape[0],)](
                    logits, vis, logits.shape[1], logits.stride(0), 256
                )
                scores, sel = logits.topk(k, dim=-1)
                out_ref[sl, :k] = torch.where(scores.isfinite(), sel, -1).int()
            elif sel is not out_ref[sl, :k]:
                out_ref[sl, :k].copy_(sel)

        # Compare selected index sets (sorting handles tie order identically if any)
        torch.testing.assert_close(
            out_batched.sort(dim=-1).values,
            out_ref.sort(dim=-1).values,
            rtol=0,
            atol=0,
        )

    def test_sm90_batched_fp8_select_with_candidates_publish_and_filter(self):
        torch.manual_seed(23)
        counts = [320, 640, 512]
        row_counts = [48, 64, 32]
        ratio = 2
        candidate_size = 8
        candidate_blocks = 32
        padded = [(c + 255) // 256 * 256 for c in counts]
        total_k = sum(padded)
        k_quant_slab = torch.randn((total_k, 128), device="cuda").to(torch.float8_e4m3fn)
        k_scale_slab = torch.rand((total_k,), dtype=torch.float32, device="cuda") + 0.5
        globals_by_req = []
        cursor = 0
        for c, p in zip(counts, padded):
            globals_by_req.append(
                (None, sm90_fp8.PrefillFP8IndexerKeys(k_quant_slab[cursor : cursor + c], k_scale_slab[cursor : cursor + c]))
            )
            cursor += p

        total_rows = sum(row_counts)
        q_fp8 = torch.randn((total_rows, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        weights = torch.randn((total_rows, 32), dtype=torch.float32, device="cuda")
        slices, pos_list, req_list = [], [], []
        r0 = 0
        for b, (rc, kc) in enumerate(zip(row_counts, counts)):
            slices.append(slice(r0, r0 + rc))
            pos_list.append(torch.linspace(0, kc * ratio - 1, rc, device="cuda").to(torch.int64))
            req_list.append(torch.full((rc,), b, dtype=torch.int64, device="cuda"))
            r0 += rc
        positions = torch.cat(pos_list)
        req_ids = torch.cat(req_list)
        prefix_lens = torch.zeros(len(counts), dtype=torch.int64, device="cuda")
        input_lens = torch.tensor([c * ratio for c in counts], dtype=torch.int64, device="cuda")
        topk = 128

        # Layer 2 publishes candidates, Layer 8 filters by them
        shared_b = {"candidates": torch.full((total_rows, candidate_blocks), -1, dtype=torch.int32, device="cuda")}
        shared_r = {"candidates": torch.full((total_rows, candidate_blocks), -1, dtype=torch.int32, device="cuda")}
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as prefill_topk
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import _apply_prefill_candidates

        for layer_id, publish in ((2, True), (8, False)):
            out_b = torch.full((total_rows, topk), -1, dtype=torch.int32, device="cuda")
            attn = SimpleNamespace(
                _shared_attention=shared_b,
                _cp_ctx=SimpleNamespace(prefix_lengths=prefix_lens, input_lengths_global=input_lens),
                compress_ratio=ratio,
                layer_id=layer_id,
                index_topk=topk,
            )
            ok = batched_select.try_select_batched(
                attn, q_fp8, None, weights, globals_by_req, slices, positions, out_b,
                candidate_source=2, publish_candidates=publish,
                candidate_size=candidate_size, candidate_blocks=candidate_blocks,
                req_ids=req_ids,
            )
            self.assertTrue(ok)

            out_r = torch.full((total_rows, topk), -1, dtype=torch.int32, device="cuda")
            for b, ((_, keys), sl) in enumerate(zip(globals_by_req, slices)):
                vis = (positions[sl] + 1) // ratio
                logits = sm90_fp8.score_fp8_prefill_chunk(q_fp8[sl], weights[sl], keys, vis)
                _apply_prefill_candidates(shared_r, logits, vis, sl, candidate_size, candidate_blocks, publish, mask_tail=True)
                k = min(topk, len(keys))
                sel = prefill_topk.try_select_tokens(logits, vis, k, out=out_r[sl, :k])
                if sel is None:
                    grouped_score._mask_tail_kernel[(logits.shape[0],)](logits, vis, logits.shape[1], logits.stride(0), 256)
                    scores, sel = logits.topk(k, dim=-1)
                    out_r[sl, :k] = torch.where(scores.isfinite(), sel, -1).int()
                elif sel is not out_r[sl, :k]:
                    out_r[sl, :k].copy_(sel)

            torch.testing.assert_close(out_b.sort(dim=-1).values, out_r.sort(dim=-1).values, rtol=0, atol=0)
            if publish:
                torch.testing.assert_close(
                    shared_b["candidates"].sort(dim=-1).values,
                    shared_r["candidates"].sort(dim=-1).values,
                    rtol=0, atol=0,
                )


if __name__ == "__main__":
    unittest.main()
