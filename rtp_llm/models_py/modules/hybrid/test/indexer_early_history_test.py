"""Byte-exact prefix read and new-K patch against the existing CUDA cache writer."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
    build_indexer_cp_chunk_plan,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
    _restore_packed_indexer_k,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.indexer_early_history import (
    make_history_plan,
    pack_prefix,
    patch_new,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class EarlyHistoryTest(unittest.TestCase):
    def test_host_guards(self):
        for p, t in [([0, 0], [3, 4]), ([1], [8]), ([], [])]:
            self.assertIsNone(
                make_history_plan(torch.tensor(p), torch.tensor(t), 4, 64, "cpu")
            )
        with self.assertRaises(ValueError):
            make_history_plan(torch.tensor([4]), torch.tensor([3]), 4, 64, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_byte_exact_prefix_new_cache_and_scale(self):
        torch.manual_seed(314)
        # Partial owner pages, multiple/empty requests, zero new tokens,
        # owner size > physical block size, and ranks owning no actual rows.
        cases = [
            ([65], [7]),
            ([256, 0, 129], [3, 0, 71]),
            ([3, 0], [1, 0]),
            ([128, 257, 0], [0, 1, 3]),
            ([64, 192], [0, 0]),
            ([4097], [513]),
        ]
        for cp in (2, 4, 8):
            for owner in (64, 128):
                for scale in ("ue8m0", "float32"):
                    for prefix, new in cases:
                        with self.subTest(
                            cp=cp, owner=owner, scale=scale, prefix=prefix, new=new
                        ):
                            self.check_case(cp, owner, scale, prefix, new)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_auxiliary_repeated_runs_and_allocator_reuse(self):
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        for i in range(6):
            side.wait_stream(main)
            with torch.cuda.stream(side):
                self.check_case(4, 128, "ue8m0", [257, 0, 65], [i + 1, 0, 3])
                done = torch.cuda.Event()
                done.record(side)
            main.wait_event(done)
            # Pressure allocation reuse between calls with different mappings.
            torch.empty((1024 * 1024,), device="cuda").fill_(i)
        main.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_production_finish_restores_rows_and_normalizes_cache_layout(self):
        for scale in ("ue8m0", "float32"):
            for prefix, new in (([257, 0, 65], [7, 0, 3]), ([128, 256], [0, 0])):
                with self.subTest(scale=scale, prefix=prefix, new=new):
                    self.check_case(4, 128, scale, prefix, new, production_finish=True)

    def check_case(self, cp, owner, scale, prefix, new, production_finish=False):
        dev = "cuda"
        bs = 64
        totals = [a + b for a, b in zip(prefix, new)]
        lengths = torch.tensor(totals, dtype=torch.int64, device=dev)
        early = make_history_plan(
            torch.tensor(prefix), torch.tensor(totals), cp, owner, dev
        )
        self.assertIsNotNone(early)
        metadata, temporary_slots = early
        key = torch.randn(sum(new), 128, dtype=torch.bfloat16, device=dev) * 5
        tmp = torch.empty((sum(new), 1, 132), dtype=torch.uint8, device=dev)
        if sum(new):
            rtp_llm_ops.indexer_k_quant_and_cache(key, tmp, temporary_slots, 128, scale)
        maxblocks = max(
            (t + cp * owner - 1) // (cp * owner) * owner // bs for t in totals
        )
        maxblocks = max(maxblocks, 1)
        gathered = []
        caches = []
        refs = []
        slot_list = []
        plans = []
        for rank in range(cp):
            plan = build_indexer_cp_chunk_plan(
                NS(cp_size=cp, cp_rank=rank),
                lengths,
                bs,
                torch.device(dev),
                owner,
                torch.tensor(totals),
            )
            table = torch.arange(
                len(totals) * maxblocks, dtype=torch.int32, device=dev
            ).reshape(len(totals), maxblocks)
            cache = torch.randint(
                0,
                256,
                (len(totals) * maxblocks, bs, 132),
                dtype=torch.uint8,
                device=dev,
            )
            reference = cache.clone()
            slots = []
            for b, (pre, n) in enumerate(zip(prefix, new)):
                for token in range(pre, pre + n):
                    local = token // (cp * owner) * owner + token % owner
                    slots.append(
                        (b * maxblocks + local // bs) * bs + local % bs
                        if token // owner % cp == rank
                        else -1
                    )
            slots = torch.tensor(slots, dtype=torch.int64, device=dev)
            # Reading history must not modify cache. New rows may be arbitrary.
            packed = pack_prefix(cache, table, plan, metadata)
            torch.testing.assert_close(cache, reference, rtol=0, atol=0)
            if sum(new):
                rtp_llm_ops.indexer_k_quant_and_cache(key, reference, slots, 128, scale)
            gathered.append(packed)
            caches.append(cache)
            refs.append(reference)
            slot_list.append(slots)
            plans.append(plan)
        full = torch.cat(gathered, dim=0)
        k = torch.empty((sum(totals), 128), dtype=torch.uint8, device=dev)
        s = torch.empty((sum(totals), 4), dtype=torch.uint8, device=dev)
        _restore_packed_indexer_k(full, k, s, plans[0].restore_indices)
        for rank in range(cp):
            if production_finish:
                # Exercise the production wrapper, including row restoration,
                # the original CUDA quantizer and kernel-block cache views.
                order = torch.arange(sum(new) - 1, -1, -1, device=dev)
                gathered_key = key[order]
                plan = NS(early_history=early)
                params = NS(indexer_cp_plan=plan)
                cache = NS(kv_scale_base=caches[rank].view(-1, 1, 132))
                op = NS(
                    block_size=128,
                    blocksize=bs,
                    scale_fmt=scale,
                    _head_dim_with_sf=lambda: 132,
                )
                op._kv_cache_blocks = lambda value: IndexerOp._kv_cache_blocks(
                    op, value
                )
                with patch.dict(
                    IndexerOp.finish_cp_history.__globals__,
                    {"all_gather": lambda *a, **kw: gathered_key},
                ):
                    IndexerOp.finish_cp_history(
                        op, key, cache, slot_list[rank], order, params, (k, s)
                    )
            else:
                patch_new(tmp, caches[rank], slot_list[rank], metadata, k, s)
            torch.testing.assert_close(caches[rank], refs[rank], rtol=0, atol=0)
        # Independently decode physical page layout for every logical row.
        expected_k = []
        expected_s = []
        for b, total in enumerate(totals):
            for token in range(total):
                rank = token // owner % cp
                local = token // (cp * owner) * owner + token % owner
                page = refs[rank][b * maxblocks + local // bs].reshape(-1)
                expected_k.append(page[(local % bs) * 128 : (local % bs + 1) * 128])
                expected_s.append(
                    page[bs * 128 + (local % bs) * 4 : bs * 128 + (local % bs + 1) * 4]
                )
        torch.testing.assert_close(k, torch.stack(expected_k), rtol=0, atol=0)
        torch.testing.assert_close(s, torch.stack(expected_s), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
