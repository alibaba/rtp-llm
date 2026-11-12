"""Actual native-paged SG sparse seam; CPU protocol is not GPU math evidence."""

import os
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import cp
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_readback, local_prefill_sg_paged_indexer as paged
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_native_read import fixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_sg_cache_rows import _RestoreCPU


def _case(device, rows=4, count=32768, *, padding=0, mirrored=True, blocks=None):
    def reorder(tag, table, blocks):
        table.copy_(table.flip(1))
        table[0, 0] = 0
        if table.shape[1] > 1:
            table[0, 1] = blocks - 1
        if table.shape[1] > 2:
            table[0, 2] = 0  # duplicate physical pages are legal reads
    cache = fixture(device, (count,), ratios={20: 1}, padding=padding, configure=reorder, blocks=blocks)
    pool = cache.pools[20, "indexer_kv"]
    if mirrored:
        cache.read_indexer_fp4(20, 0, global_entries=count)  # Genuine L20 bank-read publication.
    visible = torch.arange(count - rows + 1, count + 1, device=device, dtype=torch.int32)
    blocks = torch.arange(2048, device=device, dtype=torch.int32)[None].expand(rows, -1).clone()
    blocks[:, -1] = (visible + 7) // 8 - 1
    blocks = blocks.flip(1).contiguous()
    return cache, blocks, visible


class _SortCPU:
    def forward(self, blocks, lengths, table, phys, page):
        assert page == 256
        for row in range(blocks.shape[0]):
            count = (int(lengths[row]) + 7) // 8
            ids = (list(range(count)) if count <= blocks.shape[1]
                   else sorted(set(int(value) for value in blocks[row].tolist() if 0 <= value < count)))
            ids += [2147483647] * (blocks.shape[1] - len(ids))
            blocks[row].copy_(torch.tensor(ids, dtype=torch.int32))
            values = [2147483647 if value == 2147483647 else int(table[row, value // 32]) * 32 + value % 32
                      for value in ids]
            phys[row].copy_(torch.tensor(values, dtype=torch.int32))


class _BackendCPU:
    def __init__(self):
        self.metadata_calls = []
        self.score_calls = []

    def get_paged_sparse_mqa_logits_metadata(self, *args):
        self.metadata_calls.append(args)
        return torch.arange(16, dtype=torch.uint8)

    def fp8_fp4_paged_sparse_mqa_logits(self, *args):
        self.score_calls.append(args)
        return torch.zeros((args[0][0].shape[0], 16384), dtype=torch.bfloat16)


class LocalSgPagedIndexerCPU(unittest.TestCase):
    def setUp(self):
        self.backend = _BackendCPU()
        for item in (
            patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_SG_ALIGN": "1", "TP_SIZE": "1",
                                    "PREFILL_CP_SIZE": "1", "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_CACHE": "1",
                                    "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_READ": "1"}),
            patch.object(SharedNativeCache, "_local_source_native_read_device", new=lambda self, pool: True),
            patch.object(prefill_readback, "_restore_rows", _RestoreCPU()),
            patch("rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
                  new=lambda rows, device, dtype: torch.tensor(rows, device=device, dtype=dtype)),
            patch.object(paged, "_sort_module", new=lambda: _SortCPU()),
            patch.object(paged, "_deep_gemm", new=lambda: self.backend),
        ):
            item.start()
            self.addCleanup(item.stop)

    @torch.no_grad()
    def test_plan_abi_real_pages_zero_duplicates_sorted_clone_and_single_schedule(self):
        cache, candidates, visible = _case("cpu")
        before = candidates.clone()
        publication = paged.make_publication(cache, candidates, visible)
        self.assertIsNotNone(publication)
        plan = paged.prepare(cache, candidates, visible, publication=publication)
        self.assertIsNotNone(plan)
        assert_bytes(self, candidates, before)
        self.assertIs(paged.prepare(cache, candidates, visible, publication=publication), plan)
        self.assertEqual(len(self.backend.metadata_calls), 1)
        self.assertEqual(plan.pool_view.data_ptr(), cache.pools[20, "indexer_kv"].tensor.data_ptr())
        self.assertEqual(plan.pool_view.stride(), (17408, 68, 68, 1))
        self.assertEqual(plan.page_table.shape, (4, 128))
        self.assertEqual(plan.request_ids.tolist(), [0, 0, 2, 2])
        self.assertEqual(plan.blocks[0].tolist(), sorted(before[0].tolist()))
        self.assertEqual(self.backend.metadata_calls[0][3], 256)
        self.assertIs(self.backend.metadata_calls[0][4], plan.blocks)
        self.assertEqual(self.backend.metadata_calls[0][5:], (torch.int8, 8))
        q = torch.zeros((4, 32, 64), dtype=torch.int8)
        sf = torch.zeros((4, 32), dtype=torch.int32)
        weights = torch.randn((4, 32), dtype=torch.float32)
        result = paged.score(q, sf, weights, plan)
        self.assertEqual(result.dtype, torch.bfloat16)
        args = self.backend.score_calls[0]
        self.assertEqual(args[0][0].shape, (4, 1, 32, 64))
        self.assertEqual(args[0][1].shape, (4, 1, 32))
        assert_bytes(self, args[2], weights.bfloat16())

    @torch.no_grad()
    def test_unsupported_no_mirror_late_flag_layout_and_owner_replacement_prelaunch(self):
        for padding, mirrored in ((16, True), (0, False)):
            cache, candidates, visible = _case("cpu", padding=padding, mirrored=mirrored)
            self.assertIsNone(paged.make_publication(cache, candidates, visible))
        with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_SG_ALIGN": "0"}):
            cache, candidates, visible = _case("cpu")
        self.assertIsNone(paged.make_publication(cache, candidates, visible))
        cache, candidates, visible = _case("cpu")
        self.assertIsNone(paged.make_publication(cache, torch._neg_view(candidates), visible))
        self.assertIsNone(paged.make_publication(cache, candidates, torch._neg_view(visible)))
        publication = paged.make_publication(cache, candidates, visible)
        self.assertIsNone(paged.prepare(cache, candidates.clone(), visible, publication=publication))
        self.assertIsNone(paged.prepare(cache, candidates, visible.clone(), publication=publication))
        self.assertFalse(self.backend.metadata_calls)
        with torch.enable_grad():
            self.assertIsNone(paged.make_publication(cache, candidates, visible))

    @torch.no_grad()
    def test_normal_versions_pool_rebind_close_and_metadata_mutation_decline(self):
        for change in (lambda c, b, v: b.add_(0), lambda c, b, v: v.add_(0),
                       lambda c, b, v: c._invalidate_local_source_reads(),
                       lambda c, b, v: c.pools[20, "indexer_kv"].table.add_(0),
                       lambda c, b, v: c.close_local_source_native_reads(),
                       lambda c, b, v: c.pools.__setitem__((20, "indexer_kv"), replace(
                           c.pools[20, "indexer_kv"], tensor=c.pools[20, "indexer_kv"].tensor.clone()))):
            cache, blocks, visible = _case("cpu")
            publication = paged.make_publication(cache, blocks, visible)
            change(cache, blocks, visible)
            self.assertIsNone(paged.prepare(cache, blocks, visible, publication=publication))
        cache, blocks, visible = _case("cpu")
        publication = paged.make_publication(cache, blocks, visible)
        plan = paged.prepare(cache, blocks, visible, publication=publication)
        plan.page_table.add_(0)
        self.assertIsNone(paged.prepare(cache, blocks, visible, publication=publication))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            paged.score(None, None, None, plan)

    @torch.no_grad()
    def test_publication_and_cached_plan_release_without_a_reference_cycle(self):
        cache, blocks, visible = _case("cpu")
        publication = paged.make_publication(cache, blocks, visible)
        plan = paged.prepare(cache, blocks, visible, publication=publication)
        self.assertIs(plan.publication, publication)
        owner = weakref.ref(publication)
        del publication
        self.assertIsNone(owner())
        self.assertIsNone(plan.publication)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            paged.score(None, None, None, plan)

    @torch.no_grad()
    def test_metadata_only_reuse_and_postlaunch_error_propagates(self):
        cache, blocks, visible = _case("cpu")
        publication = paged.make_publication(cache, blocks, visible)
        plan = paged.prepare(cache, blocks, visible, publication=publication)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no DtoH")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("no GPU scalar")):
            self.assertIs(paged.prepare(cache, blocks, visible, publication=publication), plan)
        with patch.object(self.backend, "fp8_fp4_paged_sparse_mqa_logits", side_effect=RuntimeError("paged failed")):
            with self.assertRaisesRegex(RuntimeError, "paged failed"):
                paged.score(torch.zeros(4, 32, 64, dtype=torch.int8), torch.zeros(4, 32, dtype=torch.int32),
                            torch.zeros(4, 32), plan)
        self.assertEqual(len(self.backend.metadata_calls), 1)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit real PPU test")
class LocalSgPagedIndexerDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        env = patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_SG_ALIGN": "1", "TP_SIZE": "1",
                                      "PREFILL_CP_SIZE": "1", "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_CACHE": "1",
                                      "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_READ": "1"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_actual_paged_fp4_ieee_logits_against_independent_identity_pool_and_reuse_profile(self):
        cache, blocks, visible = _case("cuda:0", rows=140, count=131072, blocks=2880)
        # Include the actual compact-padding boundary alongside full rows.
        visible[-2:] = torch.tensor([0, 1], dtype=torch.int32, device="cuda:0")
        blocks[-2:].fill_(-1)
        blocks[-1, 0] = 0
        publication = paged.make_publication(cache, blocks, visible)
        self.assertIsNotNone(publication)
        original_blocks = blocks.clone()
        plan = paged.prepare(cache, blocks, visible, publication=publication)
        self.assertIsNotNone(plan)
        assert_bytes(self, blocks, original_blocks)
        pool = cache.pools[20, "indexer_kv"]
        before = pool.tensor.clone()
        # Independent CPU logical-page recovery; provider reference reads an
        # identity page table instead of the reordered/zero/duplicated pool.
        raw, host = pool.tensor.cpu(), pool.table[0].cpu().tolist()
        identity = torch.stack([raw[page] for page in host]).to("cuda:0")
        ref_table = torch.arange(len(host), device="cuda:0", dtype=torch.int32)[None].expand(140, -1).contiguous()
        host_candidates, host_visible = original_blocks.cpu(), visible.cpu().tolist()
        logical, expected_lens = [], []
        for row, length in enumerate(host_visible):
            count = (length + 7) // 8
            chosen = (list(range(count)) if count <= 2048 else
                      sorted(set(int(value) for value in host_candidates[row] if 0 <= value < count)))
            logical.append(chosen + [2147483647] * (2048 - len(chosen)))
            expected_lens.append(0 if not length else (min(count, 2048) - 1) * 8 + (length - 1) % 8 + 1)
        cpu_blocks = torch.tensor(logical, dtype=torch.int32, device="cuda:0")
        expected_lens = torch.tensor(expected_lens, dtype=torch.int32, device="cuda:0")
        assert_bytes(self, plan.valid_lens, expected_lens)
        dg = paged._deep_gemm()
        reference_metadata = dg.get_paged_sparse_mqa_logits_metadata(
            visible, ref_table, plan.request_ids, 256, cpu_blocks, torch.int8, 8,
        )
        q = torch.randint(-128, 128, (140, 32, 64), device="cuda:0", dtype=torch.int8)
        sf = torch.full((140, 32), 0x7f7f7f7f, device="cuda:0", dtype=torch.int32)
        weights = torch.randn((140, 32), device="cuda:0", dtype=torch.float32)
        weights[:4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0],
                                   device="cuda:0")[:, None]
        expected = dg.fp8_fp4_paged_sparse_mqa_logits(
            (q[:, None], sf[:, None]), identity.view(len(host), 256, 1, 68), weights.bfloat16(),
            reference_metadata, 2048, 8,
        )
        got = paged.score(q, sf, weights, plan)
        valid = torch.arange(16384, device="cuda:0")[None] < expected_lens[:, None]
        # SG selectors never consume score padding; it need not be initialized.
        assert_bytes(self, got[valid], expected[valid])
        with patch.object(cp.collective_torch, "_get_group", side_effect=AssertionError("no CP group")), \
                patch.object(dg, "fp8_fp4_paged_sparse_mqa_logits", wraps=dg.fp8_fp4_paged_sparse_mqa_logits) as actual, \
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            for _ in (24, 28, 32, 36):
                self.assertIs(paged.prepare(cache, blocks, visible, publication=publication), plan)
                paged.score(q, sf, weights, plan)
        self.assertEqual(actual.call_count, 4)
        events = _trace_events(profile, "sg-direct-paged-late-sparse")
        cpu_names = [event.get("name") for event in events if event.get("cat") == "cpu_op"]
        self.assertNotIn("aten::index_select", cpu_names)
        self.assertNotIn("aten::cat", cpu_names)
        kernels = [event.get("name", "") for event in events if event.get("cat") == "kernel"]
        self.assertGreaterEqual(sum("mqa" in name.lower() and "logits" in name.lower() for name in kernels), 4)
        assert_bytes(self, pool.tensor, before)


if __name__ == "__main__":
    unittest.main()
