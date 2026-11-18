"""Shared-prefix readback and unchanged BF16 attention consumer contracts."""

from datetime import timedelta
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_shared_global as shared_global
from rtp_llm.models_py.modules.dsv41.ppu import prefill
from rtp_llm.models_py.modules.dsv41.ppu import prefill_readback
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache, _source_tensor_key


def _bank(device="cpu", requests=2, prefix=8, count=14):
    total = prefix + requests * (count - prefix)
    packed = torch.arange(total, device=device).to(torch.uint8)[:, None].expand(-1, 288).contiguous()
    return shared_global.SharedGlobalBank.create(packed, prefix, (count,) * requests)


def _stats():
    return dict(legacy_materializations=0, dequant_launches=0, dequant_rows=0,
                workspace_hits=0, invalidations=0)


def _cache(bank):
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache._is_prefill, cache.sharded, cache.device = True, True, bank.packed.device
    cache.cp = dict(size=8, rank=0)
    cache.lengths = bank.counts
    cache.geometry = NS(ratio=lambda source: 1)
    cache._source_reads = {(20, request): (shared_global.SharedGlobalSlice(bank, request), None)
                           for request in range(len(bank.counts))}
    cache._local_source_reads = {}
    cache._fresh_swa = {}
    cache._cp_shared_global_epoch = 7
    cache._cp_shared_global_stats = _stats()
    cache._cp_shared_global_reads = {
        20: (bank, 7, "table-birth", _source_tensor_key(bank.packed), (bank.packed,)),
    }
    cache._cp_shared_prefix_signature = lambda key: ("table-birth", (bank.packed,))
    cache._local_source_cache_eligible = lambda: False
    return cache


def _referenced_rows(kv, indices):
    index = indices.flatten(1).long()
    return kv.flatten(0, 1)[index.clamp_min(0)] * (index >= 0).unsqueeze(-1)


def _workspace_fixture(device="cpu"):
    bank = _bank(device)
    cache = _cache(bank)
    cache.cp8_row_prep_enabled = lambda: True
    prepared = []
    cache.prepare_source = prepared.append
    cache.read_shared_global = lambda source: bank
    cache.read_global = lambda source, request, global_entries: bank.materialize(request, global_entries)
    cache.request_length = bank.counts.__getitem__
    cache.descriptors = lambda source: {"swa_kv": None, "global_kv_1": None}
    cache.try_decode_swa_range = lambda *args: False
    cache.read_swa = lambda layer, request, positions: (
        (positions + layer + 30 * request) % 200
    ).to(torch.uint8)[:, None].expand(-1, 528).contiguous()
    geometry = NS(head_dim=512, ratio=lambda layer: 1, kv_source_for=lambda layer: 20)
    positions = torch.tensor([10, 13, 11, 13], dtype=torch.int64, device=device)
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
    context = prefill.build_step_context(
        positions, requests, torch.tensor([0, 2, 4], device=device),
        valid_mask=torch.tensor([True, True, True, False], device=device),
    )
    cp = NS(chunk_lengths_per_req=(2, 2), gather_restore_positions=object(),
            input_lengths_global_host=None, prefix_lengths_host=None)
    selected = torch.tensor([[0, 8, 13, -1], [7, 9, 14, 7], [1, 10, -1, 12], [13, 13, 2, 0]],
                            dtype=torch.int32, device=device)
    ops = NS(decode_kv=lambda packed, *, pool_tag, descriptor, out: out.copy_(packed[:, :1].expand(-1, 512)))
    args = dict(ops=ops, cache=cache, geometry=geometry, context=context,
                selected_blocks=selected, window=4, swa_starts={})
    return bank, cache, prepared, cp, args


class CP8SharedGlobalContracts(unittest.TestCase):
    def test_bank_preserves_prefix_and_independent_tails_without_legacy_aliases(self):
        bank = _bank()
        self.assertEqual(bank.tail_offsets, (8, 14))
        self.assertEqual(tuple(bank.packed.shape), (20, 288))
        for request in range(2):
            rows = bank.materialize(request, 14)
            expected = torch.cat((bank.packed[:8], bank.packed[8 + 6 * request:14 + 6 * request]))
            self.assertTrue(torch.equal(rows, expected))
            self.assertFalse(torch._C._is_alias_of(rows, bank.packed))
        first = bank.materialize(0, 4)
        first.zero_()
        self.assertTrue(torch.equal(bank.materialize(1, 4), bank.packed[:4]))

    def test_compact_layout_retains_partial_prefix_in_request_tails(self):
        legacy = [item for _, _, group in prefill_readback._stripe_groups((145, 145), 64, 3, 1) for item in group]
        compact = [item for _, _, group in prefill_readback._stripe_groups((145, 145), 64, 3, 1, compact=True)
                   for item in group]
        self.assertEqual([row[:3] for row in compact], [row[:3] for row in legacy])
        self.assertEqual(compact, [(0, 0, 64, 0), (0, 1, 64, 64), (0, 2, 17, 128),
                                   (1, 1, 64, 145), (1, 2, 17, 209)])
        self.assertEqual(shared_global.compact_offsets((145, 145), 64), ((64, 145), 226))
        no_share = [item for _, _, group in prefill_readback._stripe_groups((145, 141), 64, 3) for item in group]
        self.assertEqual(sum(row[2] for row in no_share), 286)
        for counts, prefix in (((145, 141), 64), ((145, 145), 0), ((63, 63), 64)):
            with self.subTest(counts=counts, prefix=prefix), self.assertRaises(ValueError):
                shared_global.compact_offsets(counts, prefix)

    @torch.no_grad()
    def test_workspace_preserves_referenced_bf16_rows_and_refreshes_swa(self):
        bank, cache, prepared, cp, args = _workspace_fixture()
        state = {}
        actual = prefill.build_cp_prefill_workspace(**args, layer_id=20, cp=cp, shared=state)
        expected = prefill._build_reference_workspace(**args, layer_id=20)
        self.assertTrue(torch.equal(actual[2], expected[2]))
        torch.testing.assert_close(_referenced_rows(actual[0], actual[1]),
                                   _referenced_rows(expected[0], expected[1]), rtol=0, atol=0)
        workspace = state["prefill_kv_workspace"][2]
        self.assertIs(workspace._shared_global, bank)
        self.assertEqual(cache._cp_shared_global_stats["dequant_launches"], 1)
        self.assertEqual(cache._cp_shared_global_stats["dequant_rows"], 20)
        next_layer = prefill.build_cp_prefill_workspace(**args, layer_id=21, cp=cp, shared=state)
        next_expected = prefill._build_reference_workspace(**args, layer_id=21)
        torch.testing.assert_close(_referenced_rows(next_layer[0], next_layer[1]),
                                   _referenced_rows(next_expected[0], next_expected[1]), rtol=0, atol=0)
        self.assertEqual(prepared, [20])
        self.assertIs(actual[1], next_layer[1])
        self.assertIs(actual[2], next_layer[2])
        self.assertEqual(cache._cp_shared_global_stats["workspace_hits"], 1)

    @torch.no_grad()
    def test_legacy_source_and_split_indexer_abis_are_unchanged(self):
        bank = _bank()
        cache = _cache(bank)
        payload = torch.arange(14 * 64).reshape(14, 64).to(torch.int8)
        scales = torch.arange(14, dtype=torch.int32)
        cache._read_local_sg_source = lambda *args: None
        cache._cached_source_read = lambda *args: (False, (shared_global.SharedGlobalSlice(bank, 1), (payload, scales)))
        global_rows, indexer = cache.read_source(20, 1, global_entries=14)
        self.assertTrue(torch.equal(global_rows, bank.materialize(1, 14)))
        self.assertFalse(torch._C._is_alias_of(global_rows, bank.packed))
        self.assertEqual(tuple(indexer.shape), (14, 68))
        p, s = cache.read_indexer_fp4(20, 1, global_entries=14)
        self.assertEqual(p.data_ptr(), payload.data_ptr())
        self.assertEqual(s.data_ptr(), scales.data_ptr())
        self.assertEqual(cache._cp_shared_global_stats["legacy_materializations"], 1)

    def test_source_read_has_no_collective_and_declines_stale_epoch_or_metadata(self):
        bank = _bank()
        cache = _cache(bank)
        with patch("torch.distributed.all_reduce", side_effect=AssertionError("no implicit collective")):
            self.assertIs(cache.read_shared_global(20), bank)
            self.assertIsNone(cache.read_shared_global(99))
            cache._cp_shared_global_epoch += 1
            with self.assertRaisesRegex(RuntimeError, "stale CP"):
                cache.read_shared_global(20)
        cache = _cache(bank)
        cache._cp_shared_prefix_signature = lambda key: ("changed-table", ())
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cache.read_shared_global(20)
        cache = _cache(bank)
        bank.packed.add_(1)
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cache.read_shared_global(20)

    def test_prefix_proof_tracks_table_and_pool_identity_but_not_fresh_data_versions(self):
        bank = _bank()
        cache = _cache(bank)
        del cache._cp_shared_prefix_signature
        host = torch.tensor([[1], [1]], dtype=torch.int32)
        device = host.clone()
        handle = NS(kv_cache_base=bank.packed)
        pool = NS(framework_tag="global", tensor=bank.packed, table=device, handle=handle)
        key = 20, "global_kv_1"
        cache.pools = {key: pool}
        cache.placements = {key: NS(table=device, size=8, rank=0, physical_span=8, pages=1)}
        cache.inputs = {"global": NS(kv_cache_block_id=host, kv_cache_block_id_device=device)}
        before, owners = cache._cp_shared_prefix_signature(key)
        bank.packed.add_(1)
        self.assertEqual(cache._cp_shared_prefix_signature(key)[0], before)
        host.add_(1)
        self.assertNotEqual(cache._cp_shared_prefix_signature(key)[0], before)
        before = cache._cp_shared_prefix_signature(key)[0]
        pool.handle = NS(kv_cache_base=bank.packed)
        self.assertNotEqual(cache._cp_shared_prefix_signature(key)[0], before)
        self.assertIn(handle, owners)

    @torch.no_grad()
    def test_admission_excludes_q8_graph_decode_and_unproved_prefix(self):
        bank = _bank()
        cache = _cache(bank)
        key = 20, "global_kv_1"
        cache.device = "cuda:0"
        cache.cp8_row_prep_enabled = lambda: True
        cache.inputs = {"global": NS(is_cuda_graph=False, is_target_verify=False)}
        cache.shared_prefix_stripes = {key: 1}
        cache._cp_shared_prefix_proofs = {key: ("table-birth", ())}
        with patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True), \
                patch("torch.cuda.is_current_stream_capturing", return_value=False), \
                patch.dict(os.environ, RTP_V41_PPU_CP8_YOCO_Q8_ATTENTION="0"):
            self.assertTrue(cache._cp_shared_global_eligible(key))
            with patch.dict(os.environ, RTP_V41_PPU_CP8_YOCO_Q8_ATTENTION="1"):
                self.assertFalse(cache._cp_shared_global_eligible(key))
            cache.inputs["global"].is_cuda_graph = True
            self.assertFalse(cache._cp_shared_global_eligible(key))
            cache.inputs["global"].is_cuda_graph = False
            cache._is_prefill = False
            self.assertFalse(cache._cp_shared_global_eligible(key))
            cache._is_prefill = True
            cache.shared_prefix_stripes[key] = 0
            self.assertFalse(cache._cp_shared_global_eligible(key))
            cache.shared_prefix_stripes[key] = 1
            cache._cp_shared_prefix_signature = lambda key: ("changed-table", ())
            with self.assertRaisesRegex(RuntimeError, "placement changed"):
                cache._cp_shared_global_eligible(key)

    def test_write_invalidates_all_source_aliases_before_any_pool_store(self):
        bank = _bank()
        cache = _cache(bank)
        cache._cp_shared_global_reads[21] = cache._cp_shared_global_reads[20]
        cache.placements = {}

        def stored(*args, **kwargs):
            self.assertEqual(cache._cp_shared_global_reads, {})
            self.assertEqual(cache._source_reads, {})
            self.assertEqual(cache._cp_shared_global_epoch, 8)

        with patch.object(NativeCacheBridge, "_copy_rows", side_effect=stored):
            cache._copy_rows((20, "global_kv_1"), 0, torch.arange(2), torch.empty((2, 288), dtype=torch.uint8))
        self.assertEqual(bank.packed.shape[0], 20)
        cache = _cache(bank)
        cache.cp8_row_prep_enabled = lambda: False
        self.assertFalse(cache._try_store_prefill_range((20, "global_kv_1"), 0, 0, bank.packed))
        self.assertEqual(cache._cp_shared_global_reads, {})

    def test_release_invalidates_lookup_but_retained_workspace_owner_survives(self):
        bank = _bank()
        cache = _cache(bank)
        cache._fresh_swa[20, 0] = object()
        # An ordinary CP read must not erase its valid CP bank merely because
        # the separate local-Prefill cache is ineligible.
        self.assertFalse(cache._check_local_source_reads())
        self.assertIs(cache.read_shared_global(20), bank)
        cache.release_swa(20)
        self.assertEqual(cache._fresh_swa, {})
        self.assertIsNone(cache.read_shared_global(20))
        self.assertTrue(torch.equal(bank.materialize(1, 14)[:8], bank.packed[:8]))


def _world8_worker(rank, store):
    import torch.distributed as dist
    from rtp_llm.models_py.modules.dsv41.ppu import _quant, kv_codec

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="file://" + store, rank=rank, world_size=8,
                            timeout=timedelta(seconds=180))
    try:
        device = torch.device("cuda", rank)
        owner_rows, count, prefix, width = 8, 67, 64, 288
        stride = owner_rows * width + 16
        tensor = torch.full((10, stride), 231, dtype=torch.uint8, device=device)

        def rows(first, length, request):
            position = torch.arange(first, first + length, device=device)[:, None]
            columns = torch.arange(width, device=device)[None]
            result = ((position * 13 + columns + request * 23) % 251).to(torch.uint8)
            result[:, 256:] = 56  # finite E4M3 scale1; all E2M1 nibbles are finite.
            return result

        tensor[1, :owner_rows * width] = rows(rank * owner_rows, owner_rows, 0).flatten()
        if rank == 0:
            for request in range(8):
                tensor[2 + request, :3 * width] = rows(prefix, 3, request + 1).flatten()
        table = torch.tensor([[1, 2 + request] for request in range(8)], dtype=torch.int32, device=device)
        pool = NS(span=8, entries=8, tensor=tensor)
        placement = NS(size=8, rank=rank, ratio=1, physical_span=8, pages=1, table=table)
        packets = []

        def gather(local):
            packets.append(tuple(local.shape))
            output = torch.empty((8 * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=local.device)
            dist.all_gather_into_tensor(output, local, group=dist.group.WORLD)
            return output

        options = dict(width=width, planar=False, shared_prefix_stripes=1, receive=NS(gather=gather))
        legacy = prefill_readback.gather_pool(pool, placement, (count,) * 8, **options)
        compact = prefill_readback.gather_pool(pool, placement, (count,) * 8,
                                             compact_shared_prefix=True, **options)
        assert packets[0] == packets[1]
        assert compact.packed.shape == (prefix + 8 * 3, 288)
        decoded = torch.empty((compact.packed.shape[0], 512), dtype=torch.bfloat16, device=device)
        assert _quant.try_dequantize_prefill_rows(compact.packed, decoded, global_pool=True)
        for request in range(8):
            expected = torch.cat((rows(0, prefix, 0), rows(prefix, 3, request + 1)))
            assert torch.equal(legacy[request], expected)
            assert torch.equal(compact.materialize(request, count), expected)
            reference = torch.empty((count, 512), dtype=torch.bfloat16, device=device)
            kv_codec.decode_kv(expected, pool_tag="global_kv_1", out=reference)
            mapping = torch.cat((torch.arange(prefix, device=device),
                                 compact.tail_offsets[request] + torch.arange(3, device=device)))
            torch.testing.assert_close(decoded[mapping], reference, rtol=0, atol=0)
        positions = torch.full((8,), 66, dtype=torch.int64, device=device)
        selected = torch.tensor([[0, 63, 64, 66]] * 8, dtype=torch.int32, device=device)
        metadata = torch.tensor([[0, count, compact.packed.shape[0] + rid * 4, 63,
                                  prefix, compact.tail_offsets[rid]] for rid in range(8)],
                                dtype=torch.int64, device=device)
        actual, lengths = shared_global.assemble_indices(metadata, positions, selected, 1, 4, 640)
        for request in range(8):
            tail = compact.tail_offsets[request]
            swa = compact.packed.shape[0] + request * 4
            expected = torch.tensor([0, 63, tail, tail + 2, swa, swa + 1, swa + 2, swa + 3],
                                    device=device, dtype=torch.int32)
            assert torch.equal(actual[request, 0, :8], expected)
            assert bool((actual[request, 0, 8:] == -1).all()) and int(lengths[request]) == 8
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1", "explicit eight-PPU lease required")
class CP8SharedGlobalWorld8(unittest.TestCase):
    def test_actual_eight_rank_readback_preserves_packed_and_bf16_rows(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory(prefix="cp8-shared-global-") as directory:
            torch.multiprocessing.spawn(_world8_worker, args=(str(Path(directory) / "store"),),
                                        nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
