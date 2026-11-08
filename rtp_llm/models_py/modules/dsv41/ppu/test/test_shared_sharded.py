"""One PPU byte-domain boundary; synthetic pools, actual shared cache methods.

This does not emulate RDMA or prove distributed collectives. The native
one-request smoke supplies those observations independently.
"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.native_cache import _FULL, _WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.shared_native import SharedNativeCache


class _BoundaryPools:
    def __init__(self, rank, device):
        self.group_tags = ["hca_kv", "csa_kv", "indexer_kv", "swa_kv", "csa_state"]
        self.handles, self.inputs = {}, {}
        self.entries = {"hca_kv": 8, "csa_kv": 4, "indexer_kv": 4, "swa_kv": 8, "csa_state": 1}
        self.physical = {"hca_kv": 32, "csa_kv": 32, "indexer_kv": 16, "swa_kv": 64, "csa_state": 64}
        self.kernel = {"hca_kv": 8, "csa_kv": 8, "indexer_kv": 4, "swa_kv": 64, "csa_state": 64}
        for layer, tag in ((0, "hca_kv"), (1, "csa_kv"), (0, "indexer_kv"), (1, "indexer_kv"),
                           (0, "swa_kv"), (1, "swa_kv"), (2, "swa_kv"), (1, "csa_state")):
            width = {"hca_kv": 288, "csa_kv": 288, "indexer_kv": 68, "swa_kv": 528, "csa_state": 1024}[tag]
            # A deliberately unaligned SWA slice splits rows across peers.
            stride = 531 if tag == "swa_kv" else self.entries[tag] * width
            if tag in ("hca_kv", "csa_kv", "indexer_kv"):
                stride += 16
            pages = self.physical[tag] // self.kernel[tag]
            dtype = torch.float32 if tag == "csa_state" else torch.uint8
            value = torch.full((4 * pages, stride), 173, device=device, dtype=dtype)
            self.handles[layer, tag] = NS(kv_cache_base=value, tag=tag, layer_id=layer)
            ids = [2, 1, 3] if tag in ("hca_kv", "csa_kv", "indexer_kv") else [2, 2, 2]
            physical = torch.tensor([ids], device=device, dtype=torch.int32)
            kernel = (physical[:, :, None] * pages + torch.arange(pages, device=device)).reshape(1, -1).int()
            self.inputs[tag] = NS(kv_cache_block_id=physical, kv_cache_block_id_device=physical,
                                  kv_cache_kernel_block_id=kernel, kv_cache_kernel_block_id_device=kernel)

    def get_layer_cache(self, layer, tag):
        return self.handles[layer, tag]

    def get_entries_per_block(self, tag):
        return self.entries[tag]

    def get_seq_size_per_block(self, tag):
        return self.physical[tag]

    def get_kernel_seq_size_per_block(self, tag):
        return self.kernel[tag]


class SharedShardedBoundary(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_sparse_history_retains_fresh_rows_and_requires_live_blocks(self):
        self.assertTrue(torch.cuda.is_available(), "this boundary requires a PPU")
        device = torch.device("cuda", 0)
        geometry = NS(num_layers=3, kv_source_layer_ids=(0, 1), window=8,
                      ratio=lambda layer: 1 if layer == 0 else 2)
        rank = 4
        native = _BoundaryPools(rank, device)
        config = PrefillPolicy().freeze({"_ppu_cp_cache": {
            "enabled": True, "size": 8, "rank": rank, "sliced_prefill": True,
        }})
        cache = SharedNativeCache(config, geometry, native, native.inputs, (189,),
                                  device=device, prefill=True)
        swa, csa = cache.pools[0, "swa_kv"], cache.pools[1, "csa_state"]
        for pool in (swa, csa):
            pool.table[0] = torch.tensor([-1, 2, 3], device=device)
        positions = torch.arange(189, device=device)
        requests = torch.zeros_like(positions)
        fresh = (positions[:, None] % 151).expand(-1, _WIDTH["swa_kv"]).to(torch.uint8).contiguous()
        cache.stage_swa(0, positions, requests, fresh, (0,))
        self.assertTrue(torch.equal(cache.read_swa(0, 0, positions), fresh))
        self.assertEqual(cache.writes[-1]["absolute_positions"], list(range(120, 128)) + list(range(181, 189)))
        self.assertTrue(bool((swa.tensor[0] == 173).all()))

        csa_rows = positions[:, None].expand(-1, _WIDTH["csa_state"]).float().contiguous()
        cache.insert_rows(1, "csa_state", cache.slots_at_positions(1, "csa_state", 0, positions), csa_rows)
        self.assertTrue(bool((csa.tensor[0] == 173).all()))
        self.assertTrue(bool((csa.tensor[3] == 188).all()))

        read_positions = torch.tensor([127, 129, 188], device=device)
        width = _WIDTH["swa_kv"]
        assembled = (torch.arange(2 * 8 * swa.tensor.shape[1], device=device) % 151).reshape(2, -1).to(torch.uint8)
        with patch("rtp_llm.models_py.modules.dsv41.ppu.shared_sharded.cp_gather_request_pool_blocks",
                   return_value=assembled.reshape(16, -1)) as gather:
            actual = cache._copy_rows((0, "swa_kv"), 0, read_positions)
            args = gather.call_args.args
            self.assertEqual(args[1].tolist(), [2, 3])
            self.assertEqual(args[2:], (8, rank, 16))
            expected = torch.stack([assembled[i, offset * width:(offset + 1) * width]
                                    for i, offset in ((0, 7), (1, 1), (1, 4))])
            self.assertTrue(torch.equal(actual, expected))
            with self.assertRaisesRegex(ValueError, "read refers to an unallocated physical block"):
                cache._copy_rows((0, "swa_kv"), 0, positions[:1])
            self.assertEqual(gather.call_count, 1)

        swa.table[0, -1] = csa.table[0, -1] = -1
        with self.assertRaisesRegex(ValueError, "missing a live ring block"):
            cache.stage_swa(0, positions, requests, fresh, (0,))
        with self.assertRaisesRegex(ValueError, "missing a live ring block"):
            cache.insert_rows(1, "csa_state", cache.slots_at_positions(1, "csa_state", 0, positions), csa_rows)

    @torch.inference_mode()
    def test_per_tag_pages_odd_carry_and_cross_row_slices(self):
        self.assertTrue(torch.cuda.is_available(), "this boundary requires a PPU")
        device = torch.device("cuda", 0)
        geometry = NS(num_layers=3, kv_source_layer_ids=(0, 1), window=8,
                      ratio=lambda layer: 1 if layer == 0 else 2)
        total_owned = {}
        for rank in range(8):
            native = _BoundaryPools(rank, device)
            config = PrefillPolicy().freeze({"_ppu_cp_cache": {
                "enabled": True, "size": 8, "rank": rank, "sliced_prefill": True,
            }})
            cache = SharedNativeCache(config, geometry, native, native.inputs, (289,), device=device)
            for key, pool in cache.pools.items():
                tag, width = key[1], _WIDTH[key[1]]
                if tag in _FULL:
                    ratio = geometry.ratio(key[0])
                    positions = list(range(0, (289 // ratio) * ratio, ratio))
                elif tag == "swa_kv":
                    positions = list(range(57, 65))
                else:
                    # Same actual slot, unordered rows; absolute maximum wins.
                    positions = [64, 0, 63, 1, 65, 2]
                raw = torch.tensor(positions, device=device)
                rows = ((raw[:, None] * 7 + torch.arange(width, device=device)) % 151).to(pool.tensor.dtype)
                expected = pool.tensor.clone()
                owned_rows = 0
                # Independent scalar byte-domain reference; no production
                # placement/coordinate/planar helper participates in it.
                latest = {}
                for index, position in enumerate(positions):
                    if tag in _FULL:
                        physical = position // native.physical[pool.framework_tag]
                        owner = physical % 8
                        if owner != rank:
                            continue
                        page = (position % native.physical[pool.framework_tag]) // pool.span
                        physical_id = [2, 1, 3][physical // 8]
                        block = physical_id * (native.physical[pool.framework_tag] // pool.span) + page
                        offset = (position % pool.span) // geometry.ratio(key[0])
                        columns = ([offset * 64 + col for col in range(64)]
                                   + [pool.entries * 64 + offset * 4 + col for col in range(4)]) if tag == "indexer_kv" else list(range(offset * width, (offset + 1) * width))
                        expected[block, columns] = rows[index]
                        owned_rows += 1
                    else:
                        entries = 8
                        stride = expected.shape[1]
                        first = (position % entries) * width
                        left, right = max(first, rank * stride), min(first + width, (rank + 1) * stride)
                        if left >= right:
                            continue
                        slot = (2, left - rank * stride)
                        if tag == "csa_state" and slot in latest and latest[slot][0] > position:
                            continue
                        latest[slot] = (position, index, first, left, right)
                for (block, _), (position, index, first, left, right) in latest.items():
                    stride = expected.shape[1]
                    expected[block, left - rank * stride:right - rank * stride] = rows[index, left - first:right - first]
                slots = cache.slots_at_positions(*key, 0, raw)
                cache.insert_rows(*key, slots, rows)
                self.assertTrue(torch.equal(pool.tensor, expected), (rank, key))
                self.assertEqual(pool.tensor.data_ptr(), pool.handle.kv_cache_base.data_ptr())
                if tag in _FULL:
                    total_owned[key] = total_owned.get(key, 0) + owned_rows
                if tag == "csa_state" and rank == 0:
                    carry = pool.tensor[2]
                    self.assertTrue(torch.equal(carry, rows[0]), "latest odd-boundary carry was lost")
            self.assertFalse(cache._fresh_swa or cache._source_reads)
        for key, rows in total_owned.items():
            self.assertEqual(rows, 289 // geometry.ratio(key[0]))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
