"""Private-cache isolation, request lifecycle and captured stream dependencies."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.csa_cache import CsaLayerCache, CsaRequestSlots
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_kv_quant_decode_op import (
    read_model1_kv_slot_bytes,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CsaPrivateCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        entries = 32
        stride = ((entries * 584 + 575) // 576) * 576
        self.storage = torch.randint(
            0, 256, (128, stride), dtype=torch.uint8, pin_memory=True
        )
        self.source = self.storage.as_strided((128, entries, 584), (stride, 584, 1))
        self.requests = CsaRequestSlots(128, 3, "cuda")
        self.cache = CsaLayerCache(
            self.source,
            self.requests,
            budget_bytes=220000,
            topk=32,
            hot_entries=64,
            fetch_ctas=4,
        )
        self.assertLessEqual(self.cache.allocated_bytes, 220000)
        self.assertLess(self.cache.resident_tokens, self.source.shape[0] * entries)
        self.table = torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
        self.positions = torch.tensor([128, 256], dtype=torch.int32, device="cuda")
        self.requests.begin_prefill(self.table)
        self.requests.register(self.table, self.positions)
        self.cache.mirror_writes(
            torch.arange(self.cache.resident_tokens, dtype=torch.int64, device="cuda")
        )

    def _choices(self, offset=0):
        start = self.cache.resident_tokens + offset
        return torch.arange(
            start, start + 32, dtype=torch.int32, device="cuda"
        ).unsqueeze(0) + torch.tensor([[0], [512]], device="cuda", dtype=torch.int32)

    def _stage(self, choices):
        deferred = torch.full((choices.shape[0],), -1, dtype=torch.int64, device="cuda")
        result = self.cache.prefetch(choices, deferred)
        self.cache.mirror_writes(deferred)
        torch.cuda.synchronize()
        self._assert_bytes(choices)
        return result

    def _assert_bytes(self, choices):
        choices_cpu = choices.cpu()
        slots = self.cache.slots[: choices.shape[0]].cpu()
        pool = self.cache.pool.cpu()
        for row in range(choices.shape[0]):
            for col in range(choices.shape[1]):
                source_slot = int(choices_cpu[row, col])
                actual_slot = int(slots[row, 0, col])
                if source_slot < 0:
                    self.assertEqual(actual_slot, -1)
                    continue
                expected = read_model1_kv_slot_bytes(
                    self.source, source_slot // 32, source_slot % 32
                )
                actual = read_model1_kv_slot_bytes(
                    pool, actual_slot // 32, actual_slot % 32
                )
                self.assertTrue(
                    torch.equal(actual, expected), (row, col, source_slot, actual_slot)
                )

    def test_native_hits_and_cross_step_hot_hits(self):
        choices = self._choices()
        choices[0, :8] = torch.arange(8, device="cuda")
        choices[1, -1] = -1
        self._stage(choices)
        self.assertEqual(self.cache.counts[:2].cpu().tolist(), [[8, 0, 24], [0, 0, 31]])
        self._stage(choices)
        self.assertEqual(self.cache.counts[:2].cpu().tolist(), [[8, 24, 0], [0, 31, 0]])
        self.assertEqual(int(self.cache.miss_count), 0)

    def test_byte_validator_detects_payload_corruption(self):
        from rtp_llm.models_py.modules.dsv4.fp8.kv_offload import CsaByteValidator

        choices = self._choices()
        choices[0, 0] = 0
        choices[1, -1] = -1
        self._stage(choices)
        validator = CsaByteValidator("cuda")
        validator.check(self.source, choices, self.cache.pool, self.cache.slots[:2])
        self.assertEqual(int(validator.errors), 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            validator.check(
                self.source,
                choices,
                self.cache.pool,
                self.cache.slots[:2],
                assertions=False,
            )
        self.cache._storage[0, 0] ^= 1
        graph.replay()
        self.assertEqual(int(validator.errors), 1)

    def test_eviction_is_private_and_protects_current_topk(self):
        original = self._choices()
        self._stage(original)
        for step in range(1, 8):
            choices = original.clone()
            choices[0] += step * 24
            self._stage(choices)
            self.assertEqual(int(self.cache.counts[1, 2]), 0)
            rows = self.requests.rows[:2].cpu().tolist()
            for row in range(2):
                slots = self.cache.slots[row, 0] - self.cache.resident_tokens
                self.assertTrue(
                    ((slots >= rows[row] * 64) & (slots < (rows[row] + 1) * 64))
                    .all()
                    .item()
                )

    def test_batch_reorder_shrink_and_same_page_reuse(self):
        choices = self._choices()
        self._stage(choices)
        self.table.copy_(self.table.flip(0))
        self.requests.register(self.table, self.positions)
        self._stage(choices.flip(0).contiguous())
        self.assertEqual(int(self.cache.miss_count), 0)
        self.requests.register(self.table[:1], self.positions[:1])
        self._stage(choices[1:])
        self.assertEqual(int(self.cache.miss_count), 0)
        torch.cuda.synchronize()
        self.storage.random_(0, 256)
        self.requests.begin_prefill(self.table)
        self.requests.register(self.table, self.positions)
        self._stage(choices.flip(0).contiguous())
        self.assertEqual(int(self.cache.miss_count), 64)

    def test_new_compressed_entry_is_deferred_until_writer(self):
        choices = self._choices()
        pending = choices[:, -1].to(torch.int64)
        self.cache.prefetch(choices, pending)
        self.cache.wait()
        torch.cuda.synchronize()
        for token in pending.cpu().tolist():
            block, offset = divmod(token, 32)
            self.storage[block, offset * 576 : (offset + 1) * 576] = 73
            self.storage[block, 32 * 576 + offset * 8 : 32 * 576 + (offset + 1) * 8] = (
                97
            )
        self.cache.mirror_writes(pending)
        torch.cuda.synchronize()
        self._assert_bytes(choices)

    def test_graph_replay_with_cross_stream_fetch_and_request_reuse(self):
        choices = self._choices()
        deferred = torch.full((2,), -1, dtype=torch.int64, device="cuda")
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            self.requests.register(self.table, self.positions)
            self.cache.prefetch(choices, deferred)
            self.cache.mirror_writes(deferred)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.requests.register(self.table, self.positions)
            self.cache.prefetch(choices, deferred)
            self.cache.mirror_writes(deferred)
        addresses = (
            self.cache.pool.data_ptr(),
            self.cache.slots.data_ptr(),
            self.requests.rows.data_ptr(),
        )
        for step in range(4):
            choices.copy_(self._choices(step * 16))
            if step == 2:
                torch.cuda.synchronize()
                self.storage.random_(0, 256)
                self.requests.begin_prefill(self.table)
            graph.replay()
            torch.cuda.synchronize()
            self._assert_bytes(choices)
            self.assertEqual(
                addresses,
                (
                    self.cache.pool.data_ptr(),
                    self.cache.slots.data_ptr(),
                    self.requests.rows.data_ptr(),
                ),
            )

    def test_real_compressor_writes_pinned_backing_and_updates_hot_cache(self):
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import build_decode_metadata
        from rtp_llm.models_py.modules.dsv4.fp8.test.test_compressor_fp8_per_token import (
            _bind_pools,
            _build_compressor,
            _WS,
        )

        cmp = _build_compressor(
            dim=128, head_dim=512, rope_head_dim=64, compress_ratio=4
        )
        state, _, _ = _bind_pools(
            cmp, seqlen=32, head_dim=512, coff=2, compress_ratio=4
        )
        block = self.cache.resident_tokens // 32 + 2
        cmp._kv_pool_view = self.source
        cmp._kv_eb = 32
        cmp._kv_tokens_per_block = 128
        cmp._kv_block_table = torch.tensor([[block]], dtype=torch.int32, device="cuda")
        cmp.csa_offload = self.cache
        self.requests.begin_prefill(cmp._kv_block_table)
        positions = torch.tensor([32], dtype=torch.int64, device="cuda")
        self.requests.register(cmp._kv_block_table, positions)
        x = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        cmp.forward(x, 0, workspace=_WS)
        for position in range(32, 36):
            positions.fill_(position)
            meta = build_decode_metadata(cmp, positions, 1)
            choices = torch.full((1, 32), -1, dtype=torch.int32, device="cuda")
            n = (position + 1) // 4
            choices[0, :n] = torch.arange(block * 32, block * 32 + n, device="cuda")
            self.cache.prefetch(choices, meta.kv_slots)
            cmp.forward_decode_vectorized(
                x[:, :1], positions, position_ids=positions.reshape(1, 1), meta=meta
            )
            torch.cuda.synchronize()
            self._assert_bytes(choices)

    def test_pro_topk_and_private_capacity_at_batch_64(self):
        entries, batch, topk = 64, 64, 1024
        stride = ((entries * 584 + 575) // 576) * 576
        storage = torch.randint(
            0, 256, (4096, stride), dtype=torch.uint8, pin_memory=True
        )
        source = storage.as_strided((4096, entries, 584), (stride, 584, 1))
        requests = CsaRequestSlots(4096, batch, "cuda")
        cache = CsaLayerCache(source, requests, budget_bytes=100 * 1024**2, topk=topk)
        table = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda").reshape(
            batch, 1
        )
        requests.begin_prefill(table)
        requests.register(
            table, torch.full((batch,), 131072, dtype=torch.int32, device="cuda")
        )
        choices = (
            cache.resident_tokens
            + torch.arange(topk, dtype=torch.int32, device="cuda")[None, :]
            + 2048 * torch.arange(batch, dtype=torch.int32, device="cuda")[:, None]
        )
        deferred = torch.full((batch,), -1, dtype=torch.int64, device="cuda")
        for shift, misses in ((0, batch * topk), (0, 0), (128, batch * 128)):
            cache.prefetch(choices + shift, deferred)
            cache.wait()
            torch.cuda.synchronize()
            self.assertEqual(int(cache.miss_count), misses)
            for row, col in ((0, 0), (31, 512), (63, 1023)):
                src = int(choices[row, col]) + shift
                dst = int(cache.slots[row, 0, col])
                expected = read_model1_kv_slot_bytes(
                    source, src // entries, src % entries
                )
                actual = read_model1_kv_slot_bytes(
                    cache.pool, dst // entries, dst % entries
                ).cpu()
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
