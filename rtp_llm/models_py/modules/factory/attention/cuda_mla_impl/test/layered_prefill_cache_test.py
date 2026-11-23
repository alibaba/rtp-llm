import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
    ContiguousPrefillCache,
    _local_lengths,
    _plan_pages,
)


class PagePlanTest(unittest.TestCase):
    def test_stages_only_live_pages_and_preserves_partial_writes(self):
        # The next chunks' pages are already reserved. A repeated write slot
        # cannot make the partial page 10 look wholly overwritten.
        table = torch.tensor([[5, 10, 55], [17, 60, 61]])
        plan = _plan_pages(
            table,
            torch.tensor([7, 1]),
            torch.tensor([20, 21, 22, 23, 40, 40, 42, 68, -1]),
            page_size=4,
            capacity=3,
            logical_pages=64,
        )
        self.assertEqual(plan.table.tolist(), [[0, 1, -1], [2, -1, -1]])
        self.assertEqual(plan.slots.tolist(), [0, 1, 2, 3, 4, 4, 6, 8, -1])
        self.assertEqual(plan.fetch, [(10, 1, 1), (17, 2, 1)])
        self.assertEqual(plan.write, [(0, 5, 1), (1, 10, 1), (2, 17, 1)])
        self.assertEqual(table.tolist(), [[5, 10, 55], [17, 60, 61]])

    def test_empty_cp_owner_and_multi_page_physical_blocks(self):
        lengths = torch.tensor([4, 19, 33])
        self.assertEqual(_local_lengths(lengths, 8, 3, 0).tolist(), [4, 8, 16])
        self.assertEqual(_local_lengths(lengths, 8, 3, 1).tolist(), [0, 8, 9])
        self.assertEqual(_local_lengths(lengths, 8, 3, 2).tolist(), [0, 3, 8])
        plan = _plan_pages(
            torch.zeros(1, 3, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([-1, -1]),
            page_size=4,
            capacity=1,
            logical_pages=16,
        )
        self.assertEqual(plan.table.tolist(), [[-1, -1, -1]])
        self.assertEqual(plan.slots.tolist(), [-1, -1])
        self.assertEqual(plan.fetch, [])
        self.assertEqual(plan.write, [])

    def test_invalid_live_set_and_overflow(self):
        table = torch.tensor([[4, 5]])
        lengths = torch.tensor([8])
        with self.assertRaisesRegex(ValueError, "staging capacity"):
            _plan_pages(table, lengths, torch.tensor([16]), 4, 1, 6)
        with self.assertRaisesRegex(ValueError, "logical cache capacity"):
            _plan_pages(table, lengths, torch.tensor([16]), 4, 2, 5)
        with self.assertRaisesRegex(ValueError, "live prefill cache pages"):
            _plan_pages(table, lengths, torch.tensor([12]), 4, 2, 6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LayeredPrefillCacheTest(unittest.TestCase):
    def _fixture(self, sharded):
        device = torch.device("cuda", torch.cuda.current_device())
        backing = [
            torch.full(
                (8, 4, 8),
                layer,
                device=device if layer == 0 else "cpu",
                pin_memory=layer > 0,
            )
            for layer in range(3)
        ]
        cache = ContiguousPrefillCache(backing, 1, 1, device)
        table = torch.tensor([[4, 6]], dtype=torch.int32)
        inputs = SimpleNamespace(
            is_prefill=True, kv_cache_kernel_block_id=table, kv_cache_block_id=table
        )
        params = SimpleNamespace(
            prefill_ragged_kv_len_indptr_h=torch.tensor([0, 4]),
            slot_mapping=torch.tensor([17, 18], device=device, dtype=torch.int32),
        )
        op = SimpleNamespace(block_table=table.to(device))
        fmha = SimpleNamespace(
            attn_inputs=inputs,
            fmha_params=params,
            rope_params=params,
            fmha_impl=op,
            write_cache_store_impl=None,
        )
        if sharded:
            op.kv_cache_sharded = True
            op.kv_owner_tokens_per_block = 8
            op.prefill_cp_size = 2
            op.prefill_cp_rank = 0
            op.sharded_slot_mapping = params.slot_mapping.clone()
            op.attn_inputs = inputs
            op.write_cache_store_impl = None
        return cache, backing, fmha

    def test_tagged_writer_and_slot_reuse_for_cp_and_non_cp(self):
        from rtp_llm.ops.compute_ops import LayerKVCache

        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                cache, backing, fmha = self._fixture(sharded)
                op = fmha.fmha_impl
                original_table = op.block_table
                writer_owner = op if sharded else fmha
                publications = []

                def store(layer_cache):
                    ready = torch.cuda.Event()
                    ready.record()
                    publications.append((layer_cache, ready))

                writer_owner.write_cache_store_impl = store
                with cache.forward(fmha):
                    for layer in range(3):
                        canonical = LayerKVCache(backing[layer], 8, layer, "default")
                        with cache.attention(layer, fmha, canonical) as staged:
                            if layer == 0:
                                self.assertIs(staged, canonical)
                                continue
                            self.assertEqual(staged.layer_id, layer)
                            self.assertEqual(staged.tag, "default")
                            self.assertEqual(staged.seq_size_per_block, 8)
                            self.assertIsNone(writer_owner.write_cache_store_impl)
                            self.assertEqual(op.block_table.cpu().tolist(), [[0, -1]])
                            slots = (
                                op.sharded_slot_mapping
                                if sharded
                                else fmha.rope_params.slot_mapping
                            )
                            self.assertEqual(slots.cpu().tolist(), [1, 2])
                            staged.kv_cache_base[0, 1:3].fill_(10 + layer)

                torch.cuda.current_stream().synchronize()
                self.assertIs(op.block_table, original_table)
                self.assertIs(writer_owner.write_cache_store_impl, store)
                self.assertIsNone(fmha.layered_prefill_cache)
                self.assertEqual(fmha.rope_params.slot_mapping.cpu().tolist(), [17, 18])
                if sharded:
                    self.assertEqual(op.sharded_slot_mapping.cpu().tolist(), [17, 18])
                self.assertEqual([layer.layer_id for layer, _ in publications], [1, 2])
                for layer_cache, event in publications:
                    layer = layer_cache.layer_id
                    self.assertTrue(event.query())
                    self.assertEqual(
                        layer_cache.kv_cache_base.data_ptr(), backing[layer].data_ptr()
                    )
                    self.assertEqual(
                        backing[layer][4, :, 0].tolist(),
                        [layer, 10 + layer, 10 + layer, layer],
                    )
                    self.assertEqual(backing[layer][6, :, 0].tolist(), [layer] * 4)

    def test_reused_logical_page_reloads_canonical_bytes_for_next_chunk(self):
        from rtp_llm.ops.compute_ops import LayerKVCache

        cache, backing, fmha = self._fixture(False)
        for expected in (31, 42):
            for layer in range(1, 3):
                backing[layer][4, 0, :].fill_(expected)
            with cache.forward(fmha):
                for layer in range(3):
                    canonical = LayerKVCache(backing[layer], 8, layer)
                    with cache.attention(layer, fmha, canonical) as staged:
                        if layer > 0:
                            self.assertEqual(
                                staged.kv_cache_base[0, 0, 0].item(), expected
                            )
            torch.cuda.current_stream().synchronize()

    def test_empty_cp_owner_still_enters_every_layer_without_new_copies(self):
        from rtp_llm.ops.compute_ops import LayerKVCache

        cache, backing, fmha = self._fixture(True)
        fmha.fmha_impl.prefill_cp_rank = 1
        fmha.fmha_impl.sharded_slot_mapping.fill_(-1)
        with cache.forward(fmha):
            for layer in range(3):
                canonical = LayerKVCache(backing[layer], 8, layer)
                with cache.attention(layer, fmha, canonical):
                    if layer > 0:
                        self.assertEqual(
                            fmha.fmha_impl.block_table.tolist(), [[-1, -1]]
                        )
                        self.assertEqual(
                            fmha.fmha_impl.sharded_slot_mapping.tolist(), [-1, -1]
                        )
                        self.assertEqual(cache.fetch_runs, [])
                        self.assertEqual(cache.write_runs, [])
        self.assertEqual(fmha.fmha_impl.block_table.tolist(), [[4, 6]])

    def test_local_eager_decode_crosses_kernel_page_boundary(self):
        from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops

        device = torch.device("cuda", torch.cuda.current_device())
        backing = [
            torch.zeros(
                (8, 4, 8),
                dtype=torch.uint8,
                device=device if layer == 0 else "cpu",
                pin_memory=layer > 0,
            )
            for layer in range(2)
        ]
        backing[1][4, :3].fill_(5)
        cache = ContiguousPrefillCache(backing, 2, 1, device)
        # Device-only kernel table and a deliberately different physical table:
        # a physical KV block can cover more than one kernel page.
        table = torch.tensor([[4, 5, 2, 3]], dtype=torch.int32, device=device)
        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        inputs.input_lengths = torch.tensor([3], dtype=torch.int32)
        inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
        inputs.kv_cache_block_id = torch.tensor([[2, 1]], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id = table
        inputs.kv_cache_kernel_block_id_device = table
        op = SimpleNamespace(block_table=table)

        def unexpected_publication(_):
            self.fail("ordinary decode must not publish a cache-store layer")

        for position, expected_page, expected_offset in ((3, 4, 3), (4, 5, 0)):
            inputs.sequence_lengths = torch.tensor([position], dtype=torch.int32)
            params = rtp_llm_ops.SparseMlaParams()
            params.fill_params(inputs, 4, False)
            self.assertEqual(params.prefill_ragged_kv_len_indptr_h.tolist(), [0, 1])
            self.assertEqual(params.kvlen_h.tolist(), [position + 1])
            expected_slot = expected_page * 4 + expected_offset
            self.assertEqual(params.slot_mapping.tolist(), [expected_slot])
            fmha = SimpleNamespace(
                attn_inputs=inputs,
                fmha_params=params,
                rope_params=params,
                fmha_impl=op,
                write_cache_store_impl=unexpected_publication,
            )
            with cache.forward(fmha):
                with cache.attention(1, fmha, LayerKVCache(backing[1], 8, 1)) as staged:
                    staging_page = 0 if position < 4 else 1
                    self.assertEqual(
                        params.slot_mapping.tolist(),
                        [staging_page * 4 + expected_offset],
                    )
                    self.assertEqual(
                        op.block_table.tolist(),
                        [[0, -1, -1, -1]] if position < 4 else [[0, 1, -1, -1]],
                    )
                    self.assertEqual(staged.kv_cache_base[0, :3, 0].tolist(), [5, 5, 5])
                    staged.kv_cache_base[staging_page, expected_offset].fill_(
                        position + 10
                    )
            torch.cuda.current_stream().synchronize()
            self.assertIs(op.block_table, table)
            self.assertEqual(params.slot_mapping.tolist(), [expected_slot])
            self.assertEqual(
                backing[1][expected_page, expected_offset, 0].item(), position + 10
            )

    def test_failure_restores_metadata_and_drains_streams(self):
        from rtp_llm.ops.compute_ops import LayerKVCache

        cache, backing, fmha = self._fixture(False)
        original_table = fmha.fmha_impl.block_table
        with self.assertRaisesRegex(RuntimeError, "attention failed"):
            with cache.forward(fmha):
                with cache.attention(1, fmha, LayerKVCache(backing[1], 8, 1)):
                    raise RuntimeError("attention failed")
        self.assertIs(fmha.fmha_impl.block_table, original_table)
        self.assertEqual(fmha.rope_params.slot_mapping.cpu().tolist(), [17, 18])
        self.assertEqual(backing[1][4, :, 0].tolist(), [1] * 4)
        self.assertFalse(cache.active)


if __name__ == "__main__":
    unittest.main()
