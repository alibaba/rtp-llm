import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
    SparseMlaImpl,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs, rtp_llm_ops


class DecodeMetadataContractTest(unittest.TestCase):
    def test_empty_cp_rank_participates_without_restoring_unused_kv(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            flashmla_sparse_cp_impl as cp,
        )

        op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
        op._gather = SimpleNamespace(batch_size=1)
        op.kv_lora_rank, op.qk_rope_head_dim = 512, 64
        op.sharded_local_kv_lens = torch.tensor([1], dtype=torch.int32)
        op.sharded_kv_restore_indices = torch.tensor([0])
        op.sharded_actual_local_kv_lens = torch.tensor([0], dtype=torch.int32)
        op.sharded_actual_workspace_starts = torch.tensor([0], dtype=torch.int32)
        op.sharded_total_local_kv_len = 1
        op.sharded_actual_total_local_kv_len = 0
        cache = SimpleNamespace(
            kv_cache_base=torch.empty((0, 64, 656), dtype=torch.uint8)
        )
        # This sentinel permits reshape but deliberately cannot be indexed.
        gathered = Mock()
        with patch.object(cp, "all_gather", return_value=gathered) as collective:
            op._gather_sharded_kv_cache(cache, None)
        torch.testing.assert_close(
            collective.call_args.args[0], torch.zeros((1, 576), dtype=torch.bfloat16)
        )
        gathered.reshape.assert_called_once_with(-1, 576)

    def test_multi_token_ragged_metadata_and_graph_replay(self):
        for batch in (3, 257):
            with self.subTest(batch=batch):
                query = torch.tensor(
                    [1 + row % 3 for row in range(batch)], dtype=torch.int32
                )
                prefix = torch.tensor(
                    [63 + row % 3 for row in range(batch)], dtype=torch.int32
                )
                table = torch.arange(1, batch * 3 + 1, dtype=torch.int32).reshape(
                    batch, 3
                )
                inputs = PyAttentionInputs()
                inputs.is_prefill = True
                inputs.is_target_verify = True
                inputs.input_lengths = query
                inputs.prefix_lengths = prefix
                inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
                inputs.kv_cache_block_id = table
                inputs.kv_cache_kernel_block_id = table
                params = rtp_llm_ops.SparseMlaParams()
                params.fill_params(inputs, 64)
                query_d, prefix_d, table_d = query.cuda(), prefix.cuda(), table.cuda()

                def fill():
                    params.fill_multi_token_decode_cuda_graph_params(
                        query_d, prefix_d, table_d, 64
                    )

                fill()
                pointers = (
                    params.positions_d.data_ptr(),
                    params.slot_mapping.data_ptr(),
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fill()
                for step in (0, 1):
                    prefix_d.copy_(prefix + step)
                    graph.replay()
                    positions, slots, batches, ks, ke = [], [], [], [], []
                    qptr, kptr, pageptr, pages = [0], [0], [0], []
                    for row, (q, p) in enumerate(
                        zip(query.tolist(), (prefix + step).tolist())
                    ):
                        positions.extend(range(p, p + q))
                        batches.extend([row] * q)
                        slots.extend(
                            int(table[row, pos // 64]) * 64 + pos % 64
                            for pos in range(p, p + q)
                        )
                        ks.extend([kptr[-1]] * q)
                        ke.extend(kptr[-1] + pos + 1 for pos in range(p, p + q))
                        qptr.append(qptr[-1] + q)
                        kptr.append(kptr[-1] + p + q)
                        count = (p + q + 63) // 64
                        pageptr.append(pageptr[-1] + count)
                        pages.extend(table[row, :count].tolist())
                    for tensor, expected in (
                        (params.positions_d, positions),
                        (params.slot_mapping, slots),
                        (params.batch_indice_d, batches),
                        (params.expanded_seq_lens, [p + 1 for p in positions]),
                        (params.ks, ks),
                        (params.ke, ke),
                        (params.topk_indices_offset, [0] * len(positions)),
                        (params.qo_indptr_d, qptr),
                        (params.prefill_ragged_kv_len_indptr_d, kptr),
                        (params.decode_page_indptr_d, pageptr),
                        (params.page_indice_d[: len(pages)], pages),
                    ):
                        self.assertEqual(tensor.cpu().tolist(), expected)
                    self.assertEqual(
                        (params.positions_d.data_ptr(), params.slot_mapping.data_ptr()),
                        pointers,
                    )

    def test_graph_prepare_passes_device_lengths_for_each_multi_token_phase(self):
        for phase in ("verify", "draft_extend", "prefill"):
            with self.subTest(phase=phase):
                impl = SparseMlaImpl.__new__(SparseMlaImpl)
                impl.fmha_impl = SparseMlaFp8Op.__new__(SparseMlaFp8Op)
                impl.fmha_impl.plan = Mock()
                impl.seq_size_per_block = 64
                impl._cuda_dag_indexer_metadata = None
                impl._refresh_paged_mqa_schedule_metadata = Mock()

                def check_device_lengths(query, prefix, table, page_size):
                    self.assertTrue(query.is_cuda)
                    self.assertTrue(prefix.is_cuda)
                    self.assertTrue(table.is_cuda)
                    self.assertEqual(page_size, 64)

                multi = Mock(side_effect=check_device_lengths)
                prefill = Mock(side_effect=check_device_lengths)
                impl.fmha_params = SimpleNamespace(
                    multi_token_decode_total_tokens=4 if phase != "prefill" else 0,
                    prefill_tokens_per_batch=2,
                    fill_multi_token_decode_cuda_graph_params=multi,
                    fill_target_verify_cuda_graph_params=prefill,
                )
                host_query = torch.tensor([2, 2], dtype=torch.int32)
                host_prefix = torch.tensor([0, 64], dtype=torch.int32)
                inputs = SimpleNamespace(
                    is_prefill=True,
                    is_target_verify=phase == "verify",
                    is_draft_extend=phase == "draft_extend",
                    input_lengths=host_query,
                    prefix_lengths=host_prefix,
                    input_lengths_device=host_query.cuda(),
                    prefix_lengths_device=host_prefix.cuda(),
                    kv_cache_kernel_block_id_device=torch.ones(
                        (2, 3), dtype=torch.int32, device="cuda"
                    ),
                )
                impl.prepare_cuda_graph(inputs)
                selected = prefill if phase == "prefill" else multi
                selected.assert_called_once()
                self.assertIs(selected.call_args.args[0], inputs.input_lengths_device)
                self.assertIs(selected.call_args.args[1], inputs.prefix_lengths_device)

    def make_params(self, sparse):
        params = (
            rtp_llm_ops.SparseMlaParams()
            if sparse
            else rtp_llm_ops.FlashInferMlaAttnParams()
        )
        fill = (
            params.fill_sparse_mla_decode_cuda_graph_params
            if sparse
            else params.fill_decode_cuda_graph_params
        )
        # Replay may only mutate buffers allocated during eager initialization.
        capacity = 257
        rtp_llm_ops.FlashInferMlaAttnParams.fill_params(
            params,
            torch.empty(0, dtype=torch.int32),
            torch.zeros(capacity, dtype=torch.int32),
            torch.ones(capacity, dtype=torch.int32),
            torch.arange(1, capacity * 3 + 1, dtype=torch.int32).reshape(capacity, 3),
            64,
            False,
        )
        return params, fill

    def check_metadata(self, params, fill, batch, sparse):
        lengths = [max(1, [0, 1, 64, 65, 129][row % 5]) for row in range(batch)]
        table = torch.arange(1, batch * 3 + 1, dtype=torch.int32).reshape(batch, 3)
        fill(
            torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            table.cuda(),
            64,
        )
        pages = [(length + 63) // 64 for length in lengths]
        page_ptr = [0]
        page_ids = []
        slots = []
        for row, (length, count) in enumerate(zip(lengths, pages)):
            page_ptr.append(page_ptr[-1] + count)
            page_ids.extend(table[row, :count].tolist())
            position = length - 1
            slots.append(int(table[row, position // 64]) * 64 + position % 64)
        self.assertEqual(params.kvlen_d.cpu().tolist(), lengths)
        self.assertEqual(params.positions_d.cpu().tolist(), [x - 1 for x in lengths])
        self.assertEqual(params.batch_indice_d.cpu().tolist(), list(range(batch)))
        self.assertEqual(params.qo_indptr_d.cpu().tolist(), list(range(batch + 1)))
        self.assertEqual(params.decode_page_indptr_d.cpu().tolist(), page_ptr)
        self.assertEqual(params.page_indice_d[: len(page_ids)].cpu().tolist(), page_ids)
        self.assertEqual(
            params.paged_kv_last_page_len_d.cpu().tolist(),
            [(x - 1) % 64 + 1 for x in lengths],
        )
        if sparse:
            self.assertEqual(params.slot_mapping.cpu().tolist(), slots)
        else:
            self.assertIsNone(params.slot_mapping)

    def test_legacy_decode_accepts_absent_slot_output_in_both_dispatches(self):
        for batch in (5, 257):
            with self.subTest(batch=batch):
                params, fill = self.make_params(False)
                self.check_metadata(params, fill, batch, False)

    def test_sparse_decode_generates_slots_in_both_dispatches(self):
        for batch in (5, 257):
            with self.subTest(batch=batch):
                params, fill = self.make_params(True)
                self.check_metadata(params, fill, batch, True)

    def test_batch_shrink_and_grow_reuse_capture_buffer_addresses(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                params, fill = self.make_params(sparse)
                self.check_metadata(params, fill, 5, sparse)
                pointers = (params.positions_d.data_ptr(), params.kvlen_d.data_ptr())
                slot_pointer = params.slot_mapping.data_ptr() if sparse else None
                for batch in (2, 5):
                    self.check_metadata(params, fill, batch, sparse)
                    self.assertEqual(
                        (params.positions_d.data_ptr(), params.kvlen_d.data_ptr()),
                        pointers,
                    )
                    if sparse:
                        self.assertEqual(params.slot_mapping.data_ptr(), slot_pointer)

    def test_decode_replay_reserves_full_table_width_above_one_million_pages(self):
        # 1M context + MTP reserved tokens gives 16392 kernel blocks per row,
        # while the initial lengths use only 16384. The old length-only
        # reservation fit the 1M floor and failed on the first replay at BS=64.
        batch, width = 64, 16392
        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        inputs.input_lengths = torch.ones(batch, dtype=torch.int32)
        inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
        inputs.sequence_lengths = torch.full((batch,), 1048574, dtype=torch.int32)
        table = torch.arange(batch * width, dtype=torch.int32).reshape(batch, width)
        inputs.kv_cache_block_id = table
        inputs.kv_cache_kernel_block_id = table
        params = rtp_llm_ops.SparseMlaParams()
        params.fill_params(inputs, 64)
        pointers = (params.positions_d.data_ptr(), params.page_indice_d.data_ptr())
        lengths = torch.full((batch,), 65, dtype=torch.int32, device="cuda")
        table_d = table.cuda()
        params.fill_sparse_mla_decode_cuda_graph_params(lengths, table_d, 64)
        self.assertEqual(
            (params.positions_d.data_ptr(), params.page_indice_d.data_ptr()), pointers
        )
        self.assertEqual(params.kvlen_d.cpu().tolist(), [65] * batch)
        self.assertEqual(
            params.decode_page_indptr_d.cpu().tolist(), list(range(0, batch * 2 + 1, 2))
        )
        self.assertEqual(
            params.page_indice_d[: batch * 2].cpu().tolist(),
            table[:, :2].reshape(-1).tolist(),
        )

    def test_empty_batch_needs_no_kernel_inputs(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                params, fill = self.make_params(sparse)
                fill(
                    torch.empty(0, dtype=torch.int32, device="cuda"),
                    torch.empty((0, 3), dtype=torch.int32, device="cuda"),
                    64,
                )
                self.assertEqual(params.kvlen_d.numel(), 0)
                if sparse:
                    self.assertEqual(params.slot_mapping.numel(), 0)


if __name__ == "__main__":
    unittest.main()
