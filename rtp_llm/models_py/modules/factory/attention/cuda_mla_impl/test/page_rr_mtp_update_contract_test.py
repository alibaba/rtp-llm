"""Check dense MTP update cache writes across Page-RR owners and graph replay."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaDecodeMetadata,
)
from rtp_llm.models_py.modules.kimi_k3.native_mla_decode import NativeMlaDecode


@unittest.skipUnless(
    torch.cuda.is_available(), "requires CUDA and the K3 native runtime"
)
class MtpPageRRUpdateContractTest(unittest.TestCase):
    def test_fused_metadata_expands_strided_tables_and_preserves_replay_addresses(self):
        page, kernel_page, query = 4096, 128, 4
        for tp in (4, 8, 16):
            for rank in (0, tp - 1):
                for batch, columns in ((14, 144), (16, 145), (3, 0)):
                    with self.subTest(tp=tp, rank=rank, batch=batch, columns=columns):
                        storage = torch.arange(
                            1,
                            batch * (columns + 7) + 1,
                            dtype=torch.int32,
                            device="cuda",
                        ).reshape(batch, columns + 7)
                        table = storage[:, :columns]
                        table[-1].zero_()
                        bases = [65533 + (i % 5) for i in range(batch - 1)] + [0]
                        prefix = torch.tensor(bases, dtype=torch.int32, device="cuda")
                        inputs = SimpleNamespace(
                            input_lengths=torch.full(
                                (batch,), query, dtype=torch.int32
                            ),
                            physical_token_count=batch * query,
                            is_target_verify=True,
                            is_mtp_draft_update=False,
                            is_cuda_graph=True,
                            prefix_lengths_device=prefix,
                            sequence_lengths_plus_1_device=prefix,
                            kv_cache_kernel_block_id_device=table,
                        )
                        metadata = PageRRMlaDecodeMetadata(page, kernel_page, tp, rank)
                        metadata.prepare(inputs)
                        addresses = tuple(
                            x.data_ptr()
                            for x in (
                                metadata.positions_d,
                                metadata.slot_mapping,
                                metadata.local_causal_lens,
                                metadata.query_block_tables,
                            )
                        )
                        for round_index in (0, 1):
                            prefix.copy_(
                                torch.tensor(
                                    [x + round_index for x in bases],
                                    dtype=torch.int32,
                                    device="cuda",
                                )
                            )
                            metadata.prepare(inputs, forbid_realloc=True)
                            expected_table = (
                                table[:, None, :]
                                .expand(batch, query, columns)
                                .reshape(batch * query, columns)
                            )
                            if not columns:
                                expected_table = torch.zeros(
                                    (batch * query, 1), dtype=torch.int32, device="cuda"
                                )
                            torch.testing.assert_close(
                                metadata.query_block_tables,
                                expected_table,
                                rtol=0,
                                atol=0,
                            )
                            expected_positions, expected_lengths, expected_slots = (
                                [],
                                [],
                                [],
                            )
                            host_table = table.cpu().tolist()
                            for request, base in enumerate(bases):
                                for offset in range(query):
                                    position = base + round_index + offset
                                    end = position + 1
                                    expected_positions.append(position)
                                    expected_lengths.append(
                                        end // (page * tp) * page
                                        + min(
                                            max(end % (page * tp) - rank * page, 0),
                                            page,
                                        )
                                    )
                                    local_page = (
                                        position // (page * tp) * (page // kernel_page)
                                        + position % page // kernel_page
                                    )
                                    owned = (
                                        position // page % tp == rank
                                        and local_page < columns
                                    )
                                    physical_page = (
                                        host_table[request][local_page] if owned else 0
                                    )
                                    expected_slots.append(
                                        physical_page * kernel_page
                                        + position % kernel_page
                                        if physical_page > 0
                                        else -1
                                    )
                            self.assertEqual(
                                metadata.positions_d.cpu().tolist(), expected_positions
                            )
                            self.assertEqual(
                                metadata.local_causal_lens.flatten().cpu().tolist(),
                                expected_lengths,
                            )
                            self.assertEqual(
                                metadata.slot_mapping.cpu().tolist(), expected_slots
                            )
                            self.assertEqual(
                                addresses,
                                tuple(
                                    x.data_ptr()
                                    for x in (
                                        metadata.positions_d,
                                        metadata.slot_mapping,
                                        metadata.local_causal_lens,
                                        metadata.query_block_tables,
                                    )
                                ),
                            )

    def test_mixed_acceptance_rewrites_candidates_across_pages_in_graph_and_eager(self):
        torch.manual_seed(19)
        page, kernel_page, width, batch = 4096, 128, 4, 5
        initial = [65533, 65534, 65535, 65536, 0]
        accepts = [1, 2, 3, 4, 0]
        for tp in (4, 8):
            columns = ((max(initial) + width * 2) // (page * tp) + 1) * (
                page // kernel_page
            )
            for rank in range(tp):
                for graph_enabled in (False, True):
                    with self.subTest(tp=tp, rank=rank, graph=graph_enabled):
                        table = torch.arange(
                            1, batch * columns + 1, dtype=torch.int32
                        ).reshape(batch, columns)
                        table[-1].zero_()  # Whole dummy request: no cache allocation.
                        table_d = table.cuda()
                        prefix = torch.tensor(initial, dtype=torch.int32, device="cuda")
                        inputs = SimpleNamespace(
                            input_lengths=torch.full(
                                (batch,), width, dtype=torch.int32
                            ),
                            physical_token_count=batch * width,
                            is_target_verify=False,
                            is_mtp_draft_update=True,
                            is_cuda_graph=graph_enabled,
                            prefix_lengths_device=prefix,
                            sequence_lengths_plus_1_device=prefix + 1,
                            kv_cache_kernel_block_id_device=table_d,
                        )
                        metadata = PageRRMlaDecodeMetadata(page, kernel_page, tp, rank)
                        metadata.prepare(inputs)
                        addresses = (
                            metadata.positions_d.data_ptr(),
                            metadata.slot_mapping.data_ptr(),
                            metadata.local_causal_lens.data_ptr(),
                            metadata.query_block_tables.data_ptr(),
                        )
                        cache = torch.full(
                            (batch * columns + 1, kernel_page, 576),
                            -2,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        expected = cache.clone()
                        q = torch.zeros(
                            (batch * width, 2, 192), dtype=torch.bfloat16, device="cuda"
                        )
                        kv = torch.empty(
                            (batch * width, 512), dtype=torch.bfloat16, device="cuda"
                        )
                        pe = torch.empty(
                            (batch * width, 64), dtype=torch.bfloat16, device="cuda"
                        )
                        k_weight = torch.zeros(
                            (2, 128, 512), dtype=torch.bfloat16, device="cuda"
                        )
                        native = NativeMlaDecode(
                            num_heads=2,
                            kv_lora_rank=512,
                            nope_dim=128,
                            pe_dim=64,
                            page_size=kernel_page,
                            softmax_extra_scale=1.0,
                            workspace=torch.empty(
                                1024 * 1024, dtype=torch.uint8, device="cuda"
                            ),
                            max_batch=batch,
                            max_tokens=batch * width,
                            fp8_compute=False,
                        )
                        kv.zero_()
                        pe.zero_()
                        for _ in range(10):
                            native.write_cache(
                                q, kv, pe, cache, metadata.slot_mapping, k_weight
                            )
                        torch.cuda.synchronize()
                        graph = None
                        if graph_enabled:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                absorbed = native.write_cache(
                                    q, kv, pe, cache, metadata.slot_mapping, k_weight
                                )
                        cache.fill_(-2)
                        for round_index in (0, 1):
                            positions = [
                                p + (accepts[b] if round_index else 0) + step
                                for b, p in enumerate(initial)
                                for step in range(width)
                            ]
                            if round_index:
                                prefix.add_(
                                    torch.tensor(
                                        accepts, dtype=torch.int32, device="cuda"
                                    )
                                )
                            metadata.prepare(inputs, forbid_realloc=True)
                            self.assertEqual(
                                addresses,
                                (
                                    metadata.positions_d.data_ptr(),
                                    metadata.slot_mapping.data_ptr(),
                                    metadata.local_causal_lens.data_ptr(),
                                    metadata.query_block_tables.data_ptr(),
                                ),
                            )
                            kv.copy_(
                                torch.arange(batch * width, device="cuda")[:, None]
                                + 32 * round_index
                            )
                            pe.copy_(
                                torch.arange(batch * width, device="cuda")[:, None]
                                + 64 * round_index
                            )
                            slots, lengths = [], []
                            for token, position in enumerate(positions):
                                request = token // width
                                owned = (
                                    position // page % tp == rank
                                    and request < batch - 1
                                )
                                column = (
                                    position // (page * tp) * (page // kernel_page)
                                    + position % page // kernel_page
                                )
                                slot = (
                                    int(table[request, column]) * kernel_page
                                    + position % kernel_page
                                    if owned
                                    else -1
                                )
                                slots.append(slot)
                                # Independent count of causal keys in this owner's physical pages.
                                lengths.append(
                                    sum(
                                        min(page, max(0, position + 1 - g * page))
                                        for g in range(rank, position // page + 1, tp)
                                    )
                                )
                                if owned:
                                    expected.view(-1, 576)[slot, :512] = (
                                        token + 32 * round_index
                                    )
                                    expected.view(-1, 576)[slot, 512:] = (
                                        token + 64 * round_index
                                    )
                            torch.testing.assert_close(
                                metadata.positions_d.cpu(),
                                torch.tensor(positions, dtype=torch.int32),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                metadata.slot_mapping.cpu(),
                                torch.tensor(slots, dtype=torch.int64),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                metadata.local_causal_lens.cpu().reshape(-1),
                                torch.tensor(lengths, dtype=torch.int32),
                                rtol=0,
                                atol=0,
                            )
                            if graph is not None:
                                graph.replay()
                            else:
                                absorbed = native.write_cache(
                                    q, kv, pe, cache, metadata.slot_mapping, k_weight
                                )
                            torch.testing.assert_close(
                                absorbed, torch.zeros_like(absorbed), rtol=0, atol=0
                            )
                            torch.testing.assert_close(cache, expected, rtol=0, atol=0)
                            self.assertTrue((cache[0] == -2).all())
                        del graph, cache, expected, native


if __name__ == "__main__":
    unittest.main()
