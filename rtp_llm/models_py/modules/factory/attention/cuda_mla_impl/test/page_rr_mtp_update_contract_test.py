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
