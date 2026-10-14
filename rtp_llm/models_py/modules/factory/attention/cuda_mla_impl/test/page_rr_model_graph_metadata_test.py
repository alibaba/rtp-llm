"""Exercise the real Page-RR metadata kernel through K3's model-graph hook."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaDecodeMetadata,
)
from rtp_llm.models_py.modules.kimi_k3.mla_verify import KimiK3MlaVerifyImpl


def _implementation(inputs, metadata):
    # No weights, MLA workspace, or attention backend are needed to exercise
    # the production metadata methods. The actual prepare/hook remain intact.
    impl = KimiK3MlaVerifyImpl.__new__(KimiK3MlaVerifyImpl)
    impl.graph_mode = True
    impl.page_rr = True
    impl.batch = inputs.input_lengths.numel()
    impl.tokens = inputs.physical_token_count
    impl.page_rr_metadata = metadata
    impl._graph_metadata_inputs = None
    impl._record_input_contract = mock.Mock()
    return impl


class PageRRModelGraphDispatchTest(unittest.TestCase):
    def test_non_page_rr_retains_existing_prepare(self):
        impl = KimiK3MlaVerifyImpl.__new__(KimiK3MlaVerifyImpl)
        impl.graph_mode = True
        impl.page_rr = False
        impl._graph_metadata_inputs = None
        impl.prepare = mock.Mock()
        inputs = object()
        impl.prepare_model_graph_inputs(inputs)
        impl.prepare.assert_not_called()
        self.assertIsNone(impl._graph_metadata_inputs)
        impl.prepare_cuda_graph(inputs)
        impl.prepare.assert_called_once_with(inputs, forbid_realloc=True)

    def test_eager_page_rr_does_not_enable_graph_hook(self):
        impl = KimiK3MlaVerifyImpl.__new__(KimiK3MlaVerifyImpl)
        impl.graph_mode = False
        impl.page_rr = True
        impl._graph_metadata_inputs = None
        impl.prepare = mock.Mock()
        impl.prepare_model_graph_inputs(object())
        impl.prepare.assert_not_called()
        self.assertIsNone(impl._graph_metadata_inputs)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class PageRRModelGraphMetadataTest(unittest.TestCase):
    def test_replay_derives_current_lengths_tables_and_slots_without_host_prepare(self):
        batch = 9
        page, kernel_page = 4096, 128
        for tp, columns in ((4, 145), (16, 145), (16, 0)):
            for rank in (0, tp - 1):
                for phase, query in (("decode", 1), ("verify", 4), ("update", 4)):
                    with self.subTest(tp=tp, rank=rank, phase=phase, columns=columns):
                        storage = torch.zeros(
                            (batch, columns + 7), dtype=torch.int32, device="cuda"
                        )
                        table = storage[:, :columns]
                        base = torch.ones(batch, dtype=torch.int32, device="cuda")
                        inputs = SimpleNamespace(
                            input_lengths=torch.full(
                                (batch,), query, dtype=torch.int32
                            ),
                            physical_token_count=batch * query,
                            is_target_verify=phase == "verify",
                            is_mtp_draft_update=phase == "update",
                            is_cuda_graph=True,
                            prefix_lengths_device=base,
                            sequence_lengths_plus_1_device=base,
                            kv_cache_kernel_block_id_device=table,
                        )
                        metadata = PageRRMlaDecodeMetadata(page, kernel_page, tp, rank)
                        impl = _implementation(inputs, metadata)
                        impl.prepare(inputs)
                        side = torch.cuda.Stream()
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            for _ in range(3):
                                impl.prepare_cuda_graph(inputs)
                        torch.cuda.current_stream().wait_stream(side)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=side):
                            impl.prepare_model_graph_inputs(inputs)
                        self.assertIsNotNone(impl._graph_metadata_inputs)
                        outputs = (
                            metadata.positions_d,
                            metadata.slot_mapping,
                            metadata.local_causal_lens,
                            metadata.query_block_tables,
                        )
                        addresses = tuple(tensor.data_ptr() for tensor in outputs)

                        for round_index in (0, 3, 1):
                            starts = [
                                0,
                                page - 1,
                                page,
                                page * tp - 1,
                                page * tp,
                                page * tp + kernel_page - 1,
                                page * tp * 4 - 2,
                                kernel_page - 1,
                                0,
                            ]
                            starts = [value + round_index for value in starts]
                            source_base = (
                                starts if query > 1 else [value + 1 for value in starts]
                            )
                            # The ordinary-decode synthetic graph sentinel uses
                            # plus_1==0 and must still produce position zero.
                            if phase == "decode":
                                source_base[-1] = 0
                                starts[-1] = 0
                            host_table = (
                                torch.arange(1, batch * columns + 1, dtype=torch.int32)
                                .reshape(batch, columns)
                                .add_(round_index * 10000)
                            )
                            host_table[-1].zero_()
                            if round_index == 1:
                                host_table[0].zero_()
                            base.copy_(torch.tensor(source_base, dtype=torch.int32))
                            table.copy_(host_table)
                            for tensor in outputs:
                                tensor.fill_(-77)
                            with mock.patch.object(
                                metadata, "prepare", wraps=metadata.prepare
                            ) as prepare:
                                impl.prepare_cuda_graph(inputs)
                                prepare.assert_not_called()
                            self.assertIs(impl.attn_inputs, inputs)
                            # prepare_cuda_graph only binds/validates. The GPU
                            # derivation must wait for graph replay itself.
                            self.assertTrue((metadata.positions_d == -77).all().item())
                            graph.replay()

                            positions, lengths, slots = [], [], []
                            for request, start in enumerate(starts):
                                for offset in range(query):
                                    position = start + offset
                                    positions.append(position)
                                    # Independent per-owner physical-page count.
                                    lengths.append(
                                        sum(
                                            min(
                                                page,
                                                max(
                                                    0, position + 1 - owner_page * page
                                                ),
                                            )
                                            for owner_page in range(
                                                rank, position // page + 1, tp
                                            )
                                        )
                                    )
                                    column = (
                                        position // (page * tp) * (page // kernel_page)
                                        + position % page // kernel_page
                                    )
                                    owned = (
                                        position // page % tp == rank
                                        and column < columns
                                    )
                                    physical = (
                                        int(host_table[request, column]) if owned else 0
                                    )
                                    slots.append(
                                        physical * kernel_page + position % kernel_page
                                        if physical > 0
                                        else -1
                                    )
                            self.assertEqual(
                                metadata.positions_d.cpu().tolist(), positions
                            )
                            self.assertEqual(
                                metadata.local_causal_lens.flatten().cpu().tolist(),
                                lengths,
                            )
                            self.assertEqual(
                                metadata.slot_mapping.cpu().tolist(), slots
                            )
                            expected_tables = (
                                host_table.repeat_interleave(query, dim=0)
                                if columns
                                else torch.zeros((batch * query, 1), dtype=torch.int32)
                            )
                            torch.testing.assert_close(
                                metadata.query_block_tables.cpu(),
                                expected_tables,
                                rtol=0,
                                atol=0,
                            )
                            self.assertEqual(
                                addresses, tuple(t.data_ptr() for t in outputs)
                            )

                        replaced = SimpleNamespace(**vars(inputs))
                        replaced.kv_cache_kernel_block_id_device = (
                            table.clone() if columns else table.new_empty((batch, 1))
                        )
                        with self.assertRaisesRegex(
                            ValueError, "shape or storage changed"
                        ):
                            impl.prepare_cuda_graph(replaced)
                        replaced = SimpleNamespace(**vars(inputs))
                        replaced.prefix_lengths_device = base.clone()
                        replaced.sequence_lengths_plus_1_device = (
                            replaced.prefix_lengths_device
                        )
                        with self.assertRaisesRegex(
                            ValueError, "shape or storage changed"
                        ):
                            impl.prepare_cuda_graph(replaced)
                        replaced = SimpleNamespace(**vars(inputs))
                        replaced.physical_token_count -= query
                        with self.assertRaisesRegex(
                            ValueError, "shape or storage changed"
                        ):
                            impl.prepare_cuda_graph(replaced)


if __name__ == "__main__":
    unittest.main()
