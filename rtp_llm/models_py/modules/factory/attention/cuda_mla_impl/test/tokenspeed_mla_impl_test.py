"""CPU contract tests for the TokenSpeed MLA framework adapter."""

from types import SimpleNamespace
from unittest import TestCase, main, mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    TokenSpeedMlaDecodeImpl,
    TokenSpeedMlaDecodeOp,
    _TokenSpeedDecodeMetadata,
)


class _GraphParams:
    def __init__(self, calls):
        self.calls = calls
        self.slot_mapping = None

    def fill_decode_cuda_graph_params(
        self,
        sequence_lengths: torch.Tensor,
        block_table: torch.Tensor,
        seq_size_per_block: int,
    ) -> None:
        self.calls.append(("framework", sequence_lengths, block_table))
        self.positions_d = sequence_lengths.clamp_min(1) - 1
        self.batch_indice_d = torch.arange(sequence_lengths.numel(), dtype=torch.int32)
        self.page_indice_d = torch.empty(0, dtype=torch.int32)
        self.seq_size_per_block = seq_size_per_block


class TokenSpeedMlaGraphAdapterTest(TestCase):
    def test_group_refresh_updates_framework_metadata_before_backend(self) -> None:
        calls = []
        params = _GraphParams(calls)
        backend = mock.Mock()

        def refresh_backend(
            current_params,
            block_table,
            sequence_lengths,
            seq_size_per_block,
        ):
            self.assertIs(current_params, params)
            self.assertTrue(hasattr(current_params, "positions_d"))
            self.assertEqual(seq_size_per_block, 4)
            calls.append(("backend", sequence_lengths, block_table))

        backend.refresh_cuda_graph_metadata.side_effect = refresh_backend
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.seq_size_per_block = 4
        impl.fmha_params = params
        impl.fmha_impl = backend
        impl.attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.tensor(
                [[31, 32], [41, 42]], dtype=torch.int32
            )
        )
        current_inputs = SimpleNamespace(
            sequence_lengths_plus_1_d=torch.tensor([2, 5], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[11, 12], [21, 22]], dtype=torch.int32
            ),
        )

        impl.prepare_cuda_graph_group(current_inputs)

        self.assertEqual([call[0] for call in calls], ["framework", "backend"])
        self.assertIs(impl.attn_inputs, current_inputs)
        for _, sequence_lengths, block_table in calls:
            self.assertIs(sequence_lengths, current_inputs.sequence_lengths_plus_1_d)
            self.assertIs(block_table, current_inputs.kv_cache_kernel_block_id_device)
        torch.testing.assert_close(
            impl._device_decode_slot_mapping(),
            torch.tensor([45, 88], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_group_refresh_requires_device_lengths_and_block_table(self) -> None:
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.seq_size_per_block = 4
        impl.fmha_params = mock.Mock()
        impl.fmha_impl = mock.Mock()
        missing_lengths = SimpleNamespace(
            sequence_lengths_plus_1_d=torch.empty(0, dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.ones((1, 1), dtype=torch.int32),
        )
        missing_table = SimpleNamespace(
            sequence_lengths_plus_1_d=torch.ones(1, dtype=torch.int32),
            kv_cache_kernel_block_id_device=None,
        )

        with self.assertRaisesRegex(RuntimeError, "sequence_lengths_plus_1_d"):
            impl.prepare_cuda_graph_group(missing_lengths)
        with self.assertRaisesRegex(RuntimeError, "device block table"):
            impl.prepare_cuda_graph_group(missing_table)
        impl.fmha_params.fill_decode_cuda_graph_params.assert_not_called()
        impl.fmha_impl.refresh_cuda_graph_metadata.assert_not_called()

    def test_prepare_cuda_graph_uses_fixed_capacity_plan(self) -> None:
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.prepare = mock.Mock()
        inputs = SimpleNamespace()

        impl.prepare_cuda_graph(inputs)

        impl.prepare.assert_called_once_with(inputs, forbid_realloc=True)


class TokenSpeedMlaMetadataContractTest(TestCase):
    def test_graph_refresh_masks_stale_pages_without_reallocation(self) -> None:
        metadata = _TokenSpeedDecodeMetadata(
            token_per_block=64,
            max_bs=2,
            max_context_len=192,
            use_cuda_graph=True,
            device=torch.device("cpu"),
        )
        params = SimpleNamespace(
            qo_indptr_h=torch.arange(3, dtype=torch.int32),
            kvlen_h=torch.tensor([129, 65], dtype=torch.int32),
            kvlen_d=torch.tensor([129, 65], dtype=torch.int32),
            decode_page_indptr_d=torch.tensor([0, 3, 5], dtype=torch.int32),
            page_indice_d=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        )
        metadata.plan(params)
        table_ptr = metadata.block_tables.data_ptr()
        lengths_ptr = metadata.seq_lens.data_ptr()

        metadata.refresh_cuda_graph(
            torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int32),
        )

        self.assertEqual(metadata.block_tables.data_ptr(), table_ptr)
        self.assertEqual(metadata.seq_lens.data_ptr(), lengths_ptr)
        torch.testing.assert_close(
            metadata.block_tables,
            torch.tensor([[11, 0, 0], [0, 0, 0]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            metadata.seq_lens,
            torch.tensor([1, 0], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_graph_metadata_rejects_capacity_growth(self) -> None:
        metadata = _TokenSpeedDecodeMetadata(
            token_per_block=64,
            max_bs=1,
            max_context_len=64,
            use_cuda_graph=True,
            device=torch.device("cpu"),
        )
        too_many_rows = SimpleNamespace(
            qo_indptr_h=torch.arange(3, dtype=torch.int32),
            kvlen_h=torch.tensor([1, 1], dtype=torch.int32),
        )
        too_many_blocks = SimpleNamespace(
            qo_indptr_h=torch.arange(2, dtype=torch.int32),
            kvlen_h=torch.tensor([65], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "too small for batch 2"):
            metadata.plan(too_many_rows)
        with self.assertRaisesRegex(ValueError, "needs 2 blocks, has 1"):
            metadata.plan(too_many_blocks)

    def test_graph_refresh_validates_mode_and_table_shape(self) -> None:
        eager_metadata = _TokenSpeedDecodeMetadata(
            token_per_block=64,
            max_bs=0,
            max_context_len=0,
            use_cuda_graph=False,
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(RuntimeError, "requires CUDA graph"):
            eager_metadata.refresh_cuda_graph(
                torch.ones((1, 1), dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
            )

        graph_metadata = _TokenSpeedDecodeMetadata(
            token_per_block=64,
            max_bs=1,
            max_context_len=128,
            use_cuda_graph=True,
            device=torch.device("cpu"),
        )
        graph_metadata.batch_size = 1
        graph_metadata.padded_blocks = 2
        with self.assertRaisesRegex(RuntimeError, "block table of width >= 2"):
            graph_metadata.refresh_cuda_graph(
                torch.ones((1, 1), dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
            )

    def test_backend_refresh_rejects_page_size_mismatch(self) -> None:
        op = object.__new__(TokenSpeedMlaDecodeOp)
        op.token_per_block = 64
        op._metadata = mock.Mock()
        params = object()
        block_table = torch.ones((1, 1), dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "page-size mismatch"):
            op.refresh_cuda_graph_metadata(params, block_table, lengths, 128)
        op._metadata.refresh_cuda_graph.assert_not_called()

        op.refresh_cuda_graph_metadata(params, block_table, lengths, 64)
        op._metadata.refresh_cuda_graph.assert_called_once_with(block_table, lengths)


if __name__ == "__main__":
    main()
