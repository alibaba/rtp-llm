"""CPU contract tests for the TokenSpeed MLA framework adapter."""

from types import SimpleNamespace
from unittest import TestCase, main, mock, skipUnless

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    TokenSpeedMlaDecodeImpl,
    _TokenSpeedDecodeMetadata,
)


class TokenSpeedMlaGraphAdapterTest(TestCase):
    def test_prepare_cuda_graph_uses_fixed_capacity_plan(self) -> None:
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.prepare = mock.Mock()
        inputs = SimpleNamespace()

        impl.prepare_cuda_graph(inputs)

        impl.prepare.assert_called_once_with(inputs, forbid_realloc=True)

    def test_target_verify_uses_device_prefix_without_host_mirrors(self) -> None:
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.fmha_impl = mock.Mock()
        inputs = SimpleNamespace(
            is_target_verify=True,
            input_lengths=torch.empty(2, dtype=torch.int32),
            total_tokens=8,
            prefix_lengths=object(),
            kv_cache_kernel_block_id_device=object(),
        )
        impl.prepare_cuda_graph(inputs)
        impl.fmha_impl.plan_device.assert_called_once_with(
            inputs.prefix_lengths,
            inputs.kv_cache_kernel_block_id_device,
            4,
            True,
        )


class TokenSpeedMlaMetadataContractTest(TestCase):
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


@skipUnless(torch.cuda.is_available(), "requires CUDA")
class TokenSpeedDeviceMetadataTest(TestCase):
    def _check_metadata(self, metadata, prefix, table, query_length):
        expected_positions = (
            prefix[:, None] + torch.arange(query_length, dtype=torch.int32)
        ).flatten()
        expected_batch = torch.arange(prefix.numel(), dtype=torch.int32).repeat_interleave(
            query_length
        )
        kv_lens = prefix + query_length
        expected_table = torch.zeros_like(metadata.block_tables, device="cpu")
        for row, kv_len in enumerate(kv_lens.tolist()):
            pages = (kv_len + 127) // 128
            expected_table[row, :pages] = table[row, :pages]
        expected_slots = (
            table[expected_batch.long(), expected_positions.long() // 128].long() * 128
            + expected_positions.long() % 128
        )
        for actual, expected in (
            (metadata.positions_d, expected_positions),
            (metadata.batch_indice_d, expected_batch),
            (metadata.slot_mapping, expected_slots),
            (metadata.seq_lens, kv_lens),
            (metadata.block_tables, expected_table),
        ):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_graph_replay_reads_live_lengths_and_pages_without_host_access(self):
        for batch in (1, 2, 4, 8, 12, 16):
            for query_length in (1, 2, 4):
                with self.subTest(batch=batch, query_length=query_length):
                    prefix_h = torch.tensor(
                        [0, 125, 127, 128, 4093, 4095, 8192, 32764] * 2,
                        dtype=torch.int32,
                    )[:batch].clone()
                    prefix_d = prefix_h.cuda()
                    table_h = torch.arange(batch * 256, dtype=torch.int32).reshape(
                        batch, 256
                    ) + 7
                    # Hybrid-cache views can have nontrivial row/column strides.
                    table_d = torch.empty(
                        batch, 512, dtype=torch.int32, device="cuda"
                    )[:, ::2]
                    table_d.copy_(table_h)
                    metadata = _TokenSpeedDecodeMetadata(
                        128, batch, 32768, True, prefix_d.device
                    )
                    metadata.plan_device(prefix_d, table_d, query_length)
                    self._check_metadata(metadata, prefix_h, table_h, query_length)
                    pointers = [
                        tensor.data_ptr()
                        for tensor in (
                            metadata.positions_d,
                            metadata.batch_indice_d,
                            metadata.slot_mapping,
                            metadata.block_tables,
                            metadata.seq_lens,
                        )
                    ]
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with mock.patch.object(
                        torch.Tensor, "cpu", side_effect=AssertionError("D2H")
                    ), mock.patch.object(
                        torch.Tensor, "tolist", side_effect=AssertionError("host read")
                    ), mock.patch.object(
                        torch.Tensor, "item", side_effect=AssertionError("host scalar")
                    ):
                        with torch.cuda.graph(graph):
                            metadata.plan_device(
                                prefix_d, table_d, query_length, forbid_realloc=True
                            )
                    prefix_h = (prefix_h + 129).remainder(32764)
                    table_h = table_h.flip(1)
                    prefix_d.copy_(prefix_h)
                    table_d.copy_(table_h)
                    graph.replay()
                    self._check_metadata(metadata, prefix_h, table_h, query_length)
                    self.assertEqual(
                        pointers,
                        [
                            tensor.data_ptr()
                            for tensor in (
                                metadata.positions_d,
                                metadata.batch_indice_d,
                                metadata.slot_mapping,
                                metadata.block_tables,
                                metadata.seq_lens,
                            )
                        ],
                    )

    def test_device_plan_rejects_growth_before_launch(self):
        prefix = torch.zeros(1, dtype=torch.int32, device="cuda")
        metadata = _TokenSpeedDecodeMetadata(128, 1, 128, True, prefix.device)
        table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
        metadata.plan_device(prefix, table, 1)
        with self.assertRaisesRegex(ValueError, "captured capacity"):
            metadata.plan_device(prefix.expand(2), table.expand(2, 1), 1, True)
        with self.assertRaisesRegex(ValueError, "query shape"):
            metadata.plan_device(prefix, table, 4, True)

    def test_capture_table_may_include_speculative_reserve_columns(self):
        prefix = torch.tensor([124], dtype=torch.int32, device="cuda")
        metadata = _TokenSpeedDecodeMetadata(128, 1, 128, True, prefix.device)
        table = torch.arange(5, dtype=torch.int32, device="cuda").reshape(1, 5)
        metadata.plan_device(prefix, table, 4)
        metadata.plan_device(prefix, table, 4, forbid_realloc=True)
        self.assertEqual(metadata.block_tables.shape, (1, 1))
        self._check_metadata(metadata, prefix.cpu(), table.cpu(), 4)


if __name__ == "__main__":
    main()
