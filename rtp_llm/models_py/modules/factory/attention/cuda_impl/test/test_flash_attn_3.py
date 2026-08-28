import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.ops import fused_rope_kvcache_op
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpQOut
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_attn_3 import (
    FlashAttn3PagedShortGraphImpl,
    _MAX_QUERY_WIDTH,
    _fixed_width_active_prefix,
    _rope_capture_inputs,
    _short_graph_num_splits,
)


class FlashAttn3GraphGeometryTest(unittest.TestCase):
    def test_one_cache_block_query_limit(self) -> None:
        self.assertEqual(_MAX_QUERY_WIDTH, 64)

    def test_unpadded_fixed_width(self) -> None:
        self.assertEqual(
            _fixed_width_active_prefix(torch.tensor([8, 8, 8], dtype=torch.int32)),
            (3, 8),
        )

    def test_padded_fixed_width_active_prefix(self) -> None:
        self.assertEqual(
            _fixed_width_active_prefix(
                torch.tensor([8, 8, 8, 0, 0, 0], dtype=torch.int32)
            ),
            (3, 8),
        )

    def test_single_active_row(self) -> None:
        self.assertEqual(
            _fixed_width_active_prefix(torch.tensor([8, 0, 0], dtype=torch.int32)),
            (1, 8),
        )

    def test_rejects_non_prefix_activity(self) -> None:
        self.assertIsNone(
            _fixed_width_active_prefix(torch.tensor([8, 0, 8], dtype=torch.int32))
        )

    def test_rejects_mixed_positive_widths(self) -> None:
        self.assertIsNone(
            _fixed_width_active_prefix(torch.tensor([8, 7, 0], dtype=torch.int32))
        )

    def test_rejects_empty_or_inactive_geometry(self) -> None:
        self.assertIsNone(_fixed_width_active_prefix(None))
        self.assertIsNone(_fixed_width_active_prefix(torch.empty(0, dtype=torch.int32)))
        self.assertIsNone(
            _fixed_width_active_prefix(torch.tensor([0, 0], dtype=torch.int32))
        )


class FlashAttn3CudaGraphPrepareTest(unittest.TestCase):
    def test_device_only_offset_helper_is_a_real_operator_contract(self) -> None:
        impl = object.__new__(FusedRopeKVCachePrefillOpQOut)
        block_ids = torch.tensor([[3, 7]], dtype=torch.int32)
        attn_inputs = SimpleNamespace(kv_cache_kernel_block_id_device=block_ids)
        fused_op = mock.Mock()
        converted = torch.tensor([[9, 11]], dtype=torch.int32)
        fused_op.convert_offset_to_block_array.return_value = converted

        with mock.patch.object(
            fused_rope_kvcache_op,
            "_get_fused_rope_kvcache",
            return_value=fused_op,
        ):
            result = impl.prepare_kv_cache_offset(attn_inputs)

        self.assertIs(result, converted)
        fused_op.convert_offset_to_block_array.assert_called_once_with(block_ids)

    def test_rope_capture_retains_device_length_buffers(self) -> None:
        host_input = torch.tensor([8], dtype=torch.int32)
        host_prefix = torch.tensor([11], dtype=torch.int32)
        device_input = torch.tensor([8], dtype=torch.int32, device="cuda")
        device_prefix = torch.tensor([11], dtype=torch.int32, device="cuda")
        inputs = SimpleNamespace(
            input_lengths=host_input,
            prefix_lengths=host_prefix,
            input_lengths_device=device_input,
            prefix_lengths_device=device_prefix,
        )

        rope_inputs = _rope_capture_inputs(inputs)

        self.assertIs(rope_inputs.input_lengths, device_input)
        self.assertIs(rope_inputs.prefix_lengths, device_prefix)
        self.assertIs(inputs.input_lengths, host_input)
        self.assertIs(inputs.prefix_lengths, host_prefix)

    def test_replay_refreshes_only_kv_offset(self) -> None:
        impl = object.__new__(FlashAttn3PagedShortGraphImpl)
        impl.rope_kvcache_impl = mock.Mock()
        impl.rope_kvcache_impl.prepare_kv_cache_offset.return_value = torch.tensor(
            [7], dtype=torch.int32
        )
        impl.rope_params = mock.Mock()
        impl.rope_params.kv_cache_offset = torch.zeros(1, dtype=torch.int32)
        attn_inputs = mock.Mock()

        impl.prepare_cuda_graph(attn_inputs)

        impl.rope_kvcache_impl.prepare_kv_cache_offset.assert_called_once_with(
            attn_inputs
        )
        impl.rope_kvcache_impl.prepare.assert_not_called()
        self.assertEqual(impl.rope_params.kv_cache_offset.item(), 7)


class FlashAttn3SplitSchedulerTest(unittest.TestCase):
    def test_h20_qwen3_dspark_tp2_capture_buckets(self) -> None:
        expected = {1: 32, 8: 4, 16: 2, 24: 1, 32: 1}
        for batch_size, num_splits in expected.items():
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    _short_graph_num_splits(batch_size, 10, 78), num_splits
                )

    def test_caps_tiny_grids_and_rejects_invalid_geometry(self) -> None:
        self.assertEqual(_short_graph_num_splits(1, 1, 120), 32)
        for args in ((0, 1, 78), (1, 0, 78), (1, 1, 0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    _short_graph_num_splits(*args)


if __name__ == "__main__":
    unittest.main()
