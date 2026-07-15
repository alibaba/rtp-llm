import unittest

import torch

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.ops.compute_ops import (
    CacheGroupType,
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
)


class PyModelInputsCompatTest(unittest.TestCase):
    def _attn_inputs(self, is_prefill: bool, input_length: int) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = is_prefill
        attn_inputs.input_lengths = torch.tensor([input_length], dtype=torch.int32)
        return attn_inputs

    def test_default_constructor_then_assign_attention_inputs(self) -> None:
        attn_inputs = self._attn_inputs(is_prefill=True, input_length=3)

        inputs = PyModelInputs()
        inputs.attention_inputs = attn_inputs

        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(3, inputs.attention_inputs.input_lengths.item())

    def test_attention_inputs_field_updates_directly(self) -> None:
        inputs = PyModelInputs()
        inputs.attention_inputs = self._attn_inputs(is_prefill=False, input_length=2)

        self.assertFalse(inputs.attention_inputs.is_prefill)

        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.input_lengths = torch.tensor([5], dtype=torch.int32)
        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(5, inputs.attention_inputs.input_lengths.item())

    def test_select_block_map_for_layer_uses_layer_to_group_tensor(self) -> None:
        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_layer_to_group = torch.tensor([1], dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_by_group = [
            torch.tensor([[10]], dtype=torch.int32),
            torch.tensor([[20]], dtype=torch.int32),
        ]
        attn_inputs.kv_cache_kernel_block_id_device_by_group = [
            torch.tensor([[11]], dtype=torch.int32),
            torch.tensor([[21]], dtype=torch.int32),
        ]

        selected_gid = select_block_map_for_layer(attn_inputs, 0)

        self.assertEqual(1, selected_gid)
        self.assertEqual(20, attn_inputs.kv_cache_kernel_block_id.item())
        self.assertEqual(21, attn_inputs.kv_cache_kernel_block_id_device.item())

    def test_kv_cache_kernel_block_view_applies_only_to_full_layers(self) -> None:
        kv_cache = KVCache()
        kv_cache.seq_size_per_block = 8
        kv_cache.kernel_seq_size_per_block = 2
        kv_cache.num_kv_heads = 1
        kv_cache.head_dim = 4
        kv_cache.layer_attn_types = [CacheGroupType.FULL, CacheGroupType.LINEAR]
        kv_cache.layer_to_group_ids = [[0], [1]]
        kv_cache.group_types = [CacheGroupType.FULL, CacheGroupType.LINEAR]
        kv_cache.group_tags = ["full", "linear"]

        full_base = torch.arange(3 * 2 * 1 * 8 * 4, dtype=torch.float16).reshape(3, 64)
        linear_base = torch.arange(3 * 64, dtype=torch.float16).reshape(3, 64)
        kv_cache.kv_cache_base_by_layer = [full_base, linear_base]
        kv_cache.kv_cache_base_by_layer_group = [
            [full_base],
            [torch.empty(0, dtype=torch.float16), linear_base],
        ]

        full_layer = kv_cache.get_layer_cache(0)
        linear_layer = kv_cache.get_layer_cache(1)
        full_group = kv_cache.get_layer_cache_by_group(0, 0)
        linear_group = kv_cache.get_layer_cache_by_group(1, 1)

        self.assertEqual(2, full_layer.seq_size_per_block)
        self.assertEqual((12, 2, 1, 2, 4), tuple(full_layer.kv_cache_base.shape))
        self.assertEqual(tuple(full_layer.kv_cache_base.shape), tuple(full_group.kv_cache_base.shape))
        self.assertEqual(8, linear_layer.seq_size_per_block)
        self.assertEqual((3, 64), tuple(linear_layer.kv_cache_base.shape))
        self.assertEqual(tuple(linear_layer.kv_cache_base.shape), tuple(linear_group.kv_cache_base.shape))

    def _assert_mla_kernel_view(
        self,
        base: torch.Tensor,
        scale: torch.Tensor,
        kernel_page_size: int,
    ) -> None:
        physical_blocks, physical_page_size, stride = base.shape
        scale_stride = scale.shape[2]
        blocks_per_physical = physical_page_size // kernel_page_size

        kv_cache = KVCache()
        kv_cache.seq_size_per_block = physical_page_size
        kv_cache.kernel_seq_size_per_block = kernel_page_size
        kv_cache.use_mla = True
        kv_cache.kv_lora_rank = 4
        kv_cache.rope_head_dim = 2
        kv_cache.layer_attn_types = [CacheGroupType.FULL]
        kv_cache.layer_to_group_ids = [[0]]
        kv_cache.group_types = [CacheGroupType.FULL]
        kv_cache.group_tags = ["full"]
        kv_cache.kv_cache_base_by_layer = [base]
        kv_cache.kv_scale_base_by_layer = [scale]
        kv_cache.kv_cache_base_by_layer_group = [[base]]
        kv_cache.kv_scale_base_by_layer_group = [[scale]]

        legacy = kv_cache.get_layer_cache(0)
        by_group = kv_cache.get_layer_cache_by_group(0, 0)
        expected_shape = (
            physical_blocks * blocks_per_physical,
            kernel_page_size,
            stride,
        )
        expected_scale_shape = (
            physical_blocks * blocks_per_physical,
            kernel_page_size,
            scale_stride,
        )

        for view in (legacy, by_group):
            self.assertEqual(kernel_page_size, view.seq_size_per_block)
            self.assertEqual(expected_shape, tuple(view.kv_cache_base.shape))
            self.assertEqual(expected_scale_shape, tuple(view.kv_scale_base.shape))
            self.assertEqual(base.data_ptr(), view.kv_cache_base.data_ptr())
            self.assertEqual(base.numel(), view.kv_cache_base.numel())
            self.assertEqual(scale.data_ptr(), view.kv_scale_base.data_ptr())
            self.assertEqual(scale.numel(), view.kv_scale_base.numel())

            for physical_block in range(physical_blocks):
                for token in range(physical_page_size):
                    kernel_block = (
                        physical_block * blocks_per_physical
                        + token // kernel_page_size
                    )
                    kernel_token = token % kernel_page_size
                    self.assertTrue(
                        torch.equal(
                            base[physical_block, token],
                            view.kv_cache_base[kernel_block, kernel_token],
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            scale[physical_block, token],
                            view.kv_scale_base[kernel_block, kernel_token],
                        )
                    )

    def test_mla_bf16_kernel_block_view_preserves_physical_page_mapping(self) -> None:
        base = (
            torch.arange(8 * 8 * 6, dtype=torch.float32)
            .to(torch.bfloat16)
            .reshape(8, 8, 6)
        )
        scale = (
            torch.arange(8 * 8 * 3, dtype=torch.int32)
            .to(torch.uint8)
            .reshape(8, 8, 3)
        )

        self._assert_mla_kernel_view(base, scale, kernel_page_size=2)

    def test_mla_packed_fp8_kernel_block_view_preserves_physical_page_mapping(
        self,
    ) -> None:
        packed_stride = 256 + 256 // 128 * 4 + 64 * 2
        base = (
            torch.arange(8 * 8 * packed_stride, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .reshape(8, 8, packed_stride)
        )
        scale = (
            torch.arange(8 * 8 * 5, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .reshape(8, 8, 5)
        )

        self._assert_mla_kernel_view(base, scale, kernel_page_size=2)

    def test_mla_kernel_block_view_with_one_kernel_block_per_physical_block(
        self,
    ) -> None:
        base = (
            torch.arange(8 * 8 * 6, dtype=torch.float32)
            .to(torch.bfloat16)
            .reshape(8, 8, 6)
        )
        scale = (
            torch.arange(8 * 8 * 3, dtype=torch.int32)
            .to(torch.uint8)
            .reshape(8, 8, 3)
        )

        self._assert_mla_kernel_view(base, scale, kernel_page_size=8)


if __name__ == "__main__":
    unittest.main()
