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


if __name__ == "__main__":
    unittest.main()
