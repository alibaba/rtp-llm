import unittest

import torch

from rtp_llm.models_py.model_desc.block_map import (
    prepare_block_maps,
    select_block_map_for_layer,
)


class FakeAttentionInputs:
    def __init__(self):
        self.reads = {}
        self._device_groups = [object(), object()]
        self._kernel_host_groups = [object(), object()]
        self._host_groups = [object(), object()]
        self._layer_groups = torch.tensor([1, 0, 1], dtype=torch.int32)
        self.kv_cache_kernel_block_id_device = None
        self.kv_cache_kernel_block_id_host = None
        self.kv_cache_block_id_host = None

    def _read(self, name, value):
        self.reads[name] = self.reads.get(name, 0) + 1
        return value

    @property
    def kv_cache_kernel_block_id_device_by_group(self):
        return self._read("device", list(self._device_groups))

    @property
    def kv_cache_kernel_block_id_host_by_group(self):
        return self._read("kernel_host", list(self._kernel_host_groups))

    @property
    def kv_cache_block_id_host_by_group(self):
        return self._read("host", list(self._host_groups))

    @property
    def kv_cache_layer_to_group(self):
        return self._read("layer", self._layer_groups)


class BlockMapTest(unittest.TestCase):
    def test_prepared_tables_are_read_once_per_forward(self):
        inputs = FakeAttentionInputs()
        prepared = prepare_block_maps(inputs)
        self.assertIsNotNone(prepared)
        for layer_idx, expected_group in enumerate((1, 0, 1)):
            group = select_block_map_for_layer(inputs, layer_idx, prepared)
            self.assertEqual(group, expected_group)
            self.assertIs(
                inputs.kv_cache_kernel_block_id_device,
                inputs._device_groups[expected_group],
            )
            self.assertIs(
                inputs.kv_cache_kernel_block_id_host,
                inputs._kernel_host_groups[expected_group],
            )
            self.assertIs(
                inputs.kv_cache_block_id_host, inputs._host_groups[expected_group]
            )
        self.assertEqual(
            inputs.reads,
            {"device": 1, "kernel_host": 1, "host": 1, "layer": 1},
        )

    def test_unprepared_path_keeps_single_layer_behavior(self):
        inputs = FakeAttentionInputs()
        self.assertEqual(select_block_map_for_layer(inputs, 0), 1)
        self.assertEqual(
            inputs.reads,
            {"device": 1, "kernel_host": 1, "host": 1, "layer": 1},
        )


if __name__ == "__main__":
    unittest.main()
