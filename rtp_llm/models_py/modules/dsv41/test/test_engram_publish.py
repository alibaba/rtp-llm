# SPDX-License-Identifier: Apache-2.0
import unittest

from rtp_llm.models_py.modules.dsv41 import engram_publish as p
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


class Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(shape)


class ShardLayoutTest(unittest.TestCase):
    def test_weight_and_scale_share_the_row_partition(self):
        rows, dim, shards, granularity = 100000, 64, 2, 4096
        plan = plan_shards(rows, dim, shards, granularity=granularity)
        weight = Tensor('layers.1.engram.embed.weight', (rows, dim))
        scale = Tensor('layers.1.engram.embed.scale', (rows, dim // plan.block))
        for shard in range(shards):
            start, end, weight_bytes = p.shard_layout(weight, (rows, dim), shard, shards, granularity)
            self.assertEqual((start, end), plan.row_range(shard))
            self.assertEqual(weight_bytes, plan.weight_bytes)
            start, end, scale_bytes = p.shard_layout(scale, (rows, dim), shard, shards, granularity)
            self.assertEqual(scale_bytes, plan.scale_bytes)

    def test_mismatched_shapes_are_rejected(self):
        weight = Tensor('layers.1.engram.embed.weight', (100, 64))
        with self.assertRaises(ValueError):
            p.shard_layout(weight, (200, 64), 0, 2, 4096)

    def test_unknown_role_is_rejected(self):
        tensor = Tensor('layers.1.engram.embed.bias', (100, 64))
        with self.assertRaises(ValueError):
            p.shard_layout(tensor, (100, 64), 0, 2, 4096)

    def test_release_of_an_empty_generation_is_a_noop(self):
        self.assertEqual(p.release('missing-generation'), [])


class PlacementMapTest(unittest.TestCase):
    def test_absent_map_uses_the_scalar_default(self):
        self.assertEqual(p._placement_map(None, [1, 2], 3, 'device'), {1: 3, 2: 3})

    def test_partial_or_non_integer_maps_are_rejected(self):
        for raw in ({1: 0}, {1: 0, 2: 0, 3: 1}, {1: 0, 2: True}, {1: 0, 2: -1}, 'cuda:1'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                p._placement_map(raw, [1, 2], 0, 'NUMA')

    def test_explicit_map_is_honoured_per_layer(self):
        self.assertEqual(p._placement_map({1: 2, 2: 3}, [1, 2], 0, 'device'), {1: 2, 2: 3})


class LoadRowsBoundsTest(unittest.TestCase):
    def test_row_range_and_chunk_bounds_are_checked_before_any_read(self):
        tensor = Tensor('layers.1.engram.embed.weight', (100, 64))
        for start, end, columns, chunk in [(1, 1, 64, 4096), (0, 101, 64, 4096),
                                           (0, 10, 32, 4096), (0, 10, 64, 0),
                                           (0, 10, 64, (64 << 20) + 1)]:
            with self.subTest(case=(start, end, columns, chunk)), self.assertRaises(ValueError):
                p.load_rows(None, None, tensor, None, start, end, columns, chunk_bytes=chunk)