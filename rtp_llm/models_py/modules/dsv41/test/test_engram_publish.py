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