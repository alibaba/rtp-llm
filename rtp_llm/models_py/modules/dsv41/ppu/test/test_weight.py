"""Geometry and checkpoint helpers used by the inherited PPU model."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    _CheckpointSource,
    _load_engram_shard,
    synthetic_config,
)


class GeometryRolesTest(unittest.TestCase):
    """The synthetic config parses into every real-model layer role."""

    def setUp(self):
        self.geometry = V41Geometry.from_config(synthetic_config())

    def test_layer_roles(self):
        geometry = self.geometry
        self.assertEqual(geometry.num_layers, 8)
        self.assertEqual(
            [geometry.ratio(layer) for layer in range(8)], [0, 0, 2, 2, 2, 2, 1, 1]
        )
        self.assertEqual(geometry.kv_source_layer_ids, (2, 6))
        self.assertEqual(geometry.index_source_layer_ids, (2, 6))
        self.assertEqual(geometry.engram_layer_ids, (1,))
        self.assertEqual(geometry.kv_source_for(5), 2)
        self.assertEqual(geometry.kv_source_for(7), 6)
        self.assertIsNone(geometry.kv_source_for(1))

    def test_num_local_experts_split(self):
        self.assertEqual(self.geometry.num_local_experts(1), 8)
        self.assertEqual(self.geometry.num_local_experts(2), 4)
        with self.assertRaises(ValueError):
            self.geometry.num_local_experts(3)

    def test_mhc_config_presence(self):
        config = self.geometry.mhc_config()
        self.assertIsInstance(config, MHCConfig)
        self.assertEqual(config.hc_mult, 4)
        self.assertEqual(config.sinkhorn_iters, 4)
        self.assertEqual(config.hc_eps, 1e-6)
        self.assertEqual(config.rms_eps, 1e-6)

    def test_trailing_compress_ratios_are_truncated(self):
        # The real checkpoint's compress_ratios carries entries past
        # num_hidden_layers (multimodal composite config: 43 entries for 40
        # text layers). V41Geometry must match CacheGeometry and the fixture
        # manifest: truncate to num_layers.
        config = dict(synthetic_config())
        config["compress_ratios"] = list(config["compress_ratios"]) + [0, 0, 0]
        geometry = V41Geometry.from_config(config)
        self.assertEqual(geometry.num_layers, 8)
        self.assertEqual(
            len(geometry.compress_ratios),
            geometry.num_layers,
        )
        self.assertEqual(
            [geometry.ratio(layer) for layer in range(8)],
            [0, 0, 2, 2, 2, 2, 1, 1],
        )


class EngramShardTest(unittest.TestCase):
    def test_loads_only_rank_owned_rows(self):
        geometry = V41Geometry.from_config(synthetic_config())
        row_ids = torch.arange(2048, dtype=torch.int32)
        payload = (row_ids + row_ids // 1024).to(torch.uint8)[:, None].expand(-1, 64).contiguous()
        scales = torch.arange(2048 * 2, dtype=torch.int32).reshape(2048, 2).to(torch.uint8)
        source = _CheckpointSource({
            "layers.1.engram.embed.weight": payload.view(torch.float8_e4m3fn),
            "layers.1.engram.embed.scale": scales,
        })
        for rank in (0, 1):
            with self.subTest(rank=rank):
                shard, layout = _load_engram_shard(source, 1, geometry, rank, 2)
                start = 1024 * rank
                self.assertEqual(layout.row_start(rank), start)
                self.assertEqual(layout.valid_rows(rank), 1024)
                self.assertTrue(torch.equal(shard.payload.view(torch.uint8), payload[start:start + 1024]))
                self.assertTrue(torch.equal(shard.scale.view(torch.uint8), scales[start:start + 1024]))


if __name__ == "__main__":
    unittest.main()
