import unittest

import torch

from rtp_llm.models.kimi_k3.mla_cache_tp import MlaCacheShardLayout


class MlaCacheTpTest(unittest.TestCase):
    def test_tp8_uses_flat_contiguous_72_shards(self) -> None:
        latent = torch.arange(2 * 512).reshape(2, 512)
        suffix = torch.arange(2 * 64).reshape(2, 64) + 10000
        full_cache = torch.cat((latent, suffix), dim=-1)
        shards = []
        for rank in range(8):
            layout = MlaCacheShardLayout.fixed(512, 64, 8, rank)
            local_latent, local_suffix = layout.shard_components(latent, suffix)
            shard = torch.cat((local_latent, local_suffix), dim=-1)
            self.assertEqual(tuple(shard.shape), (2, 72))
            torch.testing.assert_close(shard, layout.shard_full_cache(full_cache))
            shards.append(shard)

        restored_latent, restored_suffix = layout.reconstruct_rank_major(
            torch.stack(shards)
        )
        torch.testing.assert_close(restored_latent, latent)
        torch.testing.assert_close(restored_suffix, suffix)

        rank7 = MlaCacheShardLayout.fixed(512, 64, 8, 7)
        self.assertEqual((rank7.shard_start, rank7.shard_stop), (504, 576))
        self.assertEqual((rank7.local_latent, rank7.local_suffix), (8, 64))
        torch.testing.assert_close(shards[7][:, :8], latent[:, 504:512])
        torch.testing.assert_close(shards[7][:, 8:], suffix)

    def test_tp16_shard_can_cross_component_boundary(self) -> None:
        crossing = MlaCacheShardLayout.fixed(512, 64, 16, 14)
        self.assertEqual((crossing.shard_start, crossing.shard_stop), (504, 540))
        self.assertEqual((crossing.local_latent, crossing.local_suffix), (8, 28))
        layout = MlaCacheShardLayout.fixed(512, 64, 16, 15)
        self.assertEqual((layout.local_latent, layout.local_suffix), (0, 36))
        self.assertEqual(layout.local_width, 36)

    def test_rejects_packed_width_not_divisible_by_tp(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide packed cache width"):
            MlaCacheShardLayout.fixed(512, 64, 7, 0)


if __name__ == "__main__":
    unittest.main()
