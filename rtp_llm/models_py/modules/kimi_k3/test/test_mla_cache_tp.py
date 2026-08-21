import unittest

import torch

from rtp_llm.models.kimi_k3.mla_cache_tp import MlaCacheShardLayout


class MlaCacheTpTest(unittest.TestCase):
    def test_tp8_shards_512_and_64_independently(self) -> None:
        latent = torch.arange(2 * 512).reshape(2, 512)
        suffix = torch.arange(2 * 64).reshape(2, 64) + 10000
        shards = []
        for rank in range(8):
            layout = MlaCacheShardLayout.fixed(512, 64, 8, rank)
            local_latent, local_suffix = layout.shard_components(latent, suffix)
            self.assertEqual(tuple(local_latent.shape), (2, 64))
            self.assertEqual(tuple(local_suffix.shape), (2, 8))
            shards.append(torch.cat((local_latent, local_suffix), dim=-1))

        restored_latent, restored_suffix = layout.reconstruct_rank_major(
            torch.stack(shards)
        )
        torch.testing.assert_close(restored_latent, latent)
        torch.testing.assert_close(restored_suffix, suffix)

    def test_tp16_uses_32_plus_4_layout(self) -> None:
        layout = MlaCacheShardLayout.fixed(512, 64, 16, 15)
        self.assertEqual((layout.local_latent, layout.local_suffix), (32, 4))
        self.assertEqual(layout.local_width, 36)

    def test_rejects_component_not_divisible_by_tp(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide latent and suffix"):
            MlaCacheShardLayout.fixed(512, 64, 12, 0)


if __name__ == "__main__":
    unittest.main()
