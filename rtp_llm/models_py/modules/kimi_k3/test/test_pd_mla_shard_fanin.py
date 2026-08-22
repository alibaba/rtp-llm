import unittest

import torch

from rtp_llm.models.kimi_k3.mla_cache_tp import MlaCacheShardLayout
from rtp_llm.models.kimi_k3.pd_mla_shard_fanin import (
    FixedMlaOwnerRegistry,
    MlaShardFanIn,
    MlaShardMetadata,
    PDMlaTransferPlan,
)


class PDMlaShardFanInTest(unittest.TestCase):
    def _full_cache_and_shards(self, token_count: int = 3):
        latent = torch.arange(token_count * 512, dtype=torch.float32).reshape(
            token_count, 512
        ).to(torch.bfloat16)
        suffix = (
            torch.arange(token_count * 64, dtype=torch.float32).reshape(token_count, 64)
            + 100000
        ).to(torch.bfloat16)
        full_cache = torch.cat((latent, suffix), dim=-1)
        packed = []
        for rank in range(8):
            layout = MlaCacheShardLayout.fixed(512, 64, 8, rank)
            packed.append(layout.shard_full_cache(full_cache))
        return full_cache, packed

    def _metadata(self, rank: int, **overrides) -> MlaShardMetadata:
        values = dict(
            request_id=123,
            target_owner_rank=5,
            layer_id=7,
            shard_rank=rank,
            shard_count=8,
            token_count=3,
        )
        values.update(overrides)
        return MlaShardMetadata.for_kimi_k3(**values)

    def _fanin(self) -> MlaShardFanIn:
        return MlaShardFanIn(
            request_id=123,
            target_owner_rank=5,
            layer_id=7,
            shard_count=8,
            token_count=3,
            decode_tp_size=8,
        )

    def test_tp8_fan_in_directly_reconstructs_token_major_576(self) -> None:
        full_cache, shards = self._full_cache_and_shards()
        fanin = self._fanin()
        for rank in (4, 0, 7, 2, 6, 1, 5, 3):
            fanin.add(self._metadata(rank), shards[rank])
        self.assertTrue(fanin.complete)
        self.assertEqual(fanin.missing_shards(), ())
        torch.testing.assert_close(fanin.finalize(), full_cache)

    def test_metadata_wire_round_trip_is_exact(self) -> None:
        metadata = self._metadata(3)
        self.assertEqual(
            MlaShardMetadata.from_wire_dict(metadata.to_wire_dict()), metadata
        )
        payload = metadata.to_wire_dict()
        payload.pop("layout_version")
        with self.assertRaisesRegex(ValueError, "missing=.*layout_version"):
            MlaShardMetadata.from_wire_dict(payload)

    def test_missing_and_duplicate_shards_fail_fast(self) -> None:
        _, shards = self._full_cache_and_shards()
        fanin = self._fanin()
        fanin.add(self._metadata(0), shards[0])
        with self.assertRaisesRegex(ValueError, "duplicate MLA shard rank 0"):
            fanin.add(self._metadata(0), shards[0])
        with self.assertRaisesRegex(ValueError, "missing shards"):
            fanin.finalize()

    def test_version_owner_and_offsets_are_validated(self) -> None:
        _, shards = self._full_cache_and_shards()
        metadata = self._metadata(1)
        for replacement, regex in (
            ({"layout_version": 99}, "layout version"),
            ({"target_owner_rank": 8}, "owner rank"),
            ({"shard_offset": metadata.shard_offset + 1}, "offset mismatch"),
        ):
            payload = metadata.to_wire_dict()
            payload.update(replacement)
            with self.assertRaisesRegex(ValueError, regex):
                self._fanin().add(MlaShardMetadata(**payload), shards[1])

    def test_rank7_directly_writes_last_72_columns(self) -> None:
        full_cache, shards = self._full_cache_and_shards()
        metadata = self._metadata(7)
        self.assertEqual((metadata.shard_offset, metadata.shard_width), (504, 72))
        torch.testing.assert_close(shards[7], full_cache[:, 504:576])

    def test_fan_in_rejects_non_bf16_payload(self) -> None:
        _, shards = self._full_cache_and_shards()
        with self.assertRaisesRegex(ValueError, "BF16 only"):
            self._fanin().add(self._metadata(0), shards[0].float())

    def test_request_layer_and_token_count_must_match_fan_in(self) -> None:
        _, shards = self._full_cache_and_shards()
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self._fanin().add(self._metadata(0, request_id=124), shards[0])
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self._fanin().add(self._metadata(0, layer_id=8), shards[0])
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self._fanin().add(self._metadata(0, token_count=2), shards[0][:2])

    def test_owner_binding_is_explicit_and_sticky(self) -> None:
        registry = FixedMlaOwnerRegistry(8)
        with self.assertRaisesRegex(KeyError, "no explicit MLA owner"):
            registry.owner(123)
        registry.bind(123, 5)
        registry.bind(123, 5)
        self.assertEqual(registry.owner(123), 5)
        with self.assertRaisesRegex(ValueError, "sticky on owner 5"):
            registry.bind(123, 6)
        registry.release(123)
        registry.bind(123, 6)
        self.assertEqual(registry.owner(123), 6)

    def test_kda_stays_rank_to_rank_while_mla_fans_in(self) -> None:
        plan = PDMlaTransferPlan(request_id=123, target_owner_rank=5, tp_size=8)
        self.assertEqual([plan.kda_destination(rank) for rank in range(8)], list(range(8)))
        self.assertEqual([plan.mla_destination(rank) for rank in range(8)], [5] * 8)


if __name__ == "__main__":
    unittest.main()
