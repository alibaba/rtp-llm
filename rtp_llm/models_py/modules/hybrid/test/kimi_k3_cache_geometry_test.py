import unittest
from types import SimpleNamespace

from rtp_llm.models_py.modules.kimi_k3.cache_geometry import bind_kimi_k3_cache_geometry
from rtp_llm.ops.compute_ops import CacheGroupType


class KimiK3CacheGeometryTest(unittest.TestCase):

    def _bind(self, *, decode=False, spans=(128, 1024), kinds=None, local_shards=None):
        cache = SimpleNamespace(
            seq_size_per_block=128,
            local_shard_count=(
                (1 if decode else 8) if local_shards is None else local_shards
            ),
            group_seq_size_per_block=spans,
            layer_group_types=kinds or [CacheGroupType.FULL, CacheGroupType.LINEAR],
            get_layer_cache=lambda layer: SimpleNamespace(group_id=layer),
        )
        parallelism = SimpleNamespace(
            tp_size=8,
            tp_rank=5,
            kv_page_rr_enabled=lambda: not decode,
            prefill_cp_config=SimpleNamespace(prefill_cp_size=8),
        )
        return bind_kimi_k3_cache_geometry(
            cache,
            [SimpleNamespace(is_kda=False), SimpleNamespace(is_kda=True)],
            parallelism,
            is_decode_role=decode,
        )

    def test_prefill_and_replicated_decode_preserve_checkpoint(self):
        for decode in (False, True):
            with self.subTest(decode=decode):
                self.assertEqual(self._bind(decode=decode), (128, 1024))

    def test_layer_kind_and_span_must_match_model(self):
        for options in (
            {"kinds": [CacheGroupType.LINEAR, CacheGroupType.FULL]},
            {"spans": [256, 1024]},
            {"spans": [128, 512]},
            {"spans": []},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._bind(**options)

    def test_manager_and_model_local_shards_must_agree(self):
        with self.assertRaisesRegex(ValueError, "local shard"):
            self._bind(decode=True, local_shards=8)


if __name__ == "__main__":
    unittest.main()
