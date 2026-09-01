import pickle
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.kimi_k3.cache_geometry import (
    validate_kimi_k3_page_rr_target,
)
from rtp_llm.ops import CPRotateMethod, KVCacheConfig, KvCacheDataType, ParallelismConfig


class KimiK3CacheConfigTest(unittest.TestCase):

    def test_kda_pool_cap_survives_spawn_pickle_round_trip(self):
        config = KVCacheConfig()
        config.kimi_k3_kda_pool_blocks = 112
        self.assertEqual(
            pickle.loads(pickle.dumps(config)).kimi_k3_kda_pool_blocks, 112
        )


class KimiK3PageRRTargetTest(unittest.TestCase):
    @staticmethod
    def _valid(page=128, shards=8, decode=False, kernel_page=None):
        parallelism = ParallelismConfig()
        parallelism.tp_size = parallelism.ep_size = shards
        parallelism.prefill_cp_config.kv_cache_sharded = not decode
        parallelism.prefill_cp_config.prefill_cp_size = shards
        return SimpleNamespace(
            parallelism=parallelism,
            model_config=SimpleNamespace(
                compute_dtype=torch.bfloat16,
                attn_config=SimpleNamespace(
                    kv_cache_dtype=KvCacheDataType.BASE,
                    mla_prefill_expanded_kv_budget_bytes=5 * 1024**3,
                    kv_lora_rank=512,
                    rope_head_dim=64,
                    nope_head_dim=128,
                    v_head_dim=128,
                ),
            ),
            kv_cache=SimpleNamespace(
                kernel_seq_size_per_block=page if kernel_page is None else kernel_page,
                linear_step=1,
            ),
            page_tokens=page,
            checkpoint_tokens=page * shards,
            is_decode_role=decode,
            kda_head_dim=128,
            whole_model_query_budget_tokens=0 if decode else 65536,
            compute_capability=(10, 3),
        )

    def test_supported_prefill_and_decode_layouts(self):
        for page in (128, 256):
            for shards in (2, 4, 8):
                for decode in (False, True):
                    with self.subTest(page=page, shards=shards, decode=decode):
                        validate_kimi_k3_page_rr_target(
                            **vars(self._valid(page, shards, decode))
                        )
        validate_kimi_k3_page_rr_target(**vars(self._valid(256, 4, True, 128)))

    def test_rejects_incompatible_runtime_layouts(self):
        cases = (
            ("parallelism.ep_size", 4, "TP == EP"),
            (
                "parallelism.prefill_cp_config.method",
                CPRotateMethod.ALL_GATHER,
                "Query CP",
            ),
            ("is_decode_role", True, "role"),
            ("checkpoint_tokens", 512, "checkpoint"),
            ("kv_cache.linear_step", 2, "linear_step"),
            ("model_config.compute_dtype", torch.float16, "BF16"),
            ("model_config.attn_config.kv_cache_dtype", KvCacheDataType.FP8, "BASE"),
            ("model_config.attn_config.kv_lora_rank", 256, "512\+64"),
            ("model_config.attn_config.nope_head_dim", 256, "head dimensions"),
            ("kda_head_dim", 64, "128"),
            ("compute_capability", (9, 0), "SM100"),
            ("whole_model_query_budget_tokens", 512, "checkpoint"),
            ("whole_model_query_budget_tokens", 65535, "checkpoint"),
            (
                "model_config.attn_config.mla_prefill_expanded_kv_budget_bytes",
                -1,
                "non-negative",
            ),
        )
        for path, value, message in cases:
            with self.subTest(path=path):
                target = self._valid()
                owner = target
                *attributes, field = path.split(".")
                for attribute in attributes:
                    owner = getattr(owner, attribute)
                setattr(owner, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_kimi_k3_page_rr_target(**vars(target))

    def test_rejects_undeclared_page_and_shard_bounds(self):
        for options in (
            {"shards": 3},
            {"page": 1024},
            {"page": 256, "kernel_page": 128},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError, "page/shard"
            ):
                validate_kimi_k3_page_rr_target(**vars(self._valid(**options)))

    def test_leaves_expanded_kv_capacity_and_alignment_to_planner(self):
        for budget in (0, 120 * 1024**2, 5 * 1024**3 + 1):
            with self.subTest(budget=budget):
                target = self._valid()
                target.model_config.attn_config.mla_prefill_expanded_kv_budget_bytes = (
                    budget
                )
                validate_kimi_k3_page_rr_target(**vars(target))


if __name__ == "__main__":
    unittest.main()
