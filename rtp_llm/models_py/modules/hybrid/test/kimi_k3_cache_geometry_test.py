import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.kimi_k3.cache_geometry import (
    bind_kimi_k3_cache_geometry,
    validate_kimi_k3_page_rr_target,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import CacheGroupType


class KimiK3CacheGeometryTest(unittest.TestCase):

    def _bind(
        self,
        *,
        decode=False,
        tp_size=8,
        upstream_shards=8,
        spans=None,
        kinds=None,
        local_shards=None,
    ):
        if spans is None:
            spans = (128, 128 * upstream_shards)
        cache = SimpleNamespace(
            seq_size_per_block=128,
            local_shard_count=(
                (1 if decode else tp_size) if local_shards is None else local_shards
            ),
            group_seq_size_per_block=spans,
            layer_group_types=kinds or [CacheGroupType.FULL, CacheGroupType.LINEAR],
            get_layer_cache=lambda layer: SimpleNamespace(group_id=layer),
        )
        parallelism = SimpleNamespace(
            tp_size=tp_size,
            tp_rank=0 if tp_size == 1 else 5,
            kv_page_rr_enabled=lambda: not decode,
            prefill_cp_config=SimpleNamespace(prefill_cp_size=upstream_shards),
        )
        return bind_kimi_k3_cache_geometry(
            cache,
            [SimpleNamespace(is_kda=False), SimpleNamespace(is_kda=True)],
            parallelism,
            is_decode_role=decode,
        )

    def test_local_and_upstream_geometry_are_independent(self):
        for decode, tp_size, upstream_shards in (
            (False, 8, 8),
            (True, 8, 8),
            (True, 1, 8),
            (True, 1, 16),
        ):
            with self.subTest(
                decode=decode,
                tp_size=tp_size,
                upstream_shards=upstream_shards,
            ):
                self.assertEqual(
                    self._bind(
                        decode=decode,
                        tp_size=tp_size,
                        upstream_shards=upstream_shards,
                    ),
                    (128, 128 * upstream_shards),
                )

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


class KimiK3PageRRTargetTest(unittest.TestCase):

    @staticmethod
    def _target(
        *,
        tp_size=8,
        ep_size=None,
        upstream_shards=None,
        local_page_rr=None,
        is_decode_role=False,
        cache_dtype=KvCacheDataType.BASE,
        mla_fp8_compute=False,
        page_tokens=128,
        kernel_page_tokens=None,
        linear_step=1,
        query_budget_tokens=0,
        compute_dtype=torch.bfloat16,
    ):
        if ep_size is None:
            ep_size = tp_size
        if upstream_shards is None:
            upstream_shards = tp_size
        if local_page_rr is None:
            local_page_rr = not is_decode_role
        parallelism = SimpleNamespace(
            tp_size=tp_size,
            ep_size=ep_size,
            kv_page_rr_enabled=lambda: local_page_rr,
            prefill_cp_config=SimpleNamespace(
                prefill_cp_size=upstream_shards,
                is_enabled=lambda: False,
            ),
        )
        return SimpleNamespace(
            parallelism=parallelism,
            model_config=SimpleNamespace(
                compute_dtype=compute_dtype,
                attn_config=SimpleNamespace(
                    kv_cache_dtype=cache_dtype,
                    mla_fp8_compute=mla_fp8_compute,
                    mla_prefill_expanded_kv_budget_bytes=0,
                    kv_lora_rank=512,
                    rope_head_dim=64,
                    nope_head_dim=128,
                    v_head_dim=128,
                ),
            ),
            kv_cache=SimpleNamespace(
                kernel_seq_size_per_block=(
                    page_tokens if kernel_page_tokens is None else kernel_page_tokens
                ),
                linear_step=linear_step,
            ),
            page_tokens=page_tokens,
            checkpoint_tokens=page_tokens * upstream_shards,
            is_decode_role=is_decode_role,
            kda_head_dim=128,
            whole_model_query_budget_tokens=query_budget_tokens,
            compute_capability=(10, 3),
        )

    def _validate(self, **options):
        target = self._target(**options)
        validate_kimi_k3_page_rr_target(**vars(target))
        return target

    def test_accepts_prefill_tp8_and_tp16_page_rr(self):
        for shards in (8, 16):
            with self.subTest(shards=shards):
                target = self._validate(
                    tp_size=shards,
                    ep_size=shards,
                    upstream_shards=shards,
                    local_page_rr=True,
                )
                self.assertEqual(target.checkpoint_tokens, 128 * shards)

    def test_accepts_equal_tp_and_replicated_decode_geometry(self):
        self._validate(
            tp_size=8,
            ep_size=8,
            upstream_shards=8,
            local_page_rr=False,
            is_decode_role=True,
        )
        for upstream_shards in (8, 16):
            for ep_size in (8, 16):
                with self.subTest(upstream_shards=upstream_shards, ep_size=ep_size):
                    target = self._validate(
                        tp_size=1,
                        ep_size=ep_size,
                        upstream_shards=upstream_shards,
                        local_page_rr=False,
                        is_decode_role=True,
                    )
                    self.assertEqual(target.checkpoint_tokens, 128 * upstream_shards)

    def test_accepts_only_matching_base_or_plain_fp8_cache(self):
        for cache_dtype, mla_fp8_compute in (
            (KvCacheDataType.BASE, False),
            (KvCacheDataType.FP8, True),
        ):
            with self.subTest(cache_dtype=cache_dtype, mla_fp8_compute=mla_fp8_compute):
                self._validate(
                    cache_dtype=cache_dtype,
                    mla_fp8_compute=mla_fp8_compute,
                )

        for cache_dtype, mla_fp8_compute in (
            (KvCacheDataType.BASE, True),
            (KvCacheDataType.FP8, False),
            (KvCacheDataType.INT8, False),
            (KvCacheDataType.INT8, True),
        ):
            with self.subTest(
                cache_dtype=cache_dtype, mla_fp8_compute=mla_fp8_compute
            ), self.assertRaisesRegex(ValueError, "cache precision"):
                self._validate(
                    cache_dtype=cache_dtype,
                    mla_fp8_compute=mla_fp8_compute,
                )

        for compute_dtype in (torch.float16, torch.float32):
            with self.subTest(compute_dtype=compute_dtype), self.assertRaisesRegex(
                ValueError, "BF16"
            ):
                self._validate(compute_dtype=compute_dtype)

    def test_rejects_role_and_topology_mismatches(self):
        invalid = (
            {"tp_size": 3, "ep_size": 3, "upstream_shards": 3},
            {"tp_size": 8, "ep_size": 16, "upstream_shards": 8},
            {"tp_size": 8, "ep_size": 8, "local_page_rr": False},
            {
                "tp_size": 1,
                "ep_size": 8,
                "upstream_shards": 8,
                "local_page_rr": True,
                "is_decode_role": True,
            },
            {
                "tp_size": 4,
                "ep_size": 8,
                "upstream_shards": 8,
                "local_page_rr": False,
                "is_decode_role": True,
            },
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._validate(**options)

    def test_rejects_invalid_page_and_linear_geometry(self):
        for options in (
            {"page_tokens": 64},
            {"page_tokens": 128, "kernel_page_tokens": 256},
            {"linear_step": 2},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._validate(**options)

    def test_decode_budget_alignment_uses_upstream_shards(self):
        target = self._target(
            tp_size=1,
            ep_size=16,
            upstream_shards=16,
            local_page_rr=False,
            is_decode_role=True,
            query_budget_tokens=128 * 16 + 16,
        )
        validate_kimi_k3_page_rr_target(**vars(target))

        target.whole_model_query_budget_tokens = 128 * 16 + 1
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_kimi_k3_page_rr_target(**vars(target))


if __name__ == "__main__":
    unittest.main()
