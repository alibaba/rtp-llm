import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.kimi_k3.cache_geometry import (
    bind_kimi_k3_cache_geometry,
    validate_kimi_k3_page_rr_target,
)
from rtp_llm.ops import KvCacheDataType, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import CacheGroupType


class KimiK3CacheGeometryTest(unittest.TestCase):

    def _bind(
        self,
        *,
        decode=False,
        tp_size=8,
        upstream_shards=8,
        page_tokens=128,
        spans=None,
        kinds=None,
        local_shards=None,
        dcp=False,
    ):
        if spans is None:
            spans = (
                page_tokens,
                page_tokens * max(tp_size if dcp else 1, upstream_shards),
            )
        cache = SimpleNamespace(
            seq_size_per_block=page_tokens,
            local_shard_count=(
                (tp_size if dcp or not decode else 1) if local_shards is None else local_shards
            ),
            group_seq_size_per_block=spans,
            layer_group_types=kinds or [CacheGroupType.FULL, CacheGroupType.LINEAR],
            get_layer_cache=lambda layer: SimpleNamespace(group_id=layer),
        )
        parallelism = ParallelismConfig()
        parallelism.tp_size = tp_size
        parallelism.tp_rank = tp_size - 1
        parallelism.role_type = RoleType.DECODE if decode else RoleType.PREFILL
        parallelism.decode_cp_kv_cache_sharded = dcp
        parallelism.prefill_cp_config.kv_cache_sharded = not decode
        parallelism.prefill_cp_config.prefill_cp_size = upstream_shards
        return bind_kimi_k3_cache_geometry(
            cache,
            [SimpleNamespace(is_kda=False), SimpleNamespace(is_kda=True)],
            parallelism,
        )

    def test_local_and_upstream_geometry_are_independent(self):
        for decode, tp_size, upstream_shards, dcp in (
            (False, 8, 8, False),
            (True, 8, 8, False),
            (True, 1, 8, False),
            (True, 1, 16, False),
            (True, 8, 1, True),
            (True, 8, 8, True),
            (True, 16, 16, True),
            (True, 8, 16, True),
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
                        dcp=dcp,
                    ),
                    (128, 128 * max(tp_size if dcp else 1, upstream_shards)),
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

    def test_large_pages_bind_checkpoint_span_from_upstream_not_local_tp(self):
        for page_tokens, shards, checkpoint_tokens in (
            (512, 8, 4096),
            (8192, 8, 65536),
            (8192, 16, 131072),
        ):
            for decode, tp_size in ((False, shards), (True, shards), (True, 1)):
                with self.subTest(page_tokens=page_tokens, decode=decode, tp=tp_size):
                    self.assertEqual(
                        self._bind(
                            page_tokens=page_tokens,
                            decode=decode,
                            tp_size=tp_size,
                            upstream_shards=shards,
                        ),
                        (page_tokens, checkpoint_tokens),
                    )

    def test_rejects_overflowing_checkpoint_span(self):
        with self.assertRaisesRegex(ValueError, "overflowing"):
            self._bind(page_tokens=1 << 28, upstream_shards=16, tp_size=16)


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
        parallelism = ParallelismConfig()
        parallelism.tp_size = tp_size
        parallelism.ep_size = ep_size
        parallelism.role_type = RoleType.DECODE if is_decode_role else RoleType.PREFILL
        parallelism.decode_cp_kv_cache_sharded = is_decode_role and local_page_rr
        parallelism.prefill_cp_config.kv_cache_sharded = not is_decode_role and local_page_rr
        parallelism.prefill_cp_config.prefill_cp_size = upstream_shards
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
            checkpoint_tokens=page_tokens * max(tp_size if local_page_rr else 1, upstream_shards),
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

    def test_accepts_power_of_two_physical_pages_with_128_kernel_pages(self):
        for page_tokens in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            for shards in (2, 4, 8, 16):
                for decode, tp_size in ((False, shards), (True, shards), (True, 1)):
                    for fp8 in (False, True):
                        with self.subTest(
                            page_tokens=page_tokens,
                            shards=shards,
                            decode=decode,
                            tp_size=tp_size,
                            fp8=fp8,
                        ):
                            self._validate(
                                tp_size=tp_size,
                                ep_size=shards,
                                upstream_shards=shards,
                                is_decode_role=decode,
                                page_tokens=page_tokens,
                                kernel_page_tokens=128,
                                cache_dtype=(
                                    KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
                                ),
                                mla_fp8_compute=fp8,
                            )

    def test_rejects_invalid_physical_kernel_page_ratios_in_both_roles(self):
        for page_tokens, kernel_page_tokens in (
            (0, 128),
            (-128, 128),
            (64, 128),
            (129, 128),
            (255, 128),
            (384, 128),
            (640, 128),
            (768, 128),
            (128, 0),
            (128, -128),
            (128, 64),
            (256, 256),
            (1024, 256),
            (1024, 1024),
        ):
            for decode, tp_size in ((False, 8), (True, 8), (True, 1)):
                with self.subTest(
                    page_tokens=page_tokens,
                    kernel_page_tokens=kernel_page_tokens,
                    decode=decode,
                    tp_size=tp_size,
                ), self.assertRaises(ValueError):
                    self._validate(
                        tp_size=tp_size,
                        ep_size=8,
                        upstream_shards=8,
                        is_decode_role=decode,
                        page_tokens=page_tokens,
                        kernel_page_tokens=kernel_page_tokens,
                    )

    def test_large_page_chunk_budget_must_reach_upstream_checkpoint(self):
        for decode, tp_size in ((False, 16), (True, 16), (True, 1)):
            target = self._target(
                tp_size=tp_size,
                ep_size=16,
                upstream_shards=16,
                is_decode_role=decode,
                page_tokens=8192,
                kernel_page_tokens=128,
                query_budget_tokens=131072,
            )
            with self.subTest(decode=decode, tp_size=tp_size):
                validate_kimi_k3_page_rr_target(**vars(target))
                target.whole_model_query_budget_tokens = 65536
                with self.assertRaisesRegex(ValueError, "query budget"):
                    validate_kimi_k3_page_rr_target(**vars(target))

    def test_accepts_equal_tp_and_replicated_decode_geometry(self):
        for tp_size, ep_size, source in (
            (4, 4, 4), (8, 8, 1), (8, 8, 8), (16, 16, 16), (8, 16, 16), (32, 32, 32)
        ):
            with self.subTest(tp_size=tp_size, ep_size=ep_size, source=source):
                self._validate(
                    tp_size=tp_size,
                    ep_size=ep_size,
                    upstream_shards=source,
                    local_page_rr=True,
                    is_decode_role=True,
                )
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
