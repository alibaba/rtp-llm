import math
import os
import random
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from navi_sched.adapters.vllm.VLLMCachePolicy import (
    VLLMCacheDelta,
    VLLMCachePolicy,
    VLLMCacheState,
    get_vllm_config_full_attention_query_drop_tokens,
    get_vllm_full_attention_query_drop_tokens,
    get_vllm_hash_block_size,
    get_vllm_prefill_routing_policy,
    resolve_vllm_hybrid_geometry,
)


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class MambaSpec:
    def __init__(self, block_size, mode="align"):
        self.block_size = block_size
        self.mamba_cache_mode = mode


class SlidingWindowSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class UniformTypeKVCacheSpecs:
    def __init__(self, block_size, *specs):
        self.block_size = block_size
        self.kv_cache_specs = {
            f"layer_{index}": spec for index, spec in enumerate(specs)
        }


def make_vllm_config(block_size=16, dcp_size=1, pcp_size=1):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def make_kv_cache_config(*specs, num_blocks=1024):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs],
    )


def block_hash(value):
    return bytes([value]) * 32


class VLLMCachePolicyTest(unittest.TestCase):
    def test_r50_disabled_drop_clears_annotation_without_disabling_mtp(self):
        config = make_vllm_config()
        config.speculative_config = SimpleNamespace(use_eagle=lambda: True)
        full = FullAttentionSpec(64)
        cache = make_kv_cache_config(full, MambaSpec(64))
        cache.kv_cache_groups[0].is_eagle_group = True
        plugin = types.ModuleType("vllm_torchtpu")
        plugin.envs = SimpleNamespace(VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=True)
        with (
            mock.patch.dict(os.environ, {"VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": "1"}),
            mock.patch.dict(sys.modules, {"vllm_torchtpu": plugin}),
        ):
            self.assertEqual(
                0,
                get_vllm_config_full_attention_query_drop_tokens(
                    config, cache, ("full", "tail"), (4, 4), 16, 64
                ),
            )
            cache.kv_cache_groups[0].is_eagle_group = False
            self.assertEqual(
                0,
                get_vllm_config_full_attention_query_drop_tokens(
                    config, cache, ("full", "tail"), (4, 4), 16, 64
                ),
            )
            self.assertTrue(config.speculative_config.use_eagle())
            plugin.envs.VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP = False
            self.assertEqual(
                16,
                get_vllm_config_full_attention_query_drop_tokens(
                    config, cache, ("full", "tail"), (4, 4), 16, 64
                ),
            )

    def test_full_query_drop_uses_actual_coordinator_groups(self):
        full = FullAttentionSpec(16)
        tail = MambaSpec(16)
        coordinator = SimpleNamespace(
            attention_groups=[
                SimpleNamespace(spec=full, group_ids=[0], use_eagle=True),
                SimpleNamespace(spec=tail, group_ids=[1], use_eagle=True),
            ],
            single_type_managers=[
                SimpleNamespace(block_size=64),
                SimpleNamespace(block_size=64),
            ],
            _cache_hit_alignment_tokens=32,
            enable_partial_hash_hits=False,
        )
        self.assertEqual(32, get_vllm_full_attention_query_drop_tokens(coordinator, 16))
        coordinator.attention_groups[0].use_eagle = False
        self.assertEqual(0, get_vllm_full_attention_query_drop_tokens(coordinator, 16))
        coordinator.attention_groups[1].use_eagle = False
        coordinator.attention_groups[0].use_eagle = True
        coordinator._cache_hit_alignment_tokens = 128
        self.assertEqual(64, get_vllm_full_attention_query_drop_tokens(coordinator, 16))

    def test_full_query_drop_config_follows_group_annotation_and_fallback(self):
        config = make_vllm_config()
        config.speculative_config = SimpleNamespace(use_eagle=lambda: True)
        full = FullAttentionSpec(16)
        groups = [
            SimpleNamespace(kv_cache_spec=full, is_eagle_group=False),
            SimpleNamespace(kv_cache_spec=MambaSpec(16), is_eagle_group=True),
        ]
        cache = SimpleNamespace(kv_cache_groups=groups)
        self.assertEqual(
            0,
            get_vllm_config_full_attention_query_drop_tokens(
                config, cache, ("full", "tail"), (4, 4), 16, 64
            ),
        )
        groups[0].is_eagle_group = True
        self.assertEqual(
            64,
            get_vllm_config_full_attention_query_drop_tokens(
                config, cache, ("full", "tail"), (4, 4), 16, 64
            ),
        )
        groups[0].is_eagle_group = groups[1].is_eagle_group = False
        self.assertEqual(
            64,
            get_vllm_config_full_attention_query_drop_tokens(
                config, cache, ("full", "tail"), (4, 4), 16, 64
            ),
        )
        config.speculative_config = None
        self.assertEqual(
            0,
            get_vllm_config_full_attention_query_drop_tokens(
                config, cache, ("full", "tail"), (4, 4), 16, 64
            ),
        )

    def test_full_query_drop_partial_hash_alignment_and_unitary(self):
        config = make_vllm_config()
        config.speculative_config = SimpleNamespace(use_eagle=lambda: True)
        cache = make_kv_cache_config(FullAttentionSpec(64), MambaSpec(64))
        self.assertEqual(
            16,
            get_vllm_config_full_attention_query_drop_tokens(
                config, cache, ("full", "tail"), (4, 4), 16, 64
            ),
        )
        coordinator = SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(16),
            block_size=16,
            eagle_group_ids={0},
        )
        self.assertEqual(16, get_vllm_full_attention_query_drop_tokens(coordinator, 16))
        coordinator.eagle_group_ids.clear()
        self.assertEqual(0, get_vllm_full_attention_query_drop_tokens(coordinator, 16))

    def test_full_query_drop_rejects_inexact_geometry(self):
        coordinator = SimpleNamespace(
            attention_groups=[
                SimpleNamespace(
                    spec=FullAttentionSpec(16), group_ids=[0], use_eagle=True
                )
            ],
            single_type_managers=[SimpleNamespace(block_size=24)],
            _cache_hit_alignment_tokens=24,
            enable_partial_hash_hits=False,
        )
        with self.assertRaises(ValueError):
            get_vllm_full_attention_query_drop_tokens(coordinator, 16)

    def test_prefill_routing_policy_uses_runtime_connector_predicates(self):
        module_name = "test_vllm_prefill_routing_connector"
        module = types.ModuleType(module_name)
        scheduler_type = type("Scheduler", (), {"__module__": module_name})
        for producer in (False, True):
            for transport in (False, True):
                for stage3 in (False, True):
                    with self.subTest(
                        producer=producer, transport=transport, stage3=stage3
                    ):
                        scheduler = scheduler_type()
                        scheduler.is_producer = producer
                        scheduler._stage3_truncates_prompt = mock.Mock(
                            return_value=stage3
                        )
                        module._use_raiden_stage3_transport = mock.Mock(
                            return_value=transport
                        )
                        connector = SimpleNamespace(connector_scheduler=scheduler)
                        with mock.patch.dict(sys.modules, {module_name: module}):
                            self.assertEqual(
                                {
                                    "trim_last_prompt_token": producer
                                    and transport
                                    and stage3
                                },
                                get_vllm_prefill_routing_policy(connector),
                            )

    def test_prefill_routing_policy_combines_multi_connector_once(self):
        module_name = "test_vllm_prefill_routing_multi"
        module = types.ModuleType(module_name)
        module._use_raiden_stage3_transport = lambda: True
        scheduler_type = type("Scheduler", (), {"__module__": module_name})
        scheduler = scheduler_type()
        scheduler.is_producer = True
        scheduler._stage3_truncates_prompt = lambda: True
        trimming = SimpleNamespace(connector_scheduler=scheduler)
        connector = SimpleNamespace(
            _connectors=[
                object(),
                trimming,
                SimpleNamespace(_connectors=[trimming]),
            ]
        )
        with mock.patch.dict(sys.modules, {module_name: module}):
            self.assertEqual(
                {"trim_last_prompt_token": True},
                get_vllm_prefill_routing_policy(connector),
            )

    def test_prefill_routing_policy_unknown_is_identity(self):
        for connector in (
            None,
            object(),
            SimpleNamespace(_connectors=[]),
            SimpleNamespace(connector_scheduler=object()),
        ):
            with self.subTest(connector=connector):
                self.assertEqual(
                    {"trim_last_prompt_token": False},
                    get_vllm_prefill_routing_policy(connector),
                )

    def test_prefill_routing_policy_rejects_missing_transport_api(self):
        module_name = "test_vllm_prefill_routing_missing"
        scheduler_type = type("Scheduler", (), {"__module__": module_name})
        scheduler = scheduler_type()
        scheduler.is_producer = True
        scheduler._stage3_truncates_prompt = lambda: True
        connector = SimpleNamespace(connector_scheduler=scheduler)
        for module in (None, types.ModuleType(module_name)):
            with self.subTest(module=module):
                with mock.patch.dict(sys.modules, {module_name: module}):
                    with self.assertRaisesRegex(RuntimeError, "transport predicate"):
                        get_vllm_prefill_routing_policy(connector)

    def setUp(self):
        def fake_vllm_geometry(config, kv_config):
            dcp = config.parallel_config.decode_context_parallel_size
            sizes = tuple(
                (
                    spec.block_size * dcp
                    if isinstance(spec, FullAttentionSpec)
                    else spec.block_size
                )
                for spec in (group.kv_cache_spec for group in kv_config.kv_cache_groups)
            )
            return math.lcm(*sizes), math.gcd(*sizes), sizes

        patcher = mock.patch(
            "navi_sched.adapters.vllm.VLLMCachePolicy.resolve_vllm_hybrid_geometry",
            side_effect=fake_vllm_geometry,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_full_attention_group_uses_flat_cache(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(16), num_blocks=1000),
        )

        self.assertEqual(16, policy.block_size)
        self.assertEqual(1000, policy.total_blocks)
        self.assertEqual(1, policy.linear_step)
        self.assertFalse(policy.hybrid_tail_required)

        state = VLLMCacheState(policy)
        value = block_hash(1)
        self.assertEqual(
            VLLMCacheDelta(added=(value,)),
            state.apply("stored", 0, (value,)),
        )
        self.assertEqual(
            VLLMCacheDelta(removed=(value,)),
            state.apply("removed", 0, (value,)),
        )

    def test_all_full_attention_groups_must_contain_a_hash(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(
                FullAttentionSpec(16),
                FullAttentionSpec(16),
                num_blocks=2000,
            ),
        )
        state = VLLMCacheState(policy)
        value = block_hash(2)

        self.assertEqual(VLLMCacheDelta(), state.apply("stored", 0, (value,)))
        self.assertEqual(
            VLLMCacheDelta(added=(value,)),
            state.apply("stored", 1, (value,)),
        )
        self.assertEqual(2000, policy.total_blocks)

    def test_context_parallelism_defines_hash_block_size(self):
        config = make_vllm_config(block_size=16, dcp_size=2, pcp_size=2)
        policy = VLLMCachePolicy.from_engine_config(
            config,
            make_kv_cache_config(FullAttentionSpec(16)),
        )

        self.assertEqual(64, get_vllm_hash_block_size(config))
        self.assertEqual(64, policy.block_size)
        self.assertEqual((1,), policy.group_block_scales)

    def test_hybrid_context_parallel_uses_fine_hash_geometry(self):
        config = make_vllm_config(block_size=768, dcp_size=8)
        config.kv_transfer_config = object()
        policy = VLLMCachePolicy.from_engine_config(
            config,
            make_kv_cache_config(FullAttentionSpec(768), MambaSpec(768)),
        )
        self.assertEqual(768, policy.block_size)
        self.assertEqual((8, 1), policy.group_block_scales)
        self.assertEqual(8, policy.linear_step)

        hashes = tuple(block_hash(value) for value in range(8))
        self.assertEqual(hashes, policy.expand_group_hash(0, b"".join(hashes)))
        self.assertEqual((hashes[-1],), policy.expand_group_hash(1, hashes[-1]))

    def test_hybrid_prefill_parallelism_does_not_scale_cache_blocks(self):
        config = make_vllm_config(block_size=768, dcp_size=1, pcp_size=8)
        config.kv_transfer_config = object()
        policy = VLLMCachePolicy.from_engine_config(
            config,
            make_kv_cache_config(FullAttentionSpec(768), MambaSpec(768)),
        )
        self.assertEqual(768, policy.block_size)
        self.assertEqual((1, 1), policy.group_block_scales)
        self.assertEqual(1, policy.linear_step)

    def test_tpu_pcp_runtime_patch_defines_logical_group_blocks(self):
        config = make_vllm_config(block_size=768, dcp_size=1, pcp_size=8)
        kv = make_kv_cache_config(FullAttentionSpec(768), MambaSpec(768))
        runtime = types.ModuleType("vllm.v1.core.kv_cache_utils")
        runtime._tpu_hybrid_pcp_block_sizes_patch = True
        runtime.resolve_kv_cache_block_sizes = lambda *_: (6144, 6144)
        runtime.resolve_dcp_kv_block_size = lambda spec, _: spec.block_size
        core = types.ModuleType("vllm.v1.core")
        core.kv_cache_utils = runtime
        with (
            mock.patch.dict(
                os.environ,
                {"TPU_RAIDEN_JOB_NAME": "prefill", "DS_ENABLE_DSERV": "1"},
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "vllm": types.ModuleType("vllm"),
                    "vllm.v1": types.ModuleType("vllm.v1"),
                    "vllm.v1.core": core,
                    "vllm.v1.core.kv_cache_utils": runtime,
                },
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMCachePolicy._vllm_layer_specs",
                side_effect=lambda spec: (spec,),
            ),
        ):
            self.assertEqual(
                (6144, 6144, (6144, 6144)),
                resolve_vllm_hybrid_geometry(config, kv),
            )

    def test_native_prefill_uses_runtime_geometry_without_kv_transfer_config(self):
        config = make_vllm_config(block_size=768, dcp_size=1, pcp_size=8)
        with mock.patch.dict(
            os.environ,
            {"TPU_RAIDEN_JOB_NAME": "prefill", "DS_ENABLE_DSERV": "1"},
        ):
            policy = VLLMCachePolicy.from_engine_config(
                config,
                make_kv_cache_config(FullAttentionSpec(768), MambaSpec(768)),
            )
        self.assertEqual(768, policy.block_size)
        self.assertEqual((1, 1), policy.group_block_scales)
        self.assertEqual(1, policy.linear_step)

    def test_native_prefill_full_attention_ignores_prefill_parallel_factor(self):
        config = make_vllm_config(block_size=768, dcp_size=1, pcp_size=8)
        with mock.patch.dict(
            os.environ,
            {"TPU_RAIDEN_JOB_NAME": "prefill", "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1"},
        ):
            policy = VLLMCachePolicy.from_engine_config(
                config, make_kv_cache_config(FullAttentionSpec(768))
            )
        self.assertEqual(768, policy.block_size)
        self.assertEqual((1,), policy.group_block_scales)
        self.assertEqual(1, policy.linear_step)

    def test_native_prefill_expands_uniform_type_cache_groups(self):
        config = make_vllm_config(block_size=768, pcp_size=8)
        kv = make_kv_cache_config(
            UniformTypeKVCacheSpecs(768, FullAttentionSpec(768)),
            UniformTypeKVCacheSpecs(768, MambaSpec(768)),
        )

        def layers(spec):
            return tuple(spec.kv_cache_specs.values())

        with (
            mock.patch.dict(
                os.environ,
                {
                    "TPU_RAIDEN_JOB_NAME": "prefill",
                    "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                },
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMCachePolicy._vllm_layer_specs",
                side_effect=layers,
            ),
        ):
            policy = VLLMCachePolicy.from_engine_config(config, kv)
        self.assertEqual(768, policy.block_size)
        self.assertEqual(("full", "tail"), policy.group_roles)
        self.assertEqual((1, 1), policy.group_block_scales)
        self.assertEqual(1, policy.linear_step)

    def test_larger_group_blocks_are_split_into_hash_blocks(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(32)),
        )
        state = VLLMCacheState(policy)
        first = block_hash(3)
        second = block_hash(4)

        self.assertEqual(
            VLLMCacheDelta(added=(first, second)),
            state.apply("stored", 0, (first + second,)),
        )

    def test_v029_larger_group_uses_terminal_hash(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(32)),
        )
        state = VLLMCacheState(policy)
        terminal = block_hash(4)

        self.assertEqual(
            VLLMCacheDelta(added=(terminal,)),
            state.apply("stored", 0, (terminal,)),
        )
        self.assertEqual(
            VLLMCacheDelta(removed=(terminal,)),
            state.apply("removed", 0, (terminal,)),
        )

    def test_full_attention_and_mamba_map_to_full_and_tail_readiness(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(
                FullAttentionSpec(16),
                MambaSpec(64),
                num_blocks=2000,
            ),
        )
        state = VLLMCacheState(policy)
        hashes = tuple(block_hash(value) for value in range(1, 5))

        self.assertEqual(2000, policy.total_blocks)
        self.assertEqual(4, policy.linear_step)
        self.assertTrue(policy.hybrid_tail_required)
        self.assertEqual(
            VLLMCacheDelta(added_full_ready=(hashes[3],)),
            state.apply("stored", 0, (hashes[3],)),
        )
        self.assertEqual(
            VLLMCacheDelta(added_tail_ready=(hashes[3],)),
            state.apply("stored", 1, (b"".join(hashes),)),
        )
        self.assertEqual(
            VLLMCacheDelta(
                removed_full_ready=(hashes[3],),
                removed_tail_ready=(hashes[3],),
            ),
            state.apply("cleared", None, ()),
        )

    def test_duplicate_hashes_are_reference_counted(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(16)),
        )
        state = VLLMCacheState(policy)
        value = block_hash(5)

        self.assertEqual(
            VLLMCacheDelta(added=(value,)),
            state.apply("stored", 0, (value, value)),
        )
        self.assertEqual(VLLMCacheDelta(), state.apply("removed", 0, (value,)))
        self.assertEqual(
            VLLMCacheDelta(removed=(value,)),
            state.apply("removed", 0, (value,)),
        )

    def test_unsupported_hybrid_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exactly represent"):
            VLLMCachePolicy.from_engine_config(
                make_vllm_config(),
                make_kv_cache_config(SlidingWindowSpec(16)),
            )

    def test_mamba_only_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mamba-only"):
            VLLMCachePolicy.from_engine_config(
                make_vllm_config(),
                make_kv_cache_config(MambaSpec(16)),
            )

    def test_snapshot_bulk_matches_event_reduction_across_topologies(self):
        rng = random.Random(20261022)
        for roles, scales in [
            (("full",), (1,)),
            (("full", "full"), (2, 1)),
            (("full", "tail"), (2, 2)),
            (("full", "full", "tail", "tail"), (4, 2, 2, 4)),
        ]:
            policy = VLLMCachePolicy(
                block_size=16,
                total_blocks=1024,
                linear_step=1,
                hybrid_tail_required="tail" in roles,
                group_roles=roles,
                group_block_scales=scales,
                spec_names=roles,
            )
            state = VLLMCacheState(policy)
            for turn in range(40):
                incoming = {}
                for group_id, scale in enumerate(scales):
                    if rng.randrange(4) == 0:
                        continue
                    values = tuple(
                        b"".join(block_hash(rng.randrange(1, 9)) for _ in range(scale))
                        for _ in range(rng.randrange(6))
                    )
                    incoming[group_id] = values + values[:1]
                reference = VLLMCacheState(policy)
                for group_id, hashes in incoming.items():
                    reference.apply("stored", group_id, hashes)
                expected = VLLMCacheDelta(
                    added=tuple(sorted(reference._flat_ready - state._flat_ready)),
                    removed=tuple(sorted(state._flat_ready - reference._flat_ready)),
                    added_full_ready=tuple(
                        sorted(reference._full_ready - state._full_ready)
                    ),
                    removed_full_ready=tuple(
                        sorted(state._full_ready - reference._full_ready)
                    ),
                    added_tail_ready=tuple(
                        sorted(reference._tail_ready - state._tail_ready)
                    ),
                    removed_tail_ready=tuple(
                        sorted(state._tail_ready - reference._tail_ready)
                    ),
                )
                with self.subTest(roles=roles, turn=turn):
                    self.assertEqual(expected, state.replace(incoming))
                    self.assertEqual(reference._group_counts, state._group_counts)
                    self.assertEqual(reference._flat_ready, state._flat_ready)
                    self.assertEqual(reference._full_ready, state._full_ready)
                    self.assertEqual(reference._tail_ready, state._tail_ready)
                    for group_id, hashes in incoming.items():
                        self.assertEqual(
                            reference.apply("removed", group_id, hashes[:1]),
                            state.apply("removed", group_id, hashes[:1]),
                        )

    def test_snapshot_bulk_avoids_per_block_membership_reduction(self):
        policy = VLLMCachePolicy(
            block_size=16,
            total_blocks=1024,
            linear_step=1,
            hybrid_tail_required=True,
            group_roles=("full", "full", "tail"),
            group_block_scales=(1, 1, 1),
            spec_names=("full", "full", "tail"),
        )
        a, b = block_hash(3), block_hash(4)
        state = VLLMCacheState(policy)
        with mock.patch.object(
            VLLMCacheState,
            "_membership",
            side_effect=AssertionError("incremental membership on bulk snapshot"),
        ):
            delta = state.replace({0: (a, b), 1: (a,), 2: (a, b)})
        self.assertEqual(
            VLLMCacheDelta(added_full_ready=(a,), added_tail_ready=(a, b)), delta
        )
        self.assertEqual(
            VLLMCacheDelta(removed_full_ready=(a,)), state.apply("removed", 1, (a,))
        )

    def test_snapshot_replace_is_idempotent_and_preserves_reference_counts(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(), make_kv_cache_config(FullAttentionSpec(16))
        )
        state = VLLMCacheState(policy)
        h = block_hash(3)
        self.assertEqual(VLLMCacheDelta(added=(h,)), state.replace({0: (h, h)}))
        self.assertTrue(state.replace({0: (h, h)}).empty())
        self.assertTrue(state.apply("removed", 0, (h,)).empty())
        self.assertEqual(VLLMCacheDelta(removed=(h,)), state.apply("removed", 0, (h,)))

    def test_snapshot_replace_expands_full_and_keeps_tail_endpoint(self):
        policy = VLLMCachePolicy(
            block_size=16,
            total_blocks=10,
            linear_step=2,
            hybrid_tail_required=True,
            group_roles=("full", "tail"),
            group_block_scales=(2, 2),
            spec_names=("full", "tail"),
        )
        state = VLLMCacheState(policy)
        a, b = block_hash(3), block_hash(4)
        delta = state.replace({0: (a + b,), 1: (a + b,)})
        self.assertEqual((a, b), delta.added_full_ready)
        self.assertEqual((b,), delta.added_tail_ready)
        removed = state.replace({})
        self.assertEqual((a, b), removed.removed_full_ready)
        self.assertEqual((b,), removed.removed_tail_ready)

    def test_invalid_snapshot_does_not_partially_replace_state(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(), make_kv_cache_config(FullAttentionSpec(16))
        )
        state = VLLMCacheState(policy)
        a, b = block_hash(3), block_hash(4)
        state.apply("stored", 0, (a,))
        with self.assertRaises(ValueError):
            state.replace({0: (b,), 8: (b,)})
        self.assertEqual(VLLMCacheDelta(removed=(a,)), state.apply("removed", 0, (a,)))


if __name__ == "__main__":
    unittest.main()
