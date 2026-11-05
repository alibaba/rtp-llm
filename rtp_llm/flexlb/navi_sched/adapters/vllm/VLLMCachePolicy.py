from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass

_HASH_BYTES = 32


def get_vllm_prefill_routing_policy(connector) -> dict[str, bool]:
    """Export the runtime connector's prompt-length rule without mutating it."""
    trim_last_prompt_token = False
    for child in getattr(connector, "_connectors", ()):
        child_policy = get_vllm_prefill_routing_policy(child)
        trim_last_prompt_token |= child_policy["trim_last_prompt_token"]

    scheduler = getattr(connector, "connector_scheduler", None)
    stage3_predicate = getattr(scheduler, "_stage3_truncates_prompt", None)
    if stage3_predicate is not None:
        module = sys.modules.get(type(scheduler).__module__)
        transport_predicate = getattr(module, "_use_raiden_stage3_transport", None)
        if not callable(transport_predicate):
            raise RuntimeError("Raiden stage3 transport predicate is unavailable")
        if not callable(stage3_predicate):
            raise RuntimeError("Raiden stage3 truncation predicate is unavailable")
        if not hasattr(scheduler, "is_producer"):
            raise RuntimeError("Raiden stage3 producer capability is unavailable")
        trim_last_prompt_token |= bool(
            scheduler.is_producer and transport_predicate() and stage3_predicate()
        )
    return {"trim_last_prompt_token": trim_last_prompt_token}


def is_vllm_native_pd_prefill() -> bool:
    """Match the existing vLLM adapter's native Prefill ingress switch."""
    return os.environ.get("TPU_RAIDEN_JOB_NAME", "").strip().lower() == "prefill" and (
        os.environ.get("VLLM_NAVI_SCHED_NATIVE_INGRESS") == "1"
        or os.environ.get("DS_ENABLE_DSERV") == "1"
    )


def is_vllm_eagle_block_drop_enabled(vllm_config) -> bool:
    """Resolve r50's cache-drop policy without disabling MTP lookahead."""
    speculative = getattr(vllm_config, "speculative_config", None)
    use_eagle = (
        callable(getattr(speculative, "use_eagle", None)) and speculative.use_eagle()
    )
    if not use_eagle:
        return False
    return not is_vllm_eagle_block_drop_disabled()


def is_vllm_eagle_block_drop_disabled() -> bool:
    """Use the TPU plugin's strict flag resolver and its r50 default."""
    if "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP" in os.environ:
        from vllm_torchtpu import envs

        return bool(envs.VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP)
    return False


def get_vllm_hash_block_size(vllm_config) -> int:
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    block_size = (
        int(cache_config.block_size)
        * int(parallel_config.decode_context_parallel_size)
        * int(getattr(parallel_config, "prefill_context_parallel_size", 1))
    )
    if block_size <= 0:
        raise ValueError("vLLM scheduler block size must be positive")
    return block_size


def _is_spec(spec, class_name: str) -> bool:
    return any(base.__name__ == class_name for base in type(spec).__mro__)


def _full_query_drop_tokens(drops, hash_block_size):
    drops = set(drops)
    if len(drops) > 1:
        raise ValueError("Full attention groups have different cache query drops")
    drop = next(iter(drops), 0)
    if drop < 0 or hash_block_size <= 0 or drop % hash_block_size:
        raise ValueError("Full attention query drop is not hash-block aligned")
    return drop


def get_vllm_full_attention_query_drop_tokens(coordinator, hash_block_size):
    """Read the actual coordinator's Full lookup rule, independently of Mamba."""
    if not hasattr(coordinator, "attention_groups"):
        if not _is_spec(coordinator.kv_cache_spec, "FullAttentionSpec"):
            return 0
        drop = coordinator.block_size if 0 in coordinator.eagle_group_ids else 0
        return _full_query_drop_tokens((drop,), hash_block_size)
    drops = []
    for group in coordinator.attention_groups:
        if not _is_spec(group.spec, "FullAttentionSpec"):
            continue
        manager = coordinator.single_type_managers[group.group_ids[0]]
        drop = 0
        if group.use_eagle:
            drop = min(coordinator._cache_hit_alignment_tokens, manager.block_size)
        drops.append(drop)
    return _full_query_drop_tokens(drops, hash_block_size)


def get_vllm_config_full_attention_query_drop_tokens(
    vllm_config,
    kv_cache_config,
    roles,
    scales,
    hash_block_size,
    scheduler_block_size,
):
    """Mirror coordinator group annotation before the scheduler is constructed."""
    groups = tuple(kv_cache_config.kv_cache_groups)
    annotated = {
        i for i, group in enumerate(groups) if getattr(group, "is_eagle_group", False)
    }
    use_eagle = is_vllm_eagle_block_drop_enabled(vllm_config)
    if is_vllm_eagle_block_drop_disabled():
        annotated.clear()
    if use_eagle and not annotated:
        annotated = set(range(len(groups)))
    dcp_size = int(vllm_config.parallel_config.decode_context_parallel_size)
    # FullAttentionManager and MambaManager both support fine-grained lookup.
    # Match the coordinator's direct MambaSpec check (uniform wrappers do not
    # enable partial hits), using spec sizes before DCP/PCP scaling.
    partial_hits = any(
        _is_spec(group.kv_cache_spec, "MambaSpec")
        and getattr(group.kv_cache_spec, "mamba_cache_mode", None) == "align"
        and (
            group.kv_cache_spec.block_size > hash_block_size
            if dcp_size == 1
            else group.kv_cache_spec.block_size >= hash_block_size
        )
        for group in groups
    )
    alignment = hash_block_size if partial_hits else scheduler_block_size
    drops = []
    for i, (group, role, scale) in enumerate(zip(groups, roles, scales, strict=True)):
        if role != "full":
            continue
        # r39 combines equal specs and ORs their eagle bit before lookup.
        eagle = any(
            j in annotated and other.kv_cache_spec == group.kv_cache_spec
            for j, other in enumerate(groups)
        )
        drops.append(min(alignment, scale * hash_block_size) if eagle else 0)
    return _full_query_drop_tokens(drops, hash_block_size)


def resolve_vllm_hybrid_geometry(vllm_config, kv_cache_config):
    """Use vLLM's scheduler/cache-manager block geometry for hybrid models."""
    from vllm.v1.core import kv_cache_utils

    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    parallel = vllm_config.parallel_config
    dcp_size = int(parallel.decode_context_parallel_size)
    pcp_size = int(parallel.prefill_context_parallel_size)
    groups = tuple(kv_cache_config.kv_cache_groups)
    tpu_hybrid_pcp = (
        is_vllm_native_pd_prefill()
        and getattr(kv_cache_utils, "_tpu_hybrid_pcp_block_sizes_patch", False)
        and pcp_size > 1
        and dcp_size == 1
        and len(groups) > 1
        and any(
            _is_spec(layer, "MambaSpec")
            for group in groups
            for layer in _vllm_layer_specs(group.kv_cache_spec)
        )
        and any(
            _is_spec(layer, "FullAttentionSpec")
            for group in groups
            for layer in _vllm_layer_specs(group.kv_cache_spec)
        )
    )
    if tpu_hybrid_pcp:
        # vllm-torchtpu patches the scheduler resolver and its coordinator:
        # both attention and Mamba rank-local blocks span PCP ranks logically.
        # The generic vLLM DCP-only helper still reports rank-local Mamba
        # blocks, so use the same PCP factor and verify against the patched
        # scheduler's returned geometry before reporting it to Master.
        group_block_sizes = tuple(
            int(group.kv_cache_spec.block_size) * pcp_size for group in groups
        )
        if math.lcm(*group_block_sizes) != scheduler_block_size:
            raise ValueError("TPU PCP group geometry differs from vLLM scheduler")
    else:
        group_block_sizes = tuple(
            kv_cache_utils.resolve_dcp_kv_block_size(group.kv_cache_spec, dcp_size)
            for group in groups
        )
    return scheduler_block_size, hash_block_size, group_block_sizes


def _vllm_layer_specs(spec):
    """Expand vLLM's uniform-type group wrapper to its per-layer specs."""
    from vllm.v1.kv_cache_interface import iter_layer_specs

    return tuple(iter_layer_specs(spec))


@dataclass(frozen=True)
class VLLMCachePolicy:
    block_size: int
    total_blocks: int
    linear_step: int
    hybrid_tail_required: bool
    group_roles: tuple[str, ...]
    group_block_scales: tuple[int, ...]
    spec_names: tuple[str, ...]
    full_attention_query_drop_tokens: int = 0

    @classmethod
    def from_engine_config(cls, vllm_config, kv_cache_config) -> VLLMCachePolicy:
        groups = tuple(kv_cache_config.kv_cache_groups)
        if not groups:
            raise ValueError("NaviSched requires a vLLM model with KV cache groups")

        native_pd = is_vllm_native_pd_prefill()
        hybrid = any(_is_spec(group.kv_cache_spec, "MambaSpec") for group in groups)
        runtime_geometry = native_pd or (
            hybrid and getattr(vllm_config, "kv_transfer_config", None) is not None
        )
        if runtime_geometry:
            scheduler_block_size, block_size, group_block_sizes = (
                resolve_vllm_hybrid_geometry(vllm_config, kv_cache_config)
            )
        else:
            block_size = get_vllm_hash_block_size(vllm_config)
            scheduler_block_size = block_size
            group_block_sizes = tuple(
                int(group.kv_cache_spec.block_size)
                * int(vllm_config.parallel_config.decode_context_parallel_size)
                * int(vllm_config.parallel_config.prefill_context_parallel_size)
                for group in groups
            )
        if block_size <= 0 or scheduler_block_size % block_size != 0:
            raise ValueError("invalid vLLM scheduler/hash block geometry")
        roles = []
        scales = []
        spec_names = []
        for group, effective_block_size in zip(groups, group_block_sizes, strict=True):
            spec = group.kv_cache_spec
            spec_name = type(spec).__name__
            layer_specs = _vllm_layer_specs(spec) if native_pd else (spec,)
            if not layer_specs:
                raise ValueError("vLLM KV cache group has no layer specs")
            if effective_block_size % block_size != 0:
                raise ValueError(
                    "vLLM KV cache group block size is not aligned with the "
                    f"scheduler hash block size: {effective_block_size} vs {block_size}; "
                    f"group={spec_name} "
                    f"group_modes={[getattr(layer, 'mamba_cache_mode', None) for layer in layer_specs]} "
                    f"runtime_geometry={runtime_geometry} "
                    f"scheduler_block_size={scheduler_block_size} "
                    f"group_block_sizes={group_block_sizes} "
                    f"groups={[(type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size, [(type(s).__name__, s.block_size, getattr(s, 'mamba_cache_mode', None), getattr(s, 'prefix_cacheable', None)) for s in _vllm_layer_specs(g.kv_cache_spec)]) for g in groups]} "
                    f"dcp={vllm_config.parallel_config.decode_context_parallel_size} "
                    f"pcp={vllm_config.parallel_config.prefill_context_parallel_size} "
                    f"prefix_match_unit={getattr(vllm_config.cache_config, 'prefix_match_unit', None)}"
                )
            if all(_is_spec(layer, "FullAttentionSpec") for layer in layer_specs):
                role = "full"
            elif all(_is_spec(layer, "MambaSpec") for layer in layer_specs):
                modes = {
                    str(getattr(layer, "mamba_cache_mode", "none"))
                    for layer in layer_specs
                }
                if not modes.issubset({"all", "align"}):
                    raise ValueError(
                        "unsupported vLLM Mamba cache modes for NaviSched: "
                        f"{sorted(modes)}"
                    )
                role = "tail"
            else:
                raise ValueError(
                    "NaviSched cannot exactly represent vLLM cache semantics for "
                    f"{spec_name}; supported specs are FullAttentionSpec and MambaSpec"
                )
            roles.append(role)
            scales.append(effective_block_size // block_size)
            spec_names.append(spec_name)

        has_full = "full" in roles
        has_tail = "tail" in roles
        if has_tail and not has_full:
            raise ValueError(
                "NaviSched does not support a vLLM Mamba-only cache topology"
            )
        hybrid_tail_required = has_full and has_tail
        if hybrid_tail_required and runtime_geometry:
            linear_step = scheduler_block_size // block_size
        elif hybrid_tail_required:
            linear_step = math.lcm(
                *(scale for scale, role in zip(scales, roles) if role == "tail")
            )
        else:
            linear_step = 1
        total_blocks = int(kv_cache_config.num_blocks)
        if total_blocks <= 0:
            raise ValueError("vLLM KV cache has no schedulable blocks")
        return cls(
            block_size=block_size,
            total_blocks=total_blocks,
            linear_step=linear_step,
            hybrid_tail_required=hybrid_tail_required,
            group_roles=tuple(roles),
            group_block_scales=tuple(scales),
            spec_names=tuple(spec_names),
            full_attention_query_drop_tokens=(
                get_vllm_config_full_attention_query_drop_tokens(
                    vllm_config,
                    kv_cache_config,
                    roles,
                    scales,
                    block_size,
                    scheduler_block_size,
                )
                if native_pd
                else 0
            ),
        )

    def expand_group_hash(self, group_id: int, block_hash: bytes) -> tuple[bytes, ...]:
        if not 0 <= group_id < len(self.group_roles):
            raise ValueError(f"invalid vLLM KV cache group id: {group_id}")
        scale = self.group_block_scales[group_id]
        raw_hash = bytes(block_hash)
        if len(raw_hash) == _HASH_BYTES:
            return (raw_hash,)
        if len(raw_hash) != _HASH_BYTES * scale:
            raise ValueError(
                "unexpected vLLM block hash size for cache group "
                f"{group_id}: {len(raw_hash)} != {_HASH_BYTES * scale}"
            )
        hashes = tuple(
            raw_hash[offset : offset + _HASH_BYTES]
            for offset in range(0, len(raw_hash), _HASH_BYTES)
        )
        if self.group_roles[group_id] == "tail":
            return hashes[-1:]
        return hashes


@dataclass(frozen=True)
class VLLMCacheDelta:
    added: tuple[bytes, ...] = ()
    removed: tuple[bytes, ...] = ()
    added_full_ready: tuple[bytes, ...] = ()
    removed_full_ready: tuple[bytes, ...] = ()
    added_tail_ready: tuple[bytes, ...] = ()
    removed_tail_ready: tuple[bytes, ...] = ()

    def empty(self) -> bool:
        return not any(
            (
                self.added,
                self.removed,
                self.added_full_ready,
                self.removed_full_ready,
                self.added_tail_ready,
                self.removed_tail_ready,
            )
        )


class VLLMCacheState:
    def __init__(self, policy: VLLMCachePolicy):
        self._policy = policy
        self._group_counts = tuple(Counter() for _ in policy.group_roles)
        self._full_group_ids = tuple(
            index for index, role in enumerate(policy.group_roles) if role == "full"
        )
        self._tail_group_ids = tuple(
            index for index, role in enumerate(policy.group_roles) if role == "tail"
        )
        self._flat_ready: set[bytes] = set()
        self._full_ready: set[bytes] = set()
        self._tail_ready: set[bytes] = set()

    def replace(
        self, group_block_hashes: dict[int, tuple[bytes, ...]]
    ) -> VLLMCacheDelta:
        """Rebuild counts and group intersections once for a complete snapshot."""
        replacement = VLLMCacheState(self._policy)
        for group_id, block_hashes in group_block_hashes.items():
            expanded_hashes = tuple(
                block_hash
                for value in block_hashes
                for block_hash in self._policy.expand_group_hash(group_id, value)
            )
            replacement._group_counts[group_id].update(expanded_hashes)

        def ready_intersection(group_ids: tuple[int, ...]) -> set[bytes]:
            if not group_ids:
                return set()
            ready = set(replacement._group_counts[group_ids[0]])
            for group_id in group_ids[1:]:
                if not ready:
                    break
                ready.intersection_update(replacement._group_counts[group_id])
            return ready

        full_ready = ready_intersection(self._full_group_ids)
        if self._policy.hybrid_tail_required:
            replacement._full_ready = full_ready
        else:
            replacement._flat_ready = full_ready
        replacement._tail_ready = ready_intersection(self._tail_group_ids)
        delta = VLLMCacheDelta(
            added=tuple(sorted(replacement._flat_ready - self._flat_ready)),
            removed=tuple(sorted(self._flat_ready - replacement._flat_ready)),
            added_full_ready=tuple(sorted(replacement._full_ready - self._full_ready)),
            removed_full_ready=tuple(
                sorted(self._full_ready - replacement._full_ready)
            ),
            added_tail_ready=tuple(sorted(replacement._tail_ready - self._tail_ready)),
            removed_tail_ready=tuple(
                sorted(self._tail_ready - replacement._tail_ready)
            ),
        )
        self._group_counts = replacement._group_counts
        self._flat_ready = replacement._flat_ready
        self._full_ready = replacement._full_ready
        self._tail_ready = replacement._tail_ready
        return delta

    def apply(
        self,
        event_type: str,
        group_id: int | None,
        block_hashes: tuple[bytes, ...],
    ) -> VLLMCacheDelta:
        if event_type == "cleared":
            delta = VLLMCacheDelta(
                removed=tuple(sorted(self._flat_ready)),
                removed_full_ready=tuple(sorted(self._full_ready)),
                removed_tail_ready=tuple(sorted(self._tail_ready)),
            )
            for counts in self._group_counts:
                counts.clear()
            self._flat_ready.clear()
            self._full_ready.clear()
            self._tail_ready.clear()
            return delta
        if event_type not in ("stored", "removed") or group_id is None:
            raise ValueError(f"invalid vLLM cache lifecycle event: {event_type}")

        expanded_hashes = tuple(
            block_hash
            for value in block_hashes
            for block_hash in self._policy.expand_group_hash(group_id, value)
        )
        affected = set(expanded_hashes)
        before = {block_hash: self._membership(block_hash) for block_hash in affected}
        counts = self._group_counts[group_id]
        for block_hash in expanded_hashes:
            if event_type == "stored":
                counts[block_hash] += 1
            elif counts[block_hash] <= 1:
                counts.pop(block_hash, None)
            else:
                counts[block_hash] -= 1

        after = {block_hash: self._membership(block_hash) for block_hash in affected}
        for block_hash, (flat, full, tail) in after.items():
            self._set_membership(self._flat_ready, block_hash, flat)
            self._set_membership(self._full_ready, block_hash, full)
            self._set_membership(self._tail_ready, block_hash, tail)

        return VLLMCacheDelta(
            added=self._changed(before, after, 0, True),
            removed=self._changed(before, after, 0, False),
            added_full_ready=self._changed(before, after, 1, True),
            removed_full_ready=self._changed(before, after, 1, False),
            added_tail_ready=self._changed(before, after, 2, True),
            removed_tail_ready=self._changed(before, after, 2, False),
        )

    def _membership(self, block_hash: bytes) -> tuple[bool, bool, bool]:
        full = bool(self._full_group_ids) and all(
            self._group_counts[index][block_hash] > 0 for index in self._full_group_ids
        )
        tail = bool(self._tail_group_ids) and all(
            self._group_counts[index][block_hash] > 0 for index in self._tail_group_ids
        )
        flat = full and not self._policy.hybrid_tail_required
        return flat, full if self._policy.hybrid_tail_required else False, tail

    @staticmethod
    def _set_membership(values: set[bytes], value: bytes, present: bool) -> None:
        if present:
            values.add(value)
        else:
            values.discard(value)

    @staticmethod
    def _changed(
        before: dict[bytes, tuple[bool, bool, bool]],
        after: dict[bytes, tuple[bool, bool, bool]],
        field: int,
        added: bool,
    ) -> tuple[bytes, ...]:
        return tuple(
            sorted(
                block_hash
                for block_hash in before
                if before[block_hash][field] != after[block_hash][field]
                and after[block_hash][field] is added
            )
        )
