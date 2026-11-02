from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

_HASH_BYTES = 32


def get_vllm_hash_block_size(vllm_config) -> int:
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    block_size = (
        int(cache_config.block_size)
        * int(parallel_config.decode_context_parallel_size)
        * int(parallel_config.prefill_context_parallel_size)
    )
    if block_size <= 0:
        raise ValueError("vLLM scheduler block size must be positive")
    return block_size


def _is_spec(spec, class_name: str) -> bool:
    return any(base.__name__ == class_name for base in type(spec).__mro__)


@dataclass(frozen=True)
class VLLMCachePolicy:
    block_size: int
    total_blocks: int
    linear_step: int
    hybrid_tail_required: bool
    group_roles: tuple[str, ...]
    group_block_scales: tuple[int, ...]
    spec_names: tuple[str, ...]

    @classmethod
    def from_engine_config(cls, vllm_config, kv_cache_config) -> VLLMCachePolicy:
        groups = tuple(kv_cache_config.kv_cache_groups)
        if not groups:
            raise ValueError("NaviSched requires a vLLM model with KV cache groups")

        block_size = get_vllm_hash_block_size(vllm_config)
        parallel_config = vllm_config.parallel_config
        context_parallel_size = int(parallel_config.decode_context_parallel_size) * int(
            parallel_config.prefill_context_parallel_size
        )
        roles = []
        scales = []
        spec_names = []
        for group in groups:
            spec = group.kv_cache_spec
            spec_name = type(spec).__name__
            effective_block_size = int(spec.block_size) * context_parallel_size
            if effective_block_size % block_size != 0:
                raise ValueError(
                    "vLLM KV cache group block size is not aligned with the "
                    f"scheduler hash block size: {effective_block_size} vs {block_size}"
                )
            if _is_spec(spec, "FullAttentionSpec"):
                role = "full"
            elif _is_spec(spec, "MambaSpec"):
                mode = str(getattr(spec, "mamba_cache_mode", "none"))
                if mode not in ("all", "align"):
                    raise ValueError(
                        f"unsupported vLLM Mamba cache mode for NaviSched: {mode}"
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
        linear_step = (
            math.lcm(*(scale for scale, role in zip(scales, roles) if role == "tail"))
            if hybrid_tail_required
            else 1
        )
        total_blocks = int(kv_cache_config.num_blocks) // len(groups)
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
        )

    def expand_group_hash(self, group_id: int, block_hash: bytes) -> tuple[bytes, ...]:
        if not 0 <= group_id < len(self.group_roles):
            raise ValueError(f"invalid vLLM KV cache group id: {group_id}")
        scale = self.group_block_scales[group_id]
        raw_hash = bytes(block_hash)
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
