import logging
from math import gcd
from typing import Sequence

from rtp_llm.ops import HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType


def calculate_hybrid_group_layer_num(linear_count: int, full_count: int) -> int:
    if linear_count > 0 and full_count > 0:
        group_layer_num = gcd(linear_count, full_count)
        if group_layer_num < full_count:
            group_layer_num = full_count
    else:
        group_layer_num = max(linear_count, full_count)
    return max(group_layer_num, 1)


def build_hybrid_kv_cache_spec_descs(
    hybrid_attention_types: Sequence[HybridAttentionType],
    full_cache_type: KVCacheSpecType,
    linear_cache_type: KVCacheSpecType = KVCacheSpecType.LINEAR,
) -> list[list[KVCacheSpecDesc]]:
    linear_count = sum(
        attn_type == HybridAttentionType.LINEAR for attn_type in hybrid_attention_types
    )
    full_count = len(hybrid_attention_types) - linear_count
    group_layer_num = calculate_hybrid_group_layer_num(linear_count, full_count)

    full_desc = KVCacheSpecDesc()
    full_desc.tag = "full"
    full_desc.cache_type = full_cache_type

    linear_descs: dict[int, KVCacheSpecDesc] = {}

    def get_linear_desc(group_idx: int) -> KVCacheSpecDesc:
        if full_count == 0:
            group_idx = 0
        desc = linear_descs.get(group_idx)
        if desc is None:
            desc = KVCacheSpecDesc()
            desc.tag = "linear" if full_count == 0 else f"linear{group_idx}"
            desc.cache_type = linear_cache_type
            linear_descs[group_idx] = desc
        return desc

    layer_descs = []
    linear_seen = 0
    for attn_type in hybrid_attention_types:
        if attn_type == HybridAttentionType.LINEAR:
            linear_group_idx = linear_seen // group_layer_num
            layer_descs.append([get_linear_desc(linear_group_idx)])
            linear_seen += 1
        else:
            layer_descs.append([full_desc])
    return layer_descs


# LinearKVCacheGroup.retainedTailBlockCount() retains at least two tails.
_MIN_RETAINED_TAIL_BLOCKS = 2
# A new tail is allocated before decode cleanup releases the old one.
_TAIL_REPLACEMENT_BLOCKS = 1
# BlockPool::totalBlocksNum() returns block_num - 1: block 0 is reserved.
_RESERVED_NULL_BLOCKS = 1


def configure_recurrent_cache_capacity(model_config, max_active_requests: int) -> None:
    """Apply an opt-in active-request budget, preserving operator overrides.

    None keeps token-based capacity. An integer explicitly budgets reusable
    prefix snapshots per active request, in addition to tails and SP reserve.
    """
    from rtp_llm.ops import CacheCapacityPolicyDesc

    if max_active_requests <= 0:
        raise ValueError("max_active_requests must be positive")
    prefix_blocks = model_config.recurrent_cache_prefix_blocks_per_request
    if prefix_blocks is not None and (
        type(prefix_blocks) is not int or prefix_blocks < 0
    ):
        raise ValueError("recurrent prefix allowance must be a non-negative integer")
    # NormalEngine uses reserve_step=proposals+1. LinearKVCacheGroup reserves
    # reserve_step-1 extra state slots, hence exactly `proposals` here.
    proposals = max(0, int(model_config.gen_num_per_cycle))
    generated = getattr(model_config, "_generated_recurrent_capacities", {})
    descs = model_config.kv_cache_spec_descs
    groups = {}
    for layer_descs in descs:
        for desc in layer_descs:
            if desc.cache_type == KVCacheSpecType.LINEAR:
                groups.setdefault(desc.tag, []).append(desc)

    def signature(capacity):
        if capacity is None:
            return None
        return (
            capacity.explicit_block_num,
            capacity.reservable,
            capacity.charge_to_paged_budget,
        )

    def policy(values):
        if values is None:
            return None
        result = CacheCapacityPolicyDesc()
        result.explicit_block_num, result.reservable, result.charge_to_paged_budget = (
            values
        )
        return result

    changed = False
    for tag, members in groups.items():
        current = signature(members[0].capacity)
        if any(signature(member.capacity) != current for member in members[1:]):
            raise ValueError(f"Inconsistent recurrent capacities for shared tag {tag}")
        previous = generated.get(tag)
        ours = previous is not None and current == previous["generated"]
        if not ours:
            generated.pop(tag, None)
        if prefix_blocks is None:
            if not ours:
                continue
            desired = previous["original"]
            generated.pop(tag, None)
        else:
            # Presence, including an explicit zero, identifies an operator
            # setting. Our own previous value is tracked separately and updated.
            if not ours and current is not None and current[0] is not None:
                continue
            tail = members[0].tail
            active_tails = tail.active_tail_blocks if tail is not None else None
            retained = max(_MIN_RETAINED_TAIL_BLOCKS, active_tails or 1)
            blocks = (
                max_active_requests
                * (retained + proposals + _TAIL_REPLACEMENT_BLOCKS + prefix_blocks)
                + _RESERVED_NULL_BLOCKS
            )
            desired = (blocks, False, True)
            generated[tag] = {
                "original": previous["original"] if ours else current,
                "generated": desired,
            }
            logging.info(
                "Recurrent tag %s: %d blocks (%d requests, %d tails, %d SP, "
                "%d replacement, %d prefixes, %d reserved null)",
                tag,
                blocks,
                max_active_requests,
                retained,
                proposals,
                _TAIL_REPLACEMENT_BLOCKS,
                prefix_blocks,
                _RESERVED_NULL_BLOCKS,
            )
        for desc in members:
            desc.capacity = policy(desired)
        changed = True
    model_config._generated_recurrent_capacities = generated
    if changed:
        # Optional/vector pybind properties return copies.
        model_config.kv_cache_spec_descs = descs
