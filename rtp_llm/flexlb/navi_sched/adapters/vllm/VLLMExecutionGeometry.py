"""Project one streaming-PCP TPU runner chunk from the actual runtime config.

This does not choose scheduler steps or infer a complete batch execution plan.
The rank ownership and padding rules follow the TPU runner's PCP sequence layout.
"""

from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TPUStepGeometry:
    scheduled_tokens: int
    pcp_size: int
    interleave_size: int
    max_local_tokens: int
    local_padded_tokens: int
    global_padded_tokens: int
    dp_target_bucket: int | None


def capture_execution_policy(scheduler, vllm_config) -> dict:
    """Record resolved scheduler values, keeping unavailable fields explicit.

    Physical cache-config blocks and logical scheduler blocks are distinct.
    TPU's Mamba wrapper temporarily selects the latter only during splitting.
    """
    cache = getattr(scheduler, "cache_config", None)
    schedule = getattr(scheduler, "scheduler_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    compilation = getattr(vllm_config, "compilation_config", None)
    return {
        "cache_config_block_size": getattr(cache, "block_size", None),
        "scheduler_block_size": getattr(scheduler, "block_size", None),
        "mamba_split_scheduler_block_size_patch": getattr(
            getattr(scheduler, "_mamba_block_aligned_split", None),
            "_tpu_scheduler_block_size_patch",
            None,
        ),
        "hash_block_size": getattr(scheduler, "hash_block_size", None),
        "max_num_scheduled_tokens": getattr(
            scheduler, "max_num_scheduled_tokens", None
        ),
        "long_prefill_token_threshold": getattr(
            schedule, "long_prefill_token_threshold", None
        ),
        "use_eagle": getattr(scheduler, "use_eagle", None),
        "use_eagle_block_drop": getattr(
            scheduler, "use_eagle_block_drop", getattr(scheduler, "use_eagle", None)
        ),
        "mamba_has_prefill_checkpoint_blocks": getattr(
            scheduler, "mamba_has_prefill_checkpoint_blocks", None
        ),
        "mamba_partial_cache_hit": getattr(scheduler, "mamba_partial_cache_hit", None),
        "mamba_cache_mode": getattr(cache, "mamba_cache_mode", None),
        "pcp_size": getattr(parallel, "prefill_context_parallel_size", None),
        "interleave_size": getattr(parallel, "cp_kv_cache_interleave_size", None),
        "compile_sizes": getattr(compilation, "compile_sizes", None),
    }


def resolved_prefill_execution_policy(scheduler, vllm_config, cache_policy) -> dict:
    """Read the constructed Scheduler's effective policy before Agent startup."""
    parallel = vllm_config.parallel_config
    compilation = vllm_config.compilation_config
    policy = {
        "pcp_size": parallel.prefill_context_parallel_size,
        "interleave_size": parallel.cp_kv_cache_interleave_size,
        "compile_sizes": compilation.compile_sizes,
        "max_scheduled_tokens": scheduler.max_num_scheduled_tokens,
        "long_prefill_token_threshold": scheduler.scheduler_config.long_prefill_token_threshold,
        "hash_block_size": scheduler.hash_block_size,
        "scheduler_block_size": scheduler.block_size,
        # The existing native field gates Mamba's one-block back-off only;
        # it does not control the engine's speculative lookahead/encoder shift.
        "use_eagle": getattr(scheduler, "use_eagle_block_drop", scheduler.use_eagle),
        "mamba_has_prefill_checkpoint_blocks": scheduler.mamba_has_prefill_checkpoint_blocks,
        "mamba_partial_cache_hit": scheduler.mamba_partial_cache_hit,
    }
    for name in (
        "pcp_size",
        "interleave_size",
        "max_scheduled_tokens",
        "hash_block_size",
        "scheduler_block_size",
    ):
        if type(policy[name]) is not int or policy[name] <= 0:
            raise ValueError(f"invalid resolved TPU prefill execution policy: {name}")
    threshold = policy["long_prefill_token_threshold"]
    if type(threshold) is not int or threshold < 0:
        raise ValueError("invalid resolved TPU prefill long threshold")
    buckets = policy["compile_sizes"]
    if (
        not isinstance(buckets, (list, tuple))
        or not buckets
        or any(type(v) is not int or v <= 0 for v in buckets)
        or any(a >= b for a, b in zip(buckets, buckets[1:]))
    ):
        raise ValueError("invalid resolved TPU compile sizes")
    policy["compile_sizes"] = list(buckets)
    for name in (
        "use_eagle",
        "mamba_has_prefill_checkpoint_blocks",
        "mamba_partial_cache_hit",
    ):
        if type(policy[name]) is not bool:
            raise ValueError(f"invalid resolved TPU prefill execution policy: {name}")
    if (
        policy["hash_block_size"] != cache_policy.block_size
        or policy["scheduler_block_size"] % policy["hash_block_size"] != 0
        or (
            cache_policy.hybrid_tail_required
            and policy["scheduler_block_size"]
            != cache_policy.block_size * cache_policy.linear_step
        )
    ):
        raise ValueError("actual Scheduler and NaviSched cache geometry disagree")
    if (
        getattr(scheduler, "need_mamba_block_aligned_split", False)
        and scheduler.cache_config.block_size != scheduler.block_size
        and not getattr(
            scheduler._mamba_block_aligned_split,
            "_tpu_scheduler_block_size_patch",
            False,
        )
    ):
        raise ValueError(
            "TPU Mamba splitter is not aligned to resolved scheduler blocks"
        )
    fixed = bool(getattr(scheduler, "_navi_fixed_prefill_plan", False))
    schedule = (
        getattr(scheduler, "_navi_fixed_prefill_schedule", None) if fixed else None
    )
    if schedule is not None and (
        not isinstance(schedule, (list, tuple))
        or len(schedule) != 3
        or any(type(v) is not int or v <= 0 for v in schedule)
    ):
        raise ValueError("invalid fixed-prefill attention schedule")
    policy.update(
        fixed_prefill_chunk_plan=fixed,
        prefill_lookahead=(
            getattr(scheduler, "num_prefill_lookahead", 0) if fixed else 0
        ),
        max_input_tokens=(
            scheduler.scheduler_config.max_num_batched_tokens if fixed else 0
        ),
        draft_slots_per_request=(
            getattr(
                getattr(vllm_config, "speculative_config", None),
                "max_num_new_slots_for_drafting",
                0,
            )
            if fixed
            else 0
        ),
        mamba_block_aligned_split=bool(
            getattr(
                scheduler,
                "_mamba_split",
                getattr(scheduler, "need_mamba_block_aligned_split", True),
            )
        ),
        attention_schedule_capacity=schedule[0] if schedule is not None else 0,
        attention_query_tile=schedule[1] if schedule is not None else 0,
        attention_kv_tile=schedule[2] if schedule is not None else 0,
    )
    if type(policy["prefill_lookahead"]) is not int or policy["prefill_lookahead"] < 0:
        raise ValueError("invalid fixed-prefill MTP lookahead")
    for name in ("max_input_tokens", "draft_slots_per_request"):
        if type(policy[name]) is not int or policy[name] < 0:
            raise ValueError(f"invalid fixed-prefill input budget: {name}")
    policy.update(resolved_async_kv_lookup_policy(scheduler))
    return policy


def resolved_async_kv_lookup_policy(scheduler) -> dict:
    """Read the actual connector's lookup implementation and hash/Eagle config."""
    values = {
        "async_kv_lookup_enabled": False,
        "async_kv_lookup_hash_block_size": 0,
        "async_kv_lookup_uses_eagle": False,
    }
    module = sys.modules.get("kvs_connector.v6d_object_connector")
    if module is None:
        return values
    connectors = [getattr(scheduler, "connector", None)]
    while connectors:
        connector = connectors.pop()
        if isinstance(connector, module.V6dObjectConnector):
            source = connector.scheduler
            if getattr(source, "_async_runtime", None) is None:
                return values
            block = source.hash_block_size
            if type(block) is not int or block <= 0:
                raise ValueError("invalid actual async KV lookup hash block size")
            speculative = source.vllm_config.speculative_config
            values.update(
                async_kv_lookup_enabled=True,
                async_kv_lookup_hash_block_size=block,
                async_kv_lookup_uses_eagle=(
                    speculative is not None and bool(speculative.use_eagle())
                ),
            )
            return values
        connectors.extend(getattr(connector, "_connectors", ()))
    return values


def project_tpu_streaming_step(
    vllm_config,
    scheduled_tokens: int,
    *,
    dp_target_bucket: int | None = None,
) -> TPUStepGeometry:
    """Use the configured buckets and request-major PCP ownership for one chunk."""
    parallel = vllm_config.parallel_config
    pcp_size = int(parallel.prefill_context_parallel_size)
    interleave_size = int(parallel.cp_kv_cache_interleave_size)
    fixed_pp = hasattr(vllm_config, "_navi_fixed_prefill_schedule")
    if (
        scheduled_tokens <= 0
        or pcp_size <= 0
        or interleave_size <= 0
        or (pcp_size == 1 and not fixed_pp)
    ):
        raise ValueError(
            "TPU projection requires positive work and PCP > 1 or fixed PP"
        )
    configured = vllm_config.compilation_config.compile_sizes
    if not configured:
        raise ValueError("TPU platform compile sizes are unavailable")
    paddings = sorted(int(size) for size in configured)
    cycles, remaining = divmod(scheduled_tokens, pcp_size * interleave_size)
    required = cycles * interleave_size + min(remaining, interleave_size)
    index = bisect.bisect_left(paddings, required)
    if index == len(paddings):
        raise ValueError("PCP rank token count exceeds configured padding buckets")
    padded = paddings[index]
    if dp_target_bucket is not None and dp_target_bucket > padded:
        padded = int(dp_target_bucket)
    # The runner allocates its token buffer from the final configured bucket.
    if padded > int(configured[-1]):
        raise ValueError("PCP local padding exceeds the runner token buffer")
    return TPUStepGeometry(
        scheduled_tokens,
        pcp_size,
        interleave_size,
        required,
        padded,
        padded * pcp_size,
        dp_target_bucket,
    )


def fixed_prefill_native_policy(scheduler):
    """Use the same resolved policy and native arithmetic as Master/cost."""
    from types import SimpleNamespace

    from navi_sched.util import naviSchedModule

    block, hashed = scheduler.block_size, scheduler.hash_block_size
    if hashed <= 0 or block % hashed:
        raise ValueError("invalid fixed-prefill cache block geometry")
    cache = SimpleNamespace(
        block_size=hashed,
        linear_step=block // hashed,
        hybrid_tail_required=bool(scheduler._mamba_split),
    )
    values = resolved_prefill_execution_policy(scheduler, scheduler.vllm_config, cache)
    if not values["fixed_prefill_chunk_plan"]:
        raise ValueError("fixed-prefill planner requested without its policy marker")
    module = naviSchedModule()
    native = module.TPUPrefillExecutionPolicyConfig()
    for key, value in values.items():
        setattr(native, key, value)
    return module._VLLMAgentBridge, native
