"""Bounded real-request warmup before advertising ROCm readiness."""

import asyncio
import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Sequence, Tuple


def rocm_startup_warmup_enabled(
    override: str,
    *,
    is_rocm: bool,
    model_type: str,
    is_pdfusion: bool,
    world_size: int,
    reuse_cache: bool,
    speculative: bool,
) -> bool:
    """Auto-enable only the supported Qwen3 serving topology.

    An explicit enable is validated, never silently ignored. Other engines and
    topologies keep their old startup behavior; zero is an emergency opt-out.
    The existing WARM_UP / MODEL_WARM_UP master switches are checked by caller.
    """
    value = (override or "auto").strip().lower()
    if value in ("0", "false", "off", "no"):
        return False
    if value not in ("auto", "1", "true", "on", "yes"):
        raise ValueError("RTP_LLM_ROCM_STARTUP_WARMUP must be auto, 1 or 0")
    reason = None
    if not is_rocm:
        reason = "a ROCm build"
    elif not is_pdfusion:
        reason = "PDFUSION"
    elif world_size != 1:
        reason = "single-rank deployments"
    elif not reuse_cache:
        reason = "REUSE_CACHE=1"
    elif speculative:
        reason = "non-speculative decoding"
    if value == "auto":
        return model_type == "qwen_3" and reason is None
    if reason:
        raise ValueError(f"RTP_LLM_ROCM_STARTUP_WARMUP requires {reason}")
    return True


@dataclass(frozen=True)
class RocmWarmupCase:
    name: str
    token_ids: Tuple[int, ...]
    reuse_cache: bool
    expected_reuse: int = 0
    max_new_tokens: int = 2


def resolve_rocm_warmup_max_seq_len(
    configured_max_seq_len: Optional[int], model_max_seq_len: Optional[int]
) -> int:
    """Use the backend's resolved limit, not an optional CLI argument.

    Direct callers with an explicit limit remain supported. Missing metadata
    must fail closed rather than guess a limit larger than the loaded model.
    """
    limit = (
        model_max_seq_len if model_max_seq_len is not None else configured_max_seq_len
    )
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(
            "ROCm startup warmup needs a positive model max_seq_len from the "
            "backend startup status (or an explicit MAX_SEQ_LEN); "
            f"model={model_max_seq_len!r}, configured={configured_max_seq_len!r}"
        )
    return limit


def rocm_warmup_cases(
    max_seq_len: int, prefix_len: int = 256, *, tokens_per_block: int = 16
):
    # Warm real KV geometry, not just q_len: a 256-token prefix does not warm
    # the workspace strides/partition counts used by an 832-token prefix.
    if tokens_per_block <= 0 or prefix_len <= 0:
        raise ValueError(
            "ROCm startup warmup needs positive cache block/prefix lengths"
        )
    prefix_len = math.ceil(prefix_len / tokens_per_block) * tokens_per_block
    if max_seq_len < prefix_len + 16 + 2:
        raise ValueError("ROCm startup warmup needs prefix_len + 18 context tokens")
    cases = [
        RocmWarmupCase("uncached", (100,) * min(1024, max_seq_len - 2), False),
    ]

    # Bound startup work independently of a checkpoint's potentially huge
    # context limit. Cover every 256-token PA partition through 8K for q=1..4,
    # plus the measured cache geometries. Use physical cache-block alignment:
    # a non-aligned seed cannot guarantee expected_reuse on the next request.
    max_prefix = min(8192 - 18, max_seq_len - 18)
    partition_stride = math.lcm(256, tokens_per_block)
    prefixes = {prefix_len}
    prefixes.update(range(partition_stride, max_prefix + 1, partition_stride))
    if tokens_per_block <= max_prefix:
        prefixes.add(tokens_per_block)
    hot_prefixes = {prefix_len}
    for observed in (816, 832):
        aligned = observed // tokens_per_block * tokens_per_block
        if 0 < aligned <= max_prefix:
            prefixes.add(aligned)
            hot_prefixes.add(aligned)

    # Different groups must not reuse an earlier, longer warmup seed. Within a
    # group all suffixes are distinct, so q=16/32 never increases another case's
    # prefix hit. Only synthetic vocabulary IDs are used, never business text.
    for group, length in enumerate(sorted(prefixes)):
        prefix = (100 + group,) * length
        cases.append(RocmWarmupCase(f"seed_p{length}", prefix, True))
        query_lengths = range(1, 33) if length in hot_prefixes else range(1, 5)
        for q_len in query_lengths:
            if length + q_len + 2 > max_seq_len:
                continue
            cases.append(
                RocmWarmupCase(
                    f"cached_p{length}_q{q_len}",
                    prefix + (200 + q_len,) * q_len,
                    True,
                    length,
                )
            )
    return cases


async def run_rocm_warmup_cases(
    cases: Sequence[RocmWarmupCase],
    run_request: Callable[[RocmWarmupCase], Awaitable[Tuple[int, int]]],
    *,
    timeout_s: Optional[float] = None,
) -> None:
    """Fail closed on RPC failures, early EOS, or a missing cache-path warmup.

    The caller owns the startup readiness gate and must only release it when
    this function returns successfully. No service traffic or business prompts
    are used. Returned fields are output_len and reuse_len from final aux_info.
    """
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("ROCm warmup timeout must be finite and positive")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s if timeout_s is not None else None
    for case in cases:
        # Also bound the client coroutine, not only the server/RPC deadline.
        # A stuck stream must leave readiness closed rather than hang startup.
        if timeout_s is None:
            output_len, reuse_len = await run_request(case)
        else:
            # A larger shape plan must not multiply the startup deadline by
            # its case count. Keep the whole readiness gate time-bounded.
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"ROCm startup warmup deadline before {case.name}")
            output_len, reuse_len = await asyncio.wait_for(run_request(case), remaining)
        if output_len < case.max_new_tokens:
            raise RuntimeError(f"ROCm warmup {case.name} did not exercise decode")
        if reuse_len != case.expected_reuse:
            raise RuntimeError(
                f"ROCm warmup {case.name} cache path mismatch: "
                f"expected reuse={case.expected_reuse}, actual={reuse_len}"
            )
