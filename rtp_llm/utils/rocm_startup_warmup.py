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


def rocm_warmup_cases(max_seq_len: int, prefix_len: int = 256):
    # 256 aligns with the Qwen ROCm block sizes. Reject unsupported small
    # contexts instead of silently skipping the cached-CK or decode path.
    if prefix_len <= 0 or max_seq_len < prefix_len + 16 + 2:
        raise ValueError("ROCm startup warmup needs prefix_len + 18 context tokens")
    prefix = (100,) * prefix_len
    cases = [
        RocmWarmupCase("uncached", (100,) * min(1024, max_seq_len - 2), False),
        RocmWarmupCase("seed_device_cache", prefix, True),
    ]
    # Distinct suffixes prevent earlier warmup requests from increasing the
    # shared prefix. q=1..4 covers Triton's specialized short-query shapes;
    # Cover the observed 14/15/16-token serving shapes as well as both backend
    # boundaries. Every case also executes decode before readiness can open.
    for index, q_len in enumerate((1, 2, 3, 4, 8, 14, 15, 16)):
        cases.append(
            RocmWarmupCase(
                f"cached_q{q_len}", prefix + (101 + index,) * q_len, True, prefix_len
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
    for case in cases:
        # Also bound the client coroutine, not only the server/RPC deadline.
        # A stuck stream must leave readiness closed rather than hang startup.
        if timeout_s is None:
            output_len, reuse_len = await run_request(case)
        else:
            output_len, reuse_len = await asyncio.wait_for(run_request(case), timeout_s)
        if output_len < case.max_new_tokens:
            raise RuntimeError(f"ROCm warmup {case.name} did not exercise decode")
        if reuse_len != case.expected_reuse:
            raise RuntimeError(
                f"ROCm warmup {case.name} cache path mismatch: "
                f"expected reuse={case.expected_reuse}, actual={reuse_len}"
            )
