"""Opt-in, bounded real-request warmup before advertising ROCm readiness."""

from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence, Tuple


@dataclass(frozen=True)
class RocmWarmupCase:
    name: str
    token_ids: Tuple[int, ...]
    reuse_cache: bool
    expected_reuse: int = 0
    max_new_tokens: int = 2


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
    # q=8,16 covers the CK paged-prefill path. Every case also executes decode.
    for index, q_len in enumerate((1, 2, 3, 4, 8, 16)):
        cases.append(
            RocmWarmupCase(
                f"cached_q{q_len}", prefix + (101 + index,) * q_len, True, prefix_len
            )
        )
    return cases


async def run_rocm_warmup_cases(
    cases: Sequence[RocmWarmupCase],
    run_request: Callable[[RocmWarmupCase], Awaitable[Tuple[int, int]]],
) -> None:
    """Fail closed on RPC failures, early EOS, or a missing cache-path warmup.

    The caller owns the startup readiness gate and must only release it when
    this function returns successfully. No service traffic or business prompts
    are used. Returned fields are output_len and reuse_len from final aux_info.
    """
    for case in cases:
        output_len, reuse_len = await run_request(case)
        if output_len < case.max_new_tokens:
            raise RuntimeError(f"ROCm warmup {case.name} did not exercise decode")
        if reuse_len != case.expected_reuse:
            raise RuntimeError(
                f"ROCm warmup {case.name} cache path mismatch: "
                f"expected reuse={case.expected_reuse}, actual={reuse_len}"
            )
