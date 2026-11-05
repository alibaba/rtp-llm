from dataclasses import dataclass
from typing import List, Optional


class NaviSchedScheduleError(RuntimeError):
    def __init__(self, error_code: int, message: str) -> None:
        self.error_code = int(error_code)
        super().__init__(message or "navi sched request failed")


@dataclass
class NaviSchedRequest:
    prompt_token_count: int
    block_hashes: List[int]
    request_id: int
    block_size: int = 256
    forward_payload: bytes = b""
    explicit_cache_checkpoint_position: int = 0


@dataclass
class NaviSchedTarget:
    biz_name: str
    address: str


@dataclass
class NaviSchedRequestTiming:
    batch_wait_ms: float = 0.0
    queue_wait_ms: float = 0.0
    execution_latency_ms: float = 0.0
    client_total_latency_ms: float = 0.0


@dataclass
class NaviSchedDecision:
    request_id: int
    schedule_id: int = 0
    schedule_loss: float = 0.0
    schedule_latency_us: int = 0
    schedule_time_us: int = 0
    batch_id: int = 0
    expected_batch_size: int = 0
    predicted_cache_hit_tokens: int = 0
    predicted_queue_latency_ms: float = 0.0
    predicted_execution_latency_ms: float = 0.0
    reason: str = ""
    target: Optional[NaviSchedTarget] = None
    request_timing: Optional[NaviSchedRequestTiming] = None
    replay: Optional[object] = None


def decisionFromNative(result) -> NaviSchedDecision:
    if result.error_code != 0:
        raise NaviSchedScheduleError(result.error_code, result.error_message)
    decision = result.decision
    target = None
    if decision.target.address:
        target = NaviSchedTarget(
            biz_name=decision.target.biz_name,
            address=decision.target.address,
        )
    request_timing = None
    if result.request_timing is not None:
        timing = result.request_timing
        request_timing = NaviSchedRequestTiming(
            batch_wait_ms=timing.batch_wait_ms,
            queue_wait_ms=timing.queue_wait_ms,
            execution_latency_ms=timing.execution_latency_ms,
            client_total_latency_ms=timing.client_total_latency_ms,
        )
    return NaviSchedDecision(
        request_id=decision.request_id,
        schedule_id=decision.schedule_id,
        schedule_loss=decision.schedule_loss,
        schedule_latency_us=decision.schedule_latency_us,
        schedule_time_us=decision.schedule_time_us,
        batch_id=decision.batch_id,
        expected_batch_size=decision.expected_batch_size,
        predicted_cache_hit_tokens=decision.predicted_cache_hit_tokens,
        predicted_queue_latency_ms=decision.predicted_queue_latency_ms,
        predicted_execution_latency_ms=decision.predicted_execution_latency_ms,
        reason=decision.reason,
        target=target,
        request_timing=request_timing,
        replay=result.replay,
    )


def makeRequest(
    prompt_token_count: int,
    block_hashes: List[int],
    request_id: Optional[int] = None,
    block_size: int = 256,
    forward_payload: bytes = b"",
    explicit_cache_checkpoint_position: int = 0,
) -> NaviSchedRequest:
    if request_id is None or not 0 < int(request_id) < (1 << 64):
        raise ValueError("request_id must be a nonzero uint64")
    if int(prompt_token_count) < 0:
        raise ValueError("prompt_token_count must be nonnegative")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    if len(block_hashes) > int(prompt_token_count) // int(block_size):
        raise ValueError("block_hashes exceed complete prompt blocks")
    if type(
        explicit_cache_checkpoint_position
    ) is not int or not 0 <= explicit_cache_checkpoint_position <= int(
        prompt_token_count
    ):
        raise ValueError("explicit cache checkpoint position must be within the prompt")
    return NaviSchedRequest(
        prompt_token_count=int(prompt_token_count),
        block_hashes=block_hashes,
        request_id=int(request_id),
        block_size=int(block_size),
        forward_payload=bytes(forward_payload),
        explicit_cache_checkpoint_position=explicit_cache_checkpoint_position,
    )
