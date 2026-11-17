from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.request import Request


def emit_request_transition(
    self, event: str, request: Request, slot_delta: int = 0, **details: Any
) -> None:
    reporter = self.request_observer
    listener = reporter.listener
    if listener is None:
        return
    # Scheduler-thread transitions only; no retained requests or extra lock.
    reporter.sequence += 1
    try:
        listener(
            {
                "event": event,
                "sequence": reporter.sequence,
                "audit_errors": reporter.errors,
                "event_time_us": time.time_ns() // 1000,
                "monotonic_ns": time.monotonic_ns(),
                "request_id": request.request_id,
                "arrival_time": request.arrival_time,
                "status": request.status.name,
                "slot_delta": slot_delta,
                "running_count": len(self.running),
                "waiting_count": len(self.waiting),
                "max_num_seqs": self.max_num_running_reqs,
                "prompt_tokens": request.num_prompt_tokens,
                "computed_tokens_including_scheduled": request.num_computed_tokens,
                "output_tokens": request.num_output_tokens,
                "output_placeholders": request.num_output_placeholders,
                "cached_tokens": request.num_cached_tokens,
                "max_tokens": request.max_tokens,
                "priority": request.priority,
                "num_preemptions": request.num_preemptions,
                **details,
            }
        )
    except Exception:
        # A failed audit must not abort a request; the next event exposes the gap.
        reporter.errors += 1


class VLLMSchedulerObserver:
    def __init__(self, listener):
        self.listener = listener
        self.sequence = 0
        self.errors = 0
        self.before_output = {}
        self.iteration_tokens = None

    def on_transition(self, scheduler, event, request, slot_delta=0, **details):
        emit_request_transition(scheduler, event, request, slot_delta, **details)

    def start_iteration(self):
        self.before_output.clear()
        self.iteration_tokens = None

    def before_request_output(self, request):
        self.before_output[request.request_id] = (
            request.num_output_tokens,
            request.num_prompt_tokens,
        )

    def after_token_output(self, scheduler, request):
        before, _ = self.before_output[request.request_id]
        if before == 0 and request.num_output_tokens > 0:
            self.on_transition(scheduler, "first_output", request)

    def record_output(self, request, token_ids):
        before, prompt_tokens = self.before_output[request.request_id]
        count = len(token_ids) + (prompt_tokens if before == 0 else 0)
        self.iteration_tokens = (self.iteration_tokens or 0) + count

    def record_failed_output(self, request):
        count = request.num_prompt_tokens if request.num_output_tokens == 0 else 0
        self.iteration_tokens = (self.iteration_tokens or 0) + count
