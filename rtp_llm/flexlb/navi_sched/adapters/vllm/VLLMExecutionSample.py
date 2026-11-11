"""Immutable observations of a physical prefill step, independent of routing batches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillStepRequest:
    request_id: int
    context_tokens: int
    compute_tokens: int
    external_read_tokens: int
    routing_batch_id: int | None = None
    prompt_tokens: int | None = None
    prefix_tokens: int | None = None

    def __post_init__(self):
        if type(self.request_id) is not int or not 0 <= self.request_id < 1 << 64:
            raise ValueError("invalid prefill request identity")
        if (
            type(self.context_tokens) is not int
            or self.context_tokens < 0
            or type(self.compute_tokens) is not int
            or self.compute_tokens <= 0
            or type(self.external_read_tokens) is not int
            or not 0 <= self.external_read_tokens <= self.context_tokens
        ):
            raise ValueError("invalid observed prefill token counts")
        if self.routing_batch_id is not None and (
            type(self.routing_batch_id) is not int or self.routing_batch_id < 0
        ):
            raise ValueError("invalid routing batch identity")
        if self.prompt_tokens is not None or self.prefix_tokens is not None:
            if (
                type(self.prompt_tokens) is not int
                or type(self.prefix_tokens) is not int
                or not 0 <= self.prefix_tokens <= self.context_tokens
                or self.context_tokens + self.compute_tokens > self.prompt_tokens
                or self.external_read_tokens > self.prefix_tokens
            ):
                raise ValueError("step does not belong to the observed prompt prefix")


@dataclass(frozen=True)
class PrefillExecutionSample:
    step_id: int
    requests: tuple[PrefillStepRequest, ...]
    started_ns: int
    completed_ns: int
    global_padded_tokens: int
    execution_step_count: int = 1

    def __post_init__(self):
        if type(self.step_id) is not int or self.step_id <= 0:
            raise ValueError("invalid physical execution identity")
        if (
            not isinstance(self.requests, tuple)
            or not self.requests
            or any(not isinstance(r, PrefillStepRequest) for r in self.requests)
            or len({r.request_id for r in self.requests}) != len(self.requests)
        ):
            raise ValueError("physical execution requires unique request observations")
        if (
            type(self.started_ns) is not int
            or type(self.completed_ns) is not int
            or not 0 < self.started_ns < self.completed_ns
        ):
            raise ValueError("invalid observed execution clock")
        if (
            type(self.global_padded_tokens) is not int
            or self.global_padded_tokens < self.compute_tokens
            or self.global_padded_tokens >= 1 << 63
            or type(self.execution_step_count) is not int
            or self.execution_step_count <= 0
        ):
            raise ValueError("runner geometry does not cover the scheduled prefill")

    @property
    def compute_tokens(self):
        return sum(r.compute_tokens for r in self.requests)

    @property
    def context_tokens(self):
        return sum(r.context_tokens for r in self.requests)

    @property
    def compute_token_square_sum(self):
        return sum(r.compute_tokens * r.compute_tokens for r in self.requests)

    @property
    def context_compute_token_product_sum(self):
        return sum(r.context_tokens * r.compute_tokens for r in self.requests)

    @property
    def elapsed_ms(self):
        return (self.completed_ns - self.started_ns) / 1_000_000

    @property
    def shared(self):
        return len({r.routing_batch_id for r in self.requests}) > 1

    def wire_requests(self):
        return tuple(
            (
                r.request_id,
                r.context_tokens + r.compute_tokens,
                r.context_tokens,
                r.external_read_tokens,
            )
            for r in self.requests
        )
