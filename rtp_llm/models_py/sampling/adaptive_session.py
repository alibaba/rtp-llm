import logging
import os

import numpy as np
import torch
from rtp_llm.models_py.sampling.adaptive_controller import AdaptiveDraftController


class AdaptiveVerificationSession:
    def __init__(self, capacity: int, steps: int, device: str) -> None:
        self.controller = AdaptiveDraftController(capacity, steps, torch.device(device))
        self.steps = steps
        self.fixed_cap = int(os.environ.get("RTP_LLM_ADAPTIVE_VERIFY_CAP", "-1"))
        if not -1 <= self.fixed_cap <= steps:
            raise ValueError(
                "RTP_LLM_ADAPTIVE_VERIFY_CAP must be -1 or between 0 and draft steps"
            )
        self.ids: list[int] = []
        self.registered: set[int] = set()
        self.target_costs: dict[int, dict[int, float]] = {}
        self.draft_costs: dict[int, float] = {}
        self.events: list[torch.cuda.Event] | None = None
        self.pending_costs: list[tuple[int, int, list[torch.cuda.Event]]] = []
        self.event_pool: list[list[torch.cuda.Event]] = []
        self.total = 0
        self.round = 0

    def _collect_costs(self) -> None:
        pending = []
        for batch, total, events in self.pending_costs:
            if not events[3].query():
                pending.append((batch, total, events))
                continue
            self.draft_costs[batch] = events[0].elapsed_time(events[1])
            measured = max(events[2].elapsed_time(events[3]), 1e-6)
            costs = self.target_costs.setdefault(batch, {})
            costs[total] = (
                0.8 * costs[total] + 0.2 * measured if total in costs else measured
            )
            self.event_pool.append(events)
        self.pending_costs = pending

    def begin(self, ids: list[int]) -> None:
        if self.controller.abort_pending():
            logging.warning(
                "[adaptive-verify] discarding an unfinished verification round"
            )
        for stale in self.registered - set(ids):
            self.controller.release(stale)
        self.registered.intersection_update(ids)
        for request_id in ids:
            self.controller.register(request_id)
            self.registered.add(request_id)
        self.ids = ids
        self._collect_costs()
        self.events = None
        if self.fixed_cap < 0 and len(self.pending_costs) < 16:
            self.events = (
                self.event_pool.pop()
                if self.event_pool
                else [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            )
            self.events[0].record()

    def predict(self, ids: list[int], step: int, logits: torch.Tensor) -> None:
        if not ids or self.fixed_cap >= 0:
            return
        # The draft sampler is top-1 without request temperature or penalties.
        # Estimate the distribution that produced those proposals, not Target's.
        if any(request_id not in self.registered for request_id in ids):
            return
        self.controller.predict(ids, step, logits)

    def plan(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.events is not None:
            self.events[1].record()
        self._collect_costs()
        batch = len(self.ids)
        costs = self.target_costs.setdefault(batch, {})
        cap = self.fixed_cap
        if cap < 0:
            in_flight = {
                total for size, total, _ in self.pending_costs if size == batch
            }
            cap = next(
                (
                    n
                    for n in range(self.steps, -1, -1)
                    if batch * (n + 1) not in costs and batch * (n + 1) not in in_flight
                ),
                -1,
            )
            if cap < 0 and any(
                batch * (n + 1) not in costs for n in range(self.steps + 1)
            ):
                cap = self.steps
        draft = np.zeros(batch + 1)
        verify = np.ones(batch * (self.steps + 1) + 1)
        if cap < 0:
            draft[batch] = self.draft_costs[batch]
            points = sorted(costs)
            verify[batch:] = np.interp(
                np.arange(batch, len(verify)), points, [costs[n] for n in points]
            )
        caps, rows, self.total = self.controller.plan(
            self.ids,
            [self.steps] * batch,
            draft_cost_ms=draft,
            verify_cost_ms=verify,
            fixed_cap=cap if cap >= 0 else None,
        )
        logging.info(
            "[adaptive-verify] round=%d batch=%d draft_steps=%d target_rows=%d fixed_cap=%d calibration=%s",
            self.round,
            batch,
            self.steps,
            self.total,
            self.fixed_cap,
            self.fixed_cap < 0 and cap >= 0,
        )
        if self.events is not None:
            self.events[2].record()
        return caps, rows, self.total

    def target_done(self) -> None:
        if self.events is not None:
            self.events[3].record()

    def observe(self, accepted: torch.Tensor, forced: list[bool]) -> None:
        self.controller.observe(self.ids, accepted, forced, asynchronous=True)
        if self.events is not None:
            self.pending_costs.append((len(self.ids), self.total, self.events))
            self.events = None
        self.round += 1


def create_session(
    capacity: int, steps: int, device: str
) -> AdaptiveVerificationSession:
    return AdaptiveVerificationSession(capacity, steps, device)
