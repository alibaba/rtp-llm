from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from rtp_llm.models_py.sampling.acceptance_estimator import OnlineAcceptanceEstimator
from rtp_llm.models_py.sampling.adaptive_verification import (
    AdaptiveBudgetInputError,
    assign_draft_token_budget,
    select_draft_token_budget,
)


@dataclass(frozen=True, slots=True)
class PendingVerification:
    request_ids: tuple[int, ...]
    slots: tuple[int, ...]
    capacities: torch.Tensor
    trainable: tuple[bool, ...]


class AdaptiveDraftController:
    def __init__(
        self, max_num_requests: int, num_speculative_steps: int, device: torch.device
    ) -> None:
        if max_num_requests < 1 or num_speculative_steps < 1:
            raise AdaptiveBudgetInputError("dimensions", "must be positive")
        self.estimator = OnlineAcceptanceEstimator(
            max_num_requests, num_speculative_steps, device
        )
        self.device = self.estimator.predictions.device
        self.num_steps = num_speculative_steps
        self.confidence_probs = self.estimator.predictions
        self._request_slots: dict[int, int] = {}
        self._free_slots = list(reversed(range(max_num_requests)))
        self._ready = np.zeros((max_num_requests, num_speculative_steps), dtype=bool)
        self._temperature = torch.ones(max_num_requests, device=self.device)
        self._batch_confidence = torch.empty_like(self.confidence_probs)
        self._survival = torch.empty(
            max_num_requests * num_speculative_steps,
            dtype=torch.int32,
            device=self.device,
        )
        self._pending: PendingVerification | None = None
        self._index_cache: OrderedDict[
            tuple[tuple[int, ...], torch.dtype],
            tuple[torch.Tensor, torch.cuda.Event, torch.cuda.Stream],
        ] = OrderedDict()

    def _device_indices(
        self, values: Sequence[int], dtype: torch.dtype = torch.int64
    ) -> torch.Tensor:
        key = (tuple(int(value) for value in values), dtype)
        stream = torch.cuda.current_stream(self.device)
        if key not in self._index_cache:
            host = torch.tensor(key[0], dtype=dtype, pin_memory=True)
            tensor = host.to(self.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
            self._index_cache[key] = (tensor, ready, stream)
            if len(self._index_cache) > 32:
                self._index_cache.popitem(last=False)
        self._index_cache.move_to_end(key)
        tensor, ready, producer = self._index_cache[key]
        if stream != producer:
            stream.wait_event(ready)
            tensor.record_stream(stream)
        return tensor

    def register(self, request_id: int) -> int:
        if isinstance(request_id, bool) or not isinstance(request_id, Integral):
            raise AdaptiveBudgetInputError("request_id", "expected an integer")
        if request_id in self._request_slots:
            return self._request_slots[request_id]
        if not self._free_slots:
            raise AdaptiveBudgetInputError("request_id", "request slots exhausted")
        slot = self._free_slots.pop()
        self._request_slots[request_id] = slot
        self._ready[slot] = False
        self.estimator.features[slot].zero_()
        self.confidence_probs[slot].fill_(1.0)
        self._temperature[slot].fill_(1.0)
        return slot

    def abort_pending(self) -> bool:
        if self._pending is None:
            return False
        self._ready[list(self._pending.slots)] = False
        self._pending = None
        return True

    def release(self, request_id: int) -> None:
        slot = self._slots((request_id,))[0]
        if self._pending is not None and request_id in self._pending.request_ids:
            raise AdaptiveBudgetInputError("request_id", "observe before release")
        del self._request_slots[request_id]
        self._ready[slot] = False
        self._free_slots.append(slot)

    def _slots(self, request_ids: Sequence[int]) -> tuple[int, ...]:
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in request_ids
        ):
            raise AdaptiveBudgetInputError("request_ids", "expected integer IDs")
        if len(set(request_ids)) != len(request_ids):
            raise AdaptiveBudgetInputError("request_ids", "duplicate requests")
        if any(request_id not in self._request_slots for request_id in request_ids):
            raise AdaptiveBudgetInputError("request_ids", "request is not registered")
        return tuple(self._request_slots[request_id] for request_id in request_ids)

    def predict(
        self,
        request_ids: Sequence[int],
        step: int,
        logits: torch.Tensor,
        temperature: torch.Tensor | None = None,
    ) -> None:
        slots = self._slots(request_ids)
        if (
            isinstance(step, bool)
            or not isinstance(step, Integral)
            or not 0 <= step < self.num_steps
        ):
            raise AdaptiveBudgetInputError("step", "outside speculative positions")
        if self._pending is not None and set(request_ids).intersection(
            self._pending.request_ids
        ):
            raise AdaptiveBudgetInputError("request_ids", "observe before predict")
        if logits.ndim != 2 or logits.shape[0] != len(slots) or logits.shape[1] < 1:
            raise AdaptiveBudgetInputError("logits", "expected batch vocabulary rows")
        if (
            logits.device != self.device
            or logits.stride(1) != 1
            or not logits.is_floating_point()
        ):
            raise AdaptiveBudgetInputError("logits", "invalid device, dtype or stride")
        if not slots:
            return
        mapping = self._device_indices(slots)
        if temperature is None:
            self._temperature.index_fill_(0, mapping, 1.0)
        else:
            if temperature.shape != (len(slots),) or temperature.device != self.device:
                raise AdaptiveBudgetInputError("temperature", "invalid shape or device")
            if ((temperature < 0) | ~torch.isfinite(temperature)).any().item():
                raise AdaptiveBudgetInputError("temperature", "invalid value")
            self._temperature[mapping] = temperature.float()
        self.estimator.predict(
            logits,
            mapping,
            self._device_indices((step,), torch.int32).view(()),
            self._batch_confidence,
            self._temperature,
        )
        self._ready[list(slots), step] = True

    def plan(
        self,
        request_ids: Sequence[int],
        capacities: Sequence[int],
        *,
        draft_cost_ms: np.ndarray,
        verify_cost_ms: np.ndarray,
        fixed_cap: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        slots = self._slots(request_ids)
        scheduled = np.asarray(capacities)
        if scheduled.shape != (len(slots),):
            raise AdaptiveBudgetInputError("capacities", "batch length mismatch")
        if not slots:
            return (
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
                0,
            )
        if (
            scheduled.dtype.kind not in "iu"
            or not ((scheduled >= 0) & (scheduled <= self.num_steps)).all()
        ):
            raise AdaptiveBudgetInputError("capacities", "invalid integer counts")
        if self._pending is not None:
            raise AdaptiveBudgetInputError("request_ids", "observe previous plan first")
        known = tuple(
            bool(self._ready[slot, :capacity].all())
            for slot, capacity in zip(slots, scheduled)
        )
        mapping = self._device_indices(slots)
        caps = self._device_indices(scheduled, torch.int32).clone()
        budget = sum(int(capacity) for capacity in scheduled)
        if fixed_cap is not None:
            if type(fixed_cap) is not int or not 0 <= fixed_cap <= self.num_steps:
                raise AdaptiveBudgetInputError(
                    "fixed_cap", "outside speculative positions"
                )
            caps.clamp_(max=fixed_cap)
            budget = int(np.minimum(scheduled, fixed_cap).sum())
        elif all(known):
            confidence = self.confidence_probs.index_select(0, mapping).cpu().numpy()
            budget = select_draft_token_budget(
                confidence,
                scheduled,
                draft_cost_ms=draft_cost_ms,
                verify_cost_ms=verify_cost_ms,
                num_non_draft_tokens=len(slots),
                num_sampling_requests=len(slots),
            )
            if budget == 0:
                caps.zero_()
            elif budget < int(scheduled.sum()):
                assign_draft_token_budget(
                    self.confidence_probs, mapping, caps, budget, self._survival
                )
        total_tokens = len(slots) + budget
        dense = torch.arange(len(slots) * (self.num_steps + 1), device=self.device)
        valid = dense % (self.num_steps + 1) <= caps.repeat_interleave(
            self.num_steps + 1
        )
        order = torch.argsort(torch.where(valid, dense, dense + dense.numel()))
        packed = dense.index_select(0, order[:total_tokens])
        self._pending = PendingVerification(
            tuple(request_ids), slots, caps.clone(), known
        )
        return caps, packed, total_tokens

    def observe(
        self,
        request_ids: Sequence[int],
        accepted_drafts: torch.Tensor,
        forced: Sequence[bool],
        *,
        asynchronous: bool = False,
    ) -> None:
        slots = self._slots(request_ids)
        if accepted_drafts.shape != (len(slots),) or accepted_drafts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise AdaptiveBudgetInputError("accepted_drafts", "invalid integer counts")
        if accepted_drafts.device != self.device:
            raise AdaptiveBudgetInputError("accepted_drafts", "device mismatch")
        if len(forced) != len(slots) or any(
            type(value) not in (bool, np.bool_) for value in forced
        ):
            raise AdaptiveBudgetInputError("forced", "expected one boolean per request")
        if not slots:
            return
        pending = self._pending
        if pending is None or set(request_ids) != set(pending.request_ids):
            raise AdaptiveBudgetInputError("request_ids", "must match the pending plan")
        positions = {
            request_id: index for index, request_id in enumerate(pending.request_ids)
        }
        order = [positions[request_id] for request_id in request_ids]
        admitted = pending.capacities.index_select(0, self._device_indices(order))
        valid = ((accepted_drafts >= 0) & (accepted_drafts <= admitted)).all()
        if asynchronous:
            torch._assert_async(valid, "accepted_drafts outside admitted counts")
        elif not valid.item():
            raise AdaptiveBudgetInputError("accepted_drafts", "outside admitted counts")
        eligible = [
            i
            for i, position in enumerate(order)
            if pending.trainable[position] and not forced[i]
        ]
        if eligible:
            indices = self._device_indices(eligible)
            accepted = accepted_drafts.index_select(0, indices)
            self.estimator.step(
                self._device_indices([slots[i] for i in eligible]),
                accepted + 1,
                admitted.index_select(0, indices) - accepted,
            )
        self._ready[list(pending.slots)] = False
        self._pending = None
