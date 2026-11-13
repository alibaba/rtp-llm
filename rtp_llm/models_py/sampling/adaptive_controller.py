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
    pack_verification_rows,
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
        packed = pack_verification_rows(caps, self.num_steps, total_tokens)
        self._pending = PendingVerification(
            tuple(request_ids), slots, caps.clone(), known
        )
        return caps, packed, total_tokens

    def prepare_from_host(
        self,
        request_ids: Sequence[int],
        *,
        confidence: np.ndarray | None,
        draft_cost_ms: np.ndarray,
        verify_cost_ms: np.ndarray,
        fixed_cap: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Plan from completed prior-round feedback, never read the GPU here.

        This is a performance prediction, not a sampling decision. A missing
        snapshot keeps every draft; rejection still uses the current logits.
        Planning does not freeze training features: current-round predict()
        runs before commit_prepared() marks the verification pending.
        """
        slots = self._slots(request_ids)
        batch = len(slots)
        scheduled = np.full(batch, self.num_steps, dtype=np.int32)
        if fixed_cap is not None:
            if type(fixed_cap) is not int or not 0 <= fixed_cap <= self.num_steps:
                raise AdaptiveBudgetInputError(
                    "fixed_cap", "outside speculative positions"
                )
            scheduled.fill(fixed_cap)
        elif confidence is not None:
            budget = select_draft_token_budget(
                confidence,
                scheduled,
                draft_cost_ms=draft_cost_ms,
                verify_cost_ms=verify_cost_ms,
                num_non_draft_tokens=batch,
                num_sampling_requests=batch,
            )
            # Stable ties keep earlier positions before later positions of the
            # same request, so every selected set remains a valid prefix.
            survival = np.cumprod(confidence.astype(np.float64), axis=1)
            chosen = np.argsort(-survival.reshape(-1), kind="stable")[:budget]
            scheduled = np.bincount(chosen // self.num_steps, minlength=batch).astype(
                np.int32
            )
        self.host_cap_counts = (
            int(np.count_nonzero(scheduled == 0)),
            int(np.count_nonzero(scheduled == self.num_steps)),
        )
        width = self.num_steps + 1
        dense = np.arange(batch * width, dtype=np.int64)
        rows = dense[dense % width <= np.repeat(scheduled, width)]
        return (
            self._device_indices(scheduled, torch.int32),
            self._device_indices(rows, torch.int64),
            len(rows),
        )

    def commit_prepared(
        self, request_ids: Sequence[int], capacities: torch.Tensor
    ) -> None:
        """Commit after current predictions, before verification feedback."""
        slots = self._slots(request_ids)
        if self._pending is not None:
            raise AdaptiveBudgetInputError("request_ids", "observe previous plan first")
        if (
            capacities.shape != (len(slots),)
            or capacities.device != self.device
            or capacities.dtype != torch.int32
        ):
            raise AdaptiveBudgetInputError(
                "capacities", "prepared batch/device mismatch"
            )
        known = tuple(bool(self._ready[slot].all()) for slot in slots)
        # Prepared index-cache tensors are immutable, including across eviction.
        self._pending = PendingVerification(
            tuple(request_ids), slots, capacities, known
        )

    def observe(
        self,
        request_ids: Sequence[int],
        accepted_drafts: torch.Tensor,
        forced: Sequence[bool],
        *,
        asynchronous: bool = False,
        effective_capacities: torch.Tensor | None = None,
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
            if self._pending is not None and self._pending.request_ids:
                raise AdaptiveBudgetInputError(
                    "request_ids", "must match the pending plan"
                )
            self._pending = None
            return
        pending = self._pending
        if pending is None or set(request_ids) != set(pending.request_ids):
            raise AdaptiveBudgetInputError("request_ids", "must match the pending plan")
        positions = {
            request_id: index for index, request_id in enumerate(pending.request_ids)
        }
        order = [positions[request_id] for request_id in request_ids]
        admitted = pending.capacities.index_select(0, self._device_indices(order))
        planned = admitted
        if effective_capacities is not None:
            if (
                effective_capacities.shape != (len(slots),)
                or effective_capacities.device != self.device
                or effective_capacities.dtype not in (torch.int32, torch.int64)
            ):
                raise AdaptiveBudgetInputError(
                    "effective_capacities", "invalid shape, device or dtype"
                )
            # Grammar can censor a planned suffix. It is not a rejection label.
            # The supplied caps follow request_ids order, like accepted_drafts.
            admitted = effective_capacities
        valid = (
            (accepted_drafts >= 0)
            & (accepted_drafts <= admitted)
            & (admitted >= 0)
            & (admitted <= planned)
        ).all()
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
