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
        self._unit_temperature = torch.ones(max_num_requests, device=self.device)
        self._temperature = self._unit_temperature.clone()
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
        self._explicit_host_plans: OrderedDict[
            tuple[int, ...], tuple[np.ndarray, tuple[int, int], tuple[int, ...]]
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
        temperatures = self._unit_temperature
        if temperature is not None:
            if temperature.shape != (len(slots),) or temperature.device != self.device:
                raise AdaptiveBudgetInputError("temperature", "invalid shape or device")
            if ((temperature < 0) | ~torch.isfinite(temperature)).any().item():
                raise AdaptiveBudgetInputError("temperature", "invalid value")
            self._temperature[mapping] = temperature.float()
            temperatures = self._temperature
        self.estimator.predict(
            logits,
            mapping,
            self._device_indices((step,), torch.int32).view(()),
            self._batch_confidence,
            temperatures,
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
        fixed_budget: int | None = None,
        grouped_tail_cut: bool = False,
        explicit_caps: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Plan from completed prior-round feedback, never read the GPU here.

        This is a performance prediction, not a sampling decision. A missing
        snapshot keeps every draft; rejection still uses the current logits.
        Planning does not freeze training features: current-round predict()
        runs before commit_prepared() marks the verification pending.
        """
        slots = self._slots(request_ids)
        batch = len(slots)
        if explicit_caps is not None:
            if (
                fixed_budget is not None
                or fixed_cap is not None
                or confidence is not None
                or grouped_tail_cut
                or len(explicit_caps) != batch
                or any(
                    type(cap) is not int or not 0 <= cap <= self.num_steps
                    for cap in explicit_caps
                )
            ):
                raise AdaptiveBudgetInputError(
                    "explicit_caps", "expected one valid prefix cap per request"
                )
            return self._prepare_explicit_host_caps(explicit_caps)
        scheduled = np.full(batch, self.num_steps, dtype=np.int32)
        if fixed_budget is not None:
            if (
                fixed_cap is not None
                or type(fixed_budget) is not int
                or not 0 <= fixed_budget <= batch * self.num_steps
            ):
                raise AdaptiveBudgetInputError(
                    "fixed_budget", "invalid or conflicts with fixed_cap"
                )
            if grouped_tail_cut:
                cut = batch * self.num_steps - fixed_budget
                if self.num_steps != 5 or not 0 < cut <= batch:
                    raise AdaptiveBudgetInputError(
                        "fixed_budget", "invalid grouped MTP5 tail cut"
                    )
                weakest = np.arange(batch)
                if confidence is not None:
                    survival = np.cumprod(confidence.astype(np.float64), axis=1)
                    weakest = np.argsort(survival[:, -2:].sum(axis=1), kind="stable")
                pairs, leftover = divmod(cut, 2)
                scheduled[weakest[:pairs]] -= 2
                if leftover:
                    scheduled[weakest[pairs]] -= 1
            elif confidence is not None and 0 < fixed_budget < batch * self.num_steps:
                scheduled = self._assign_host_budget(confidence, fixed_budget, batch)
            else:
                # Balanced prefixes can visit buckets unreachable by a uniform cap.
                scheduled.fill(fixed_budget // batch if batch else 0)
                if batch:
                    scheduled[: fixed_budget % batch] += 1
        elif fixed_cap is not None:
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
            scheduled = self._assign_host_budget(confidence, budget, batch)
        self.host_capacities = scheduled
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

    def _prepare_explicit_host_caps(
        self, caps: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # The ordered cap vector determines both the packed indices and the
        # feedback bounds. Cache that immutable host plan, never request IDs.
        key = tuple(caps)
        if key not in self._explicit_host_plans:
            scheduled = np.asarray(key, dtype=np.int32)
            scheduled.setflags(write=False)
            width = self.num_steps + 1
            rows = tuple(
                row
                for index, cap in enumerate(key)
                for row in range(index * width, index * width + cap + 1)
            )
            counts = (key.count(0), key.count(self.num_steps))
            self._explicit_host_plans[key] = scheduled, counts, rows
            if len(self._explicit_host_plans) > 32:
                self._explicit_host_plans.popitem(last=False)
        self._explicit_host_plans.move_to_end(key)
        self.host_capacities, self.host_cap_counts, rows = self._explicit_host_plans[
            key
        ]
        # Keep the existing device cache's event and stream ownership checks.
        return (
            self._device_indices(key, torch.int32),
            self._device_indices(rows, torch.int64),
            len(rows),
        )

    def _assign_host_budget(
        self, confidence: np.ndarray, budget: int, batch: int
    ) -> np.ndarray:
        # Stable ties keep earlier positions before later positions of the
        # same request, so every selected set remains a valid prefix.
        survival = np.cumprod(confidence.astype(np.float64), axis=1)
        chosen = np.argsort(-survival.reshape(-1), kind="stable")[:budget]
        return np.bincount(chosen // self.num_steps, minlength=batch).astype(np.int32)

    def reallocate_current(self, request_ids: Sequence[int], total_tokens: int):
        """Use current draft confidences for a host-sized budget, without D2H.

        Unknown/incomplete rows stay optimistic, as in vLLM. The host budget
        remains unchanged, so tensor shapes and TP communication are fixed.
        """
        slots = self._slots(request_ids)
        batch = len(slots)
        if not batch <= total_tokens <= batch * (self.num_steps + 1):
            raise AdaptiveBudgetInputError("total_tokens", "outside prefix bounds")
        if self._pending is not None:
            raise AdaptiveBudgetInputError("request_ids", "observe before reallocation")
        caps = torch.full(
            (batch,), self.num_steps, dtype=torch.int32, device=self.device
        )
        budget = total_tokens - batch
        if budget == 0:
            caps.zero_()
        elif budget < batch * self.num_steps:
            mapping = self._device_indices(slots)
            ready = self._ready[list(slots)].all(axis=1)
            known = None if ready.all() else self._device_indices(ready, torch.bool)
            assign_draft_token_budget(
                self.confidence_probs,
                mapping,
                caps,
                budget,
                self._survival,
                known=known,
            )
        return (
            caps,
            pack_verification_rows(caps, self.num_steps, total_tokens),
            total_tokens,
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

    def _validate_observation(
        self, batch: int, accepted_drafts: torch.Tensor, forced: Sequence[bool]
    ) -> None:
        if accepted_drafts.shape != (batch,) or accepted_drafts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise AdaptiveBudgetInputError("accepted_drafts", "invalid integer counts")
        if accepted_drafts.device != self.device:
            raise AdaptiveBudgetInputError("accepted_drafts", "device mismatch")
        if len(forced) != batch or any(
            type(value) not in (bool, np.bool_) for value in forced
        ):
            raise AdaptiveBudgetInputError("forced", "expected one boolean per request")

    def _validate_effective_capacities(
        self, batch: int, effective_capacities: torch.Tensor
    ) -> None:
        if (
            effective_capacities.shape != (batch,)
            or effective_capacities.device != self.device
            or effective_capacities.dtype not in (torch.int32, torch.int64)
        ):
            raise AdaptiveBudgetInputError(
                "effective_capacities", "invalid shape, device or dtype"
            )

    def consume_measured_plan(
        self,
        request_ids: Sequence[int],
        accepted_drafts: torch.Tensor,
        forced: Sequence[bool],
        effective_capacities: torch.Tensor | None,
    ) -> np.ndarray:
        """Consume a host-sized plan without preparing GPU estimator feedback."""
        pending = self._pending
        if (
            pending is None
            or pending.request_ids != tuple(request_ids)
            or any(pending.trainable)
        ):
            raise AdaptiveBudgetInputError(
                "request_ids", "must match an ordered, untrained host plan"
            )
        self._validate_observation(len(request_ids), accepted_drafts, forced)
        if effective_capacities is not None:
            self._validate_effective_capacities(len(request_ids), effective_capacities)
        self._ready[list(pending.slots)] = False
        self._pending = None
        return self.host_capacities

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
        self._validate_observation(len(slots), accepted_drafts, forced)
        if not slots:
            if self._pending is not None and self._pending.request_ids:
                raise AdaptiveBudgetInputError(
                    "request_ids", "must match the pending plan"
                )
            self._pending = None
            return
        pending = self._pending
        if pending is None:
            raise AdaptiveBudgetInputError("request_ids", "must match the pending plan")
        if tuple(request_ids) == pending.request_ids:
            order = range(len(slots))
            planned = pending.capacities
        else:
            if set(request_ids) != set(pending.request_ids):
                raise AdaptiveBudgetInputError(
                    "request_ids", "must match the pending plan"
                )
            positions = {
                request_id: index
                for index, request_id in enumerate(pending.request_ids)
            }
            order = [positions[request_id] for request_id in request_ids]
            planned = pending.capacities.index_select(0, self._device_indices(order))
        admitted = planned
        if effective_capacities is not None:
            self._validate_effective_capacities(len(slots), effective_capacities)
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
        if len(eligible) == len(slots):
            self.estimator.step(
                self._device_indices(slots),
                accepted_drafts + 1,
                admitted - accepted_drafts,
            )
        elif eligible:
            indices = self._device_indices(eligible)
            accepted = accepted_drafts.index_select(0, indices)
            self.estimator.step(
                self._device_indices([slots[i] for i in eligible]),
                accepted + 1,
                admitted.index_select(0, indices) - accepted,
            )
        self._ready[list(pending.slots)] = False
        self._pending = None
