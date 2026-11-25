import logging
import math
import os
import time
from collections import deque

import numpy as np
import torch

from rtp_llm.models_py.sampling.adaptive_controller import AdaptiveDraftController
from rtp_llm.models_py.sampling.adaptive_tail_budget import TailRiskEpPolicy
from rtp_llm.models_py.sampling.adaptive_verification import (
    pack_ep_margin_rows,
    pack_ep_tail_rows,
)

_EP_VERIFY_ROWS_ENV = "RTP_LLM_ADAPTIVE_EP_VERIFY_ROWS_PER_REQUEST"


def _configured_ep_verify_rows_per_request(
    steps: int,
    fixed_cap: int,
    ep_size: int,
    allow_ep_current_scores: bool,
    has_capture_buckets: bool,
) -> float | None:
    raw = os.environ.get(_EP_VERIFY_ROWS_ENV)
    supports_budget = (
        allow_ep_current_scores
        and ep_size > 1
        and steps > 0
        and fixed_cap == -1
        and has_capture_buckets
    )
    competing_policy = (
        os.environ.get("RTP_LLM_ADAPTIVE_EP_TAIL_RISK") == "1"
        or os.environ.get("RTP_LLM_ADAPTIVE_EP_CURRENT_SCORES") == "1"
    )
    if raw is None:
        # The default is tuned for five draft steps. Other widths keep their
        # existing policy unless the operator supplies a budget.
        return 4.5 if supports_budget and steps == 5 and not competing_policy else None
    range_error = f"{_EP_VERIFY_ROWS_ENV} requires a finite number in [1, {steps + 1}]"
    try:
        rows = float(raw)
    except ValueError:
        raise ValueError(range_error) from None
    if not math.isfinite(rows) or not 1 <= rows <= steps + 1:
        raise ValueError(range_error)
    if not supports_budget:
        raise ValueError(
            f"{_EP_VERIFY_ROWS_ENV} requires GLM5 TP1 Decode with EP > 1, "
            "positive draft steps, CAP=-1 and CUDA Graph captures"
        )
    if competing_policy:
        raise ValueError(
            f"{_EP_VERIFY_ROWS_ENV} conflicts with EP TailRisk or current scores"
        )
    return rows


def _configured_ep_bucket_row_quota(ep_size: int) -> float | None:
    raw = os.environ.get("RTP_LLM_ADAPTIVE_EP_BUCKET_ROW_QUOTAS")
    if raw is None:
        return None
    # This session is only selected on GLM5 Decode with TP1; the process
    # group's world rank is then the DP rank. Reading group metadata is local.
    dist = torch.distributed
    if ep_size != 4 or not dist.is_initialized() or dist.get_world_size() != ep_size:
        raise ValueError(
            "EP bucket row quotas require an initialized four-rank TP1 EP group"
        )
    rank = dist.get_rank()
    quotas = tuple(map(float, raw.split(",")))
    if (
        not 0 <= rank < ep_size
        or len(quotas) != ep_size
        or not all(map(TailRiskEpPolicy.valid_bucket_row_quota, quotas))
        or not math.isclose(sum(quotas), 252, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(
            "provide four feasible EP row quotas in world-rank order totaling 252"
        )
    logging.info(
        "[adaptive-verify] EP bucket row quota rank=%d rows=%s", rank, quotas[rank]
    )
    return quotas[rank]


class _MeasuredEpPolicy:
    """Use realized tokens per round under the live expert-parallel workload."""

    _SAMPLES = 3
    _WARMUP_SAMPLES = 2
    _REFRESH = 16
    _SAMPLE_GRACE_ROUNDS = 16
    _MIN_GAIN = 1.025

    def __init__(
        self, steps: int, execution_rows: np.ndarray, *, grouped_tail_cut: bool = False
    ) -> None:
        self.steps = steps
        self.execution_rows = execution_rows
        self.grouped_tail_cut = grouped_tail_cut
        self.ids: tuple[int, ...] = ()
        self.epoch = 0
        self.phase = 0
        self.dense = 0
        self.candidates: list[int] = []
        self.winner: int | None = None
        self.reference_deadline: int | None = None
        self.initial_reference_waits = 0
        self.samples: dict[int, deque[tuple[float, int]]] = {}
        self.warmup_samples: dict[int, int] = {}
        self.accepted: dict[int, np.ndarray] = {}

    def begin(self, ids: list[int]) -> tuple[int, int]:
        cohort = tuple(sorted(ids))
        if cohort != self.ids:
            self.ids = cohort
            self.epoch += 1
            self.phase = 0
            self.winner = None
            self.reference_deadline = None
            self.initial_reference_waits = 0
            self.dense = len(ids) * (self.steps + 1)
            self.samples.clear()
            self.warmup_samples.clear()
            self.accepted.clear()
            # Probe modest cuts at actual graph boundaries. A smaller logical
            # count in the dense execution bucket cannot save graph work.
            lower = max(len(ids), math.ceil(0.8 * self.dense))
            representatives = {
                int(self.execution_rows[n]): n
                for n in range(lower, self.dense)
                if self.execution_rows[n] < self.execution_rows[self.dense]
            }
            if self.grouped_tail_cut:
                # Start with the nearest distinct physical bucket. For MTP5
                # a two-row cut per request can remove a complete query tile.
                candidates = [
                    n for n in representatives.values() if self.dense - n <= len(ids)
                ]
                self.candidates = []
                if candidates:
                    nearest = max(candidates)
                    if (
                        (self.dense - nearest) % 2
                        and self.dense - nearest < len(ids)
                        and self.execution_rows[nearest - 1]
                        == self.execution_rows[nearest]
                    ):
                        nearest -= 1
                    self.candidates.append(nearest)
            else:
                self.candidates = sorted(
                    {
                        min(representatives.values(), key=lambda n: (abs(n - goal), -n))
                        for goal in (0.9 * self.dense, 0.85 * self.dense)
                        if representatives
                    },
                    reverse=True,
                )

        if not self.needs_measurement():
            return self.dense, 3
        if len(self.samples.get(self.dense, ())) < self._SAMPLES:
            # An enqueued event is not a completed reference. Keep all drafts
            # while asynchronous sampling is pending. A cohort without usable
            # feedback, for example forced acceptance, stays full-width.
            self.initial_reference_waits += 1
            if self.initial_reference_waits > self._SAMPLE_GRACE_ROUNDS:
                self.winner = self.dense
                return self.dense, 3
            return self.dense, 1
        phase = max(self.phase, self._SAMPLES)
        self.phase = phase + 1
        trial_rounds = self._SAMPLES
        if self.grouped_tail_cut:
            trial_rounds += self._WARMUP_SAMPLES
        probe = (phase - self._SAMPLES) // trial_rounds
        if probe < len(self.candidates):
            return self.candidates[probe], 1
        if self.reference_deadline is not None:
            if phase >= self.reference_deadline:
                self.winner = self.dense
            return self.dense, 3

        def score(rows: int) -> float:
            samples = self.samples.get(rows, ())
            if len(samples) < self._SAMPLES:
                return 0.0
            return sum(tokens for _, tokens in samples) / sum(
                duration for duration, _ in samples
            )

        reference = score(self.dense) * self._MIN_GAIN
        if self.winner is None:
            if any(
                len(self.samples.get(rows, ())) < self._SAMPLES
                for rows in self.candidates
            ):
                deadline = (
                    self._SAMPLES
                    + len(self.candidates) * trial_rounds
                    + self._SAMPLE_GRACE_ROUNDS
                )
                if phase >= deadline:
                    self.winner = self.dense
                return self.dense, 3
            self.winner = max(self.candidates, key=score)
        if score(self.winner) <= reference:
            # Do not probe losing buckets again while the same cohort is live.
            self.winner = self.dense
            return self.dense, 3
        if phase % self._REFRESH == 0:
            self.reference_deadline = phase + self._SAMPLE_GRACE_ROUNDS
            return self.dense, 1
        return self.winner, 2

    def needs_measurement(self) -> bool:
        return bool(self.candidates) and self.winner != self.dense

    def record(
        self,
        epoch: int,
        ids: tuple[int, ...],
        total: int,
        ms: float,
        values: np.ndarray,
        effective_caps: np.ndarray | None = None,
    ) -> None:
        if epoch != self.epoch or not math.isfinite(ms) or ms <= 0:
            return
        if not ((values >= 0) & (values <= self.steps)).all():
            return
        # Prime each execution shape before pricing it. A submitted round
        # can only fill a warmup slot once its feedback completes.
        warmed = self.warmup_samples.get(total, 0)
        if self.grouped_tail_cut and warmed < self._WARMUP_SAMPLES:
            self.warmup_samples[total] = warmed + 1
        else:
            self.samples.setdefault(total, deque(maxlen=self._SAMPLES)).append(
                (ms, int(values.sum()) + len(ids))
            )
        if total == self.dense:
            self.reference_deadline = None
            for index, (request_id, accepted) in enumerate(zip(ids, values)):
                # The measured throughput includes grammar-capped tokens,
                # but those rows cannot tell us whether the suffix would pass.
                if effective_caps is not None and effective_caps[index] < self.steps:
                    continue
                self.accepted.setdefault(
                    request_id, np.zeros(self.steps + 1, dtype=np.float64)
                )[accepted] += 1

    def confidence(self, ids: list[int]) -> np.ndarray | None:
        if not self.accepted:
            return None
        pooled = np.sum(list(self.accepted.values()), axis=0)
        pooled_survival = np.cumsum(pooled[::-1])[::-1]
        pooled_survival /= pooled_survival[0]
        counts = np.stack(
            [self.accepted.get(request_id, np.zeros_like(pooled)) for request_id in ids]
        )
        # Sparse requests shrink to the cohort prior. Only full-width
        # observations train the tail, so truncated drafts are not rejections.
        survival = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1] + 4 * pooled_survival
        return np.divide(
            survival[:, 1:],
            survival[:, :-1],
            out=np.zeros_like(survival[:, 1:]),
            where=survival[:, :-1] > 0,
        )


class AdaptiveVerificationSession:
    # Fade unrefreshed exact samples back to the live-scaled startup curve.
    # This changes estimates, never forces a verification budget.
    _COST_MAX_AGE = 128

    def __init__(
        self,
        capacity: int,
        steps: int,
        device: str,
        capture_rows: list[int] | None = None,
        ep_size: int = 1,
        allow_ep_current_scores: bool = False,
    ) -> None:
        # C++ passes its actual target graph grid, avoiding a second grid formula.
        rows = list(capture_rows or [])
        maximum = capacity * (steps + 1)
        if rows and (
            any(type(n) is not int or n <= 0 for n in rows)
            or rows != sorted(set(rows))
            or rows[-1] != maximum
        ):
            raise ValueError(
                "capture_rows must be sorted unique positive buckets ending at capacity * (steps + 1)"
            )
        self._has_capture_buckets = bool(rows)
        self._execution_rows = np.arange(maximum + 1)
        if rows:
            self._execution_rows[1:] = np.asarray(rows)[
                np.searchsorted(rows, self._execution_rows[1:])
            ]
        self.steps = steps
        self.fixed_cap = int(os.environ.get("RTP_LLM_ADAPTIVE_VERIFY_CAP", "-1"))
        if not -1 <= self.fixed_cap <= steps:
            raise ValueError(
                "RTP_LLM_ADAPTIVE_VERIFY_CAP must be -1 or between 0 and draft steps"
            )
        self._ep_verify_rows_per_request = _configured_ep_verify_rows_per_request(
            steps,
            self.fixed_cap,
            ep_size,
            allow_ep_current_scores,
            self._has_capture_buckets,
        )
        self._capture_buckets = rows
        self.controller = AdaptiveDraftController(capacity, steps, torch.device(device))
        self.ids: list[int] = []
        self.registered: set[int] = set()
        self.round_costs: dict[int, dict[int, float]] = {}
        self._cost_sample_round: dict[tuple[int, int], int] = {}
        self._profile_target: dict[int, float] = {}
        self._profile_draft: dict[int, float] = {}
        self._profile_scale: dict[int, float] = {}
        self.events: list[torch.cuda.Event | None] | None = None
        self.pending_costs: list[tuple[int, int, list[torch.cuda.Event], float]] = []
        self.event_pool: list[list[torch.cuda.Event | None]] = []
        self.total = 0
        self.round = 0
        self._prepared = None
        self._feedback_stream = torch.cuda.Stream(device=device)
        self._pending_predictions = []
        self._prediction_buffers = []
        self._host_predictions = {}
        self._request_epochs = {}
        self._next_epoch = 0
        self._ep_tail_risk = (
            allow_ep_current_scores
            and steps == 5
            and ep_size > 1
            and self.fixed_cap < 0
            and self._has_capture_buckets
            and os.environ.get("RTP_LLM_ADAPTIVE_EP_TAIL_RISK") == "1"
        )
        self._ep_current_scores = (
            allow_ep_current_scores
            and steps == 5
            and not self._ep_tail_risk
            and os.environ.get("RTP_LLM_ADAPTIVE_EP_CURRENT_SCORES") == "1"
        )
        self._ep_reallocated_caps = None
        self._ep_policy = None
        price = None
        if self._ep_tail_risk:
            price = os.environ.get("RTP_LLM_ADAPTIVE_EP_ROW_PRICE")
            bucket_bid_price = os.environ.get("RTP_LLM_ADAPTIVE_EP_BUCKET_BID_PRICE")
            prior = os.environ.get("RTP_LLM_ADAPTIVE_EP_ROW_PRICE_PRIOR")
            if prior is not None and price is None:
                raise ValueError("an EP row price prior requires an EP row price")
            self._ep_policy = TailRiskEpPolicy(
                steps,
                self._execution_rows,
                near_full=os.environ.get("RTP_LLM_ADAPTIVE_EP_TAIL_RISK_NEAR_FULL")
                == "1",
                row_price=float(price) if price is not None else None,
                prior=(
                    tuple(map(float, prior.split(","))) if prior is not None else None
                ),
                bucket_bid_price=(
                    float(bucket_bid_price) if bucket_bid_price is not None else None
                ),
                bucket_row_quota=_configured_ep_bucket_row_quota(ep_size),
            )
        elif (
            ep_size > 1
            and self.fixed_cap < 0
            and self._ep_verify_rows_per_request is None
        ):
            self._ep_policy = _MeasuredEpPolicy(
                steps, self._execution_rows, grouped_tail_cut=self._ep_current_scores
            )
        self._ep_margin_rerank = (
            self._ep_tail_risk
            and capacity == 14
            and price is not None
            and os.environ.get("RTP_LLM_ADAPTIVE_EP_MARGIN_RERANK") == "1"
        )
        self._ep_margin_four_gaps = (
            self._ep_margin_rerank
            and os.environ.get("RTP_LLM_ADAPTIVE_EP_MARGIN_FOUR_GAPS") == "1"
        )
        if self._ep_margin_four_gaps:
            logging.info("[adaptive-verify] EP margin model uses all four Draft gaps")
        self._ep_margin_prior = None
        self._ep_pending = []
        self._ep_buffers = []
        self._ep_accepted = None

    def _collect_predictions(self) -> None:
        pending = []
        for ids, epochs, generation, host, ready in self._pending_predictions:
            if not ready.query():
                pending.append((ids, epochs, generation, host, ready))
                continue
            values = host.numpy()
            for index, request_id in enumerate(ids):
                if self._request_epochs.get(request_id) == epochs[index]:
                    row = values[index]
                    if np.all(np.isfinite(row)) and np.all((row >= 0) & (row <= 1)):
                        self._host_predictions[request_id] = (generation, row.copy())
                    else:
                        # Prediction failure must not strand TP peers waiting
                        # for a budget broadcast. Sampling still uses real logits.
                        self._host_predictions.pop(request_id, None)
            self._prediction_buffers.append(host)
        self._pending_predictions = pending

    def _queue_predictions(self, ids: tuple[int, ...] | None = None) -> None:
        if ids is None:
            ids = tuple(self.ids)
        if not ids or len(self._pending_predictions) >= 16:
            return
        slots = self.controller._slots(ids)
        source_stream = torch.cuda.current_stream(self.controller.device)
        values = self.controller.confidence_probs.index_select(
            0, self.controller._device_indices(slots)
        )
        host = next(
            (
                buffer
                for buffer in self._prediction_buffers
                if buffer.shape == values.shape
            ),
            None,
        )
        if host is None:
            host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
        else:
            self._prediction_buffers = [
                b for b in self._prediction_buffers if b is not host
            ]
        # Bound reusable pinned storage when the online batch size varies.
        self._prediction_buffers = self._prediction_buffers[-16:]
        produced = torch.cuda.Event()
        produced.record(source_stream)
        ready = torch.cuda.Event()
        with torch.cuda.stream(self._feedback_stream):
            self._feedback_stream.wait_event(produced)
            host.copy_(values, non_blocking=True)
            values.record_stream(self._feedback_stream)
            ready.record()
        self._pending_predictions.append(
            (
                ids,
                tuple(self._request_epochs[i] for i in ids),
                self.round,
                host,
                ready,
            )
        )

    def seed_cost_curves(self, target_curve, draft_curve) -> None:
        """Seed graph-body costs; online whole-round timing supplies the tail.

        Both curves are measured graph replays, not assumed token-linear costs.
        Sampling, LM head and CPU bookkeeping are not in these graphs and must
        not be claimed as profiled. Whole-round observations calibrate the curve
        in either direction for each batch, including shorter live KV contexts.
        """

        def validated(curve):
            result = {}
            for rows, cost in curve:
                if (
                    type(rows) is not int
                    or rows <= 0
                    or rows in result
                    or not math.isfinite(cost)
                    or cost <= 0
                ):
                    raise ValueError("invalid adaptive graph cost curve")
                result[rows] = float(cost)
            return result

        target = validated(target_curve)
        draft = validated(draft_curve)
        required = set(int(n) for n in self._execution_rows[1:])
        if not self._has_capture_buckets or set(target) != required or not draft:
            raise ValueError(
                "adaptive profiles must cover all target buckets and have draft samples"
            )
        self._profile_target, self._profile_draft = target, draft
        self._profile_scale.clear()

    def _has_profile_for(self, batch: int) -> bool:
        return bool(batch and self._profile_draft and batch <= max(self._profile_draft))

    def _profile_cost(self, batch: int, bucket: int) -> float:
        draft_bucket = min(n for n in self._profile_draft if n >= batch)
        return (
            self._profile_target[bucket]
            + self.steps * self._profile_draft[draft_bucket]
        )

    def _decision_costs(self, batch: int) -> dict[int, float]:
        costs = {}
        if self._has_profile_for(batch):
            scale = self._profile_scale.get(batch, 1.0)
            for bucket in set(
                int(n)
                for n in self._execution_rows[batch : batch * (self.steps + 1) + 1]
            ):
                costs[bucket] = self._profile_cost(batch, bucket) * scale
        for bucket, measured in self.round_costs.setdefault(batch, {}).items():
            if bucket in costs:
                age = max(
                    0,
                    self.round
                    - self._cost_sample_round.get((batch, bucket), self.round),
                )
                weight = max(0.0, 1.0 - age / self._COST_MAX_AGE)
                costs[bucket] = weight * measured + (1.0 - weight) * costs[bucket]
            else:
                # Without a profile there is no defensible replacement estimate.
                costs[bucket] = measured
        return costs

    def _cost_key(self, total: int) -> int:
        return int(self._execution_rows[total])

    def _collect_costs(self) -> None:
        pending = []
        for batch, total, events, cpu_ms in self.pending_costs:
            if not events[1].query():
                pending.append((batch, total, events, cpu_ms))
                continue
            # CPU and GPU work overlap; adding them double counts execution.
            # Include the whole round's GPU queue and the host submission cost,
            # rather than pricing only target forward and ignoring its tail.
            measured = max(events[0].elapsed_time(events[1]), cpu_ms, 1e-6)
            if self._has_profile_for(batch):
                scale = measured / self._profile_cost(batch, total)
                old = self._profile_scale.get(batch)
                self._profile_scale[batch] = (
                    scale if old is None else 0.8 * old + 0.2 * scale
                )
            costs = self.round_costs.setdefault(batch, {})
            stale = self._has_profile_for(batch) and (
                self.round - self._cost_sample_round.get((batch, total), self.round)
                >= self._COST_MAX_AGE
            )
            costs[total] = (
                0.8 * costs[total] + 0.2 * measured
                if total in costs and not stale
                else measured
            )
            self._cost_sample_round[(batch, total)] = self.round
            self.event_pool.append(events)
        self.pending_costs = pending

    def _collect_ep_samples(self) -> None:
        pending = []
        for entry in self._ep_pending:
            epoch, ids, rows, events, cpu_ms, host, planned, has_forced, bonus = entry
            if not events[1].query():
                pending.append(entry)
                continue
            values = host.numpy()
            # Tail-risk receives emitted tokens. Convert after the same D2H
            # readiness fence instead of launching a decode-stream subtraction.
            accepted = values[0] - bonus if bonus else values[0]
            if len(values) == 3:
                planned = values[2]
            effective = values[1] if len(values) >= 2 else planned
            if not (
                (accepted >= 0) & (accepted <= effective) & (effective <= planned)
            ).all():
                # Ignore broken feedback without throwing between TP collectives.
                logging.error(
                    "[adaptive-verify] accepted drafts outside admitted counts"
                )
            elif not has_forced:
                if events[0] is None:
                    self._ep_policy.record_acceptance(
                        epoch, ids, rows, accepted, effective
                    )
                else:
                    self._ep_policy.record(
                        epoch,
                        ids,
                        rows,
                        max(events[0].elapsed_time(events[1]), cpu_ms),
                        accepted,
                        effective,
                    )
            self.event_pool.append(events)
            self._ep_buffers.append(host)
        self._ep_pending = pending
        self._ep_buffers = self._ep_buffers[-16:]

    def _start_round_timing(self) -> None:
        self.events = (
            self.event_pool.pop()
            if self.event_pool
            else [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        )
        self.events[0].record()

    def _prepare_acceptance_feedback(self) -> None:
        # This policy has no timing input. The event fences the existing
        # asynchronous D2H copies when finish records it on the decode stream.
        self.events = (
            self.event_pool.pop()
            if self.event_pool
            else [None, torch.cuda.Event(enable_timing=False)]
        )

    def begin(self, ids: list[int]) -> None:
        self._cpu_round_start = time.perf_counter()
        if self.controller.abort_pending():
            logging.warning(
                "[adaptive-verify] discarding an unfinished verification round"
            )
        for stale in self.registered - set(ids):
            self.controller.release(stale)
            self._host_predictions.pop(stale, None)
            self._request_epochs.pop(stale, None)
        self.registered.intersection_update(ids)
        for request_id in ids:
            if request_id not in self.registered:
                self._next_epoch += 1
                self._request_epochs[request_id] = self._next_epoch
            self.controller.register(request_id)
            self.registered.add(request_id)
        self.ids = list(ids)
        if self._ep_policy is None:
            self._collect_costs()
            self._collect_predictions()
        else:
            self._collect_ep_samples()
        self.events = None
        self._ep_accepted = None
        self._ep_reallocated_caps = None
        self._ep_margin_prior = None
        if (
            self._ep_verify_rows_per_request is None
            and self.needs_prediction()
            and len(self.pending_costs) < 16
        ):
            self._start_round_timing()
        self._prepare_plan()

    def begin_for_executor(self, ids: list[int]) -> tuple:
        self.begin(ids)
        plan = self.prepared_plan()
        # Tail-risk uses only completed past feedback. There are no current
        # draft scores to collect before committing these exact prefixes.
        if self._ep_tail_risk and not self.needs_current_ep_margin():
            self.controller.commit_prepared(self.ids, plan[0])
        return (
            plan,
            self.can_reuse_dense_fallback(),
            self.needs_current_allocation(),
            self.needs_current_ep_scores(),
            self.metrics_snapshot(),
            self._ep_tail_risk and not self.needs_current_ep_margin(),
            self.needs_current_ep_margin(),
        )

    def needs_prediction(self) -> bool:
        return self.fixed_cap < 0 and self._ep_policy is None

    def can_reuse_dense_fallback(self) -> bool:
        return self._ep_policy is not None and not self._ep_policy.needs_measurement()

    def predict(self, ids: list[int], step: int, logits: torch.Tensor) -> None:
        if not ids or not self.needs_prediction():
            return
        # The draft sampler is top-1 without request temperature or penalties.
        # Estimate the distribution that produced those proposals, not Target's.
        if any(request_id not in self.registered for request_id in ids):
            return
        self.controller.predict(ids, step, logits)

    def _prepare_ep_verify_rows(self, requested: float, plan_start: float) -> None:
        batch = len(self.ids)
        dense = batch * (self.steps + 1)
        full_bucket = self._cost_key(dense)
        target = batch * requested
        # Full width competes using its logical rows. A smaller budget is only
        # useful if it is closer to the request and runs a smaller graph.
        candidates = [dense]
        candidates.extend(
            rows
            for rows in self._capture_buckets
            if batch <= rows < dense and rows < full_bucket
        )
        bucket = min(candidates, key=lambda rows: (abs(rows - target), -rows))
        snapshots = [self._host_predictions.get(request_id) for request_id in self.ids]
        sparse = bucket < dense and all(
            snapshot is not None and 0 <= self.round - snapshot[0] <= 16
            for snapshot in snapshots
        )
        # A partial budget is committed only after current Draft scores re-rank
        # it. Reuse a shape-only prefix plan before Draft instead of sorting
        # stale scores and uploading caps that will immediately be replaced.
        planned_rows = bucket if sparse else dense
        cap, remainder = divmod(planned_rows - batch, max(batch, 1))
        caps = (cap + 1,) * remainder + (cap,) * (batch - remainder)
        self._prepared = self.controller.prepare_from_host(
            self.ids,
            confidence=None,
            draft_cost_ms=np.empty(0),
            verify_cost_ms=np.empty(0),
            explicit_caps=caps,
        )
        self.total = self._prepared[2]
        self._calibration = False
        zero, full = self.controller.host_cap_counts
        mode = 2 if sparse else 3
        self._metrics = (mode, zero, full, (time.perf_counter() - plan_start) * 1e6)

    def _prepare_plan(self) -> None:
        plan_start = time.perf_counter()
        if self._ep_verify_rows_per_request is not None:
            self._prepare_ep_verify_rows(self._ep_verify_rows_per_request, plan_start)
            return
        batch = len(self.ids)
        if self._ep_policy is not None:
            total, mode = self._ep_policy.begin(self.ids)
            if self._ep_policy.needs_measurement() and len(self._ep_pending) < 16:
                if self._ep_tail_risk:
                    self._prepare_acceptance_feedback()
                else:
                    self._start_round_timing()
            partial = total < self._ep_policy.dense
            self._prepared = self.controller.prepare_from_host(
                self.ids,
                confidence=(
                    self._ep_policy.confidence(self.ids)
                    if partial and not self._ep_tail_risk
                    else None
                ),
                draft_cost_ms=np.empty(0),
                verify_cost_ms=np.empty(0),
                fixed_budget=None if self._ep_tail_risk else total - batch,
                grouped_tail_cut=self._ep_current_scores and partial,
                explicit_caps=(
                    self._ep_policy.capacities if self._ep_tail_risk else None
                ),
            )
            self.total = total
            if (
                self._ep_margin_rerank
                and batch == 14
                and partial
                and self._ep_policy.margin_prior is not None
            ):
                self._ep_margin_prior = torch.tensor(
                    self._ep_policy.margin_prior,
                    dtype=torch.float32,
                    pin_memory=True,
                ).to(self.controller.device, non_blocking=True)
            self._calibration = mode == 1
            zero, full = self.controller.host_cap_counts
            self._metrics = (mode, zero, full, (time.perf_counter() - plan_start) * 1e6)
            return
        costs = self._decision_costs(batch)
        cap = self.fixed_cap
        calibration_budget = None
        if cap < 0:
            in_flight = {
                total for size, total, _, _ in self.pending_costs if size == batch
            }
            # Sample each reachable execution bucket, not just K+1 uniform
            # request depths. Use the fullest logical shape in each bucket.
            representatives = {}
            sample_rows = (
                range(batch, batch * (self.steps + 1) + 1)
                if self._has_capture_buckets
                else (batch * (depth + 1) for depth in range(self.steps + 1))
            )
            for total in sample_rows:
                representatives[self._cost_key(total)] = total
            missing = [key for key in representatives if key not in costs]
            available = [key for key in missing if key not in in_flight]
            if available:
                calibration_budget = representatives[max(available)] - batch
            elif missing:
                # GPU samples are still in flight. Do not wait on the host.
                calibration_budget = batch * self.steps
        draft = np.zeros(batch + 1)
        verify = np.ones(batch * (self.steps + 1) + 1)
        if cap < 0 and calibration_budget is None:
            points = sorted(costs)
            # Captured graphs execute discrete padded row counts. Interpolating
            # invents savings while replay can still use the larger graph.
            # Unknown sizes pay the higher adjacent measured cost. Preserve exact
            # measurements: real kernels need not have monotonic latency, and
            # inflating the full-cap baseline can falsely favor a smaller cap.
            measured = np.asarray([costs[n] for n in points])
            # Every live row count in one graph bucket pays the same cost,
            # including when the full-cap logical count itself needs padding.
            sizes = self._execution_rows[batch : len(verify)]
            buckets = np.minimum(
                np.searchsorted(points, sizes, side="left"), len(points) - 1
            )
            exact = np.asarray(points)[buckets] == sizes
            ceiling = np.maximum(
                measured[buckets], measured[np.maximum(buckets - 1, 0)]
            )
            verify[batch:] = np.where(exact, measured[buckets], ceiling)
        confidence = None
        snapshots = [self._host_predictions.get(request_id) for request_id in self.ids]
        fresh = [
            snapshot is not None and 0 <= self.round - snapshot[0] <= 16
            for snapshot in snapshots
        ]
        if any(fresh):
            # A new or stale request must not discard other requests' learned
            # predictions. Unknown rows use optimistic acceptance, as in vLLM;
            # the cost model still decides the batch budget. Keep the existing
            # full-width fallback when no request has a usable prediction.
            confidence = np.ones((batch, self.steps), dtype=np.float64)
            for index, (snapshot, ready) in enumerate(zip(snapshots, fresh)):
                if ready:
                    confidence[index] = snapshot[1]
        self._prepared = self.controller.prepare_from_host(
            self.ids,
            confidence=confidence if calibration_budget is None else None,
            draft_cost_ms=draft,
            verify_cost_ms=verify,
            fixed_cap=cap if cap >= 0 else None,
            fixed_budget=calibration_budget,
        )
        self.total = self._prepared[2]
        self._calibration = calibration_budget is not None
        mode = (
            0
            if self.fixed_cap >= 0
            else (1 if self._calibration else (3 if confidence is None else 2))
        )
        zero, full = self.controller.host_cap_counts
        self._metrics = (mode, zero, full, (time.perf_counter() - plan_start) * 1e6)

    def metrics_snapshot(self) -> tuple[int, int, int, float]:
        """Host-only decision counters; no GPU reductions or synchronization."""
        return self._metrics

    def prepared_plan(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._prepared is None:
            raise RuntimeError("begin must precede adaptive planning")
        return self._prepared

    def needs_current_ep_margin(self) -> bool:
        return self._ep_margin_prior is not None

    def needs_current_ep_scores(self) -> bool:
        return self.needs_current_ep_margin() or (
            self._ep_current_scores
            and self._ep_policy is not None
            and 0 < self._ep_policy.dense - self.total <= len(self.ids)
        )

    def needs_current_allocation(self) -> bool:
        return self.needs_current_ep_scores() or (
            self._ep_policy is None
            and self._metrics[0] == 2
            and len(self.ids) < self.total < len(self.ids) * (self.steps + 1)
        )

    def plan(
        self, ep_scores: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        caps, rows, total = self.prepared_plan()
        if self.needs_current_ep_margin():
            if ep_scores is not None:
                caps, rows = pack_ep_margin_rows(
                    ep_scores,
                    self._ep_margin_prior,
                    caps,
                    total,
                    four_gaps=self._ep_margin_four_gaps,
                )
                self._ep_reallocated_caps = caps
        elif self.needs_current_ep_scores():
            if ep_scores is not None:
                caps, rows = pack_ep_tail_rows(ep_scores, caps, total)
                self._ep_reallocated_caps = caps
        elif self.needs_current_allocation():
            caps, rows, total = self.controller.reallocate_current(self.ids, total)
        self._prepared = caps, rows, total
        self.controller.commit_prepared(self.ids, caps)
        logging.debug(
            "[adaptive-verify] round=%d batch=%d draft_steps=%d target_rows=%d fixed_cap=%d calibration=%s",
            self.round,
            len(self.ids),
            self.steps,
            self.total,
            self.fixed_cap,
            self._calibration,
        )
        return caps, rows, total

    def finish(self, host_elapsed_ms: float | None = None) -> None:
        """Record after draft-prefill, sampling and dispatch, without waiting.

        This is a round service-cost estimate, not a client latency measurement.
        The caller may include host gather/dispatch time excluded by begin().
        """
        if self.events is None:
            return
        elapsed = (time.perf_counter() - self._cpu_round_start) * 1000
        if (
            host_elapsed_ms is not None
            and math.isfinite(host_elapsed_ms)
            and host_elapsed_ms >= 0
        ):
            elapsed = host_elapsed_ms
        # A wall-clock adjustment in C++ telemetry must not fail a valid forward
        # on rank0 while its TP peers continue. Fall back to our monotonic clock.
        if self._ep_policy is None:
            self.events[1].record()
            self.pending_costs.append(
                (len(self.ids), self._cost_key(self.total), self.events, elapsed)
            )
        elif self._ep_accepted is not None:
            self.events[1].record()
            self._ep_pending.append(
                (
                    self._ep_policy.epoch,
                    tuple(self.ids),
                    self.total,
                    self.events,
                    elapsed,
                    *self._ep_accepted,
                )
            )
        else:
            self.event_pool.append(self.events)
        self.events = None

    def observe_emitted_tokens(
        self,
        emitted: torch.Tensor,
        forced: list[bool],
        effective_capacities: torch.Tensor | None = None,
    ) -> None:
        if self._ep_tail_risk:
            self._observe(emitted, forced, effective_capacities, bonus=1)
        else:
            self.observe(emitted - 1, forced, effective_capacities)

    def observe(
        self,
        accepted: torch.Tensor,
        forced: list[bool],
        effective_capacities: torch.Tensor | None = None,
    ) -> None:
        self._observe(accepted, forced, effective_capacities)

    def _observe(
        self,
        accepted: torch.Tensor,
        forced: list[bool],
        effective_capacities: torch.Tensor | None,
        *,
        bonus: int = 0,
    ) -> None:
        if self._ep_policy is None:
            completed_ids = None
            if self._ep_verify_rows_per_request is not None:
                # observe() clears the readiness mask. A completed copy alone
                # does not mean a new request has a prediction for every hop.
                slots = self.controller._slots(self.ids)
                completed_ids = tuple(
                    request_id
                    for request_id, slot in zip(self.ids, slots)
                    if self.controller._ready[slot].all()
                )
            self.controller.observe(
                self.ids,
                accepted,
                forced,
                asynchronous=True,
                effective_capacities=effective_capacities,
            )
            if self.needs_prediction():
                self._queue_predictions(completed_ids)
        else:
            planned = self.controller.consume_measured_plan(
                self.ids, accepted, forced, effective_capacities
            )
            if self.events is not None:
                self._queue_ep_feedback(
                    accepted, effective_capacities, planned, any(forced), bonus
                )
        self.round += 1
        self._prepared = None

    def _queue_ep_feedback(
        self,
        accepted: torch.Tensor,
        effective_capacities: torch.Tensor | None,
        planned: np.ndarray,
        has_forced: bool,
        bonus: int,
    ) -> None:
        actual_caps = self._ep_reallocated_caps
        columns = (
            3 if actual_caps is not None else (1 if effective_capacities is None else 2)
        )
        shape = (columns, len(self.ids))
        dtype = accepted.dtype
        if effective_capacities is not None:
            dtype = torch.promote_types(dtype, effective_capacities.dtype)
        host = next(
            (
                buffer
                for buffer in self._ep_buffers
                if buffer.shape == shape and buffer.dtype == dtype
            ),
            None,
        )
        if host is None:
            host = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            self._ep_buffers = [b for b in self._ep_buffers if b is not host]
        # finish() records readiness on this same decode stream. No
        # synchronization or logits-sized feedback is needed.
        host[0].copy_(accepted, non_blocking=True)
        if actual_caps is not None:
            effective = (
                actual_caps if effective_capacities is None else effective_capacities
            )
            host[1].copy_(effective, non_blocking=True)
            host[2].copy_(actual_caps, non_blocking=True)
        elif effective_capacities is not None:
            host[1].copy_(effective_capacities, non_blocking=True)
        self._ep_accepted = (host, planned, has_forced, bonus)


def create_session(
    capacity: int,
    steps: int,
    device: str,
    capture_rows: list[int] | None = None,
    ep_size: int = 1,
    allow_ep_current_scores: bool = False,
) -> AdaptiveVerificationSession:
    return AdaptiveVerificationSession(
        capacity, steps, device, capture_rows, ep_size, allow_ep_current_scores
    )
