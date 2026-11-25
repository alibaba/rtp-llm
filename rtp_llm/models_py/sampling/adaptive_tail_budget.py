"""Choose Target verification prefixes from completed acceptance feedback."""

from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from math import isfinite


@dataclass(frozen=True)
class TailBudget:
    rows: int
    caps: tuple[int, ...]
    lost_per_request: float


class RollingAcceptance:
    """Count complete prefix observations without rescanning the past window."""

    def __init__(self, prior, *, observations=8, prior_weight=0.5, decay=1.0):
        if (
            not prior
            or type(observations) is not int
            or observations < 1
            or not isfinite(prior_weight)
            or prior_weight <= 0
            or not isfinite(decay)
            or not 0 < decay <= 1
            or any(not isfinite(value) or not 0 <= value <= 1 for value in prior)
        ):
            raise ValueError(
                "use a positive observation window and valid conditional prior"
            )
        self.history = deque(maxlen=observations)
        self.chances = [0] * len(prior)
        self.accepted = [0] * len(prior)
        self.prior_weight = prior_weight
        self.decay = decay
        self.expired_weight = decay**observations
        self.prior_successes = tuple(prior_weight * value for value in prior)

    def _update(self, successes, cap, delta):
        # A miss is known only if a verified position failed. Reaching a
        # partial cap supplies no evidence on the next position.
        for hop in range(min(cap, successes + 1)):
            self.chances[hop] += delta
            self.accepted[hop] += delta * (hop < successes)

    def observe(self, successes, cap):
        if not 0 <= successes <= cap <= len(self.chances):
            raise ValueError("feedback must be within its actual verification cap")
        if self.decay < 1:
            for hop in range(len(self.chances)):
                self.chances[hop] *= self.decay
                self.accepted[hop] *= self.decay
        if len(self.history) == self.history.maxlen:
            first_successes, first_cap = self.history[0]
            self._update(first_successes, first_cap, -self.expired_weight)
        self.history.append((successes, cap))
        self._update(successes, cap, 1)

    def survival(self):
        product = 1.0
        probabilities = []
        for actual, opportunity, prior in zip(
            self.accepted, self.chances, self.prior_successes
        ):
            # Bound tiny round-off from expiring weighted observations.
            opportunity = max(0, opportunity)
            actual = max(0, min(actual, opportunity))
            product *= (actual + prior) / (opportunity + self.prior_weight)
            probabilities.append(product)
        return tuple(probabilities)


def survival_from_history(history, prior, *, observations=8, prior_weight=0.5):
    """Estimate P(accepted >= hop) for an offline replay."""
    window = RollingAcceptance(
        prior, observations=observations, prior_weight=prior_weight
    )
    for successes, cap in history[-observations:]:
        window.observe(successes, cap)
    return window.survival()


def choose_tail_budget(
    probabilities, widths, max_mean_loss, *, max_request_loss=None, known=None
):
    """Minimize exact row width within a predicted output-loss budget.

    Every request keeps its mandatory Target row. Removed positions are always
    a suffix. Greedy removal is exact because survival probabilities are
    non-increasing within each request, so removing deeper positions is cheaper.
    """
    batch = len(probabilities)
    if not batch or not probabilities[0]:
        raise ValueError("verification requires at least one request and draft")
    steps = len(probabilities[0])
    if any(
        len(p) != steps
        or any(not isfinite(v) or not 0 <= v <= 1 for v in p)
        or any(left < right for left, right in zip(p, p[1:]))
        for p in probabilities
    ):
        raise ValueError("each request needs non-increasing survival probabilities")
    if not isfinite(max_mean_loss) or max_mean_loss < 0:
        raise ValueError("the mean expected output loss must be nonnegative")
    if max_request_loss is not None and (
        not isfinite(max_request_loss) or max_request_loss < 0
    ):
        raise ValueError("the per-request expected output loss must be nonnegative")

    full_width = batch * (steps + 1)
    candidate_widths = set(widths)
    if full_width not in candidate_widths or not all(
        type(width) is int and batch <= width <= full_width
        for width in candidate_widths
    ):
        raise ValueError("provide exact captured widths, including the full width")
    known = (True,) * batch if known is None else tuple(known)
    if len(known) != batch:
        raise ValueError("known mask must match the request list")

    result = TailBudget(full_width, (steps,) * batch, 0.0)
    caps = [steps] * batch
    total_lost = 0.0
    request_lost = [0.0] * batch
    tails = [(p[-1], i) for i, p in enumerate(probabilities) if known[i]]
    heapify(tails)
    for width in range(full_width - 1, min(candidate_widths) - 1, -1):
        while tails:
            cost, index = heappop(tails)
            if (
                max_request_loss is None
                or request_lost[index] + cost <= max_request_loss
            ):
                break
        else:
            break
        if total_lost + cost > max_mean_loss * batch:
            break
        total_lost += cost
        request_lost[index] += cost
        caps[index] -= 1
        if caps[index]:
            heappush(tails, (probabilities[index][caps[index] - 1], index))
        if width in candidate_widths:
            result = TailBudget(width, tuple(caps), total_lost / batch)
    return result


class TailRiskEpPolicy:
    """Use rolling acceptance with a conservative per-request loss limit."""

    _OBSERVATION_WINDOW = 32
    _MIN_OBSERVATIONS = 16
    _REFERENCE_INTERVAL = 32
    _MAX_MEAN_LOSS = 0.04
    _MAX_LOSS_PER_REQUEST = 0.08
    _NEAR_FULL_WIDTH = 82
    _NEAR_FULL_MEAN_LOSS = 0.08
    _NEAR_FULL_REQUEST_LOSS = 0.18
    _BUCKET_WIDTHS = (56, 64)
    _QUOTA_WIDTHS = (48, 56, 64, 72)
    _BUCKET_DENSE_WIDTH = 84

    @classmethod
    def valid_bucket_row_quota(cls, quota):
        # A mandatory dense reference occurs once in each 32-step period.
        lower, upper = (
            ((cls._REFERENCE_INTERVAL - 1) * width + cls._BUCKET_DENSE_WIDTH)
            / cls._REFERENCE_INTERVAL
            for width in (cls._QUOTA_WIDTHS[0], cls._QUOTA_WIDTHS[-1])
        )
        return isfinite(quota) and lower <= quota <= upper

    def __init__(
        self,
        steps,
        execution_rows,
        *,
        near_full=False,
        row_price=None,
        prior=None,
        bucket_bid_price=None,
        bucket_row_quota=None,
    ):
        if row_price is not None and (
            near_full or not isfinite(row_price) or not 0 < row_price < 1
        ):
            raise ValueError(
                "row price must be between zero and one and exclude near-full mode"
            )
        if bucket_bid_price is not None and (
            row_price is None
            or not isfinite(bucket_bid_price)
            or not 0 < bucket_bid_price < 1
        ):
            raise ValueError(
                "bucket bid price requires a row price and must be between zero and one"
            )
        if bucket_row_quota is not None and (
            row_price is None
            or bucket_bid_price is not None
            or not self.valid_bucket_row_quota(bucket_row_quota)
        ):
            raise ValueError(
                "bucket row quota requires a row price, no bucket bid price and "
                "a feasible mean paid row count"
            )
        self.steps = steps
        self._row_price = row_price
        self._bucket_bid_price = bucket_bid_price
        self._bucket_row_quota = bucket_row_quota
        self._bucket_row_debt = None
        self._bucket_widths = ()
        self._prior = tuple(prior) if prior is not None else (1.0,) * steps
        if len(self._prior) != steps or any(
            not isfinite(value) or not 0 <= value <= 1 for value in self._prior
        ):
            raise ValueError("the prior needs one probability for each draft hop")
        self.execution_rows = execution_rows
        self._near_full = near_full
        self.ids = ()
        self.epoch = 0
        self.phase = 0
        self.dense = 0
        self._widths = ()
        self._history = {}
        self.capacities = ()
        self.margin_prior = None

    def begin(self, ids):
        cohort = tuple(sorted(ids))
        if cohort != self.ids:
            self.epoch += 1
            self.ids = cohort
            self.dense = len(ids) * (self.steps + 1)
            self._history = {
                key: self._history[key] for key in cohort if key in self._history
            }
            if self._row_price is not None:
                self.phase = 0
                self._bucket_row_debt = None
                candidate_widths = (
                    self._QUOTA_WIDTHS
                    if self._bucket_row_quota is not None
                    else self._BUCKET_WIDTHS
                )
                self._bucket_widths = (
                    candidate_widths
                    if (
                        self._bucket_bid_price is not None
                        or self._bucket_row_quota is not None
                    )
                    and self.steps == 5
                    and self.dense == self._BUCKET_DENSE_WIDTH
                    and all(
                        self.execution_rows[rows] == rows
                        for rows in (*candidate_widths, self.dense)
                    )
                    else ()
                )
                self._widths = self._bucket_widths or tuple(
                    rows
                    for rows in range(2 * len(ids), self.dense)
                    if self.execution_rows[rows] < self.execution_rows[self.dense]
                )
            elif self._near_full:
                # This B14/MTP5 experiment admits only the 82-row
                # candidate. Do not reduce a different batch or pad to 84 rows.
                rows = self._NEAR_FULL_WIDTH
                self._widths = (
                    (rows,)
                    if self.steps == 5
                    and self.dense == rows + 2
                    and self.execution_rows[self.dense] == self.dense
                    and self.execution_rows[rows] == rows
                    else ()
                )
            else:
                minimum = max(len(ids), (2 * self.dense + 2) // 3)
                self._widths = tuple(
                    rows
                    for rows in range(minimum, self.dense)
                    if self.execution_rows[rows] == rows
                    and self.execution_rows[self.dense] - rows >= 4
                )

        self.capacities = (self.steps,) * len(ids)
        self.margin_prior = None
        if not self.needs_measurement():
            return self.dense, 3

        self.phase += 1
        records = [self._history.get(key) for key in ids]
        known = [
            item is not None and len(item.history) >= self._MIN_OBSERVATIONS
            for item in records
        ]
        if (
            not any(known)
            or (self._bucket_widths and not all(known))
            or self.phase % self._REFERENCE_INTERVAL == 0
        ):
            if self._bucket_row_debt is not None:
                self._bucket_row_debt += self.dense - self._bucket_row_quota
            return self.dense, 1

        plans = [
            item.survival() if ready else (1.0,) * self.steps
            for item, ready in zip(records, known)
        ]
        if self._bucket_widths:
            return self._begin_bucket(plans)
        if self._row_price is not None:
            # One shared price lets lower-confidence requests give up more
            # positions without fixing the row count or adding a collective.
            caps = tuple(
                max(1, sum(prob > self._row_price for prob in plan)) for plan in plans
            )
            rows = len(ids) + sum(caps)
            if self.execution_rows[rows] < self.execution_rows[self.dense]:
                self.capacities = caps
                if all(known) and 2 * len(ids) < rows:
                    self.margin_prior = tuple(plans)
                return rows, 2
            return self.dense, 3

        max_mean_loss = (
            self._NEAR_FULL_MEAN_LOSS if self._near_full else self._MAX_MEAN_LOSS
        )
        max_request_loss = (
            self._NEAR_FULL_REQUEST_LOSS
            if self._near_full
            else self._MAX_LOSS_PER_REQUEST
        )
        budget = choose_tail_budget(
            plans,
            (*self._widths, self.dense),
            max_mean_loss,
            max_request_loss=max_request_loss,
            known=known,
        )
        self.capacities = budget.caps
        return budget.rows, 2 if budget.rows < self.dense else 3

    def _begin_bucket(self, plans):
        # One bonus row and at least one draft verification per request.
        # Descending survival and stable ties retain a prefix for every request.
        tails = sorted(
            (-probability, index)
            for index, plan in enumerate(plans)
            for probability in plan[1:]
        )
        minimum = 2 * len(plans)
        if self._bucket_row_quota is None:
            lower, upper = self._bucket_widths
            # Price only the eight additional positions before current Draft.
            bid = -sum(
                probability
                for probability, _ in tails[lower - minimum : upper - minimum]
            ) / (upper - lower)
            rows = upper if bid >= self._bucket_bid_price else lower
        else:
            rows = self._select_quota_width(tails, minimum)
        caps = [1] * len(plans)
        for _, index in tails[: rows - minimum]:
            caps[index] += 1
        self.capacities = tuple(caps)
        self.margin_prior = tuple(plans)
        return rows, 2

    def _select_quota_width(self, tails, minimum):
        debt = self._bucket_row_debt or 0.0
        target = self._bucket_row_quota - debt
        candidates = [
            width for width in self._bucket_widths if abs(width - target) <= 16
        ]
        if candidates:
            # Staying below budget makes verified rows cheaper on the next step.
            price = 0.25 + debt / 64
            rows = max(
                candidates,
                key=lambda width: -sum(
                    probability for probability, _ in tails[: width - minimum]
                )
                - price * (width - minimum),
            )
        else:
            # A dense reference can temporarily exceed the soft debt band.
            rows = min(self._bucket_widths, key=lambda width: abs(width - target))
        self._bucket_row_debt = debt + rows - self._bucket_row_quota
        return rows

    def needs_measurement(self):
        return bool(self._widths)

    def record(self, epoch, ids, total, ms, values, effective_caps=None):
        if not isfinite(ms) or ms <= 0:
            return
        self.record_acceptance(epoch, ids, total, values, effective_caps)

    def record_acceptance(self, epoch, ids, total, values, effective_caps=None):
        if epoch != self.epoch or not self.needs_measurement():
            return
        if len(ids) != len(values) or tuple(sorted(ids)) != self.ids:
            return
        # An unclipped fallback describes a dense native validation round.
        if effective_caps is None:
            if total != self.dense:
                return
            effective_caps = (self.steps,) * len(ids)
        if len(effective_caps) != len(ids):
            return
        for key, accepted, cap in zip(ids, values, effective_caps):
            if not 0 <= accepted <= cap <= self.steps:
                continue
            history = self._history.get(key)
            if history is None:
                # Unobserved suffix positions start optimistic. A request with
                # poor acceptance must earn confidence from its own feedback.
                price_mode = self._row_price is not None
                history = RollingAcceptance(
                    self._prior,
                    observations=self._OBSERVATION_WINDOW,
                    prior_weight=1.0 if price_mode else 0.5,
                    decay=0.5 ** (1 / 8) if price_mode else 1.0,
                )
                self._history[key] = history
            history.observe(int(accepted), int(cap))
