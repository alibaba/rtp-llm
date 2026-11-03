"""Exact, production-independent probability oracle for bounded speculation.

Two requests, binary vocabulary, gamma=2. Every elementary draw is enumerated
with Fraction weights, including rejection and residual resampling. A second
round uses the *committed* prefix, so the assertion concerns each request's
first two actual output tokens, not just an accept-count histogram.

The offline scheduler sees all draft tokens. Its independent reference exposes
only the next frontier, selects it, then draws that token and exposes its
successor. Both use (-score, stable_request_id, position). No production module,
Torch, sampled frequency, floating-point tolerance, or production planner is
used as the expected distribution. Deliberately incorrect schedulers below
establish that the oracle detects information leaks and boundary mistakes.
"""

import itertools
import unittest
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

F = Fraction
GAMMA = 2
REQUESTS = (0, 1)
EMPTY = ((), ())


def _binary(probability_one):
    return (1 - probability_one, probability_one)


def _draws(probabilities):
    return [(token, mass) for token, mass in enumerate(probabilities) if mass]


class _Model:
    """Tiny conditional p/q tables; confidence depends on the previous token."""

    def __init__(self, kind):
        self.kind = kind

    def target(self, request, history):
        if self.kind == "fair":
            return _binary(F(1, 2))
        if self.kind in ("target_point", "disjoint_points"):
            return _binary(F((request + len(history)) % 2))
        if not history:
            return _binary((F(1, 3), F(2, 3))[request])
        return _binary(((F(1, 4), F(4, 5)), (F(3, 4), F(2, 5)))[request][history[-1]])

    def draft(self, request, history):
        if self.kind in ("fair", "equal"):
            return self.target(request, history)
        if self.kind in ("draft_point", "disjoint_points"):
            return _binary(F(1 - (request + len(history)) % 2))
        return _binary(F(1, 3) if history and history[-1] else F(2, 3))

    @staticmethod
    def confidence(request, history):
        previous = history[-1] if history else request
        # Includes exact ties at 1; a successor can equal its predecessor.
        return ((F(1), F(1, 4)), (F(1, 2), F(1)))[request][previous]


def _draft_paths(model, request, history, width=GAMMA):
    if not width:
        return [((), F(1))]
    paths = []
    for token, mass in _draws(model.draft(request, history)):
        for suffix, suffix_mass in _draft_paths(
            model, request, history + (token,), width - 1
        ):
            paths.append(((token,) + suffix, mass * suffix_mass))
    return paths


def _offline_lengths(model, histories, proposals, budget, mutation=None):
    candidates = []
    active = [request for request in REQUESTS if len(histories[request]) < 2]
    if mutation == "future_budget":
        # The budget is chosen after seeing the first draft token: forbidden.
        budget = int(proposals[active[0]][0] == 1)
    for request in active:
        score = F(1)
        for position in range(GAMMA):
            history = histories[request] + proposals[request][:position]
            confidence = model.confidence(request, history)
            if mutation == "current_token":
                confidence = F(1) if proposals[request][position] else F(1, 4)
            elif mutation == "bad_tie":
                # A0 = B0 = 1/2; A1 ties only when A's first draw is zero.
                confidence = F(1, 2) if position == 0 else F(not history[-1])
            score *= confidence
            if mutation == "bad_tie":
                # Stable and token-independent, but A1 < B0 < A0 violates
                # predecessor priority. Counting this mask is not a prefix.
                tie_rank = {(0, 1): 0, (1, 0): 1, (0, 0): 2, (1, 1): 3}
                key = (-score, tie_rank[(request, position)])
            else:
                key = (-score, request, position)
            candidates.append((key, request, position))
    selected = sorted(candidates)[: min(budget, len(candidates))]
    lengths = tuple(
        sum(row == request for _, row, _ in selected) for request in REQUESTS
    )
    if mutation != "bad_tie":
        for request in active:
            assert sorted(pos for _, row, pos in selected if row == request) == list(
                range(lengths[request])
            )
    return lengths


def _offline_schedules(model, histories, budget, mutation=None):
    paths = [
        _draft_paths(model, request, history) if len(history) < 2 else [((), F(1))]
        for request, history in enumerate(histories)
    ]
    for combination in itertools.product(*paths):
        proposals = tuple(item[0] for item in combination)
        mass = combination[0][1] * combination[1][1]
        lengths = _offline_lengths(model, histories, proposals, budget, mutation)
        yield proposals, lengths, mass


def _frontier_schedules(model, histories, budget):
    """Construct the law without drawing or inspecting any unselected token."""
    active = [request for request in REQUESTS if len(histories[request]) < 2]

    def visit(prefixes, products, remaining, mass):
        if not remaining:
            yield prefixes, tuple(map(len, prefixes)), mass
            return
        available = []
        for request in active:
            position = len(prefixes[request])
            if position < GAMMA:
                confidence = model.confidence(
                    request, histories[request] + prefixes[request]
                )
                score = products[request] * confidence
                available.append((-score, request, position))
        negative_score, request, _ = min(available)
        conditional = model.draft(request, histories[request] + prefixes[request])
        for token, token_mass in _draws(conditional):
            next_prefixes = list(prefixes)
            next_prefixes[request] += (token,)
            next_products = list(products)
            next_products[request] = -negative_score
            yield from visit(
                tuple(next_prefixes),
                tuple(next_products),
                remaining - 1,
                mass * token_mass,
            )

    yield from visit(EMPTY, (F(1), F(1)), min(budget, GAMMA * len(active)), F(1))


def _selected_prefix_law(schedules):
    result = defaultdict(F)
    for proposals, lengths, mass in schedules:
        result[tuple(row[:length] for row, length in zip(proposals, lengths))] += mass
    return dict(result)


def _verify(model, request, history, proposals, length, wrong_bonus=False):
    """Exact standard p/q rejection, stopping after the required real tokens."""
    needed = 2 - len(history)
    result = defaultdict(F)

    def visit(output, mass):
        if len(output) == needed:
            result[output] += mass
            return
        position = len(output)
        if position == length:
            # Dense target rows beyond L exist but are not the bonus context.
            bonus_prefix = proposals if wrong_bonus else proposals[:length]
            for token, token_mass in _draws(
                model.target(request, history + bonus_prefix)
            ):
                result[output + (token,)] += mass * token_mass
            return
        p = model.target(request, history + output)
        q = model.draft(request, history + output)
        token = proposals[position]
        acceptance = min(F(1), p[token] / q[token])
        if acceptance:
            visit(output + (token,), mass * acceptance)
        if acceptance != 1:
            residual = tuple(max(F(0), p_t - q_t) for p_t, q_t in zip(p, q))
            denominator = sum(residual)
            assert denominator > 0
            for replacement, residual_mass in _draws(residual):
                result[output + (replacement,)] += (
                    mass * (1 - acceptance) * residual_mass / denominator
                )

    visit((), F(1))
    assert sum(result.values()) == 1
    return tuple(result.items())


def _output_law(model, budget, scheduler="offline", mutation=None):
    """Commit accepted/resampled/bonus tokens, then start the next real round."""

    @lru_cache(None)
    def verify(request, history, proposals, length):
        return _verify(
            model, request, history, proposals, length, mutation == "bonus_gamma"
        )

    @lru_cache(None)
    def continue_from(histories):
        if all(len(history) == 2 for history in histories):
            return ((histories, F(1)),)
        if scheduler == "frontier":
            schedules = _frontier_schedules(model, histories, budget)
        else:
            schedules = _offline_schedules(model, histories, budget, mutation)
        result = defaultdict(F)
        for proposals, lengths, schedule_mass in schedules:
            outputs = [
                (
                    verify(request, history, proposals[request], lengths[request])
                    if len(history) < 2
                    else (((), F(1)),)
                )
                for request, history in enumerate(histories)
            ]
            for combination in itertools.product(*outputs):
                committed = tuple(
                    history + item[0] for history, item in zip(histories, combination)
                )
                mass = schedule_mass * combination[0][1] * combination[1][1]
                for final, suffix_mass in continue_from(committed):
                    result[final] += mass * suffix_mass
        assert sum(result.values()) == 1
        return tuple(result.items())

    return dict(continue_from(EMPTY))


def _target_only_law(model):
    # Direct autoregressive product measure; no scheduler or rejector calls.
    result = {}
    for a0, a1, b0, b1 in itertools.product((0, 1), repeat=4):
        mass = (
            model.target(0, ())[a0]
            * model.target(0, (a0,))[a1]
            * model.target(1, ())[b0]
            * model.target(1, (b0,))[b1]
        )
        if mass:
            result[((a0, a1), (b0, b1))] = mass
    return result


def _total_variation(left, right):
    return (
        sum(
            abs(left.get(k, F(0)) - right.get(k, F(0)))
            for k in left.keys() | right.keys()
        )
        / 2
    )


class DSparkBoundedDistributionTest(unittest.TestCase):
    def test_offline_selected_prefix_law_matches_independent_frontier(self):
        for kind in ("different", "equal", "fair", "draft_point", "disjoint_points"):
            model = _Model(kind)
            # Includes re-entering with committed prefixes and one finished row.
            for histories in (EMPTY, ((0,), (1,)), ((1, 0), (0,))):
                for budget in range(2 * GAMMA + 1):
                    with self.subTest(kind=kind, histories=histories, budget=budget):
                        offline = _selected_prefix_law(
                            _offline_schedules(model, histories, budget)
                        )
                        frontier = _selected_prefix_law(
                            _frontier_schedules(model, histories, budget)
                        )
                        self.assertEqual(sum(offline.values()), 1)
                        self.assertEqual(offline, frontier)

    def test_two_real_tokens_per_request_match_target_product_at_all_budgets(self):
        for kind in (
            "different",
            "equal",
            "fair",
            "target_point",
            "draft_point",
            "disjoint_points",
        ):
            model = _Model(kind)
            expected = _target_only_law(model)
            for budget in range(2 * GAMMA + 1):
                for scheduler in ("offline", "frontier"):
                    with self.subTest(kind=kind, budget=budget, scheduler=scheduler):
                        self.assertEqual(
                            _output_law(model, budget, scheduler), expected
                        )

    def test_previous_token_confidence_changes_legal_prefix_allocation(self):
        model = _Model("fair")
        left = ((0, 1), (1, 0))
        right = ((1, 1), (1, 0))
        # A0 is selected before A's token is known; A1 depends on that token.
        self.assertEqual(_offline_lengths(model, EMPTY, left, 1), (1, 0))
        self.assertEqual(_offline_lengths(model, EMPTY, right, 1), (1, 0))
        self.assertEqual(_offline_lengths(model, EMPTY, left, 2), (2, 0))
        self.assertEqual(_offline_lengths(model, EMPTY, right, 2), (1, 1))

    def test_four_invalid_policies_change_actual_output_distribution(self):
        for mutation, kind, budget in (
            ("future_budget", "fair", 1),
            ("current_token", "fair", 1),
            ("bad_tie", "fair", 1),
            ("bonus_gamma", "different", 0),
        ):
            with self.subTest(mutation=mutation):
                model = _Model(kind)
                wrong = _output_law(model, budget, mutation=mutation)
                expected = _target_only_law(model)
                self.assertEqual(sum(wrong.values()), 1)
                self.assertGreater(_total_variation(wrong, expected), 0)

    def test_shared_draft_and_acceptance_uniform_is_a_counterexample(self):
        # q=(1/2,1/2), p=(3/4,1/4). Four equal-width intervals enumerate
        # the SAME U reused for q's inverse CDF and the acceptance test.
        # With independent U's, standard rejection gives p exactly.
        wrong = defaultdict(F)
        p, q = _binary(F(1, 4)), _binary(F(1, 2))
        for u in (F(1, 8), F(3, 8), F(5, 8), F(7, 8)):
            proposal = int(u >= q[0])
            accepted = u < min(F(1), p[proposal] / q[proposal])
            # Rejection residual is the point mass at token 0.
            wrong[proposal if accepted else 0] += F(1, 4)
        self.assertEqual(dict(wrong), {0: F(1)})
        self.assertNotEqual(dict(wrong), dict(enumerate(p)))

    def test_shared_request_seed_preserves_marginals_but_not_joint_law(self):
        wrong = {(0, 0): F(1, 2), (1, 1): F(1, 2)}
        expected = {tokens: F(1, 4) for tokens in itertools.product((0, 1), repeat=2)}
        for request in REQUESTS:
            marginal = defaultdict(F)
            for tokens, mass in wrong.items():
                marginal[tokens[request]] += mass
            self.assertEqual(dict(marginal), {0: F(1, 2), 1: F(1, 2)})
        self.assertEqual(_total_variation(wrong, expected), F(1, 2))


if __name__ == "__main__":
    unittest.main()
