#!/usr/bin/env python3
"""Restricted-library symbolic regression for prefill latency.

This module searches a versioned, FlexLB-compatible feature library with
greedy forward selection.  It deliberately does not perform unrestricted
expression-tree search.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Sequence, TypeVar

LIBRARY_VERSION = "prefill-request-list-v3"
DEFAULT_HINGE_TOKENS = (16384, 32768, 65536, 131072, 262144, 524288)
DEFAULT_EXP_DECAY_TOKENS = (16384, 65536, 262144)
PARSER_OPERATORS = ("+", "-", "*", "/", "^")
PARSER_FUNCTIONS = ("sqrt", "log", "exp", "abs", "max", "min", "pow")
PARSER_AGGREGATES = ("sum",)
PARSER_PER_REQUEST_VARIABLES = (
    "inputTokens",
    "hitCacheTokens",
    "computeTokens",
    "hasHitCache",
)
PARSER_BATCH_VARIABLES = (
    "batchSize",
    "totalInputTokens",
    "totalHitCacheTokens",
    "totalComputeTokens",
    "maxInputTokens",
    "maxComputeTokens",
)
_POWERS = (0.25, 0.75, 1.25, 1.5, 1.75, 2.5)

ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class CandidateTerm:
    name: str
    expression: str
    evaluate: Callable[[object], float]


@dataclass(frozen=True)
class RestrictedSymbolicModel:
    terms: tuple[CandidateTerm, ...]
    coefficients: tuple[float, ...]
    formula: str
    backend: str
    search_report: dict[str, object]

    def predict(self, row: object) -> float:
        return sum(
            coefficient * term.evaluate(row)
            for coefficient, term in zip(self.coefficients, self.terms)
        )


def _variables(row: object, token_unit: int) -> tuple[float, float, float, float]:
    input_len = float(getattr(row, "input_len"))
    cache_len = float(getattr(row, "cache_len"))
    compute_len = input_len - cache_len
    return (
        compute_len / token_unit,
        cache_len / token_unit,
        input_len / token_unit,
        float(cache_len > 0),
    )


def _requests(row):
    return getattr(row, "requests", ()) or ((row.input_len, row.cache_len),)


def _sum_term(row, token_unit, fn):
    return sum(
        fn(_variables(SimpleNamespace(input_len=i, cache_len=c), token_unit))
        for i, c in _requests(row)
    )


def build_candidate_library(
    token_unit: int,
    hinge_tokens: Sequence[int] = DEFAULT_HINGE_TOKENS,
    exp_decay_tokens: Sequence[int] = DEFAULT_EXP_DECAY_TOKENS,
) -> tuple[CandidateTerm, ...]:
    if token_unit <= 0:
        raise ValueError("token_unit must be positive")
    if any(value <= 0 for value in hinge_tokens):
        raise ValueError("hinge token thresholds must be positive")
    if any(value <= 0 for value in exp_decay_tokens):
        raise ValueError("exp decay token scales must be positive")

    u = f"(computeTokens / {token_unit}.0)"
    c = f"(hitCacheTokens / {token_unit}.0)"
    s = f"(inputTokens / {token_unit}.0)"
    h = "hasHitCache"
    terms: list[CandidateTerm] = [CandidateTerm("1", "1", lambda row: 1.0)]

    def add(
        name: str,
        expression: str,
        fn: Callable[[tuple[float, float, float, float]], float],
    ) -> None:
        terms.append(
            CandidateTerm(
                name,
                f"sum({expression})",
                lambda row, fn=fn: _sum_term(row, token_unit, fn),
            )
        )

    add("u", u, lambda values: values[0])
    add("c", c, lambda values: values[1])
    add("hasHitCache", h, lambda values: values[3])
    add("u**2", f"({u} ^ 2)", lambda values: values[0] ** 2)
    add("u*c", f"{u} * {c}", lambda values: values[0] * values[1])
    add("c**2", f"({c} ^ 2)", lambda values: values[1] ** 2)
    add("u**3", f"({u} ^ 3)", lambda values: values[0] ** 3)
    add("u**2*c", f"{u} * {u} * {c}", lambda values: values[0] ** 2 * values[1])
    add("u*c**2", f"{u} * {c} * {c}", lambda values: values[0] * values[1] ** 2)
    add("c**3", f"({c} ^ 3)", lambda values: values[1] ** 3)

    variables = (("u", u, 0), ("c", c, 1), ("s", s, 2))
    for name, expression, index in variables:
        add(
            f"sqrt({name})",
            f"sqrt({expression})",
            lambda values, index=index: math.sqrt(values[index]),
        )
        add(
            f"log1p({name})",
            f"log(1 + {expression})",
            lambda values, index=index: math.log1p(values[index]),
        )
        for power in _POWERS:
            add(
                f"{name}**{power}",
                f"pow({expression}, {power})",
                lambda values, index=index, power=power: values[index] ** power,
            )
        for scale_tokens in exp_decay_tokens:
            scale = scale_tokens / float(token_unit)
            scale_label = f"{scale:.15g}"
            add(
                f"exp(-{name}/{scale_label})",
                f"exp(-{expression} / {scale_label})",
                lambda values, index=index, scale=scale: math.exp(
                    -values[index] / scale
                ),
            )

    add("abs(u-c)", f"abs({u} - {c})", lambda values: abs(values[0] - values[1]))
    add("max(u,c)", f"max({u}, {c})", lambda values: max(values[0], values[1]))
    add("min(u,c)", f"min({u}, {c})", lambda values: min(values[0], values[1]))
    add("u/(1+c)", f"{u} / (1 + {c})", lambda values: values[0] / (1 + values[1]))
    add("c/(1+u)", f"{c} / (1 + {u})", lambda values: values[1] / (1 + values[0]))

    for name, expression, index in (("c", c, 1), ("s", s, 2), ("u", u, 0)):
        add(
            f"u*sqrt({name})",
            f"{u} * sqrt({expression})",
            lambda values, index=index: values[0] * math.sqrt(values[index]),
        )
        add(
            f"u*log1p({name})",
            f"{u} * log(1 + {expression})",
            lambda values, index=index: values[0] * math.log1p(values[index]),
        )

    for threshold_tokens in hinge_tokens:
        threshold = threshold_tokens / float(token_unit)
        label = f"{threshold:.15g}"
        add(
            f"max(u-{label},0)",
            f"max({u} - {label}, 0)",
            lambda values, threshold=threshold: max(values[0] - threshold, 0.0),
        )
        add(
            f"max(s-{label},0)",
            f"max({s} - {label}, 0)",
            lambda values, threshold=threshold: max(values[2] - threshold, 0.0),
        )
    terms.extend(
        [
            CandidateTerm("batchSize", "batchSize", lambda row: float(row.batch_size)),
            CandidateTerm(
                "maxComputeTokens",
                f"(maxComputeTokens / {token_unit}.0)",
                lambda row: max(i - c for i, c in _requests(row)) / token_unit,
            ),
            CandidateTerm(
                "maxInputTokens",
                f"(maxInputTokens / {token_unit}.0)",
                lambda row: max(i for i, c in _requests(row)) / token_unit,
            ),
        ]
    )
    return tuple(terms)


def _formula_text(terms: Sequence[CandidateTerm], coefficients: Sequence[float]) -> str:
    rendered: list[str] = []
    for term, coefficient in zip(terms, coefficients):
        if abs(coefficient) < 1e-14:
            continue
        magnitude = f"{abs(coefficient):.15g}"
        value = (
            magnitude if term.expression == "1" else f"{magnitude} * {term.expression}"
        )
        if not rendered:
            rendered.append(("-" if coefficient < 0 else "") + value)
        else:
            rendered.append((" - " if coefficient < 0 else " + ") + value)
    return "".join(rendered) if rendered else "0"


def fit_restricted_symbolic(
    train_rows: Sequence[ObservationT],
    validation_rows: Sequence[ObservationT],
    refit_rows: Sequence[ObservationT],
    *,
    token_unit: int,
    hinge_tokens: Sequence[int] = DEFAULT_HINGE_TOKENS,
    exp_decay_tokens: Sequence[int] = DEFAULT_EXP_DECAY_TOKENS,
    max_terms: int = 15,
    complexity_tolerance_pct: float = 5.0,
) -> RestrictedSymbolicModel:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "restricted symbolic fitting requires CPU PyTorch"
        ) from error

    if not train_rows or not validation_rows or not refit_rows:
        raise ValueError(
            "restricted symbolic fitting requires non-empty train, validation, and refit rows"
        )
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    if complexity_tolerance_pct < 0:
        raise ValueError("complexity_tolerance_pct must be nonnegative")

    terms = build_candidate_library(token_unit, hinge_tokens, exp_decay_tokens)
    max_terms = min(max_terms, len(terms))

    def matrix(rows: Sequence[ObservationT]):
        values = [[term.evaluate(row) for term in terms] for row in rows]
        for row_index, row_values in enumerate(values):
            for term, value in zip(terms, row_values):
                if not math.isfinite(value):
                    raise ValueError(
                        f"candidate {term.name!r} produced non-finite value "
                        f"at row {row_index}"
                    )
        return torch.tensor(
            values,
            dtype=torch.float64,
            device="cpu",
        )

    x_train = matrix(train_rows)
    x_validation = matrix(validation_rows)
    y_train = torch.tensor(
        [float(getattr(row, "target_ms")) for row in train_rows], dtype=torch.float64
    )
    y_validation = torch.tensor(
        [float(getattr(row, "target_ms")) for row in validation_rows],
        dtype=torch.float64,
    )
    scales = torch.sqrt(torch.mean(x_train * x_train, dim=0)).clamp_min(1.0)
    z_train = x_train / scales
    z_validation = x_validation / scales
    relative_train = z_train / y_train[:, None].clamp_min(1e-12)

    def weights(rows):
        counts = Counter(getattr(row, "batch_size", 1) for row in rows)
        geometry_counts = Counter(
            getattr(row, "geometry_key", "") or str(i) for i, row in enumerate(rows)
        )
        raw = [
            1.0 / geometry_counts[getattr(row, "geometry_key", "") or str(i)]
            for i, row in enumerate(rows)
        ]
        totals = {
            b: sum(w for w, row in zip(raw, rows) if getattr(row, "batch_size", 1) == b)
            for b in counts
        }
        return torch.tensor(
            [w / totals[getattr(row, "batch_size", 1)] for w, row in zip(raw, rows)],
            dtype=torch.float64,
        )

    train_weights = weights(train_rows)
    validation_weights = weights(validation_rows)
    relative_train = relative_train * train_weights.sqrt()[:, None]
    gram = relative_train.T @ relative_train
    rhs = relative_train.T @ train_weights.sqrt()

    def solve(z, y, indices: Sequence[int], row_weights):
        relative_design = z[:, indices] / y[:, None].clamp_min(1e-12)
        return torch.linalg.lstsq(
            relative_design * row_weights.sqrt()[:, None],
            row_weights.sqrt(),
            rcond=1e-10,
        ).solution

    def solve_train(indices: Sequence[int]):
        return torch.linalg.lstsq(
            gram[list(indices)][:, list(indices)],
            rhs[list(indices)],
            rcond=1e-12,
        ).solution

    selected = [0]
    history: list[dict[str, object]] = []
    for _ in range(max_terms):
        beta = solve_train(selected)
        train_relative_mse = float(
            torch.sum(
                train_weights
                * ((z_train[:, selected].mv(beta) - y_train) / y_train) ** 2
            )
            / train_weights.sum()
        )
        validation_relative_mse = float(
            torch.sum(
                validation_weights
                * (
                    ((z_validation[:, selected].mv(beta) - y_validation) / y_validation)
                    ** 2
                )
            )
            / validation_weights.sum()
        )
        history.append(
            {
                "term_count": len(selected),
                "term_names": [terms[index].name for index in selected],
                "train_relative_mse": train_relative_mse,
                "validation_relative_mse": validation_relative_mse,
            }
        )
        if len(selected) >= max_terms:
            break
        choices: list[tuple[float, int]] = []
        for index in range(1, len(terms)):
            if index in selected:
                continue
            candidate = [*selected, index]
            if torch.linalg.matrix_rank(z_train[:, candidate], tol=1e-8) < len(
                candidate
            ):
                continue
            candidate_beta = solve_train(candidate)
            candidate_gram = gram[candidate][:, candidate]
            candidate_rhs = rhs[candidate]
            loss = float(
                (
                    candidate_beta @ candidate_gram @ candidate_beta
                    - 2 * candidate_beta @ candidate_rhs
                    + train_weights.sum()
                )
                / train_weights.sum()
            )
            choices.append((loss, index))
        if not choices:
            break
        selected.append(min(choices, key=lambda item: (item[0], item[1]))[1])

    best_validation = min(float(item["validation_relative_mse"]) for item in history)
    limit = best_validation * (1.0 + complexity_tolerance_pct / 100.0)
    chosen = next(
        item for item in history if float(item["validation_relative_mse"]) <= limit
    )
    chosen_names = set(chosen["term_names"])
    chosen_indices = [
        index for index, term in enumerate(terms) if term.name in chosen_names
    ]

    x_refit = matrix(refit_rows) / scales
    y_refit = torch.tensor(
        [float(getattr(row, "target_ms")) for row in refit_rows], dtype=torch.float64
    )
    beta = solve(x_refit, y_refit, chosen_indices, weights(refit_rows))
    coefficients = tuple(
        float(value) for value in (beta / scales[chosen_indices]).tolist()
    )
    chosen_terms = tuple(terms[index] for index in chosen_indices)
    condition_number = float(torch.linalg.cond(x_refit[:, chosen_indices]).item())
    report: dict[str, object] = {
        "library_version": LIBRARY_VERSION,
        "candidate_count": len(terms),
        "candidate_names": [term.name for term in terms],
        "hinge_tokens": list(hinge_tokens),
        "exp_decay_tokens": list(exp_decay_tokens),
        "parser_coverage": {
            "operators": list(PARSER_OPERATORS),
            "functions": list(PARSER_FUNCTIONS),
            "aggregates": list(PARSER_AGGREGATES),
            "per_request_variables": list(PARSER_PER_REQUEST_VARIABLES),
            "batch_variables": list(PARSER_BATCH_VARIABLES),
        },
        "max_terms": max_terms,
        "complexity_tolerance_pct": complexity_tolerance_pct,
        "selection_rule": "smallest expression within tolerance of best validation relative MSE",
        "objective": "mean_squared_relative_error",
        "feature_scaling": "train RMS, clamped to a minimum of 1",
        "weighting": "equal batch-size weight; equal geometry weight within batch",
        "rank_policy": "skip linearly dependent candidate columns on training data",
        "refit_policy": "selected terms refit on train+validation",
        "test_usage": "test rows are excluded from selection and refit",
        "history": history,
        "selected_terms": [term.name for term in chosen_terms],
        "selected_validation_relative_mse": chosen["validation_relative_mse"],
        "design_condition_number": condition_number,
    }
    return RestrictedSymbolicModel(
        chosen_terms,
        coefficients,
        _formula_text(chosen_terms, coefficients),
        "torch_cpu_forward_selection_relative_wls",
        report,
    )
