"""Graded assertion infrastructure for result-property test cases (task #61).

Balance/affinity cases assert *result properties* (P-series) rather than
mechanism narratives.  A property is measured once per case (or once per
in-case variant — "batched assertions"), evaluated against a three-tier band
table, and rolled up into:

  * per-property achieved grade — the tightest tier the measured value hits
    (``strict`` / ``normal`` / ``loose``, or ``fail`` when it exceeds even the
    loose floor);
  * per-case achieved grade — the worst tier across the case's properties;
  * suite-level overall verdict — 优异 (excellent) / 良好 (good) /
    边缘 (marginal) / 不可用 (unusable).

Run-grade semantics (``--grade`` on the runner): the runner is started at one
grade and every property must be met *at that grade or tighter*; exceeding the
run grade's bound fails the case while the achieved grade still records what
the system actually demonstrated.  strict at run time ⇒ only strict-level
values pass (达到=优异); normal (default) ⇒ standard bounds (达到=良好);
loose ⇒ floor bounds, the widest that can still pronounce 不可用.

Hard invariants (P2 no-starvation, P6 completeness) carry no band: any
violation is an immediate 不可用 regardless of the run grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

GRADES = ("strict", "normal", "loose")

# Rank for "worst tier" comparisons; fail sits below everything.
_GRADE_RANK = {"strict": 3, "normal": 2, "loose": 1, "fail": 0}

VERDICT_EXCELLENT = "excellent"  # 优异 — every graded case achieved strict
VERDICT_GOOD = "good"  # 良好 — every graded case achieved >= normal
VERDICT_MARGINAL = "marginal"  # 边缘 — some case only reached loose (fill-in
#   between the user-mandated three tiers; no case exceeded loose)
VERDICT_UNUSABLE = "unusable"  # 不可用 — some case exceeded loose / broke an invariant

VERDICT_LABELS = {
    VERDICT_EXCELLENT: "优异",
    VERDICT_GOOD: "良好",
    VERDICT_MARGINAL: "边缘",
    VERDICT_UNUSABLE: "不可用",
}

# ---------------------------------------------------------------------------
# Property band table (single source of truth — cases reference, never
# re-hardcode).  Initial values calibrated where noted; re-calibrate from
# observed run data and update the annotation (task #61 discipline).
#
# kind:
#   upper    — measured value must stay <= band (share / ratio / multiplier)
#   lower    — measured value must stay >= band (hit-rate style)
#   invariant — hard contract, no band: violation is directly unusable
# ---------------------------------------------------------------------------

GRADE_BANDS: Dict[str, dict] = {
    # P1 request-uniformity max-share (fraction of requests on the busiest
    # engine).  Calibration: 2 engines, 20 serial samples, uniform random
    # tie-window sampling — P(share > 0.85) = 2 * P(X >= 17 | B(20, .5))
    # ~= 0.26% (< 1% mandated false-fail floor); 0.75 corresponds to
    # P(X >= 15) ~= 4.1%, 0.65 to P(X >= 13) ~= 26% (strict tier is a
    # quality bar, not a statistical guarantee).
    # end-to-end observed (task #61, balance_uniform_serial, 4 profiles x normal):
    # plain 0.50-0.70 (13/20..15/20), speed_hetero 0.50 — tiers separate
    # exactly as the binomial model predicts (strict sometimes, normal
    # sometimes, never near loose).
    "P1": {
        "kind": "upper",
        "bands": {"strict": 0.65, "normal": 0.75, "loose": 0.85},
    },
    # P2 no-starvation: hard invariant (an engine receiving zero of the
    # offered homogeneous traffic is starved).
    "P2": {"kind": "invariant"},
    # P3 token-weighted max-share (client-side Σ input_len per engine / total).
    # First end-to-end calibration (task #62, balance_len_mixed — bimodal 5-wave, per
    # wave 2 long @32768..49152 + 6 short @512, all prefills set_perf 3s):
    #   batch-window 0.507, single-nonbatch 0.528, single-batch 0.517,
    #   window-nonbatch 0.502 — all four inside the predicted 0.5 ± 0.02.
    # Calibration rationale: the wave choreography deterministically pairs
    # each long request with ~1.5 shorts on its engine and ~4.5 shorts on the
    # other (ledger diversion), so the aggregate share concentrates near 0.5
    # with only the uniform-split tail adding binomial noise — the design
    # bands (0.65/0.70/0.80) keep ~0.15 of headroom above the deterministic
    # baseline, tolerating a whole wave's diversion failing before the normal
    # tier trips; request-count balance is deliberately NOT asserted (it
    # genuinely conflicts with token balance in this scene).
    "P3": {
        "kind": "upper",
        "bands": {"strict": 0.65, "normal": 0.70, "loose": 0.80},
    },
    # P5 overload-avoidance hot-engine share (fraction of the wave landing
    # on the deliberately overloaded engine; 0 = deterministic avoidance).
    # end-to-end observed (task #61, balance_overload_avoid_prefill, 4 profiles): 0.0
    # every run — ledger-priced avoidance is deterministic; the nonzero
    # tiers only tolerate an in-flight straggler racing the injection
    # snapshot.  (Decode-KV caliber uses a case override: delta bands
    # 0/1/2 — see balance_overload_avoid_decode.)
    "P5": {
        "kind": "upper",
        "bands": {"strict": 0.0, "normal": 0.05, "loose": 0.10},
    },
    # P6 completeness: hard invariant (every issued request reaches a
    # terminal state — completed, no loss/hang).
    "P6": {"kind": "invariant"},
    # P7 short-request protection: TTFT (or, under BATCH dispatch, completion
    # duration — see balance_overload_avoid_prefill) as a multiple of the
    # unloaded baseline.
    # end-to-end observed (task #61, balance_overload_avoid_prefill): 0.97 (batch-window
    # completion-duration caliber, wave_max 0.152s vs base 0.157s) — with
    # successful avoidance the wave rides the cool engine and the ratio
    # hovers near 1.0; a swallowed request pays the hot engine's ~5s and
    # blows past every tier.
    "P7": {
        "kind": "upper",
        "bands": {"strict": 2.0, "normal": 3.0, "loose": 5.0},
    },
    # P9 affinity fidelity (fraction of prefix-reuse requests that land on
    # the engine holding the prefix cache).
    # end-to-end observed (task #61, kv_prefix_stickiness, 4 profiles): 10/10
    # hits every run — cache-affinity leader selection is deterministic in
    # the serial single-family form; the lower tiers tolerate tie-window
    # overrides observed historically in concurrent forms.
    "P9": {
        "kind": "lower",
        "bands": {"strict": 0.95, "normal": 0.90, "loose": 0.80},
    },
    # M2 concentration cap (upper bound on the hot-family holder's TOTAL
    # request share).  First end-to-end calibration: kv_hot_prefix_tension — 70%
    # family traffic pinned to the holder + 30% uniform free flow.  The
    # holder's share = (29 + k)/41 with k ~ B(12, .5) over the free requests
    # scattered onto it (29 = seed + 28 continuations under perfect P9
    # stickiness), i.e. expected ~0.854 ± 0.042 (1σ).
    # First measured (task #62, four profiles): 0.805 / 0.902 / 0.902 / 0.829
    # (batch-window / single-nonbatch / single-batch / window-nonbatch; the
    # free flow scattered 4/12, 8/12, 8/12, 5/12 onto the holder — all inside
    # 2σ of the binomial model; an extra batch-window run measured 0.878,
    # free 7/12, same distribution).
    # Band derivation (false-fail probabilities from B(12, .5)):
    #   strict 0.88 -> P(k >= 8)  ≈ 19.4% (quality bar, same philosophy as
    #                                P1's strict: not a statistical guarantee)
    #   normal 0.93 -> P(k >= 10) ≈ 1.93% (standard regression stays green)
    #   loose   0.96 -> P(k >= 11) ≈ 0.317% (< the 1% false-fail floor)
    # The bands police the CAP only — a share far above ~0.9 means the free
    # flow collapsed onto the holder (tie-window spread gone wrong), while
    # P9 separately polices the floor (stickiness itself).
    "M2": {
        "kind": "upper",
        "bands": {"strict": 0.88, "normal": 0.93, "loose": 0.96},
    },
    # M3 hit-tier concentration (lower bound on the same-engine share of the
    # full-hit / half-hit tiers, kv_match_mixed).  Design values (task #62):
    # the estimate discount (0.7 * hitTokens ms — ~5.0s full-hit, ~2.9s
    # half-hit) dwarfs the tie window (~0.3s), so a correct affinity router
    # concentrates both tiers deterministically; a value near 0.5 is the
    # zero-hit baseline (no affinity signal at all).
    # First measured (task #62, four profiles): 1.00 / 1.00 on every profile
    # for BOTH tiers — concentration is fully deterministic (same caliber as
    # P9's 10/10 in task #61); the lower tiers tolerate tie-window overrides
    # observed historically in concurrent forms.
    "M3": {
        "kind": "lower",
        "bands": {"strict": 0.8, "normal": 0.7, "loose": 0.6},
    },
}


@dataclass
class PropertyResult:
    """One measured property (or one batch/variant of it)."""

    prop: str  # "P1".."P9", "M2"
    context: str  # in-case batch label ("" when the case measures it once)
    value: Optional[float]  # None for invariant checks without a scalar
    achieved: str  # strict|normal|loose|fail  (invariant: strict|fail)
    passed: bool  # within the run grade's bound
    detail: str


def _resolve_bands(
    prop: str,
    override: Optional[dict],
    relax: int,
) -> dict:
    """Effective bands for *prop* after per-case override/relaxation.

    ``override`` replaces the tier values outright (unit conversions, e.g.
    P5 as an absolute delta).  ``relax`` shifts every tier right by N steps
    (concurrent-burst scenes inherit the loose floor — the loose bound
    itself never widens past the calibrated floor).
    """
    spec = GRADE_BANDS[prop]
    if spec["kind"] == "invariant":
        raise ValueError(f"property {prop} is an invariant and has no bands")
    bands = dict(override or spec["bands"])
    if relax > 0:
        ordered = list(GRADES)  # strict, normal, loose
        shifted = {}
        for i, tier in enumerate(ordered):
            shifted[tier] = bands[ordered[min(i + relax, len(ordered) - 1)]]
        bands = shifted
    return bands


def _achieved_for_value(kind: str, bands: dict, value: float) -> str:
    if kind == "upper":
        if value <= bands["strict"]:
            return "strict"
        if value <= bands["normal"]:
            return "normal"
        if value <= bands["loose"]:
            return "loose"
        return "fail"
    # lower
    if value >= bands["strict"]:
        return "strict"
    if value >= bands["normal"]:
        return "normal"
    if value >= bands["loose"]:
        return "loose"
    return "fail"


def _passed_for_run_grade(achieved: str, run_grade: str) -> bool:
    return _GRADE_RANK[achieved] >= _GRADE_RANK[run_grade]


@dataclass
class GradeReport:
    """Per-case collector: batched property checks + achieved roll-up.

    Cases call :meth:`check` / :meth:`invariant` as many times as needed
    (per variant / per wave — each call is reported independently), then
    return ``report.finish(detail)`` to produce the runner's
    ``(passed, detail, report)`` tuple.
    """

    run_grade: str = "normal"
    results: List[PropertyResult] = field(default_factory=list)

    def check(
        self,
        prop: str,
        value: float,
        *,
        context: str = "",
        detail: str = "",
        bands: Optional[dict] = None,
        relax: int = 0,
    ) -> bool:
        """Evaluate a band property at the run grade.

        Returns True when the value is within the run grade's bound.  The
        achieved tier (tightest tier the value hits) is always recorded.
        """
        spec = GRADE_BANDS[prop]
        if spec["kind"] == "invariant":
            raise ValueError(f"property {prop} is invariant — use report.invariant()")
        eff = _resolve_bands(prop, bands, relax)
        achieved = _achieved_for_value(spec["kind"], eff, value)
        passed = _passed_for_run_grade(achieved, self.run_grade)
        bound_note = (
            f"bands={ {t: eff[t] for t in GRADES} }"
            f"{' (case override)' if bands else ''}"
            f"{' (relaxed +%d)' % relax if relax else ''}"
        )
        self.results.append(
            PropertyResult(
                prop=prop,
                context=context,
                value=value,
                achieved=achieved,
                passed=passed,
                detail=f"{detail} {bound_note}".strip(),
            )
        )
        return passed

    def invariant(
        self, prop: str, ok: bool, *, context: str = "", detail: str = ""
    ) -> bool:
        """Record a hard-invariant property.  Any violation is unusable."""
        if GRADE_BANDS[prop]["kind"] != "invariant":
            raise ValueError(f"property {prop} has bands — use report.check()")
        self.results.append(
            PropertyResult(
                prop=prop,
                context=context,
                value=None,
                achieved="strict" if ok else "fail",
                passed=ok,
                detail=detail,
            )
        )
        return ok

    # -- roll-up -----------------------------------------------------------

    @property
    def achieved(self) -> str:
        """Worst tier across all recorded checks ('strict' when only passing
        invariants are present — a hard invariant is the tightest contract)."""
        if not self.results:
            return "strict"
        return min((r.achieved for r in self.results), key=lambda g: _GRADE_RANK[g])

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        """Compact per-property line for case detail output."""
        parts = []
        for r in self.results:
            label = r.prop + (f"[{r.context}]" if r.context else "")
            value = "n/a" if r.value is None else f"{r.value:.3f}"
            parts.append(f"{label}={value}:{r.achieved}{'✓' if r.passed else '✗'}")
        return " ".join(parts)

    def finish(self, detail: str):
        """Runner-facing return value: (passed, detail, self)."""
        return self.passed, detail, self

    def to_dict(self) -> dict:
        return {
            "run_grade": self.run_grade,
            "achieved": self.achieved,
            "properties": [
                {
                    "prop": r.prop,
                    "context": r.context,
                    "value": r.value,
                    "achieved": r.achieved,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }


def overall_verdict(achieved_per_case: List[str]) -> Optional[str]:
    """Suite-level verdict from the achieved grades of graded cases.

    * every case achieved strict  -> excellent (优异)
    * every case >= normal        -> good (良好)
    * any case failed (beyond loose / invariant break) -> unusable (不可用)
    * otherwise (some loose, none failed) -> marginal (边缘)
    """
    if not achieved_per_case:
        return None
    ranks = [_GRADE_RANK[a] for a in achieved_per_case]
    if min(ranks) == 0:
        return VERDICT_UNUSABLE
    if min(ranks) == _GRADE_RANK["strict"]:
        return VERDICT_EXCELLENT
    if min(ranks) >= _GRADE_RANK["normal"]:
        return VERDICT_GOOD
    return VERDICT_MARGINAL
