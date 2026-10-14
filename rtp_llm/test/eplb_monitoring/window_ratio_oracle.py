"""Independent pure-Python oracle for EPLB expert-load monitoring.

This module is the reference implementation used by the EPLB monitoring tests.
It intentionally depends on NOTHING outside the Python standard library (no
torch, no numpy) so it can never be "contaminated" by the production code it
validates, and so it runs on any CPU-only machine:

    /opt/conda310/bin/python -I rtp_llm/test/eplb_monitoring/window_ratio_oracle.py

Semantics implemented (from the approved EPLB monitoring contract):

* Counts are physical-expert histograms.  Every actually routed token-expert
  assignment counts, including CUDA-graph padding rows and fake inputs; the
  caller simply must not feed invalid sentinel ids.
* ``StepRecorder`` accumulates one model forward ("step").  Layer chunks may
  be recorded in any number of calls; they accumulate within the step.
* ``finish_step`` folds the step histogram into per-(step, layer, rank)
  totals ``x[t][l][r]`` using an explicit physical-expert -> owner-rank map,
  then clears the step histogram.
* Multiple EP sources each own a ``WindowTracker`` history.  Histories are
  summed element-wise (cross-source SUM) BEFORE any ratio is computed.
* For a window of W finished steps, per layer l:

      moe_lb        = sum_t( max_r x[t,l,r] ) / sum_t( mean_r x[t,l,r] )
      moe_lb_window = max_r( sum_t x[t,l,r] ) / mean_r( sum_t x[t,l,r] )

  If a denominator is 0 (no samples) the layer reports valid=False and its
  ratios are None -- the oracle never fabricates 1.0.
* Layers are always reported independently; there is no cross-layer sum and
  no token-kind/prefill/decode bucketing anywhere in this module.

Ratios are computed with fractions.Fraction so results are exact.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Sequence


def sum_histograms(hists: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    """Element-wise sum of per-source [L][Ephys] histograms."""
    if not hists:
        return []
    num_layers = len(hists[0])
    num_experts = len(hists[0][0])
    out = [[0] * num_experts for _ in range(num_layers)]
    for h in hists:
        assert len(h) == num_layers, "source layer count mismatch"
        for l in range(num_layers):
            assert len(h[l]) == num_experts, "source expert count mismatch"
            for e in range(num_experts):
                out[l][e] += h[l][e]
    return out


class StepRecorder:
    """Accumulates routed token-expert assignments for one forward step."""

    def __init__(self, num_layers: int, num_physical_experts: int):
        self.num_layers = num_layers
        self.num_physical_experts = num_physical_experts
        self._counts: List[List[int]] = [
            [0] * num_physical_experts for _ in range(num_layers)
        ]

    def record(self, layer: int, expert_ids: Sequence[int]) -> None:
        """Record routed assignments for one layer chunk.

        ``expert_ids`` holds one physical expert id per routed token-expert
        pair (a token routed to topk experts contributes topk entries).
        Padding rows that were really routed MUST be included; invalid
        sentinel ids MUST NOT be passed here.
        """
        assert 0 <= layer < self.num_layers, f"bad layer {layer}"
        row = self._counts[layer]
        for e in expert_ids:
            assert 0 <= e < self.num_physical_experts, f"bad expert id {e}"
            row[e] += 1

    def snapshot(self) -> List[List[int]]:
        return [list(r) for r in self._counts]


class WindowTracker:
    """Tracks per-step rank totals and closes windows of W steps.

    One tracker models ONE EP source (one rank's locally produced counts).
    """

    def __init__(
        self,
        num_layers: int,
        num_physical_experts: int,
        expert_to_rank: Sequence[int],
        num_ranks: int,
        window_size: int,
    ):
        assert window_size > 0, "window_size must be > 0 (0 disables monitoring)"
        assert len(expert_to_rank) == num_physical_experts
        for r in expert_to_rank:
            assert 0 <= r < num_ranks, f"bad rank {r}"
        self.num_layers = num_layers
        self.num_physical_experts = num_physical_experts
        self.expert_to_rank = list(expert_to_rank)
        self.num_ranks = num_ranks
        self.window_size = window_size
        self._recorder = StepRecorder(num_layers, num_physical_experts)
        # history[t][l][r]: total routed assignments in step t, layer l,
        # landing on experts owned by rank r (this source only).
        self.history: List[List[List[int]]] = []

    def begin_step(self) -> StepRecorder:
        self._recorder = StepRecorder(self.num_layers, self.num_physical_experts)
        return self._recorder

    def recorder(self) -> StepRecorder:
        return self._recorder

    def finish_step(self) -> None:
        """Fold the current step histogram into history as per-rank totals."""
        step_rank_totals: List[List[int]] = [
            [0] * self.num_ranks for _ in range(self.num_layers)
        ]
        snap = self._recorder.snapshot()
        for l in range(self.num_layers):
            for e in range(self.num_physical_experts):
                if snap[l][e]:
                    step_rank_totals[l][self.expert_to_rank[e]] += snap[l][e]
        self.history.append(step_rank_totals)

    def pending_steps(self) -> int:
        return len(self.history) % self.window_size

    def close_window(self) -> Optional["WindowReport"]:
        """Close the current window if exactly W steps have accumulated.

        Returns None when fewer than W steps are pending (window not due).
        On success the consumed steps are removed from the pending history.
        """
        if len(self.history) < self.window_size:
            return None
        steps = self.history[: self.window_size]
        del self.history[: self.window_size]
        return WindowReport(steps, self.num_layers, self.num_ranks)


def sum_step_histories(
    histories: Sequence[List[List[List[int]]]],
) -> List[List[List[int]]]:
    """Cross-source SUM of per-source histories: x[t][l][r] element-wise.

    All sources must have aligned step ranges (the contract requires EP
    participants to align window begin/end); lengths must match.
    """
    if not histories:
        return []
    num_steps = len(histories[0])
    num_layers = len(histories[0][0])
    num_ranks = len(histories[0][0][0])
    for h in histories:
        assert len(h) == num_steps, "misaligned step ranges across sources"
    out: List[List[List[int]]] = [
        [[0] * num_ranks for _ in range(num_layers)] for _ in range(num_steps)
    ]
    for h in histories:
        for t in range(num_steps):
            for l in range(num_layers):
                for r in range(num_ranks):
                    out[t][l][r] += h[t][l][r]
    return out


class WindowReport:
    """One closed window for ONE source or for the summed multi-source view."""

    def __init__(
        self,
        steps: List[List[List[int]]],
        num_layers: int,
        num_ranks: int,
    ):
        assert len(steps) > 0
        self.steps = steps  # x[t][l][r]
        self.num_steps = len(steps)
        self.num_layers = num_layers
        self.num_ranks = num_ranks

    @classmethod
    def from_sources(cls, reports: Sequence["WindowReport"]) -> "WindowReport":
        """Combine aligned per-source reports by summing counts first."""
        assert reports, "need at least one source report"
        for r in reports:
            assert r.num_steps == reports[0].num_steps, "misaligned windows"
            assert r.num_layers == reports[0].num_layers
            assert r.num_ranks == reports[0].num_ranks
        summed = sum_step_histories([r.steps for r in reports])
        return cls(summed, reports[0].num_layers, reports[0].num_ranks)

    def layer_rank_totals(self, layer: int) -> List[int]:
        """sum_t x[t, layer, r] for each rank r."""
        totals = [0] * self.num_ranks
        for t in range(self.num_steps):
            for r in range(self.num_ranks):
                totals[r] += self.steps[t][layer][r]
        return totals

    def layer_ratios(self, layer: int) -> Dict[str, Optional[Fraction]]:
        """Both window ratios for one layer.

        Returns {"valid": bool, "moe_lb": Fraction|None,
        "moe_lb_window": Fraction|None, "rank_totals": [...],
        "sample_count": int}.
        """
        sum_max = 0
        sum_mean_num = 0  # sum_t sum_r x[t,l,r]; mean = /num_ranks
        for t in range(self.num_steps):
            x = self.steps[t][layer]
            sum_max += max(x)
            sum_mean_num += sum(x)
        rank_totals = self.layer_rank_totals(layer)
        total_samples = sum(rank_totals)

        if sum_mean_num == 0 or total_samples == 0:
            return {
                "valid": False,
                "moe_lb": None,
                "moe_lb_window": None,
                "rank_totals": rank_totals,
                "sample_count": 0,
            }

        # moe_lb = sum_t(max_r x) / sum_t(mean_r x)
        #        = sum_max / (sum_mean_num / num_ranks)
        moe_lb = Fraction(sum_max * self.num_ranks, sum_mean_num)
        # moe_lb_window = max_r(sum_t x) / mean_r(sum_t x)
        moe_lb_window = Fraction(max(rank_totals) * self.num_ranks, total_samples)
        return {
            "valid": True,
            "moe_lb": moe_lb,
            "moe_lb_window": moe_lb_window,
            "rank_totals": rank_totals,
            "sample_count": total_samples,
        }

    def all_layers(self) -> List[Dict[str, Optional[Fraction]]]:
        return [self.layer_ratios(l) for l in range(self.num_layers)]


def _self_check() -> int:
    """Standalone sanity check; exits non-zero on failure."""
    failures = []

    def expect(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # --- Contract acceptance case 1 -------------------------------------
    # W=2, 4 steps with per-rank totals [4,0],[0,4],[2,2],[0,4]
    #   window 1 -> (moe_lb=2,   moe_lb_window=1)
    #   window 2 -> (moe_lb=3/2, moe_lb_window=3/2)
    # Model: 1 layer, 4 physical experts, experts 0,1 -> rank0, 2,3 -> rank1.
    tracker = WindowTracker(
        num_layers=1,
        num_physical_experts=4,
        expert_to_rank=[0, 0, 1, 1],
        num_ranks=2,
        window_size=2,
    )
    steps = [[4, 0], [0, 4], [2, 2], [0, 4]]
    for rank_totals in steps:
        rec = tracker.begin_step()
        # Expand per-rank totals into concrete expert ids: put all of rank0's
        # tokens on expert 0 and all of rank1's on expert 2.
        rec.record(0, [0] * rank_totals[0])
        rec.record(0, [2] * rank_totals[1])
        tracker.finish_step()
    w1 = tracker.close_window()
    w2 = tracker.close_window()
    assert w1 is not None and w2 is not None
    r1 = w1.layer_ratios(0)
    r2 = w2.layer_ratios(0)
    expect("w1.moe_lb", r1["moe_lb"], Fraction(2))
    expect("w1.moe_lb_window", r1["moe_lb_window"], Fraction(1))
    expect("w2.moe_lb", r2["moe_lb"], Fraction(3, 2))
    expect("w2.moe_lb_window", r2["moe_lb_window"], Fraction(3, 2))
    expect("no residual steps", tracker.pending_steps(), 0)

    # --- Contract acceptance case 2 -------------------------------------
    # physical 0,1 -> rank0; 2,3 -> rank1.
    # source0 routes [[2,3],[2,3]] (two tokens each to experts 2 and 3)
    # source1 routes [[0,1]]      (one token to experts 0 and 1)
    # summed histogram = [1,1,2,2]; rank totals = [2,4]; total = 6.
    src0 = StepRecorder(num_layers=1, num_physical_experts=4)
    src1 = StepRecorder(num_layers=1, num_physical_experts=4)
    src0.record(0, [2, 3, 2, 3])
    src1.record(0, [0, 1])
    hist = sum_histograms([src0.snapshot(), src1.snapshot()])
    expect("summed histogram", hist[0], [1, 1, 2, 2])
    expect("histogram total", sum(hist[0]), 6)

    # Same through the window machinery: ratio must come from summed counts.
    t0 = WindowTracker(1, 4, [0, 0, 1, 1], 2, 1)
    t1 = WindowTracker(1, 4, [0, 0, 1, 1], 2, 1)
    t0.begin_step().record(0, [2, 3, 2, 3])
    t0.finish_step()
    t1.begin_step().record(0, [0, 1])
    t1.finish_step()
    rep = WindowReport.from_sources([t0.close_window(), t1.close_window()])
    lr = rep.layer_ratios(0)
    expect("case2 rank_totals", lr["rank_totals"], [2, 4])
    expect("case2 sample_count", lr["sample_count"], 6)
    expect("case2 moe_lb", lr["moe_lb"], Fraction(4 * 2, 6))
    expect("case2 moe_lb_window", lr["moe_lb_window"], Fraction(4 * 2, 6))

    # --- All-zero window -> valid=False ---------------------------------
    tz = WindowTracker(2, 4, [0, 0, 1, 1], 2, 2)
    for _ in range(2):
        tz.begin_step()
        tz.finish_step()
    wz = tz.close_window()
    for l in range(2):
        rz = wz.layer_ratios(l)
        expect(f"zero layer {l} valid", rz["valid"], False)
        expect(f"zero layer {l} moe_lb", rz["moe_lb"], None)
        expect(f"zero layer {l} moe_lb_window", rz["moe_lb_window"], None)

    # --- Layers independent ---------------------------------------------
    tl = WindowTracker(2, 4, [0, 0, 1, 1], 2, 1)
    rec = tl.begin_step()
    rec.record(0, [0, 0, 0, 2])  # layer0: rank totals [3,1]
    rec.record(1, [2, 2, 3, 3])  # layer1: rank totals [0,4]
    tl.finish_step()
    wl = tl.close_window()
    expect("layer0 totals", wl.layer_ratios(0)["rank_totals"], [3, 1])
    expect("layer1 totals", wl.layer_ratios(1)["rank_totals"], [0, 4])
    expect("layer0 moe_lb", wl.layer_ratios(0)["moe_lb"], Fraction(3 * 2, 4))
    expect("layer1 moe_lb", wl.layer_ratios(1)["moe_lb"], Fraction(4 * 2, 4))

    # --- Chunked accumulation within one step ----------------------------
    tc = WindowTracker(1, 4, [0, 0, 1, 1], 2, 1)
    rec = tc.begin_step()
    rec.record(0, [0])
    rec.record(0, [0, 1])  # same layer, second chunk
    rec.record(0, [3])
    tc.finish_step()
    wc = tc.close_window()
    expect("chunked totals", wc.layer_ratios(0)["rank_totals"], [3, 1])

    if failures:
        for f in failures:
            print(f"ORACLE SELF-CHECK FAIL: {f}")
        return 1
    print("ORACLE SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
