"""Warmup precision-gate consumer: the decode gate (dispatcher's upfront
precision filter).

Consumes the records produced by the warmup framework. For each decode candidate
backend, at bs=1 across the full kv grid times all layers, it compares against
golden layer by layer and forms a global AND to produce a set of passing
backends, which it hands to the dispatcher so that eligible = passed intersected
with support(bs).

Key points:
  - Only the decode phase is judged; prefill keeps fixed priority and does not
    enter the gate.
  - Each backend is a global AND over (kv bucket x layer); failing any bucket at
    any layer eliminates the backend overall.
  - A backend must be verified at least once to be eligible to PASS; a backend
    with no decode record cannot pass the gate just by "not failing".
  - The output is (passed, detail, verified); the gate only produces an
    allowlist, it does not select a backend or fall back, and the empty set is
    handled by the dispatcher's fallback.
  - Under TP, each rank runs locally and the intersection is taken across ranks.
    The intersection's bitmask encode/decode (gate_to_mask / mask_to_gate) lives
    in this module and is pure CPU; the actual NCCL-holding
    reduce_gate_across_tp lives in backend_selector, and this module stays pure
    CPU to keep it unit-testable.

This module imports no rtp_llm or CUDA modules, runs on GPU-less machines and in
CI, and reuses the metrics from precision_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    evaluate_fp8_quality,
    evaluate_precision,
)

GOLDEN = "golden"
DECODE = "decode"


# ─── Input contract mirror (aligned with the warmup framework's fields) ────────
# Only mirror the fields the consumer actually uses; q/k/v do not participate in
# the gate's judgment (which looks at output), kept to match the contract.


@dataclass
class AttentionLayerRecord:
    layer_idx: int
    output: torch.Tensor  # [T, H*D] (already flattened by the wrapper)
    q: Optional[torch.Tensor] = None  # [T, H, D]
    k: Optional[torch.Tensor] = None  # [T, KVH, D]
    v: Optional[torch.Tensor] = None  # [T, KVH, D]
    is_prefill: bool = False
    sequence_lengths: Optional[torch.Tensor] = None
    input_lengths: Optional[torch.Tensor] = None
    prefix_lengths: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None


@dataclass
class AttentionForwardRecord:
    scenario_name: str  # "p0_c512_d1::XQADecodeImpl::decode_0"
    impl_name: str
    phase: str  # "plain" | "prefix" | "decode"
    layer_records: Dict[int, AttentionLayerRecord]
    head_num: int = 0
    kv_head_num: int = 0
    head_dim: int = 0
    dtype: Optional[torch.dtype] = None


# records: scenario_base -> impl_name -> [AttentionForwardRecord]
Records = Dict[str, Dict[str, List["AttentionForwardRecord"]]]


# ─── Per-layer judgment ────────────────────────────────────────────────────────


@dataclass
class LayerVerdict:
    impl: str
    scenario: str
    phase: str
    layer_idx: int
    kv_len: int
    overall_pass: bool
    cos_sim: float
    nrmse: float
    mean_ulp: float
    snr: float
    snr_regime: str
    fail_reason: str


def _normalize_kv_dtype(kv_cache_dtype) -> str:
    """Normalize to 'BASE' (strict bf16 bucket) or 'FP8' (loose quantization-quality bucket).

    Accepts a string ('BASE'/'BF16'/'FP8'/...) or an enum with a .name
    (rtp_llm KvCacheDataType). INT8 explicitly raises NotImplementedError
    (consistent with GoldenCacheWriter).
    """
    name = getattr(kv_cache_dtype, "name", kv_cache_dtype)
    s = str(name).upper()
    if "FP8" in s:
        return "FP8"
    if "INT8" in s:
        raise NotImplementedError(
            "decode gate: INT8 KV cache not supported (only BASE/FP8)"
        )
    if "BASE" in s or "BF16" in s or "BFLOAT16" in s:
        return "BASE"
    raise ValueError(f"decode gate: unrecognized kv_cache_dtype={kv_cache_dtype!r}")


def _infer_kv_len(lr: AttentionLayerRecord) -> int:
    """kv_len for diagnostics: decode uses sequence_lengths (current sequence length = history + current)."""
    sl = lr.sequence_lengths
    if sl is not None and hasattr(sl, "numel") and sl.numel() > 0:
        return int(sl.flatten()[0].item())
    return 0


def judge_layer(
    golden: AttentionForwardRecord,
    cand: AttentionForwardRecord,
    layer_idx: int,
    kv_dtype: str,
) -> LayerVerdict:
    """Single-layer golden vs candidate judgment. Judged as evaluate(candidate, golden):

    golden is the reference (b), so the SNR's ref_rms = RMS(golden). BASE uses the
    three-tier metrics (cos+nrmse+mean_ulp + SNR gate); FP8 uses cos+nrmse
    (+SNR gate, mean_ulp does not participate).
    """
    g_lr = golden.layer_records[layer_idx]
    c_lr = cand.layer_records[layer_idx]
    a = c_lr.output  # candidate
    b = g_lr.output  # golden = reference

    if kv_dtype == "FP8":
        m = evaluate_fp8_quality(a, b)
    else:
        m = evaluate_precision(a, b, "BF16")

    return LayerVerdict(
        impl=cand.impl_name,
        scenario=_scenario_base(cand.scenario_name),
        phase=cand.phase,
        layer_idx=layer_idx,
        kv_len=_infer_kv_len(g_lr),
        overall_pass=m["overall_pass"],
        cos_sim=m["cos_sim"],
        nrmse=m["nrmse"],
        mean_ulp=m["mean_ulp"],
        snr=m["snr"],
        snr_regime=m["snr_regime"],
        fail_reason=m["fail_reason"],
    )


def _scenario_base(scenario_name: str) -> str:
    """'p0_c512_d1::XQADecodeImpl::decode_0' -> 'p0_c512_d1'."""
    return scenario_name.split("::", 1)[0] if "::" in scenario_name else scenario_name


def _decode_record(
    recs: Sequence[AttentionForwardRecord],
) -> Optional[AttentionForwardRecord]:
    """Single-step pick: get this impl's decode record for this scenario (at most one decode_0)."""
    for r in recs:
        if r.phase == DECODE:
            return r
    return None


def _decode_candidate_names(records: Records) -> List[str]:
    """All non-golden impl names that have a decode record in any scenario (deduplicated, stably sorted)."""
    names: List[str] = []
    seen = set()
    for bucket in records.values():
        for impl, recs in bucket.items():
            if impl == GOLDEN or impl in seen:
                continue
            if _decode_record(recs) is not None:
                seen.add(impl)
                names.append(impl)
    return sorted(names)


# ─── Gate: build_decode_gate ───────────────────────────────────────────────────


@dataclass
class GateResult:
    passed: (
        frozenset  # set of passing backends (the decision; dispatcher consumes this)
    )
    detail: Dict[
        str, List[LayerVerdict]
    ]  # per-impl per-layer verdict (diagnostics/alerts)
    verified: frozenset = field(
        default_factory=frozenset
    )  # backends actually verified (passed is a subset of verified)

    def failures(self) -> Dict[str, List[LayerVerdict]]:
        """Per-eliminated-backend failing-layer details (for alerting/locating)."""
        out: Dict[str, List[LayerVerdict]] = {}
        for impl, vs in self.detail.items():
            if impl in self.passed:
                continue
            bad = [v for v in vs if not v.overall_pass]
            if bad:
                out[impl] = bad
        return out


def build_decode_gate(records: Records, kv_cache_dtype) -> GateResult:
    """For each decode candidate backend, produce pass/fail via a global AND over (kv bucket x layer).

    Args:
        records: warmup output, scenario_base -> impl -> [AttentionForwardRecord].
        kv_cache_dtype: 'BASE'/'FP8' or an rtp_llm KvCacheDataType enum.
    Returns:
        GateResult(passed=frozenset, detail=..., verified=...).
    """
    kv_dtype = _normalize_kv_dtype(kv_cache_dtype)
    passed = set()
    verified = set()
    detail: Dict[str, List[LayerVerdict]] = {}

    for impl in _decode_candidate_names(records):
        impl_ok = True
        impl_verified = False
        for scenario, bucket in records.items():
            golden = _decode_record(bucket.get(GOLDEN, []))
            cand = _decode_record(bucket.get(impl, []))
            if golden is None or cand is None:
                continue  # this kv point lacks golden or this backend -> not judged here
            impl_verified = True
            for layer_idx in sorted(golden.layer_records):
                if layer_idx not in cand.layer_records:
                    # structural missing layer: golden has it, candidate does not -> judged as failure
                    v = LayerVerdict(
                        impl=impl,
                        scenario=scenario,
                        phase=DECODE,
                        layer_idx=layer_idx,
                        kv_len=_infer_kv_len(golden.layer_records[layer_idx]),
                        overall_pass=False,
                        cos_sim=float("nan"),
                        nrmse=float("nan"),
                        mean_ulp=float("nan"),
                        snr=float("nan"),
                        snr_regime="N/A",
                        fail_reason="candidate missing layer present in golden",
                    )
                else:
                    v = judge_layer(golden, cand, layer_idx, kv_dtype)
                detail.setdefault(impl, []).append(v)
                impl_ok = (
                    impl_ok and v.overall_pass
                )  # global AND (no break, collect full detail for diagnostics)
        if impl_verified:
            verified.add(impl)
            if impl_ok:
                passed.add(impl)

    return GateResult(
        passed=frozenset(passed), detail=detail, verified=frozenset(verified)
    )


def merge_tp_gates(per_rank: Sequence[frozenset]) -> frozenset:
    """Cross-rank merge: take the intersection of each rank's passed set (failing on any rank fails overall).

    A pure-CPU semantic reference (tp=1 / for unit tests); in production the
    cross-rank merge is done by the integration-layer reduce_gate_across_tp
    (backend_selector, holds NCCL) via bitmask all_reduce(SUM), not through this
    function (it does not gather strings).
    """
    if not per_rank:
        return frozenset()
    out = set(per_rank[0])
    for s in per_rank[1:]:
        out &= set(s)
    return frozenset(out)


# ─── Cross-rank bitmask encode/decode (pure CPU, called by integration-layer reduce_gate_across_tp) ──
# The candidate list (DECODE_MHA_IMPS) is identical and ordered across all ranks,
# so position is identity: encode the frozenset into a fixed-length bitmask, do one
# all_reduce(SUM) on the mask, and the bits == tp are the intersection. A few dozen
# bytes, no pickle/variable-length/gloo.


def gate_to_mask(s: frozenset, registry: Sequence[str]) -> List[int]:
    """frozenset -> fixed-length 0/1 mask (position is identity; registry is the all-rank ordered registry class names)."""
    return [1 if n in s else 0 for n in registry]


def mask_to_gate(
    passed_sum: Sequence[int],
    verified_sum: Sequence[int],
    registry: Sequence[str],
    tp: int,
):
    """Decode after all_reduce(SUM): reduced==tp means all ranks passed = the intersection; 0<v<tp marks asymmetric verification (alert and exclude).

    Returns (merged: frozenset, asym: List[str]).
    """
    out, asym = set(), []
    for i, n in enumerate(registry):
        if passed_sum[i] == tp and verified_sum[i] == tp:
            out.add(n)
        elif 0 < verified_sum[i] < tp:
            asym.append(n)
    return frozenset(out), asym
