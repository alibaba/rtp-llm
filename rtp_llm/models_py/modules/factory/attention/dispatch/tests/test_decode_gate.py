"""decode_gate consumer unit tests -- synthetic records verify the gate logic is correct (pure CPU, CI-ready).

Does not depend on upstream warmup code: hand-build AttentionForwardRecord with
known "good/bad" tensors, covering all of the consumer's judgment/aggregation
paths (per-layer -> kv bucket -> per-impl AND -> frozenset -> TP intersection ->
bitmask encode/decode). The judgment kernel itself (three-tier metrics + SNR
gate) is calibrated on real data; here we only verify the consumer's
alignment/aggregation/gate logic.
"""

import unittest
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    GOLDEN,
    AttentionForwardRecord,
    AttentionLayerRecord,
    _normalize_kv_dtype,
    build_decode_gate,
    gate_to_mask,
    mask_to_gate,
    merge_tp_gates,
)


@contextmanager
def _assert_raises(exc):
    """Minimal stand-in for pytest.raises (bazel py_test has no pytest)."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


H, KVH, D = 8, 8, 128
HD = H * D
NLAYERS = 4


def _seed():
    torch.manual_seed(1234)


def _layer(layer_idx, output, kv_len):
    return AttentionLayerRecord(
        layer_idx=layer_idx,
        output=output,
        sequence_lengths=torch.tensor([kv_len], dtype=torch.int32),
        is_prefill=False,
    )


def _fwd(impl, layer_outputs, kv_len, scenario_base):
    lrs = {li: _layer(li, out, kv_len) for li, out in layer_outputs.items()}
    return AttentionForwardRecord(
        scenario_name=f"{scenario_base}::{impl}::decode_0",
        impl_name=impl,
        phase="decode",
        layer_records=lrs,
        head_num=H,
        kv_head_num=KVH,
        head_dim=D,
        dtype=torch.bfloat16,
    )


def _golden_layers():
    out = {}
    for li in range(NLAYERS):
        t = torch.randn(1, HD)
        out[li] = (t + torch.sign(t) * 0.5).to(torch.bfloat16)
    return out


def _add_noise(t, rel):
    f = t.float()
    rms = f.pow(2).mean().sqrt()
    return (f + torch.randn_like(f) * rms * rel).to(t.dtype)


def _scale(t, factor):
    return (t.float() * factor).to(t.dtype)


def _page_corrupt(t, n=64):
    out = t.clone().float()
    out[0, :n] = torch.randn(n) * 5.0
    return out.to(t.dtype)


def _head_collapse(t, head=3):
    out = t.clone().float().reshape(1, H, D)
    rms = t.float().pow(2).mean().sqrt()
    out[0, head] = torch.randn(D) * rms * 3.0
    return out.reshape(1, HD).to(t.dtype)


def _clean_candidate(golden_layers, kv_len, scenario, impl, rel=0.003):
    return _fwd(
        impl,
        {li: _add_noise(g, rel) for li, g in golden_layers.items()},
        kv_len,
        scenario,
    )


def _records_one_scenario(golden_layers, kv_len, scenario, cand_map):
    bucket = {GOLDEN: [_fwd(GOLDEN, golden_layers, kv_len, scenario)]}
    for impl, rec in cand_map.items():
        bucket[impl] = [rec]
    return {scenario: bucket}


def test_clean_candidate_passes():
    _seed()
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, "p0_c512_d1", "XQADecodeImpl")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"XQADecodeImpl": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"XQADecodeImpl"})
    assert "XQADecodeImpl" in res.verified
    assert isinstance(res.passed, frozenset)


def test_identical_candidate_passes():
    _seed()
    gl = _golden_layers()
    cand = _fwd("XQAImpl", {li: g.clone() for li, g in gl.items()}, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"XQAImpl": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"XQAImpl"})


def test_scale_error_fails():
    _seed()
    gl = _golden_layers()
    cand = _fwd(
        "Buggy", {li: _scale(g, 1.05) for li, g in gl.items()}, 512, "p0_c512_d1"
    )
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert "Buggy" in res.verified
    assert "Buggy" in res.failures()


def test_page_corruption_fails():
    _seed()
    gl = _golden_layers()
    cand = _fwd(
        "Buggy", {li: _page_corrupt(g) for li, g in gl.items()}, 512, "p0_c512_d1"
    )
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    assert build_decode_gate(recs, "BASE").passed == frozenset()


def test_single_head_collapse_fails():
    _seed()
    gl = _golden_layers()
    cand = _fwd(
        "Buggy", {li: _head_collapse(g) for li, g in gl.items()}, 512, "p0_c512_d1"
    )
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    assert build_decode_gate(recs, "BASE").passed == frozenset()


def test_and_across_layers():
    _seed()
    gl = _golden_layers()
    layers = {li: _add_noise(g, 0.003) for li, g in gl.items()}
    layers[2] = _scale(gl[2], 1.05)
    cand = _fwd("Buggy", layers, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    bad_layers = {v.layer_idx for v in res.failures()["Buggy"]}
    assert bad_layers == {2}


def test_and_across_kv_scenarios():
    _seed()
    gl_a = _golden_layers()
    gl_b = _golden_layers()
    cand_a = _clean_candidate(gl_a, 512, "p0_c512_d1", "X")
    cand_b = _fwd(
        "X", {li: _scale(g, 1.05) for li, g in gl_b.items()}, 32768, "p0_c32768_d1"
    )
    recs = {}
    recs.update(_records_one_scenario(gl_a, 512, "p0_c512_d1", {"X": cand_a}))
    recs.update(_records_one_scenario(gl_b, 32768, "p0_c32768_d1", {"X": cand_b}))
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    fail_scenarios = {v.scenario for v in res.failures()["X"]}
    assert "p0_c32768_d1" in fail_scenarios


def test_clean_across_two_kv_passes():
    _seed()
    gl_a, gl_b = _golden_layers(), _golden_layers()
    recs = {}
    recs.update(
        _records_one_scenario(
            gl_a,
            512,
            "p0_c512_d1",
            {"X": _clean_candidate(gl_a, 512, "p0_c512_d1", "X")},
        )
    )
    recs.update(
        _records_one_scenario(
            gl_b,
            32768,
            "p0_c32768_d1",
            {"X": _clean_candidate(gl_b, 32768, "p0_c32768_d1", "X")},
        )
    )
    assert build_decode_gate(recs, "BASE").passed == frozenset({"X"})


def test_unverified_backend_excluded():
    _seed()
    gl = _golden_layers()
    plain_only = AttentionForwardRecord(
        scenario_name="p0_c512_d1::PlainOnly::plain",
        impl_name="PlainOnly",
        phase="plain",
        layer_records={0: _layer(0, gl[0].clone(), 512)},
        head_num=H,
        kv_head_num=KVH,
        head_dim=D,
        dtype=torch.bfloat16,
    )
    recs = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _clean_candidate(gl, 512, "p0_c512_d1", "X")}
    )
    recs["p0_c512_d1"]["PlainOnly"] = [plain_only]
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"X"})
    assert "PlainOnly" not in res.verified
    assert "PlainOnly" not in res.detail


def test_missing_layer_in_candidate_fails():
    _seed()
    gl = _golden_layers()
    layers = {li: _add_noise(g, 0.003) for li, g in gl.items()}
    del layers[3]
    cand = _fwd("X", layers, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"X": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    reasons = [v.fail_reason for v in res.failures()["X"] if v.layer_idx == 3]
    assert reasons and "missing layer" in reasons[0]


def test_fp8_dtype_selects_loose_threshold():
    _seed()
    gl = _golden_layers()
    fp8_noise = {li: _add_noise(g, 0.05) for li, g in gl.items()}

    def fresh_recs():
        return _records_one_scenario(
            gl, 512, "p0_c512_d1", {"X": _fwd("X", dict(fp8_noise), 512, "p0_c512_d1")}
        )

    assert build_decode_gate(fresh_recs(), "BASE").passed == frozenset()
    assert build_decode_gate(fresh_recs(), "FP8").passed == frozenset({"X"})


def test_multiple_candidates_partition():
    _seed()
    gl = _golden_layers()
    good = _clean_candidate(gl, 512, "p0_c512_d1", "Good")
    bad = _fwd("Bad", {li: _scale(g, 1.05) for li, g in gl.items()}, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Good": good, "Bad": bad})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"Good"})
    assert res.verified == frozenset({"Good", "Bad"})


def test_merge_tp_gates_intersection():
    assert merge_tp_gates([frozenset({"A", "B"}), frozenset({"B", "C"})]) == frozenset(
        {"B"}
    )
    assert merge_tp_gates([frozenset({"A", "B"}), frozenset({"A", "B"})]) == frozenset(
        {"A", "B"}
    )
    assert merge_tp_gates([frozenset({"A"}), frozenset()]) == frozenset()
    assert merge_tp_gates([]) == frozenset()
    assert merge_tp_gates([frozenset({"A", "B"})]) == frozenset({"A", "B"})


def test_empty_and_golden_missing():
    assert build_decode_gate({}, "BASE").passed == frozenset()
    _seed()
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, "p0_c512_d1", "X")
    recs = {"p0_c512_d1": {"X": [cand]}}
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert res.verified == frozenset()


def test_normalize_kv_dtype():
    assert _normalize_kv_dtype("BASE") == "BASE"
    assert _normalize_kv_dtype("BF16") == "BASE"
    assert _normalize_kv_dtype("FP8") == "FP8"
    assert _normalize_kv_dtype("fp8_e4m3") == "FP8"
    with _assert_raises(NotImplementedError):
        _normalize_kv_dtype("INT8")
    with _assert_raises(ValueError):
        _normalize_kv_dtype("weird")

    class _Enum:
        name = "FP8"

    assert _normalize_kv_dtype(_Enum()) == "FP8"


def test_detail_collected_for_all_layers_even_on_pass():
    _seed()
    gl = _golden_layers()
    recs = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _clean_candidate(gl, 512, "p0_c512_d1", "X")}
    )
    res = build_decode_gate(recs, "BASE")
    assert len(res.detail["X"]) == NLAYERS
    assert all(v.overall_pass for v in res.detail["X"])
    assert res.failures() == {}


# ─── bitmask cross-rank encode/decode (the pure-CPU part of the integration-layer reduce_gate_across_tp) ──────────


def test_gate_to_mask_position_is_identity():
    reg = ["XQAImpl", "XQADecodeImpl", "PyFlashinferDecodeImpl"]
    assert gate_to_mask(frozenset({"XQAImpl", "PyFlashinferDecodeImpl"}), reg) == [
        1,
        0,
        1,
    ]
    assert gate_to_mask(frozenset(), reg) == [0, 0, 0]


def test_mask_to_gate_intersection_and_asym():
    reg = ["A", "B", "C"]
    tp = 2
    # A: passed both ranks (2,2) -> selected; B: only 1 rank passed but both ranks verified (1,2) -> not in intersection, no alert (verified is complete);
    # C: only 1 rank verified (0,1) -> asymmetric verification, alerted and excluded.
    merged, asym = mask_to_gate([2, 1, 0], [2, 2, 1], reg, tp)
    assert merged == frozenset({"A"})
    assert asym == ["C"]


def test_mask_to_gate_all_pass():
    reg = ["A", "B"]
    merged, asym = mask_to_gate([2, 2], [2, 2], reg, 2)
    assert merged == frozenset({"A", "B"})
    assert asym == []


# Bind the module-level test_* functions onto a TestCase so bazel's unittest
# runner (no pytest available) discovers and runs them.
class DecodeGateTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(DecodeGateTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    unittest.main()
