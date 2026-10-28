"""Untimed prefix-only precision decomposition; an explicit manual Bazel target.

No production kernel or installed AITER module is modified.
Outputs are evidence, not accuracy acceptance gates. Three controls distinguish:
(1) final O conversion, (2) P conversion before PV, (3) softmax/addressing errors.
Target success means the controlled experiment executed, not accuracy acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import aiter
import torch

from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade_mkt import (
    DIM,
    HEADS,
    KV_HEADS,
    PAGE,
    AttentionAB,
    error_metrics,
    make_case,
)


def rtz_bf16_fp32(tensor):
    """Exact finite FP32 -> BF16 RTZ -> FP32, independent of torch's RN cast."""
    bits = tensor.float().contiguous().view(torch.int32)
    return torch.bitwise_and(bits, -65536).view(torch.float32)


def quantize(tensor, rounding):
    if rounding == "fp32":
        return tensor.float()
    if rounding == "rn":
        return tensor.bfloat16().float()
    if rounding == "rtz":
        return rtz_bf16_fp32(tensor)
    raise ValueError(rounding)


def dense_prefix(case, rows, dtype=torch.float64):
    # Diagnostic intentionally uses one mkt request: shared prefix = 704 tokens.
    shared = (case.request_histories[0] // PAGE) * PAGE
    ids = case.logical_table[0, : shared // PAGE].to(device="cuda", dtype=torch.long)
    k = (
        case.k[ids]
        .transpose(0, 1)
        .flatten(1, 2)
        .repeat_interleave(HEADS // KV_HEADS, 0)
        .to(dtype)
    )
    v = (
        case.v[ids]
        .transpose(0, 1)
        .flatten(1, 2)
        .repeat_interleave(HEADS // KV_HEADS, 0)
        .to(dtype)
    )
    q = case.q[rows].to(dtype)
    logits = torch.einsum("rhd,hld->rhl", q, k) * DIM**-0.5
    lse = logits.logsumexp(-1)
    output = torch.einsum("rhl,hld->rhd", logits.softmax(-1), v)
    return output, lse, q.float(), k.float(), v.float()


def online_model(
    q,
    k,
    v,
    p_rounding,
    o_rounding,
    arithmetic="separate",
    *,
    denom_fma=False,
    normalization="divide",
):
    """CK N=128 online math with a separately specified P and O conversion.

    Emulates the algorithm, not MFMA accumulation order or ISA exp approximation.
    raw QK is an independent oracle, not captured CK intermediate data. Optional
    fma_p/fma_p_alpha modes isolate fused centering in P and then in alpha.
    denom_fma changes only alpha*l + rowsum(P) to one FP32 rounding.
    reciprocal_then_mul changes only final Oacc/l to Oacc * FP32(1/l).
    P is unnormalized exp2(logit - running_max), not globally normalized softmax.
    Its FP32 sum feeds denominator/LSE BEFORE P is rounded for the PV multiply.
    """
    if arithmetic not in ("separate", "fma_p", "fma_p_alpha"):
        raise ValueError(arithmetic)
    if normalization not in ("divide", "reciprocal_then_mul"):
        raise ValueError(normalization)
    rows, heads, dim = q.shape
    acc = torch.zeros((rows, heads, dim), dtype=torch.float32, device=q.device)
    maximum = torch.full((rows, heads), -float("inf"), device=q.device)
    denom = torch.zeros_like(maximum)
    scale = torch.tensor(
        DIM**-0.5 * 1.4426950408889634, dtype=torch.float32, device=q.device
    )
    for start in range(0, k.shape[1], 128):
        raw = torch.einsum("rhd,hld->rhl", q, k[:, start : start + 128])
        new_max = torch.maximum(maximum, raw.max(-1).values)
        # CK FAST_EXP2 uses exp2(scale * raw - scale * max(raw)).
        row_max = scale * new_max
        if arithmetic in ("fma_p", "fma_p_alpha"):
            # The product of two FP32 values fits FP64 exactly; convert only
            # after subtracting the separately rounded FP32 row_max. This
            # emulates one FP32 FMA without assuming torch.addcmul is fused.
            centered = (
                scale.double() * raw.double() - row_max.double()[..., None]
            ).float()
        else:
            centered = scale * raw - row_max[..., None]
        probability = torch.exp2(centered)
        if arithmetic == "fma_p_alpha":
            alpha = torch.exp2(
                (scale.double() * maximum.double() - row_max.double()).float()
            )
        else:
            alpha = torch.exp2(scale * maximum - row_max)
        rowsum = probability.sum(-1)
        if denom_fma:
            # FP32 product is represented exactly in FP64; round once after
            # adding the existing FP32 rowsum, matching FMA's arithmetic.
            # CK async pipeline:1379; M128 bload PC 0x50c8 v_fmac_f32.
            denom = (denom.double() * alpha.double() + rowsum.double()).float()
        else:
            denom = denom * alpha + rowsum
        acc = acc * alpha[..., None] + torch.einsum(
            "rhl,hld->rhd", quantize(probability, p_rounding), v[:, start : start + 128]
        )
        maximum = new_max
    if normalization == "reciprocal_then_mul":
        # Model the rounded FP32 reciprocal with independent FP64 arithmetic,
        # then perform a separate FP32 multiply. This does not emulate the
        # instruction sequence used by HIP to refine a reciprocal estimate.
        # CK async pipeline:1635-1647; M128 bload PC 0x51c8-0x5218
        # div_scale/rcp/FMA/div_fmas/div_fixup, then multiply at PC 0x5220.
        reciprocal = (1.0 / denom.double()).float()
        normalized = acc * reciprocal[..., None]
    else:
        normalized = acc / denom[..., None]
    output = quantize(normalized, o_rounding)
    lse = maximum * scale * 0.6931471805599453 + denom.log()
    return output, lse


def native_prefix(ab, kv=None, table=None, q=None):
    op = ab.on_op
    case = ab.case
    kv = case.kv if kv is None else kv
    table = op.prefix_table if table is None else table
    q = case.q if q is None else q
    k = kv[:, 0].view(kv.shape[0], KV_HEADS, DIM // 8, PAGE, 8)
    v = kv[:, 1].view(kv.shape[0], KV_HEADS, PAGE // 8, DIM, 8)
    return aiter.mha_batch_prefill_func(
        q,
        k,
        v,
        op.prefix_qptr,
        op.prefix_indptr,
        table.flatten(),
        op.BEAMS_PER_TILE,
        op.max_pages * PAGE,
        causal=False,
        return_lse=True,
        out=torch.empty(
            (case.q.shape[0], HEADS, DIM), device="cuda", dtype=torch.bfloat16
        ),
        block_table=table,
        seqlen_k=op.cuts,
    )


def native_lse_rows(lse, rows, all_rows):
    if tuple(lse.shape) != (HEADS, all_rows):
        raise AssertionError(
            f"Unexpected native LSE layout {lse.shape}; do not flatten blindly"
        )
    return lse[:, rows].T


def refresh_physical(case):
    blocks = case.k.shape[0]
    pk = (
        case.k.reshape(blocks, KV_HEADS, PAGE, DIM // 8, 8)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view_as(case.k)
    )
    pv = (
        case.v.reshape(blocks, KV_HEADS, PAGE // 8, 8, DIM)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .view_as(case.v)
    )
    case.packed = torch.stack((pk, pv), 1)
    case.kv.index_copy_(0, case.physical_ids.cuda().long(), case.packed)


def metrics(actual, reference):
    result = error_metrics(actual, reference)
    a, r = actual.double().flatten(), reference.double().flatten()
    result["signed_amplitude_bias"] = (
        (a - r).dot(r) / r.square().sum().clamp_min(1e-30)
    ).item()
    result["bitwise_value_equal_fraction"] = (
        (actual.double() == reference.double()).double().mean().item()
    )
    return result


def scenario(name, high_address=False):
    case = make_case(1, high_address=high_address)
    if name == "uniform_logits_random_v":
        case.q.zero_()
    elif name == "random_logits_unit_v":
        case.v.fill_(1)
        refresh_physical(case)
    elif name == "random_logits_channel_constants":
        # 128 exactly representable BF16 constants provide many overlapping
        # RTZ intervals for the same unrounded probability mass. This separates
        # a ~0.3% P deficit from a tiny FP32-negative error that could also make
        # the single V=1 case round down by one BF16 ULP.
        constants = 1 + torch.arange(DIM, device="cuda", dtype=torch.float32) / 128
        case.v.copy_(constants.to(torch.bfloat16)[None, None, None, :])
        refresh_physical(case)
    elif name == "two_logits_low_class_value_one":
        # CPU-selected BEFORE GPU measurement. All products in QK are exact:
        # one nonzero Q dimension; low/high K = -6.6875/0, low/high V = 1/0.
        # Each 128-token tile (and final 64) has equal counts and max=0, so
        # online alpha is exactly one. Output rounding predictions are well
        # separated from their conversion boundaries (minimum margin .000914).
        case.q.zero_()
        case.q[:, :, 0] = 1
        case.k.zero_()
        case.k[:, :, ::2, 0] = -6.6875
        case.v.zero_()
        case.v[:, :, ::2, :] = 1
        refresh_physical(case)
    elif name != "random_logits_random_v":
        raise ValueError(name)
    ab = AttentionAB(case)
    # Call native prefix directly regardless of production large-pool guard.
    output, lse = native_prefix(ab)
    rows = [0, 1, 126, 127, 128, 249, 250, 383, 384, 498, 499]
    oracle64, oracle_lse64, q, k, v = dense_prefix(case, rows, torch.float64)
    oracle32, oracle_lse32, *_ = dense_prefix(case, rows, torch.float32)
    native = output[rows].float()
    native_lse = native_lse_rows(lse, rows, case.q.shape[0])
    comparisons = {
        "native_o_vs_fp64": metrics(native, oracle64),
        "native_lse_vs_fp64": metrics(native_lse, oracle_lse64),
        "fp32_o_vs_fp64": metrics(oracle32, oracle64),
        "fp32_lse_vs_fp64": metrics(oracle_lse32, oracle_lse64),
        "native_vs_final_rn_only": metrics(native, quantize(oracle32, "rn")),
        "native_vs_final_rtz_only": metrics(native, quantize(oracle32, "rtz")),
    }
    for p_rounding, o_rounding in (
        ("fp32", "fp32"),
        ("fp32", "rtz"),
        ("rn", "rn"),
        ("rn", "rtz"),
        ("rtz", "rn"),
        ("rtz", "rtz"),
    ):
        modeled, modeled_lse = online_model(q, k, v, p_rounding, o_rounding)
        key = f"online_p_{p_rounding}_o_{o_rounding}"
        comparisons["native_vs_" + key] = metrics(native, modeled)
        comparisons[key + "_vs_fp64"] = metrics(modeled, oracle64)
        if p_rounding == "fp32":
            comparisons[key + "_lse_vs_fp64"] = metrics(modeled_lse, oracle_lse64)
    arithmetic_models = {}
    for rounding in ("rtz", "rn"):
        centered_baseline = None
        for arithmetic in ("fma_p", "fma_p_alpha"):
            modeled, modeled_lse = online_model(q, k, v, rounding, rounding, arithmetic)
            key = f"online_p_{rounding}_o_{rounding}_" + arithmetic
            comparisons["native_vs_" + key] = metrics(native, modeled)
            comparisons[key + "_vs_fp64"] = metrics(modeled, oracle64)
            comparisons[key + "_lse_vs_fp64"] = metrics(modeled_lse, oracle_lse64)
            arithmetic_models[key] = {
                "centering": arithmetic,
                "denom_fma": False,
                "normalization": "divide",
            }
            if arithmetic == "fma_p_alpha":
                centered_baseline = modeled
        for denom_fma, normalization, suffix in (
            (True, "divide", "denom_fma"),
            (False, "reciprocal_then_mul", "reciprocal_then_mul"),
            (True, "reciprocal_then_mul", "denom_fma_reciprocal_then_mul"),
        ):
            # Independent toggles against the same P+alpha-FMA model, followed
            # by their combination. Existing separate and centering controls
            # above remain unchanged; no measured error changes any threshold.
            modeled, modeled_lse = online_model(
                q,
                k,
                v,
                rounding,
                rounding,
                "fma_p_alpha",
                denom_fma=denom_fma,
                normalization=normalization,
            )
            key = f"online_p_{rounding}_o_{rounding}_fma_p_alpha_" + suffix
            comparisons["native_vs_" + key] = metrics(native, modeled)
            comparisons[key + "_vs_fp64"] = metrics(modeled, oracle64)
            comparisons[key + "_lse_vs_fp64"] = metrics(modeled_lse, oracle_lse64)
            comparisons[key + "_vs_centering_only_model"] = metrics(
                modeled, centered_baseline
            )
            arithmetic_models[key] = {
                "centering": "fma_p_alpha",
                "denom_fma": denom_fma,
                "normalization": normalization,
            }
    # Exact semantic invariance probes: change only Q stride and physical layout.
    contiguous_output, contiguous_lse = native_prefix(ab, q=case.q.contiguous())
    logical_prefix_table = case.logical_table[
        ab.on_op.qptr_host[:-1].clamp_max(case.q.shape[0] - 1).long()
    ].cuda()
    compact_output, compact_lse = native_prefix(
        ab, kv=case.packed, table=logical_prefix_table
    )
    comparisons["noncontiguous_vs_contiguous_q"] = metrics(output, contiguous_output)
    comparisons["noncontiguous_vs_contiguous_lse"] = metrics(lse, contiguous_lse)
    comparisons["random_pages_vs_compact_pages"] = metrics(output, compact_output)
    comparisons["random_pages_vs_compact_lse"] = metrics(lse, compact_lse)
    result = {
        "evidence": "RTP_MEASURED",
        "scenario": name,
        "classification": "controlled_experiment_not_acceptance",
        "high_address": high_address,
        "shape": case.describe(),
        "rows": rows,
        "prefix_tokens": (case.request_histories[0] // PAGE) * PAGE,
        "comparisons": comparisons,
        "arithmetic_model_controls": arithmetic_models,
        "native_o_min_max_mean": [
            native.min().item(),
            native.max().item(),
            native.mean().item(),
        ],
        "native_lse_min_max": [native_lse.min().item(), native_lse.max().item()],
        "online_model_limitation": "Independent FP32 QK, N128; no CK MFMA QK/PV reduction-order, rowsum reduction-order or ISA exp2 emulation. PV is a separately reduced einsum then add, whereas CK seeds MFMA with the scaled prior Oacc. FMA modes isolate P/alpha centering and denominator update; reciprocal mode uses FP64 division rounded to FP32 then FP32 multiply. Residual is not fully attributed.",
    }
    if name == "two_logits_low_class_value_one":
        result["predeclared_cpu_predictions"] = {
            "q_nonzero_dim": 0,
            "q_value": 1,
            "k_low": -6.6875,
            "k_high": 0,
            "v_low": 1,
            "v_high": 0,
            "class_counts": [352, 352],
            "p_fp32": 0.5537194609642029,
            "p_rtz": 0.55078125,
            "p_rn": 0.5546875,
            "o_only_final_rtz": 0.35546875,
            "o_p_rtz_and_final_rtz": 0.353515625,
            "o_p_rn_and_final_rtz": 0.35546875,
            "o_p_rn_and_final_rn": 0.357421875,
            "minimum_conversion_boundary_margin": 0.0009143948554992676,
        }
        result["native_unique_outputs"] = native.unique().cpu().tolist()
        result["native_fraction_matching_only_final_rtz"] = (
            (native == 0.35546875).float().mean().item()
        )
        result["native_fraction_matching_p_and_final_rtz"] = (
            (native == 0.353515625).float().mean().item()
        )
        result["native_fraction_matching_p_and_final_rn"] = (
            (native == 0.357421875).float().mean().item()
        )
    if name == "random_logits_unit_v":
        result["unit_v_below_one_fraction"] = (native < 1).float().mean().item()
        result["unit_v_above_one_fraction"] = (native > 1).float().mean().item()
        result["unit_v_unique_outputs"] = native.unique().cpu().tolist()
    if name == "random_logits_channel_constants":
        constants = 1 + torch.arange(DIM, device="cuda", dtype=torch.float32) / 128
        ulp = torch.exp2(torch.floor(torch.log2(native.abs())) - 7)
        # For positive values, RTZ output y implies y <= c*mass < y+ULP(y).
        lower = (native / constants).max(-1).values
        upper = ((native + ulp) / constants).min(-1).values
        result["probability_mass_interval_from_output_rtz"] = {
            "assumption": "Final O uses RTZ; this interval must not be interpreted for an RN native module",
            "median_lower": lower.median().item(),
            "median_upper": upper.median().item(),
            "lower_min": lower.min().item(),
            "upper_max": upper.max().item(),
            "interval_width_max": (upper - lower).max().item(),
            "inconsistent_interval_fraction": (lower > upper + 2e-6)
            .float()
            .mean()
            .item(),
            "upper_below_one_minus_1e_4_fraction": (upper < 1 - 1e-4)
            .float()
            .mean()
            .item(),
            "fp32_arithmetic_slack": 2e-6,
            "interpretation": "If intervals exclude 1 by much more than FP32 slack, final O rounding alone cannot explain mass loss",
        }
    del ab, case
    return result


def main():
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm GPU required; execute through the controlling agent's locked launcher"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    output_dir = Path(
        os.getenv(
            "MKT_CASCADE_OUTPUT_DIR",
            os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", "native_precision_results"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in (
        "uniform_logits_random_v",
        "two_logits_low_class_value_one",
        "random_logits_unit_v",
        "random_logits_channel_constants",
        "random_logits_random_v",
    ):
        result = scenario(name)
        results.append(result)
        print(json.dumps(result), flush=True)
    if os.getenv("NATIVE_DIAG_HIGH_ADDRESS", "0") == "1":
        result = scenario("random_logits_random_v", high_address=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    mapped = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "mha_batch_prefill" in line and ".so" in line:
            path = line.split()[-1]
            if path not in [item["path"] for item in mapped]:
                mapped.append(
                    {
                        "path": path,
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    }
                )
    receipt = {
        "results": results,
        "loaded_modules": mapped,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "classification": "controlled_experiment_not_acceptance",
        "float32_precision": torch.get_float32_matmul_precision(),
        "warning": "Synthetic controlled experiment; no end-to-end TopK accuracy claim",
    }
    (output_dir / "native_precision_diagnosis.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
