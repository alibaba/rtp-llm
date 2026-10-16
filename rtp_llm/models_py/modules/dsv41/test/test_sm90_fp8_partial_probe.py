"""Raw FP32 SM90 FP8 partial diagnostic, NOT an acceptance/performance test.

Keep the failed seed and tolerance. No production dispatch changes. A passing
test means the historical discrepancy was reproduced and evidence was saved;
split/antithetic variants remain hypotheses until independent broad gates.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.sm90_dense_triton import (
    _w8a8_block_fp8_matmul,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_tuning import (
    _independent_scales,
    _metrics,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


@triton.jit
def _raw_k32_partials(A, B, Out, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, SPLITS: tl.constexpr,
                      NEGATE_A: tl.constexpr):
    """Instrumentation of _w8a8_block_fp8_matmul's existing K32 tl.dot.

    Each QGMMA starts at zero and stores FP32 before scales/BF16 rounding.
    Diagnostic only: this deliberately writes all partials and is not a new
    production GEMM implementation or candidate for performance promotion.
    """
    pn = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    rm = tl.arange(0, 64)
    rn = pn * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    active_k = (rk >= split * (32 // SPLITS)) & (rk < (split + 1) * (32 // SPLITS))
    # Modulo M matches the existing generic M64 candidate's padded rows.
    a = tl.load(A + (rm[:, None] % M) * K + group * 32 + rk[None, :],
                mask=active_k[None, :], other=0.0)
    b = tl.load(B + rn[None, :] * K + group * 32 + rk[:, None],
                mask=(rn[None, :] < N) & active_k[:, None], other=0.0)
    if NEGATE_A:
        a = (-a.to(tl.float32)).to(a.dtype)
    c = tl.dot(a, b)
    offsets = ((group * SPLITS + split) * M + rm[:, None]) * N + rn[None, :]
    tl.store(Out + offsets, c, (rm[:, None] < M) & (rn[None, :] < N))


def _save_kernel(compiled, name, directory):
    ptx = compiled.asm.get("ptx", "")
    binary = compiled.asm.get("cubin")
    assert ptx and isinstance(binary, bytes), "Missing instruction evidence"
    (directory / f"{name}.ptx").write_text(ptx)
    (directory / f"{name}.cubin").write_bytes(binary)
    instructions = [line.strip() for line in ptx.splitlines()
                    if "wgmma.mma_async" in line and ".e4m3" in line]
    assert instructions, "Probe did not emit native FP8 WGMMA PTX"
    return dict(ptx_instructions=list(dict.fromkeys(instructions)),
                cubin_sha256=hashlib.sha256(binary).hexdigest(),
                sass_status="PENDING cuobjdump verification; PTX is not final SASS proof")


def _full_fp32(q, weight, activation_scales, weight_scales):
    m, k = q.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=q.device)
    compiled = _w8a8_block_fp8_matmul[(triton.cdiv(m, 64) * triton.cdiv(n, 32),)](
        q, weight, out, activation_scales, weight_scales,
        m, n, k, 32, 32,
        q.stride(0), q.stride(1), weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1), activation_scales.stride(0),
        activation_scales.stride(1), weight_scales.stride(1), weight_scales.stride(0),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=32,
        needs_masking=False, num_warps=4, num_stages=3,
    )
    return out, compiled


def _reconstruct(partials, a_scales, b_scales):
    """CPU FP32 promotion after each K32, no BF16 intermediate rounding."""
    groups, _, m, n = partials.shape
    result = torch.zeros((m, n), dtype=torch.float32)
    for group in range(groups):
        value = torch.zeros((m, n), dtype=torch.float32)
        for split in range(partials.shape[1]):
            value += partials[group, split]
        result += value * a_scales[:, group, None] * b_scales[:, group][None, :]
    return result


def _alignment_models(products, observed):
    """Falsifiable coarse mantissa-alignment models, not a claimed circuit."""
    matches = []
    for subgroup in (4, 8, 16, 32):
        for bits in range(10, 25):
            for rounding in ("toward_zero", "floor", "nearest_even"):
                pieces = []
                for start in range(0, len(products), subgroup):
                    values = products[start:start + subgroup]
                    maximum = max(abs(value) for value in values)
                    if maximum == 0:
                        pieces.append(0.0)
                        continue
                    _, exponent = math.frexp(maximum)
                    quantum = math.ldexp(1.0, exponent - bits)
                    operation = {"toward_zero": math.trunc, "floor": math.floor,
                                 "nearest_even": round}[rounding]
                    pieces.append(math.fsum(operation(value / quantum) * quantum
                                            for value in values))
                prediction = math.fsum(pieces)
                if prediction == observed:
                    matches.append(dict(significand_bits=bits, subgroup=subgroup,
                                        rounding=rounding))
    return matches


def _empirical_fp22_partials(q_cpu, weight_cpu, splits):
    """CPU model inferred from H20 observations, NOT an ISA guarantee.

    E4M3 products align to max(input exponent sums), keeping 13 fractional
    bits with round-toward-zero. Their exact aligned sum is then normalized
    and truncated to 13 fractional bits again. Subnormal input exponent is
    -6 (encoded E4M3 exponent), not the value's normalized binary exponent.
    This models a zero-initialized K32 instruction only, not chained WGMMA.
    """
    m, k = q_cpu.shape
    n = weight_cpu.shape[0]
    a = q_cpu.float().numpy().astype(np.float64).reshape(m, k // 32, 32).transpose(1, 0, 2)
    b = weight_cpu.float().numpy().astype(np.float64).reshape(n, k // 32, 32).transpose(1, 0, 2)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    products = a[:, :, None, :] * b[:, None, :, :]
    exponent_a = np.maximum(np.frexp(np.abs(a))[1] - 1, -6)
    exponent_b = np.maximum(np.frexp(np.abs(b))[1] - 1, -6)
    exponent_sums = exponent_a[:, :, None, :] + exponent_b[:, None, :, :]
    result = []
    for split in range(splits):
        start, end = split * (32 // splits), (split + 1) * (32 // splits)
        values = products[..., start:end]
        exponents = np.where(values != 0, exponent_sums[..., start:end], -100)
        quantum = np.exp2(exponents.max(-1) - 13)
        aligned = np.trunc(values / quantum[..., None]) * quantum[..., None]
        total = aligned.sum(-1)
        final_quantum = np.exp2(np.frexp(np.abs(total))[1] - 1 - 13)
        result.append(np.trunc(total / final_quantum) * final_quantum)
    return torch.from_numpy(np.stack(result, axis=1))


def _check_empirical_model(q_cpu, weight_cpu, all_partials):
    results = {}
    for name, observed in all_partials.items():
        predicted = _empirical_fp22_partials(q_cpu, weight_cpu, observed.shape[1])
        difference = predicted - observed.double()
        results[name] = dict(elements=observed.numel(),
                             unequal=int((difference != 0).sum()),
                             max_abs=float(difference.abs().max()))
    return dict(status="EMPIRICAL_MODEL; not NVIDIA architectural guarantee",
                model="product exponent alignment RTZ13 + final normalized sum RTZ13; E4M3 subnormal exponent=-6",
                variants=results)


def test_sm90_fp8_raw_partial_diagnostic():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "fp8_raw_partial_probe"
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(decision="NOT_RUN", tolerance=dict(rtol=.015, atol=.02),
                  scope="Expected-failure localization only; no timing or production promotion",
                  kernels={}, variants={})
    try:
        # Exactly the original M7/N256/K512 seed and changed-input sequence.
        m, n, k = 7, 256, 512
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, (n // 32, k // 32),
                                             device="cuda").float()).to(torch.float8_e8m0fnu)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(514100 + m)
        x.copy_(torch.randn_like(x) * 16.0)
        q, packed = V41MXFP8Linear._quantize_input(SimpleNamespace(K=k), x)
        a_scales = _independent_scales(packed, k // 32).contiguous()
        b_scales = scales.float().contiguous()
        full, compiled = _full_fp32(q, weight, a_scales, b_scales)
        record["kernels"]["full"] = _save_kernel(compiled, "full", directory)
        q_cpu, b_cpu = q.float().cpu().double(), weight.float().cpu().double()
        a_cpu = a_scales.cpu()
        b_cpu_scales = b_scales.cpu().repeat_interleave(32, 0)
        exact_partials = torch.stack([
            q_cpu[:, group * 32:(group + 1) * 32]
            @ b_cpu[:, group * 32:(group + 1) * 32].T
            for group in range(k // 32)
        ])
        exact = torch.zeros((m, n), dtype=torch.float64)
        for group in range(k // 32):
            exact += (exact_partials[group] * a_cpu[:, group, None].double()
                      * b_cpu_scales[:, group][None, :].double())
        row, col = 2, 33
        full_cpu = full.cpu()
        full_metrics = _metrics(full_cpu.bfloat16(), exact.bfloat16())
        record["original"] = dict(anchor_exact=float(exact[row, col]),
                                   anchor_fp32=float(full_cpu[row, col]),
                                   anchor_bf16=float(full_cpu.bfloat16()[row, col]),
                                   metrics=full_metrics)
        # Store evidence before the assertion: a different reproduction is not
        # a silently accepted "fix" of the original failure.
        assert float(exact[row, col]) == 2.6964111328125
        assert float(full_cpu.bfloat16()[row, col]) == 2.640625
        assert full_metrics["tolerance_violations"] > 0

        native_partials = None
        all_partials = {}
        for splits, negate in [(1, False), (1, True), (2, False), (4, False),
                                (8, False), (16, False), (32, False)]:
            name = f"split{splits}_{'neg' if negate else 'pos'}"
            output = torch.empty((k // 32, splits, m, n), dtype=torch.float32, device="cuda")
            compiled = _raw_k32_partials[(triton.cdiv(n, 32), k // 32, splits)](
                q, weight, output, m, n, k, splits, negate,
                num_warps=4, num_stages=3,
            )
            record["kernels"][name] = _save_kernel(compiled, name, directory)
            partials = output.cpu()
            if negate:
                partials = -partials
            all_partials[name] = partials
            restored = _reconstruct(partials, a_cpu, b_cpu_scales)
            raw_difference = partials.double().sum(1) - exact_partials
            record["variants"][name] = dict(
                anchor_fp32=float(restored[row, col]),
                anchor_bf16=float(restored.bfloat16()[row, col]),
                metrics=_metrics(restored.bfloat16(), exact.bfloat16()),
                raw_partial_max_abs=float(raw_difference.abs().max()),
                raw_partial_unequal=int((raw_difference != 0).sum()),
                raw_partial_negative_errors=int((raw_difference < 0).sum()),
                raw_partial_positive_errors=int((raw_difference > 0).sum()),
                anchor_partials=partials[:, :, row, col].tolist(),
            )
            if splits == 1 and not negate:
                native_partials = partials
                record["raw_reconstruction"] = dict(
                    fp32_unequal=int((restored != full_cpu).sum()),
                    max_abs=float((restored - full_cpu).abs().max()),
                    bf16_bitwise_equal=bool(torch.equal(restored.bfloat16(), full_cpu.bfloat16())),
                )
        # Sign flip is exact in E4M3. A mismatch after undoing the sign is
        # evidence against symmetric rounding, not proof of a particular rule.
        positive = all_partials["split1_pos"]
        negative = all_partials["split1_neg"]
        average = (positive + negative) * 0.5
        restored_average = _reconstruct(average, a_cpu, b_cpu_scales)
        record["antithetic"] = dict(
            sign_asymmetric_partials=int((positive != negative).sum()),
            max_sign_asymmetry=float((positive - negative).abs().max()),
            metrics=_metrics(restored_average.bfloat16(), exact.bfloat16()),
            anchor_fp32=float(restored_average[row, col]),
            anchor_bf16=float(restored_average.bfloat16()[row, col]),
        )
        anchor_details = []
        for group in range(k // 32):
            products = (q_cpu[row, group * 32:(group + 1) * 32]
                        * b_cpu[col, group * 32:(group + 1) * 32]).tolist()
            observed = float(native_partials[group, 0, row, col])
            raw_exact = math.fsum(products)
            assert raw_exact == float(exact_partials[group, row, col])
            anchor_details.append(dict(
                group=group, products=products, exact_fsum=raw_exact, observed=observed,
                raw_error=observed - raw_exact,
                activation_scale=float(a_cpu[row, group]),
                weight_scale=float(b_cpu_scales[col, group]),
                scaled_error=(observed - raw_exact) * float(a_cpu[row, group])
                             * float(b_cpu_scales[col, group]),
                alignment_model_matches=_alignment_models(products, observed),
            ))
        record["anchor_partials"] = anchor_details
        record["empirical_fp22_model"] = _check_empirical_model(q.cpu(), weight.cpu(), all_partials)
        # Retain actual payload/partials so followup hardware models can be
        # evaluated offline without rerunning or altering the failing seed.
        torch.save(dict(q=q.cpu(), weight=weight.cpu(), a_scales=a_cpu,
                        b_scales=b_cpu_scales, exact_partials=exact_partials,
                        exact=exact, full=full_cpu, partials=all_partials),
                   directory / "raw_evidence.pt")
        assert record["raw_reconstruction"]["bf16_bitwise_equal"], \
            "Raw independent QGMMA partials did not reproduce full GEMM; inspect before attribution"
        assert all(value["unequal"] == 0
                   for value in record["empirical_fp22_model"]["variants"].values()), \
            "Empirical arithmetic model falsified; inspect raw artifacts, do not generalize it"
        record["decision"] = "REPRODUCED; diagnostic only, candidates NOT_ACCEPTED"
    finally:
        (directory / "result.json").write_text(json.dumps(record, indent=2))


def test_sm90_fp8_empirical_model_independent_seed():
    """New payloads/ranges attempt to falsify the model, not tune a tolerance."""
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "fp8_partial_model_seed6141"
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(seed=6141, decision="NOT_RUN", scope="Raw instruction model only", kernels={})
    try:
        m, n, k = 7, 64, 128
        torch.manual_seed(6141)
        # E4M3 subnormals, broad exponent range, cancellation and zero operands.
        a = torch.randn(m, k, device="cuda") * torch.exp2(
            torch.randint(-9, 9, (m, k), device="cuda").float())
        b = torch.randn(n, k, device="cuda") * torch.exp2(
            torch.randint(-9, 9, (n, k), device="cuda").float())
        a[:, ::11] = 0
        b[:, ::13] = 0
        q, weight = a.clamp(-448, 448).to(torch.float8_e4m3fn), b.clamp(-448, 448).to(torch.float8_e4m3fn)
        all_partials = {}
        for splits in (1, 2, 4, 8, 16, 32):
            name = f"split{splits}"
            output = torch.empty((k // 32, splits, m, n), dtype=torch.float32, device="cuda")
            compiled = _raw_k32_partials[(triton.cdiv(n, 32), k // 32, splits)](
                q, weight, output, m, n, k, splits, False,
                num_warps=4, num_stages=3,
            )
            record["kernels"][name] = _save_kernel(compiled, name, directory)
            all_partials[name] = output.cpu()
        record["empirical_fp22_model"] = _check_empirical_model(q.cpu(), weight.cpu(), all_partials)
        torch.save(dict(q=q.cpu(), weight=weight.cpu(), partials=all_partials),
                   directory / "raw_evidence.pt")
        assert all(value["unequal"] == 0
                   for value in record["empirical_fp22_model"]["variants"].values()), \
            "Independent payload falsified empirical model; retain counterexample"
        record["decision"] = "BOUNDED_EMPIRICAL_MODEL_MATCH; not ISA precision guarantee"
    finally:
        (directory / "result.json").write_text(json.dumps(record, indent=2))
