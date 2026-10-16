"""SM90 acceptance for the existing SM100 indexer-Q leaf, without enabling it.

This test launches the actual existing Triton kernel directly. It never mocks
device capability or relaxes the production support gate. Independent eager
complex RoPE/BF16 rounding and nearest-even FP4 packing own correctness; the
old production expression is an additional compatibility/timing reference.
Parent owns GPU/Bazel execution. Component timing is not E2E TPOT.
"""

import json
import os

import pytest
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_indexer_q_triton as candidate


@pytest.fixture(autouse=True)
def _sm90_only():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (
        9,
        0,
    ), "Requires a parent-locked SM90 GPU; no capability mocking or skip-as-PASS"


def _case(batch=1, span=1, *, seed=41, weights_dtype=torch.bfloat16):
    torch.manual_seed(seed)
    q = torch.randn(batch, span, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, span, 32, device="cuda", dtype=weights_dtype)
    angles = torch.randn(4096, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(
        0, 4096, (batch * span,), device="cuda", dtype=torch.int64
    )
    return dict(q=q, weights=weights, freqs_cis=freqs, positions=positions)


def _independent_reference(q, weights, freqs_cis, positions):
    """Independent FP32 complex math, BF16 boundary, G32 power-of-two FP4.

    FP4 uses a codebook distance search with explicit even-code tie breaking,
    not the production threshold helper. frexp determines scale exponents;
    no production scale/packing/rotation reader is used.
    """
    shape = q.shape
    rotated = q.clone().reshape(-1, 32, 128)
    tail = torch.view_as_complex(rotated[..., 64:].float().reshape(-1, 32, 32, 2))
    product = tail * freqs_cis[positions.long()][:, None, :]
    rotated[..., 64:].copy_(torch.view_as_real(product).flatten(-2).bfloat16())
    groups = rotated.float().reshape(-1, 32)
    maximum = groups.abs().amax(-1).clamp_min(6.0 * (2.0**-126))
    raw_scale = maximum * (1.0 / 6.0)
    fraction, exponent = torch.frexp(raw_scale)
    power = exponent - (fraction == 0.5).to(exponent.dtype)
    scales = torch.ldexp(torch.ones_like(raw_scale), power)
    normalized = (groups / scales[:, None]).clamp(-6, 6)
    magnitude = normalized.abs()
    codebook = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    # Work in bounded tiles so the long rounding stress does not require a
    # full [B*S*32*128,8] temporary. Reference is outside all timing/capture.
    flat = magnitude.flatten()
    codes = torch.empty(flat.shape, device=q.device, dtype=torch.uint8)
    for begin in range(0, flat.numel(), 1 << 20):
        end = min(begin + (1 << 20), flat.numel())
        distances = (flat[begin:end, None] - codebook).abs()
        lower = distances.argmin(-1)
        upper = (lower + 1).clamp_max(7)
        tie = distances.gather(1, lower[:, None]) == distances.gather(1, upper[:, None])
        code = torch.where(tie[:, 0] & ((lower % 2) == 1), upper, lower)
        codes[begin:end] = code.to(torch.uint8)
    codes = codes.reshape_as(normalized) | (
        torch.signbit(normalized).to(torch.uint8) << 3
    )
    rows = codes.reshape(-1, 128)
    payload = (rows[:, 0::2] | (rows[:, 1::2] << 4)).view(torch.int8)
    scale_bytes = (power + 127).to(torch.uint8).reshape(-1, 4)
    words = sum(scale_bytes[:, j].to(torch.int64) << (8 * j) for j in range(4)).to(
        torch.int32
    )
    return (
        payload.reshape(*shape[:-1], 64),
        words.reshape(shape[:-1]),
        weights.float() * (1.0 / 64.0),
    )


def _legacy(q, weights, freqs_cis, positions):
    scaled = weights.float() * (1.0 / 64.0)
    payload, scales = indexer.prepare_indexer_q(q, scaled, freqs_cis[positions])
    return payload, scales, scaled


def _direct_leaf(q, weights, freqs_cis, positions, *, hopper=None):
    """No gate override: invoke the real existing leaf on its actual inputs."""
    assert q.dtype == torch.bfloat16 and q.shape[-2:] == (32, 128)
    assert q.is_contiguous() and weights.is_contiguous()
    assert freqs_cis.is_contiguous() and positions.is_contiguous()
    payload = torch.empty(*q.shape[:-1], 64, device=q.device, dtype=torch.int8)
    scales = torch.empty(q.shape[:-1], device=q.device, dtype=torch.int32)
    scaled_weights = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    candidate._fused_indexer_q_rope_fp4_kernel[(q.numel() // 128,)](
        q,
        weights,
        torch.view_as_real(freqs_cis),
        positions,
        payload,
        scales,
        scaled_weights,
        freqs_cis.shape[0],
        HOPPER=(
            torch.cuda.get_device_capability(q.device)[0] == 9
            if hopper is None
            else hopper
        ),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return payload, scales, scaled_weights


def _bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    a = torch.empty(actual.numel(), device=actual.device, dtype=actual.dtype)
    b = torch.empty(expected.numel(), device=expected.device, dtype=expected.dtype)
    a.copy_(actual.reshape(-1))
    b.copy_(expected.reshape(-1))
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


def _triplet_bits(actual, expected):
    for got, want, dtype in zip(
        actual, expected, (torch.int8, torch.int32, torch.float32)
    ):
        assert got.dtype == dtype and got.is_contiguous()
        _bits(got, want)


@triton.jit
def _diagnostic_rope_boundary_kernel(
    q_ptr,
    frequencies,
    positions,
    row_ids,
    old_bf16,
    alternate_bf16,
    old_fp32,
    alternate_fp32,
    MAX_POSITION: tl.constexpr,
):
    sample = tl.program_id(0)
    row = tl.load(row_ids + sample)
    position = tl.load(positions + row // 32)
    position = tl.where(position < 0, position + MAX_POSITION, position)
    columns = tl.arange(0, 128)
    q = tl.load(q_ptr + row * 128 + columns)
    # Explicit False preserves the original mismatch after adding the SM90
    # candidate. This is the actual helper, not a copied BF16 implementation.
    existing = candidate._indexer_rope_bf16(q, frequencies, position, False)
    values = q.to(tl.float32)
    real = tl.gather(values, columns & ~1, axis=0)
    imag = tl.gather(values, columns | 1, axis=0)
    rotary = columns >= 64
    angle = tl.maximum((columns - 64) // 2, 0)
    cosine = tl.load(frequencies + position * 64 + angle * 2, rotary, other=1)
    sine = tl.load(frequencies + position * 64 + angle * 2 + 1, rotary, other=0)
    product = tl.where(columns % 2 == 0, imag, real) * sine
    signed_product = (
        product.to(tl.uint32, bitcast=True) ^ tl.where(columns % 2 == 0, 0x80000000, 0)
    ).to(tl.float32, bitcast=True)
    original = tl.fma(tl.where(columns % 2 == 0, real, imag), cosine, signed_product)
    swapped_odd = tl.fma(real, sine, imag * cosine)
    alternate = tl.where(columns % 2 == 0, original, swapped_odd)
    original = tl.where(rotary, original, values)
    alternate = tl.where(rotary, alternate, values)
    tl.store(old_bf16 + sample * 128 + columns, existing)
    tl.store(alternate_bf16 + sample * 128 + columns, alternate.to(tl.bfloat16))
    tl.store(old_fp32 + sample * 128 + columns, original)
    tl.store(alternate_fp32 + sample * 128 + columns, alternate)


@pytest.mark.skipif(
    os.environ.get("DSV41_SM90_INDEXER_Q_DIAG") != "1",
    reason="explicit unchanged-seed rounding diagnosis",
)
@torch.no_grad()
def test_diagnose_original_leaf_rope_midpoint():
    case = _case(16384, 1, seed=41)
    reference = _independent_reference(**case)
    legacy = _legacy(**case)
    actual = _direct_leaf(**case, hopper=False)
    _triplet_bits(legacy, reference)
    mismatches = [
        (a != e).flatten().nonzero().flatten() for a, e in zip(actual, reference)
    ]
    print(
        "SM90_INDEXER_Q_DIAG_COUNTS "
        + json.dumps(
            dict(
                payload=mismatches[0].numel(),
                scales=mismatches[1].numel(),
                weights=mismatches[2].numel(),
                payload_offsets=mismatches[0][:32].cpu().tolist(),
                legacy_vs_independent="byte exact",
            )
        ),
        flush=True,
    )
    if not mismatches[0].numel():
        return
    rows = (mismatches[0] // 64).unique()
    n = rows.numel()
    buffers = [
        torch.empty((n, 128), device="cuda", dtype=dtype)
        for dtype in (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32)
    ]
    _diagnostic_rope_boundary_kernel[(n,)](
        case["q"],
        torch.view_as_real(case["freqs_cis"]),
        case["positions"],
        rows,
        *buffers,
        case["freqs_cis"].shape[0],
        num_warps=4,
        enable_fp_fusion=False,
    )
    q_rows = case["q"].reshape(-1, 128)[rows]
    frequency_rows = case["freqs_cis"][case["positions"][rows // 32]]
    complex_product = (
        torch.view_as_complex(q_rows[:, 64:].float().reshape(n, 32, 2)) * frequency_rows
    )
    eager_fp32 = q_rows.float()
    eager_fp32[:, 64:] = torch.view_as_real(complex_product).flatten(-2)
    eager_bf16 = eager_fp32.bfloat16()
    differences = (
        buffers[0].view(torch.int16) != eager_bf16.view(torch.int16)
    ).nonzero()
    print(
        "SM90_INDEXER_Q_DIAG_ROPE "
        + json.dumps(
            dict(
                rows=rows.cpu().tolist(),
                original_bf16_mismatches=differences.shape[0],
                alternate_bf16_mismatches=int(
                    (buffers[1].view(torch.int16) != eager_bf16.view(torch.int16)).sum()
                ),
                scope="only rows selected by actual payload mismatch; not full alternate acceptance",
            )
        ),
        flush=True,
    )
    for sample, col in differences[:32].cpu().tolist():
        pair = col & ~1
        a, b = q_rows[sample, pair : pair + 2].float().cpu().tolist()
        c, d = (
            torch.view_as_real(frequency_rows[sample, (pair - 64) // 2]).cpu().tolist()
        )
        # BF16 inputs times FP32 frequencies have exact FP64 products. Explicit
        # FP32 casts reproduce the two FMA schedules without a CUDA oracle.
        fp32 = lambda value: torch.tensor(value, dtype=torch.float32).item()
        exact64 = a * c - b * d if col % 2 == 0 else a * d + b * c
        schedule0 = (
            fp32(a * c + fp32(-b * d)) if col % 2 == 0 else fp32(b * c + fp32(a * d))
        )
        schedule1 = schedule0 if col % 2 == 0 else fp32(a * d + fp32(b * c))
        print(
            "SM90_INDEXER_Q_DIAG_ELEMENT "
            + json.dumps(
                dict(
                    row=int(rows[sample]),
                    column=col,
                    a=a,
                    b=b,
                    cosine=c,
                    sine=d,
                    eager_fp32=float(eager_fp32[sample, col]),
                    original_fp32=float(buffers[2][sample, col]),
                    alternate_fp32=float(buffers[3][sample, col]),
                    eager_bf16=float(eager_bf16[sample, col]),
                    original_bf16=float(buffers[0][sample, col]),
                    alternate_bf16=float(buffers[1][sample, col]),
                    cpu_exact64=exact64,
                    cpu_original_schedule=schedule0,
                    cpu_alternate_schedule=schedule1,
                )
            ),
            flush=True,
        )


@pytest.mark.parametrize(
    "batch,span",
    [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (1, 7), (2, 7), (1, 16), (1, 128)],
)
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_real_leaf_payload_scale_weight_and_input_bits(batch, span, weights_dtype):
    case = _case(batch, span, weights_dtype=weights_dtype)
    saved = {key: value.clone() for key, value in case.items()}
    expected = _independent_reference(**case)
    _triplet_bits(_legacy(**case), expected)
    _triplet_bits(_direct_leaf(**case), expected)
    for key, value in case.items():
        _bits(value, saved[key])


@torch.no_grad()
def test_rope_bf16_rounding_long_stress_exact_no_seed_or_amplitude_relaxation():
    # Same long random shape that exposed a one-nibble imaginary-RoPE error
    # during SM100 development. This does not change production decode gating.
    case = _case(16384, 1, seed=41)
    expected = _independent_reference(**case)
    _triplet_bits(_legacy(**case), expected)
    _triplet_bits(_direct_leaf(**case), expected)


@torch.no_grad()
def test_fp4_ties_scale_boundaries_signed_zero_tiny_and_negative_positions():
    case = _case(1, 8, weights_dtype=torch.float32)
    case["freqs_cis"].fill_(complex(1, 0))
    case["positions"].copy_(
        torch.tensor([0, 4095, -1, -4096, 17, 31, 100, 999], device="cuda")
    )
    midpoints = torch.tensor(
        [0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5],
        device="cuda",
        dtype=torch.bfloat16,
    )
    samples = torch.cat(
        (
            torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
        )
    )
    groups = case["q"].view(-1, 32)
    groups.zero_()
    groups[:, : samples.numel()] = samples
    groups[:, -1] = 6
    groups[1::8].neg_()
    six = torch.tensor(6.0, device="cuda", dtype=torch.bfloat16)
    groups[2::8, -1] = torch.nextafter(six, torch.full_like(six, torch.inf))
    groups[3::8, -1] = torch.nextafter(six, torch.full_like(six, -torch.inf))
    groups[4::8].zero_()
    groups[5::8].fill_(-0.0)
    groups[6::8].mul_(2.0**-126)
    groups[7::8].mul_(2.0**-130)
    case["weights"].view(-1)[:4] = torch.tensor(
        [0.0, -0.0, 2.0**-126, -(2.0**-126)], device="cuda"
    )
    expected = _independent_reference(**case)
    _triplet_bits(_legacy(**case), expected)
    _triplet_bits(_direct_leaf(**case), expected)


@torch.no_grad()
def test_noncontiguous_inputs_use_real_fallback_without_capability_mock():
    for key in ("q", "weights", "freqs_cis", "positions"):
        case = _case(2, 2)
        value = case[key]
        padded = torch.empty(
            (*value.shape[:-1], value.shape[-1] * 2),
            device=value.device,
            dtype=value.dtype,
        )
        padded[..., ::2].copy_(value)
        case[key] = padded[..., ::2]
        assert not case[key].is_contiguous()
        assert candidate.try_fused_indexer_q(**case) is None
        expected = _independent_reference(**case)
        _triplet_bits(indexer.prepare_indexer_q_and_weights(**case), expected)


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


@torch.no_grad()
def test_graph_replay_changes_q_weights_positions_and_frequency_table():
    for batch, span in ((1, 1), (2, 2), (1, 7)):
        case = _case(batch, span)
        graph, actual = _capture(lambda: _direct_leaf(**case))
        for step in range(4):
            replacement = _case(batch, span, seed=97 + step)
            for key in case:
                case[key].copy_(replacement[key])
            if step == 1:
                case["positions"].neg_().sub_(1)
            if step == 2:
                case["q"].zero_()
                case["weights"].zero_()
            saved = {key: value.clone() for key, value in case.items()}
            expected = _independent_reference(**case)
            graph.replay()
            torch.cuda.synchronize()
            _triplet_bits(actual, expected)
            _triplet_bits(_legacy(**case), expected)
            for key in case:
                _bits(case[key], saved[key])


@pytest.mark.skipif(
    os.environ.get("DSV41_SM90_INDEXER_Q_BENCH") != "1",
    reason="explicit clean benchmark opt-in",
)
@torch.no_grad()
def test_clean_graph_prep_benchmark_after_exact_gate():
    for span in (1, 2, 8, 16):
        case = _case(1, span)
        reference = _independent_reference(**case)
        _triplet_bits(_legacy(**case), reference)
        _triplet_bits(_direct_leaf(**case), reference)
        old_graph, old_output = _capture(lambda: _legacy(**case))
        new_graph, new_output = _capture(lambda: _direct_leaf(**case))

        def timing(graph):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(200):
                graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 200

        old_ms, new_ms = [], []
        for repeat in range(5):
            for graph, samples in ((old_graph, old_ms), (new_graph, new_ms))[
                :: 1 if repeat % 2 == 0 else -1
            ]:
                samples.append(timing(graph))
        _triplet_bits(old_output, reference)
        _triplet_bits(new_output, reference)
        print(
            "SM90_INDEXER_Q_PREP_BENCH "
            + json.dumps(
                dict(
                    B=1,
                    S=span,
                    H=32,
                    D=128,
                    old_ms=sorted(old_ms)[2],
                    fused_ms=sorted(new_ms)[2],
                    speedup=sorted(old_ms)[2] / sorted(new_ms)[2],
                    correctness="exact independent payload/scale/FP32-weight oracle before timing",
                    scope="Q preparation only; excludes scoring/topk and E2E TPOT",
                    profiling=False,
                )
            ),
            flush=True,
        )
