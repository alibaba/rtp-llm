"""Performance benchmark for DSV4 fused KV compress + norm + RoPE insert."""

from __future__ import annotations

import importlib.util
import os
import time
import unittest
from dataclasses import asdict, dataclass
from typing import Callable, List

import torch

from dsv4_kernel_perf_utils import (
    device_payload,
    env_flag,
    ensure_triton_cc,
    git_commit,
    iters_for_m,
    measure_kernel,
    measurement_payload,
    parse_int_list,
    report_path_from_env,
    trace_dir_from_report,
    write_json_report,
)

_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "_compressor_vllm_triton.py",
)
_SPEC = importlib.util.spec_from_file_location("_compressor_vllm_triton", _KERNEL_PATH)
_KERNEL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_KERNEL)
run_fused_compress_kv_write_bf16 = _KERNEL.run_fused_compress_kv_write_bf16
should_use_compact_boundary = _KERNEL._should_use_compact_boundary
select_kv_write_dispatch = _KERNEL._select_kv_write_dispatch
ensure_triton_cc()


DEFAULT_M_SWEEP = [16, 64, 256, 1024, 4096, 16384, 65536]
ROPE_HEAD_DIM = 64
RMS_NORM_EPS = 1e-6
KERNEL_REGEX = r".*fused_kv_compress_norm_rope_insert_bf16.*"


@dataclass
class BenchCase:
    name: str
    M: int
    head_dim: int
    compress_ratio: int
    overlap: bool

    @property
    def state_width(self) -> int:
        return (1 + int(self.overlap)) * self.head_dim


@dataclass
class CaseTensors:
    state_cache: torch.Tensor
    token_to_req_indices: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    rms_norm_weight: torch.Tensor
    cos_sin_cache: torch.Tensor
    kv_cache: torch.Tensor
    kv_slot_mapping: torch.Tensor
    kv_raw: torch.Tensor
    score_raw: torch.Tensor
    ape: torch.Tensor
    boundary_positions: torch.Tensor
    boundary_token_indices: torch.Tensor


def _build_cases(M: int) -> List[BenchCase]:
    cases = [
        BenchCase("ratio4_head512_overlap_prefill", M, 512, 4, True),
        BenchCase("ratio128_head512_no_overlap_prefill", M, 512, 128, False),
        BenchCase("ratio4_head128_overlap_prefill", M, 128, 4, True),
        BenchCase("ratio128_head128_no_overlap_prefill", M, 128, 128, False),
    ]
    case_filter = os.environ.get("DSV4_KV_COMPRESS_CASE_FILTER")
    if not case_filter:
        return cases
    needles = [x.strip() for x in case_filter.split(",") if x.strip()]
    return [case for case in cases if any(needle in case.name for needle in needles)]


def _make_cos_sin_cache(rows: int) -> torch.Tensor:
    torch.manual_seed(9001 + rows)
    angle = torch.rand(rows, ROPE_HEAD_DIM // 2, device="cuda") * 6.28
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return torch.cat([cos, sin], dim=-1).contiguous()


def _make_tensors(case: BenchCase) -> CaseTensors:
    torch.manual_seed(17 + case.M + case.head_dim + case.compress_ratio)
    positions = torch.arange(case.M, device="cuda", dtype=torch.long)
    boundary_mask = ((positions + 1) % case.compress_ratio) == 0
    boundary_positions = positions[boundary_mask].contiguous()
    boundary_token_indices = torch.nonzero(boundary_mask, as_tuple=False).flatten()
    boundary_count = int(boundary_positions.numel())

    token_to_req_indices = torch.zeros(case.M, device="cuda", dtype=torch.int32)
    slot_mapping = torch.zeros(case.M, device="cuda", dtype=torch.long)
    kv_slot_mapping = torch.full((case.M,), -1, device="cuda", dtype=torch.long)
    if boundary_count:
        kv_slot_mapping[boundary_positions] = torch.arange(
            boundary_count, device="cuda", dtype=torch.long
        )

    kv_raw = (
        torch.randn(case.M, case.state_width, device="cuda", dtype=torch.float32)
        * 0.25
    ).contiguous()
    score_raw = (
        torch.randn(case.M, case.state_width, device="cuda", dtype=torch.float32)
        * 0.25
    ).contiguous()
    ape = (
        torch.randn(
            case.compress_ratio,
            case.state_width,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
    ).contiguous()
    rms_norm_weight = (
        torch.rand(case.head_dim, device="cuda", dtype=torch.bfloat16) + 0.5
    ).contiguous()
    cos_sin_cache = _make_cos_sin_cache(case.M + case.compress_ratio + 1)

    state_cache = torch.zeros(
        1,
        max(case.compress_ratio, 1),
        2 * case.state_width,
        device="cuda",
        dtype=torch.float32,
    )
    block_table = torch.ones(1, 1, device="cuda", dtype=torch.int32)
    kv_cache = torch.empty(
        max(boundary_count, 1),
        case.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_cache.fill_(float("nan"))

    return CaseTensors(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_raw=kv_raw,
        score_raw=score_raw,
        ape=ape,
        boundary_positions=boundary_positions,
        boundary_token_indices=boundary_token_indices,
    )


def _launch(case: BenchCase, tensors: CaseTensors) -> None:
    run_fused_compress_kv_write_bf16(
        tensors.state_cache,
        tensors.token_to_req_indices,
        tensors.positions,
        tensors.slot_mapping,
        tensors.block_table,
        tensors.rms_norm_weight,
        RMS_NORM_EPS,
        tensors.cos_sin_cache,
        tensors.kv_cache,
        tensors.kv_slot_mapping,
        tensors.kv_raw,
        tensors.score_raw,
        tensors.ape,
        0,
        disable_raw_path=False,
        boundary_token_indices=tensors.boundary_token_indices,
        head_dim=case.head_dim,
        rope_head_dim=ROPE_HEAD_DIM,
        compress_ratio=case.compress_ratio,
        overlap=case.overlap,
    )


def _reference_one(case: BenchCase, tensors: CaseTensors, position: int) -> torch.Tensor:
    window = (1 + int(case.overlap)) * case.compress_ratio
    start = position - window + 1
    token_offsets = torch.arange(window, device="cuda", dtype=torch.long)
    pos = start + token_offsets
    valid = pos >= 0
    safe_pos = pos.clamp(min=0)
    head_offset = (token_offsets >= case.compress_ratio).to(torch.long) * case.head_dim
    cols = torch.arange(case.head_dim, device="cuda", dtype=torch.long)
    gather_cols = head_offset[:, None] + cols[None, :]

    kv = tensors.kv_raw[safe_pos[:, None], gather_cols]
    score = tensors.score_raw[safe_pos[:, None], gather_cols]
    ape_rows = pos.remainder(case.compress_ratio).clamp(min=0)
    score = score + tensors.ape[ape_rows[:, None], gather_cols]

    valid_2d = valid[:, None]
    kv = torch.where(valid_2d, kv, torch.zeros_like(kv))
    score = torch.where(valid_2d, score, torch.full_like(score, float("-inf")))
    weights = torch.softmax(score, dim=0)
    compressed = (kv * weights).sum(dim=0)

    variance = compressed.square().mean()
    normed = (
        compressed
        * torch.rsqrt(variance + RMS_NORM_EPS)
        * tensors.rms_norm_weight.float()
    )

    nope_head_dim = case.head_dim - ROPE_HEAD_DIM
    rope = normed[nope_head_dim:].view(ROPE_HEAD_DIM // 2, 2)
    compressed_pos = (position // case.compress_ratio) * case.compress_ratio
    cos = tensors.cos_sin_cache[compressed_pos, : ROPE_HEAD_DIM // 2]
    sin = tensors.cos_sin_cache[compressed_pos, ROPE_HEAD_DIM // 2 :]
    even = rope[:, 0]
    odd = rope[:, 1]
    rotated = torch.empty_like(rope)
    rotated[:, 0] = even * cos - odd * sin
    rotated[:, 1] = odd * cos + even * sin
    result = normed.clone()
    result[nope_head_dim:] = rotated.reshape(ROPE_HEAD_DIM)
    return result.to(torch.bfloat16)


def _check_correctness(case: BenchCase, tensors: CaseTensors) -> None:
    _launch(case, tensors)
    torch.cuda.synchronize()
    boundary_count = int(tensors.boundary_positions.numel())
    if boundary_count == 0:
        if not torch.isnan(tensors.kv_cache).all().item():
            raise AssertionError(f"{case.name} M={case.M} unexpectedly wrote KV cache")
        return

    if case.M <= 256:
        refs = [
            _reference_one(case, tensors, int(pos.item()))
            for pos in tensors.boundary_positions
        ]
        ref = torch.stack(refs, dim=0)
        diff = (ref.float() - tensors.kv_cache[:boundary_count].float()).abs()
        max_abs = float(diff.max().item())
        if max_abs > 5e-2:
            raise AssertionError(f"{case.name} M={case.M} max_abs={max_abs:.4e}")
    else:
        written = tensors.kv_cache[:boundary_count]
        if not torch.isfinite(written.float()).all().item():
            raise AssertionError(f"{case.name} M={case.M} produced non-finite output")


def _bench_case(case: BenchCase, trace_dir: str, profile_enabled: bool) -> dict:
    tensors = _make_tensors(case)
    _check_correctness(case, tensors)
    tensors.kv_cache.fill_(float("nan"))

    warmup = 20 if case.M <= 4096 else 8
    iters = iters_for_m(case.M)
    fused: Callable[[], None] = lambda: _launch(case, tensors)
    boundary_count = int(tensors.boundary_positions.numel())
    measure = measure_kernel(
        fused,
        label=f"{case.name}_M{case.M}",
        trace_dir=trace_dir,
        kernel_regex=KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled and boundary_count > 0,
    )
    row = {
        "case": case.name,
        "M": case.M,
        "head_dim": case.head_dim,
        "rope_head_dim": ROPE_HEAD_DIM,
        "compress_ratio": case.compress_ratio,
        "overlap": case.overlap,
        "boundary_count": boundary_count,
        "boundary_density": boundary_count / max(case.M, 1),
        "state_width": case.state_width,
        "seq_start": 0,
        "path": "prefill_raw",
        "compact_boundary": should_use_compact_boundary(
            case.M,
            boundary_count,
            head_dim=case.head_dim,
            compress_ratio=case.compress_ratio,
        ),
        "dispatch_path": select_kv_write_dispatch(
            case.M,
            boundary_count,
            has_boundary_indices=True,
            head_dim=case.head_dim,
            rope_head_dim=ROPE_HEAD_DIM,
            compress_ratio=case.compress_ratio,
            overlap=case.overlap,
        ),
        **measurement_payload(measure),
    }
    return row


def _print_summary(rows: List[dict]) -> None:
    print("\n[fused_kv_compress_norm_rope_insert_bf16]")
    print(
        "  {:>8} {:>36} {:>8} {:>8} {:>12} {:>12}".format(
            "M", "case", "head", "ratio", "boundary", "span_us"
        )
    )
    for row in rows:
        print(
            "  {:8d} {:>36} {:8d} {:8d} {:12d} {:12.2f}".format(
                row["M"],
                row["case"],
                row["head_dim"],
                row["compress_ratio"],
                row["boundary_count"],
                row["kernel_span_us"],
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedKvCompressNormRopeInsertPerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_KV_COMPRESS_M_LIST", DEFAULT_M_SWEEP)
        report_path = report_path_from_env(
            "DSV4_KV_COMPRESS_JSON",
            "dsv4_fused_kv_compress_norm_rope_insert_perf.json",
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_KV_COMPRESS_TRACE_DIR")
        profile_enabled = env_flag("DSV4_KV_COMPRESS_PROFILE", True)

        rows: List[dict] = []
        invalid: List[dict] = []
        for M in m_list:
            for case in _build_cases(M):
                try:
                    rows.append(_bench_case(case, trace_dir, profile_enabled))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    invalid.append({**asdict(case), "invalid_reason": f"OOM: {exc}"})
                torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
            "device": device_payload(),
            "m_list": m_list,
            "case_filter": os.environ.get("DSV4_KV_COMPRESS_CASE_FILTER", ""),
            "profile_enabled": profile_enabled,
            "kernel_regex": KERNEL_REGEX,
            "results": rows,
            "invalid": invalid,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote fused KV compress perf JSON: {report_path}")
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
