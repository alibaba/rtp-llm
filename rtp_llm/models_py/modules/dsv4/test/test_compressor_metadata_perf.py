"""Benchmark UT for fused DSV4 compressor metadata generation."""

from __future__ import annotations

import os
import time
import unittest
from typing import List

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
from rtp_llm.models_py.modules.dsv4._compressor_metadata_triton import (
    build_prefill_compressor_metadata,
)


ensure_triton_cc()

METADATA_KERNEL_REGEX = r".*compressor_metadata.*"


def _torch_prefill_metadata(start_pos, bsz, seqlen, state_bt, state_eb, kv_bt, kv_eb, ratio):
    n_tokens = bsz * seqlen
    positions = (
        (torch.arange(seqlen, device="cuda", dtype=torch.long).unsqueeze(0) + start_pos)
        .expand(bsz, -1)
        .reshape(n_tokens)
        .contiguous()
    )
    b_idx = (
        torch.arange(bsz, device="cuda", dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, seqlen)
        .reshape(n_tokens)
        .contiguous()
    )
    state_bt_long = state_bt.to(torch.long)
    state_block_in_seq = (positions // state_eb) % state_bt_long.shape[1]
    state_in_block = positions % state_eb
    state_block_id = state_bt_long[b_idx, state_block_in_seq]
    state_slot = state_block_id * state_eb + state_in_block
    state_slot = torch.where(
        state_block_id > 0, state_slot, torch.full_like(state_slot, -1)
    )

    kv_bt_long = kv_bt.to(torch.long)
    tokens_per_block = kv_eb * ratio
    boundary = ((positions + 1) % ratio) == 0
    kv_block_in_seq = positions // tokens_per_block
    kv_in_block = (positions % tokens_per_block) // ratio
    kv_in_capacity = kv_block_in_seq < kv_bt_long.shape[1]
    kv_block_safe = kv_block_in_seq.clamp(min=0, max=kv_bt_long.shape[1] - 1)
    kv_block_id = kv_bt_long[b_idx, kv_block_safe]
    kv_valid = boundary & kv_in_capacity & (kv_block_id > 0)
    kv_slot = kv_block_id * kv_eb + kv_in_block
    kv_slot = torch.where(kv_valid, kv_slot, torch.full_like(kv_slot, -1))
    token_to_req = b_idx.to(torch.int32)
    return positions, b_idx, state_slot, kv_slot, token_to_req


def _make_tables(bsz: int, seqlen: int, state_eb: int, kv_eb: int, ratio: int):
    state_blocks = seqlen // state_eb + 8
    kv_blocks = seqlen // (kv_eb * ratio) + 8
    state_bt = (
        torch.arange(1, bsz * state_blocks + 1, device="cuda", dtype=torch.int32)
        .reshape(bsz, state_blocks)
        .contiguous()
    )
    kv_bt = (
        torch.arange(1001, 1001 + bsz * kv_blocks, device="cuda", dtype=torch.int32)
        .reshape(bsz, kv_blocks)
        .contiguous()
    )
    state_bt[:, -1] = 0
    kv_bt[:, -1] = 0
    return state_bt, kv_bt


def _bench_m(M: int, trace_dir: str, profile_enabled: bool) -> List[dict]:
    bsz = int(os.environ.get("DSV4_METADATA_BSZ", "1"))
    start_pos = int(os.environ.get("DSV4_METADATA_START_POS", "0"))
    state_eb = int(os.environ.get("DSV4_METADATA_STATE_EB", "256"))
    kv_eb = int(os.environ.get("DSV4_METADATA_KV_EB", "16"))
    ratio = int(os.environ.get("DSV4_METADATA_RATIO", "4"))
    state_bt, kv_bt = _make_tables(bsz, M, state_eb, kv_eb, ratio)

    ref = _torch_prefill_metadata(start_pos, bsz, M, state_bt, state_eb, kv_bt, kv_eb, ratio)
    fused = build_prefill_compressor_metadata(
        start_pos,
        bsz,
        M,
        torch.device("cuda"),
        state_bt,
        state_eb,
        kv_bt,
        kv_eb,
        ratio,
    )
    assert fused is not None
    torch.cuda.synchronize()
    for actual, expected in zip(fused, ref):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

    baseline_fn = lambda: _torch_prefill_metadata(
        start_pos, bsz, M, state_bt, state_eb, kv_bt, kv_eb, ratio
    )
    fused_fn = lambda: build_prefill_compressor_metadata(
        start_pos,
        bsz,
        M,
        torch.device("cuda"),
        state_bt,
        state_eb,
        kv_bt,
        kv_eb,
        ratio,
    )

    warmup = 20 if M <= 4096 else 8
    iters = iters_for_m(M)
    baseline = measure_kernel(
        baseline_fn,
        label=f"compressor_metadata_M{M}_torch",
        trace_dir=trace_dir,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    fused_measure = measure_kernel(
        fused_fn,
        label=f"compressor_metadata_M{M}_triton",
        trace_dir=trace_dir,
        kernel_regex=METADATA_KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    speedup = baseline.kernel_span_us / fused_measure.kernel_span_us
    common = {
        "M": M,
        "bsz": bsz,
        "state_eb": state_eb,
        "kv_eb": kv_eb,
        "compress_ratio": ratio,
    }
    return [
        {
            **common,
            "impl": "torch_metadata",
            "speedup_vs_baseline": 1.0,
            **measurement_payload(baseline),
        },
        {
            **common,
            "impl": "triton_prefill_metadata",
            "speedup_vs_baseline": speedup,
            **measurement_payload(fused_measure),
        },
    ]


def _print_summary(rows: List[dict]) -> None:
    print("\n[compressor_metadata]")
    print("  {:>8} {:>24} {:>12} {:>7} {:>9}".format("M", "impl", "span_us", "kernels", "speedup"))
    for row in rows:
        print(
            "  {:8d} {:>24} {:12.2f} {:7d} {:8.3f}x".format(
                row["M"],
                row["impl"],
                row["kernel_span_us"],
                row["kernel_count"],
                row["speedup_vs_baseline"],
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CompressorMetadataPerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_METADATA_M_LIST", [16, 256, 1024, 16384, 65536])
        report_path = report_path_from_env(
            "DSV4_METADATA_JSON", "dsv4_compressor_metadata_perf.json"
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_METADATA_TRACE_DIR")
        profile_enabled = env_flag("DSV4_METADATA_PROFILE", True)

        rows: List[dict] = []
        for M in m_list:
            rows.extend(_bench_m(M, trace_dir, profile_enabled))
            torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
            "device": device_payload(),
            "m_list": m_list,
            "profile_enabled": profile_enabled,
            "results": rows,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote compressor metadata perf JSON: {report_path}")
        self.assertTrue(rows)

        if env_flag("DSV4_PERF_ASSERT", False):
            fused_rows = [r for r in rows if r["impl"] == "triton_prefill_metadata"]
            self.assertTrue(fused_rows)
            self.assertLessEqual(max(r["kernel_count"] for r in fused_rows), 1)
            self.assertGreaterEqual(min(r["speedup_vs_baseline"] for r in fused_rows), 1.5)


if __name__ == "__main__":
    unittest.main()
