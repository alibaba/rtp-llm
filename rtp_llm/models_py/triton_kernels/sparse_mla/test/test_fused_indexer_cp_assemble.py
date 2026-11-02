import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import (
    fused_indexer_cp_assemble as assemble,
)

SKIP_REASON = "CUDA required"


def _has_cuda():
    return torch.cuda.is_available()


class FusedIndexerAssembleTest(unittest.TestCase):

    @unittest.skipIf(not _has_cuda(), SKIP_REASON)
    def test_assemble_retains_fast_launch_and_collective_packing(self):
        for case in [
            {"packed": False, "fast": False},
            {"packed": False, "fast": True},
            {"packed": True, "fast": False},
            {"packed": True, "fast": True},
        ]:
            with self.subTest(**case), ExitStack() as patches:
                packed = case["packed"]
                fast = case["fast"]
                patches.enter_context(
                    patch.object(assemble, "cp_opt_enabled", lambda: packed)
                )
                patches.enter_context(
                    patch.object(assemble._cpu_launch, "ENABLED", fast)
                )
                (launches, gathers) = ([], [])
                launch = assemble._cpu_launch.launch

                def tracked_launch(jit, grid, args, **kwargs):
                    launches.append(jit)
                    return launch(jit, grid, args, **kwargs)

                def gather(value, **kwargs):
                    gathers.append(kwargs["role"])
                    return torch.cat((value, value))

                patches.enter_context(
                    patch.object(assemble._cpu_launch, "launch", tracked_launch)
                )
                patches.enter_context(patch.object(assemble, "all_gather", gather))
                k = torch.randint(0, 256, (2, 128), dtype=torch.uint8, device="cuda")
                scale = torch.randint(0, 256, (2, 4), dtype=torch.uint8, device="cuda")
                mapping = torch.tensor([0, -1, 1, -1], device="cuda")
                restore = torch.tensor([6, 0, 3, 4, 2], device="cuda")
                plan = SimpleNamespace(
                    total_local_T=4, total_actual_local_T=2, restore_indices=restore
                )
                local_k = torch.zeros((4, 128), dtype=torch.uint8, device="cuda")
                local_s = torch.zeros((4, 4), dtype=torch.uint8, device="cuda")
                (local_k[0], local_k[2]) = (k[0], k[1])
                (local_s[0], local_s[2]) = (scale[0], scale[1])
                (expected_k, expected_s) = (
                    torch.cat((local_k, local_k))[restore],
                    torch.cat((local_s, local_s))[restore],
                )
                for _ in range(3):
                    out_k = torch.empty((5, 128), dtype=torch.uint8, device="cuda")
                    out_s = torch.empty((5, 4), dtype=torch.uint8, device="cuda")
                    assemble.fused_copy_and_assemble_indexer_k(
                        plan=plan,
                        actual_k_quant=k.view(torch.float8_e4m3fn),
                        actual_k_scale=scale,
                        out_k_quant=out_k.view(torch.float8_e4m3fn),
                        out_k_scale=out_s,
                        src_for_padded=mapping,
                    )
                    assert torch.equal(out_k, expected_k)
                    assert torch.equal(out_s, expected_s)
                assert len(gathers) == (3 if packed else 6)
                assert len(launches) == 6

    def test_master_switch_on_uses_one_packed_all_gather(self):
        for case in [{}]:
            with self.subTest(**case), ExitStack() as patches:
                from rtp_llm.models_py.triton_kernels.sparse_mla import (
                    fused_indexer_cp_assemble as assemble,
                )

                q_bytes = torch.arange(12, dtype=torch.uint8).view(3, 4)
                local_q = q_bytes.view(torch.float8_e4m3fn)
                local_s = torch.tensor(
                    [[101, 102], [103, 104], [105, 106]], dtype=torch.uint8
                )
                calls = []

                def fake_all_gather(payload, group=None, **kwargs):
                    calls.append((payload.clone(), group, kwargs))
                    return payload

                patches.enter_context(
                    patch.object(assemble, "cp_opt_enabled", lambda: True)
                )
                patches.enter_context(
                    patch.object(assemble, "all_gather", fake_all_gather)
                )
                (gathered, gathered_s, q_cols, s_cols) = assemble._all_gather_indexer_k(
                    local_q, local_s, group="tp"
                )
                assert len(calls) == 1
                assert calls[0][2]["role"] == "indexer_k_packed"
                assert gathered_s is None
                assert tuple(gathered.shape) == (3, 6)
                assert (q_cols, s_cols) == (4, 2)
                assert torch.equal(gathered[:, :q_cols], q_bytes)
                assert torch.equal(gathered[:, q_cols:], local_s)

    def test_master_switch_off_uses_two_separate_all_gathers(self):
        for case in [{}]:
            with self.subTest(**case), ExitStack() as patches:
                from rtp_llm.models_py.triton_kernels.sparse_mla import (
                    fused_indexer_cp_assemble as assemble,
                )

                local_q = (
                    torch.arange(12, dtype=torch.uint8)
                    .view(3, 4)
                    .view(torch.float8_e4m3fn)
                )
                local_s = torch.arange(6, dtype=torch.uint8).view(3, 2)
                calls = []

                def fake_all_gather(payload, group=None, **kwargs):
                    calls.append((payload, group, kwargs))
                    return payload

                patches.enter_context(
                    patch.object(assemble, "cp_opt_enabled", lambda: False)
                )
                patches.enter_context(
                    patch.object(assemble, "all_gather", fake_all_gather)
                )
                (gathered_q, gathered_s, q_cols, s_cols) = (
                    assemble._all_gather_indexer_k(local_q, local_s, group="tp")
                )
                assert [call[2]["role"] for call in calls] == [
                    "indexer_k_quant",
                    "indexer_k_scale",
                ]
                assert gathered_q is local_q
                assert gathered_s is local_s
                assert (q_cols, s_cols) == (4, 2)

    @unittest.skipIf(not _has_cuda(), SKIP_REASON)
    def test_fused_padded_wire_bytes(self):
        for case in [
            {
                "actual_lens": [0],
                "padded_lens": [0],
                "width": 128,
                "scale_width": 4,
                "fast_launch": False,
            },
            {
                "actual_lens": [0],
                "padded_lens": [0],
                "width": 128,
                "scale_width": 4,
                "fast_launch": True,
            },
            {
                "actual_lens": [0],
                "padded_lens": [0],
                "width": 256,
                "scale_width": 8,
                "fast_launch": False,
            },
            {
                "actual_lens": [0],
                "padded_lens": [0],
                "width": 256,
                "scale_width": 8,
                "fast_launch": True,
            },
            {
                "actual_lens": [0],
                "padded_lens": [0],
                "width": 17,
                "scale_width": 3,
                "fast_launch": False,
            },
            {
                "actual_lens": [0],
                "padded_lens": [0],
                "width": 17,
                "scale_width": 3,
                "fast_launch": True,
            },
            {
                "actual_lens": [0],
                "padded_lens": [5],
                "width": 128,
                "scale_width": 4,
                "fast_launch": False,
            },
            {
                "actual_lens": [0],
                "padded_lens": [5],
                "width": 128,
                "scale_width": 4,
                "fast_launch": True,
            },
            {
                "actual_lens": [0],
                "padded_lens": [5],
                "width": 256,
                "scale_width": 8,
                "fast_launch": False,
            },
            {
                "actual_lens": [0],
                "padded_lens": [5],
                "width": 256,
                "scale_width": 8,
                "fast_launch": True,
            },
            {
                "actual_lens": [0],
                "padded_lens": [5],
                "width": 17,
                "scale_width": 3,
                "fast_launch": False,
            },
            {
                "actual_lens": [0],
                "padded_lens": [5],
                "width": 17,
                "scale_width": 3,
                "fast_launch": True,
            },
            {
                "actual_lens": [7],
                "padded_lens": [7],
                "width": 128,
                "scale_width": 4,
                "fast_launch": False,
            },
            {
                "actual_lens": [7],
                "padded_lens": [7],
                "width": 128,
                "scale_width": 4,
                "fast_launch": True,
            },
            {
                "actual_lens": [7],
                "padded_lens": [7],
                "width": 256,
                "scale_width": 8,
                "fast_launch": False,
            },
            {
                "actual_lens": [7],
                "padded_lens": [7],
                "width": 256,
                "scale_width": 8,
                "fast_launch": True,
            },
            {
                "actual_lens": [7],
                "padded_lens": [7],
                "width": 17,
                "scale_width": 3,
                "fast_launch": False,
            },
            {
                "actual_lens": [7],
                "padded_lens": [7],
                "width": 17,
                "scale_width": 3,
                "fast_launch": True,
            },
            {
                "actual_lens": [3],
                "padded_lens": [8],
                "width": 128,
                "scale_width": 4,
                "fast_launch": False,
            },
            {
                "actual_lens": [3],
                "padded_lens": [8],
                "width": 128,
                "scale_width": 4,
                "fast_launch": True,
            },
            {
                "actual_lens": [3],
                "padded_lens": [8],
                "width": 256,
                "scale_width": 8,
                "fast_launch": False,
            },
            {
                "actual_lens": [3],
                "padded_lens": [8],
                "width": 256,
                "scale_width": 8,
                "fast_launch": True,
            },
            {
                "actual_lens": [3],
                "padded_lens": [8],
                "width": 17,
                "scale_width": 3,
                "fast_launch": False,
            },
            {
                "actual_lens": [3],
                "padded_lens": [8],
                "width": 17,
                "scale_width": 3,
                "fast_launch": True,
            },
            {
                "actual_lens": [0, 3, 0, 2],
                "padded_lens": [4, 4, 4, 4],
                "width": 128,
                "scale_width": 4,
                "fast_launch": False,
            },
            {
                "actual_lens": [0, 3, 0, 2],
                "padded_lens": [4, 4, 4, 4],
                "width": 128,
                "scale_width": 4,
                "fast_launch": True,
            },
            {
                "actual_lens": [0, 3, 0, 2],
                "padded_lens": [4, 4, 4, 4],
                "width": 256,
                "scale_width": 8,
                "fast_launch": False,
            },
            {
                "actual_lens": [0, 3, 0, 2],
                "padded_lens": [4, 4, 4, 4],
                "width": 256,
                "scale_width": 8,
                "fast_launch": True,
            },
            {
                "actual_lens": [0, 3, 0, 2],
                "padded_lens": [4, 4, 4, 4],
                "width": 17,
                "scale_width": 3,
                "fast_launch": False,
            },
            {
                "actual_lens": [0, 3, 0, 2],
                "padded_lens": [4, 4, 4, 4],
                "width": 17,
                "scale_width": 3,
                "fast_launch": True,
            },
            {
                "actual_lens": [513, 1, 0],
                "padded_lens": [768, 256, 256],
                "width": 128,
                "scale_width": 4,
                "fast_launch": False,
            },
            {
                "actual_lens": [513, 1, 0],
                "padded_lens": [768, 256, 256],
                "width": 128,
                "scale_width": 4,
                "fast_launch": True,
            },
            {
                "actual_lens": [513, 1, 0],
                "padded_lens": [768, 256, 256],
                "width": 256,
                "scale_width": 8,
                "fast_launch": False,
            },
            {
                "actual_lens": [513, 1, 0],
                "padded_lens": [768, 256, 256],
                "width": 256,
                "scale_width": 8,
                "fast_launch": True,
            },
            {
                "actual_lens": [513, 1, 0],
                "padded_lens": [768, 256, 256],
                "width": 17,
                "scale_width": 3,
                "fast_launch": False,
            },
            {
                "actual_lens": [513, 1, 0],
                "padded_lens": [768, 256, 256],
                "width": 17,
                "scale_width": 3,
                "fast_launch": True,
            },
        ]:
            with self.subTest(**case), ExitStack() as patches:
                actual_lens = case["actual_lens"]
                padded_lens = case["padded_lens"]
                width = case["width"]
                scale_width = case["scale_width"]
                fast_launch = case["fast_launch"]
                from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
                    _pack_padded_k_scale,
                )

                patches.enter_context(
                    patch.object(assemble._cpu_launch, "ENABLED", fast_launch)
                )
                (n, rows) = (sum(actual_lens), sum(padded_lens))
                k = torch.randint(0, 256, (n, width), dtype=torch.uint8, device="cuda")
                scales = torch.randint(
                    0, 256, (n, scale_width), dtype=torch.uint8, device="cuda"
                )
                mapping = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
                ref = torch.zeros(
                    (rows, width + scale_width), dtype=torch.uint8, device="cuda"
                )
                src = dst = 0
                for actual, padded in zip(actual_lens, padded_lens):
                    mapping[dst : dst + actual] = torch.arange(
                        src, src + actual, device="cuda"
                    )
                    ref[dst : dst + actual, :width] = k[src : src + actual]
                    ref[dst : dst + actual, width:] = scales[src : src + actual]
                    src += actual
                    dst += padded
                for repeat in range(3):
                    result = _pack_padded_k_scale(
                        k.view(torch.float8_e4m3fn), scales, rows, mapping
                    )
                    assert torch.equal(result, ref)
                if n == rows or n == 0:
                    result = _pack_padded_k_scale(k, scales, rows)
                    assert torch.equal(result, ref)

    @unittest.skipIf(not _has_cuda(), SKIP_REASON)
    def test_packed_restore_fast_launch_matches_reference(self):
        for case in [
            {"fast_launch": False, "rows": 1},
            {"fast_launch": False, "rows": 17},
            {"fast_launch": False, "rows": 129},
            {"fast_launch": False, "rows": 4096},
            {"fast_launch": True, "rows": 1},
            {"fast_launch": True, "rows": 17},
            {"fast_launch": True, "rows": 129},
            {"fast_launch": True, "rows": 4096},
        ]:
            with self.subTest(**case), ExitStack() as patches:
                fast_launch = case["fast_launch"]
                rows = case["rows"]
                patches.enter_context(
                    patch.object(assemble._cpu_launch, "ENABLED", fast_launch)
                )
                patches.enter_context(
                    patch.object(
                        assemble,
                        "prefill_optimization_setting",
                        lambda name, default: (
                            "1"
                            if name == "RTP_LLM_CP_TILED_INDEXER_RESTORE"
                            else default
                        ),
                    )
                )
                packed = torch.randint(
                    0, 256, (rows, 132), dtype=torch.uint8, device="cuda"
                )
                ids = torch.arange(rows - 1, -1, -1, device="cuda")
                k = torch.empty((rows, 128), dtype=torch.uint8, device="cuda")
                s = torch.empty((rows, 4), dtype=torch.uint8, device="cuda")
                side = torch.cuda.Stream()
                for stream in (
                    torch.cuda.current_stream(),
                    side,
                    torch.cuda.current_stream(),
                ):
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        assemble._restore_packed_indexer_k(packed, k, s, ids)
                    stream.synchronize()
                    assert torch.equal(k, packed[ids, :128])
                    assert torch.equal(s, packed[ids, 128:])
                    packed = packed.clone().add_(1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_zero_scatter_exact(self):
        for case in [{"k_width": 128, "s_width": 4}, {"k_width": 65, "s_width": 3}]:
            with self.subTest(**case), ExitStack() as patches:
                k_width = case["k_width"]
                s_width = case["s_width"]
                (actual_rows, padded_rows) = (17, 24)
                k = torch.randint(
                    0, 256, (actual_rows, k_width), device="cuda", dtype=torch.uint8
                )
                s = torch.randint(
                    0, 256, (actual_rows, s_width), device="cuda", dtype=torch.uint8
                )
                dst = torch.randperm(padded_rows, device="cuda")[:actual_rows]
                inverse = torch.full(
                    (padded_rows,), -1, device="cuda", dtype=torch.int64
                )
                inverse[dst] = torch.arange(actual_rows, device="cuda")
                out_k = torch.empty(
                    (padded_rows, k_width), device="cuda", dtype=torch.uint8
                )
                out_s = torch.empty(
                    (padded_rows, s_width), device="cuda", dtype=torch.uint8
                )
                assemble._fused_zero_scatter_kernel[padded_rows,](
                    k,
                    s,
                    out_k,
                    out_s,
                    inverse,
                    k_width,
                    s_width,
                    padded_rows,
                    assemble._next_power_of_2(k_width),
                    assemble._next_power_of_2(s_width),
                )
                ref_k = torch.zeros_like(out_k)
                ref_s = torch.zeros_like(out_s)
                ref_k.index_copy_(0, dst, k)
                ref_s.index_copy_(0, dst, s)
                assert torch.equal(out_k, ref_k)
                assert torch.equal(out_s, ref_s)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_separate_restore_exact(self):
        for case in [{"k_width": 128, "s_width": 4}, {"k_width": 65, "s_width": 3}]:
            with self.subTest(**case), ExitStack() as patches:
                k_width = case["k_width"]
                s_width = case["s_width"]
                (rows, output_rows) = (37, 23)
                k = torch.randint(
                    0, 256, (rows, k_width), device="cuda", dtype=torch.uint8
                )
                s = torch.randint(
                    0, 256, (rows, s_width), device="cuda", dtype=torch.uint8
                )
                indices = torch.randperm(rows, device="cuda")[:output_rows].long()
                out_k = torch.empty(
                    (output_rows, k_width), device="cuda", dtype=torch.uint8
                )
                out_s = torch.empty(
                    (output_rows, s_width), device="cuda", dtype=torch.uint8
                )
                assemble._gather_restore_separate_kernel[output_rows,](
                    k,
                    s,
                    out_k,
                    out_s,
                    indices,
                    k_width,
                    s_width,
                    output_rows,
                    assemble._next_power_of_2(k_width),
                    assemble._next_power_of_2(s_width),
                )
                assert torch.equal(out_k, k.index_select(0, indices))
                assert torch.equal(out_s, s.index_select(0, indices))


if __name__ == "__main__":
    unittest.main()
