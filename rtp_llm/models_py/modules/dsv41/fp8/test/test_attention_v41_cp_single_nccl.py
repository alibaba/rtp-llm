"""CP4 long-request owner-view parity with real NCCL and native FP32 projection.

The oracle is the old 32-broadcast iterator. Timing covers transport, original
projection GEMMs and an identical copy consumer, not model end-to-end latency.
"""

import importlib.util
import json
import os
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_SPEC = importlib.util.spec_from_file_location(
    "cp_single_mixed_helpers",
    Path(__file__).with_name("test_attention_v41_cp_mixed_nccl.py"),
)
MIXED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MIXED)
CONTRACTS = MIXED.CONTRACTS
TOTAL = 1048448
HALF = 131056
TILE_ROWS = (32768, 32768, 32768, 32752)
WORKSPACE_CAP = 2560 * 1024**2
# Native library bookkeeping may allocate after the baseline measurement. This
# allowance is reported separately; it is not another group-sized allocation.
NATIVE_ALLOWANCE = 32 * 1024**2


def _expected_tiles():
    result = []
    offset = 0
    for _ in range(8):
        for rows in TILE_ROWS:
            result.append((offset, rows))
            offset += rows
    assert offset == TOTAL
    return result


def _expected_projections(projections):
    result = []
    for half_start in (0, HALF):
        start = half_start
        for rows in TILE_ROWS:
            for column in range(projections):
                result.append((start, rows, column))
            start += rows
    return result


def _worker(rank, rendezvous, output_dir):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + rendezvous, rank=rank, world_size=4
    )
    try:
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        attention = CONTRACTS.ATTENTION
        device = torch.device("cuda", rank)
        comm = torch.cuda.Stream(device=device)
        attention._get_cp_comm_stream = lambda unused: comm
        native_projection = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32
        cp, packed, token_oracle = CONTRACTS._case((TOTAL,))
        cp.cp_rank = rank
        cp.unpad_restore = cp.unpad_restore.to(device)
        cp.input_lengths_global = cp.input_lengths_global.to(device)
        x = packed[rank].to(device).repeat(1, 1280).contiguous()
        del packed, token_oracle
        assert tuple(x.shape) == (2 * HALF, 5120)
        expected_tiles = _expected_tiles()
        summaries = []
        # These counters retain only scalar metadata, never tensor arguments.
        recording = False
        broadcast_calls, gather_calls, projection_calls = [], [], []
        real_broadcast = dist.broadcast
        real_gather = dist.all_gather_into_tensor

        def project(raw, weight):
            assert raw.dtype == weight.dtype == torch.bfloat16
            assert raw.is_contiguous() and weight.is_contiguous()
            if recording:
                byte_offset = raw.data_ptr() - x.data_ptr()
                row_bytes = 5120 * x.element_size()
                assert byte_offset % row_bytes == 0
                column = next(
                    i
                    for i, candidate in enumerate(weights)
                    if candidate.data_ptr() == weight.data_ptr()
                )
                projection_calls.append(
                    (byte_offset // row_bytes, raw.shape[0], column)
                )
            result = native_projection(raw, weight)
            assert result.dtype == torch.float32
            assert tuple(result.shape) == (raw.shape[0], 512)
            return result

        def broadcast(tensor, *args, **kwargs):
            if recording:
                broadcast_calls.append((kwargs["src"], tuple(tensor.shape)))
            return real_broadcast(tensor, *args, **kwargs)

        def gather(output, source, *args, **kwargs):
            if recording:
                gather_calls.append((tuple(output.shape), tuple(source.shape)))
            return real_gather(output, source, *args, **kwargs)

        compressor = SimpleNamespace(_linear_bf16_bf16_fp32=project)
        with (
            patch.object(collective_torch, "_get_group", return_value=dist.group.WORLD),
            patch.object(dist, "broadcast", broadcast),
            patch.object(dist, "all_gather_into_tensor", gather),
            CONTRACTS._cpu_compressor(compressor),
        ):
            for projections in (1, 2):
                width = 512 * projections
                groups = attention._prefill_x_group_plan(
                    x, cp, allow_owner_tiles=True, output_width=width
                )
                expected_groups = (
                    ((0, 2 * HALF, 0, TOTAL),)
                    if projections == 1
                    else ((0, HALF, 0, TOTAL // 2), (HALF, 2 * HALF, TOTAL // 2, TOTAL))
                )
                assert groups == expected_groups
                logical_peak = max(
                    5 * (end - start) * width * 4 for start, end, _, _ in groups
                )
                assert logical_peak <= WORKSPACE_CAP
                torch.manual_seed(20261026)
                weights = tuple(
                    (torch.randn(512, 5120, device=device) * 0.001).bfloat16()
                    for _ in range(projections)
                )
                # Both outputs and the boundary-consumer state exist before any
                # peak baseline. Candidate workspace is never preallocated here.
                expected = torch.empty(
                    (TOTAL, width), device=device, dtype=torch.float32
                )
                output = torch.empty_like(expected)
                expected_seams = output.new_empty((32, 8, width))
                actual_seams = torch.empty_like(expected_seams)
                carry = output.new_empty((3, width))

                def iterator(candidate):
                    if candidate:
                        return attention._prefill_projected_x_groups(
                            x, cp, groups, weights, whole_groups=False
                        )
                    plan = iter(attention._prefill_x_tile_plan(cp))
                    first = next(plan)
                    buffers = [output.new_empty((32768, width)) for _ in range(2)]
                    pending = attention._start_prefill_x_gather_async(
                        x, cp, first, buffers[0], weights
                    )
                    return attention._prefill_x_tiles(
                        x, cp, plan, pending, buffers, weights
                    )

                def execute(candidate, destination, seams=None):
                    seen = []
                    if seams is not None:
                        carry.zero_()
                    tiles = iterator(candidate)
                    with torch.inference_mode():
                        try:
                            for start, tile in tiles:
                                # enumerate caches its previous result tuple;
                                # that retains the prior group's last view
                                # while the generator allocates its next group.
                                index = len(seen)
                                # Destination offsets alone could hide reordered
                                # yields. Compare the exact stream independently.
                                item = (start, tile.shape[0])
                                assert index < 32 and item == expected_tiles[index]
                                seen.append(item)
                                destination[start : start + tile.shape[0]].copy_(tile)
                                if seams is not None:
                                    seams[index, :3].copy_(carry)
                                    seams[index, 3:].copy_(tile[:5])
                                    carry.copy_(tile[-3:])
                                del tile
                        finally:
                            tiles.close()
                    assert seen == expected_tiles
                    return destination

                def clear_counts():
                    broadcast_calls.clear()
                    gather_calls.clear()
                    projection_calls.clear()

                def check_counts(candidate):
                    assert projection_calls == _expected_projections(projections)
                    if candidate:
                        assert broadcast_calls == []
                        assert gather_calls == [
                            ((4 * (end - start), width), (end - start, width))
                            for start, end, _, _ in groups
                        ]
                    else:
                        assert gather_calls == []
                        assert broadcast_calls == [
                            (owner, (rows, width))
                            for owner in (0, 1, 2, 3, 3, 2, 1, 0)
                            for rows in TILE_ROWS
                        ]

                peaks = []
                for changed in (False, True):
                    # Second pass creates and consumes the changed inputs on a
                    # non-default compute stream; the transport uses its own
                    # separate stream and must provide both dependency edges.
                    compute = (
                        torch.cuda.Stream(device=device)
                        if changed
                        else torch.cuda.current_stream()
                    )
                    previous_first = expected[0].clone() if changed else None
                    compute.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(compute):
                        if changed:
                            x.add_(1)
                            weights = tuple(weight.neg() for weight in weights)
                        recording = True
                        clear_counts()
                        expected.fill_(float("nan"))
                        execute(False, expected, expected_seams)
                        check_counts(False)
                        torch.cuda.synchronize()
                        clear_counts()
                        output.fill_(float("nan"))
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        execute(True, output, actual_seams)
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated() - before
                        peaks.append(peak)
                        check_counts(True)
                        recording = False
                        assert peak <= WORKSPACE_CAP + NATIVE_ALLOWANCE, (
                            peak,
                            WORKSPACE_CAP,
                            NATIVE_ALLOWANCE,
                        )
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            actual_seams, expected_seams, rtol=0, atol=0
                        )
                        assert torch.isfinite(output).all()
                        if changed:
                            assert not torch.equal(expected[0], previous_first)
                    torch.cuda.current_stream().wait_stream(compute)
                    torch.cuda.synchronize()
                    del previous_first, compute

                # Every rank leaves at the SAME yield boundary, including the
                # first tile after the front/back boundary. Then check both
                # allocation release and a subsequent real collective.
                for stop_after in (1, 17):
                    for action in ("close", "throw"):
                        dist.barrier()
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        tiles = iterator(True)
                        for _ in range(stop_after):
                            start, tile = next(tiles)
                            output[start : start + tile.shape[0]].copy_(tile)
                            del tile
                        if action == "close":
                            tiles.close()
                        else:
                            try:
                                tiles.throw(RuntimeError("intentional consumer stop"))
                            except RuntimeError as error:
                                assert str(error) == "intentional consumer stop"
                            else:
                                raise AssertionError(
                                    "generator swallowed consumer exception"
                                )
                        del tiles
                        torch.cuda.synchronize()
                        assert (
                            torch.cuda.memory_allocated() - before <= NATIVE_ALLOWANCE
                        )
                        probe = torch.tensor([rank + 1], device=device)
                        dist.all_reduce(probe)
                        assert probe.item() == 10
                        del probe
                        dist.barrier()

                summary = {
                    "rank": rank,
                    "projections": projections,
                    "input_tokens": TOTAL,
                    "local_rows": 2 * HALF,
                    "groups": groups,
                    "yield_order": expected_tiles,
                    "native_projection_calls": _expected_projections(projections),
                    "baseline_broadcasts": 32,
                    "candidate_broadcasts": 0,
                    "candidate_all_gathers": len(groups),
                    "exact_fp32_parity": True,
                    "boundary_carry_parity": True,
                    "changed_inputs_and_weights": True,
                    "non_default_compute_stream": True,
                    "close_and_throw_at_yields": [1, 17],
                    "logical_send_gather_peak_bytes": logical_peak,
                    "peak_temporary_bytes": peaks,
                    "workspace_cap_bytes": WORKSPACE_CAP,
                    "native_allowance_bytes": NATIVE_ALLOWANCE,
                    "memory_scope": (
                        "max_memory_allocated minus preallocated common inputs/weights/two outputs/boundary state; "
                        "includes candidate workspace; excludes reserved and external NCCL allocations"
                    ),
                    "scope": (
                        "CP transport plus original native projection tiles and identical copy consumer; "
                        "includes launch gaps; not model E2E"
                    ),
                    "torch_compile": False,
                }
                if os.getenv("RTP_V41_CP_SINGLE_BENCHMARK", "0") == "1":
                    timings = {}
                    for candidate in (False, True):
                        label = "single" if candidate else "owner"
                        timings[label] = MIXED._measure(
                            lambda: execute(candidate, output)
                        )
                        dist.barrier()
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                        ) as profiler:
                            with torch.profiler.record_function(
                                f"cp_single.{label}.w{width}"
                            ):
                                execute(candidate, output)
                                torch.cuda.synchronize()
                        profiler.export_chrome_trace(
                            str(
                                Path(output_dir)
                                / f"{label}_p{projections}_rank{rank}.json"
                            )
                        )
                    per_rank = [None] * 4
                    dist.all_gather_object(per_rank, timings)
                    max_rank = {}
                    for label in ("owner", "single"):
                        samples = [
                            max(item[label]["samples"][i] for item in per_rank)
                            for i in range(100)
                        ]
                        max_rank[label] = {
                            "samples": samples,
                            "median": statistics.median(samples),
                            "p90": sorted(samples)[89],
                            "mean": statistics.mean(samples),
                        }
                    summary.update(
                        cuda_event_ms=timings,
                        per_sample_max_rank_ms=max_rank,
                        warmups=30,
                        timed_samples=100,
                    )
                summaries.append(summary)
                del output, expected, expected_seams, actual_seams, carry, weights
        Path(output_dir, f"cp_single_rank{rank}.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CPSingleNCCLTest(unittest.TestCase):
    def test_single_long_cp4_projection_and_collectives(self):
        self.assertTrue(torch.cuda.is_available(), "This target requires CUDA")
        self.assertGreaterEqual(
            torch.cuda.device_count(), 4, "This target requires CP4"
        )
        output_dir = os.environ.get(
            "TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir()
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(
                _worker, args=(str(Path(tmp) / "rendezvous"), output_dir), nprocs=4
            )


if __name__ == "__main__":
    unittest.main()
