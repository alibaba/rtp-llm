"""CP4 mixed-length transport parity through actual NCCL and native projection."""

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
    "cp_mixed_contracts",
    Path(__file__).with_name("test_attention_v41_cp_raw_groups.py"),
)
CONTRACTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CONTRACTS)

# Ragged CP4 shape: one oversized request between two bounded request groups.
LENGTHS = (503, 159, 3263, 583, 8127, 1940, 5177, 8370, 2913, 159357, 14606, 9695)


def _measure(fn):
    # CUDA events intentionally include CPU launch gaps in this transport path.
    # Inputs, weights, output, events and cache-flush allocation are not timed;
    # production iterator temporary allocations remain part of the operation.
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    for _ in range(30):
        fn()
    flush.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "median": statistics.median(samples),
        "p90": sorted(samples)[89],
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


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
        compressor = SimpleNamespace(_linear_bf16_bf16_fp32=native_projection)
        cp, packed, _ = CONTRACTS._case(LENGTHS)
        cp.cp_rank = rank
        cp.unpad_restore = cp.unpad_restore.to(device)
        cp.input_lengths_global = cp.input_lengths_global.to(device)
        # Token identity is independently encoded before the rank-local repeat.
        x = packed[rank].to(device).repeat(1, 1280).contiguous()
        del packed
        assert x.shape == (53682, 5120)
        summaries = []
        with patch.object(
            collective_torch, "_get_group", return_value=dist.group.WORLD
        ):
            with CONTRACTS._cpu_compressor(compressor):
                for projections in (1, 2):
                    groups = attention._prefill_x_group_plan(
                        x, cp, allow_owner_tiles=True, output_width=512 * projections
                    )
                    expected_gathers = 3 if projections == 1 else 4
                    assert len(groups) == expected_gathers
                    torch.manual_seed(20261026)
                    weights = tuple(
                        (torch.randn(512, 5120, device=device) * 0.001).bfloat16()
                        for _ in range(projections)
                    )
                    output = torch.empty(
                        (sum(LENGTHS), 512 * projections),
                        device=device,
                        dtype=torch.float32,
                    )

                    def execute(mixed):
                        if mixed:
                            tiles = attention._prefill_projected_x_groups(
                                x, cp, groups, weights, whole_groups=False
                            )
                        else:
                            plan = iter(attention._prefill_x_tile_plan(cp))
                            first = next(plan)
                            buffers = [
                                output.new_empty(
                                    (
                                        attention._PRODUCE_GLOBAL_TILE_ROWS,
                                        output.shape[1],
                                    )
                                )
                                for _ in range(2)
                            ]
                            pending = attention._start_prefill_x_gather_async(
                                x, cp, first, buffers[0], weights
                            )
                            tiles = attention._prefill_x_tiles(
                                x, cp, plan, pending, buffers, weights
                            )
                        with torch.inference_mode():
                            for start, tile in tiles:
                                output[start : start + tile.shape[0]].copy_(tile)
                                # Match _produce_global's release before next().
                                del tile
                        return output

                    # Pure counters must not retain tensor arguments as Mock
                    # call histories would, which inflates cross-group peaks.
                    counts = {"broadcast": 0, "gather": 0}
                    real_broadcast = dist.broadcast
                    real_gather = dist.all_gather_into_tensor

                    def count_broadcast(*args, **kwargs):
                        counts["broadcast"] += 1
                        return real_broadcast(*args, **kwargs)

                    def count_gather(*args, **kwargs):
                        counts["gather"] += 1
                        return real_gather(*args, **kwargs)

                    with (
                        patch.object(dist, "broadcast", count_broadcast),
                        patch.object(dist, "all_gather_into_tensor", count_gather),
                    ):
                        expected = execute(False).clone()
                        torch.cuda.synchronize()
                        assert counts == {"broadcast": 96, "gather": 0}
                        counts["broadcast"] = 0
                        output.fill_(float("nan"))
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        actual = execute(True)
                        torch.cuda.synchronize()
                        peak_temporary_bytes = (
                            torch.cuda.max_memory_allocated() - before
                        )
                        assert counts == {"broadcast": 0, "gather": expected_gathers}
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        assert torch.isfinite(actual).all()
                    previous_first = expected[0].clone()
                    del expected
                    # Change all live token values and weights; repeat the parity
                    # check so a stale gather or buffer reuse cannot pass.
                    x.add_(1)
                    weights = tuple(weight.neg() for weight in weights)
                    expected = execute(False).clone()
                    assert not torch.equal(expected[0], previous_first)
                    output.fill_(float("nan"))
                    torch.testing.assert_close(execute(True), expected, rtol=0, atol=0)
                    torch.cuda.synchronize()
                    del expected, previous_first

                    if os.getenv("RTP_V41_CP_MIXED_BENCHMARK", "0") == "1":
                        timings = {}
                        for mixed in (False, True):
                            label = "mixed" if mixed else "owner"
                            timings[label] = _measure(lambda: execute(mixed))
                            # Profiling is separate from uninstrumented timing.
                            dist.barrier()
                            with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ],
                                record_shapes=True,
                            ) as profiler:
                                execute(mixed)
                                torch.cuda.synchronize()
                            profiler.export_chrome_trace(
                                str(
                                    Path(output_dir)
                                    / f"{label}_p{projections}_rank{rank}.json"
                                )
                            )
                        summaries.append(
                            {
                                "projections": projections,
                                "rank": rank,
                                "cuda_event_ms": timings,
                                "scope": "CP transport plus original projection tiles and identical copy consumer; includes launch gaps; not model E2E",
                                "baseline_broadcasts": 96,
                                "mixed_broadcasts": 0,
                                "mixed_all_gathers": expected_gathers,
                                "peak_temporary_bytes": peak_temporary_bytes,
                                "warmups": 30,
                                "timed_samples": 100,
                                "torch_compile": False,
                                "exact_parity": True,
                            }
                        )
                    del output, weights
        if summaries:
            Path(output_dir, f"cp_mixed_rank{rank}.json").write_text(
                json.dumps(summaries, indent=2) + "\n"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CPMixedNCCLTest(unittest.TestCase):
    def test_mixed_length_cp4_projection_and_collectives(self):
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
