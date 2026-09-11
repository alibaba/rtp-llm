"""Bitwise dynamic-batch regression and GPU timing for shared-L1 scales."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import unittest

import torch

# Load the production leaf module without initializing unrelated MoE wrappers.
module_path = Path(__file__).parents[2] / "models_py/modules/glm5_mega_moe/shared_fp8_scale.py"
spec = importlib.util.spec_from_file_location("shared_fp8_scale_under_test", module_path)
scales = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scales
spec.loader.exec_module(scales)


def stage_full_buffer_reference(source, destination, tokens, block_m):
    """Reproduce the old full-buffer launch only for regression and timing."""
    block_rows = 128
    destination_rows = destination.size(0)
    grid = (scales.triton.cdiv(destination_rows, block_rows), source.size(1))
    scales._stage_shared_fp8_scale_kernel[grid](
        source,
        destination,
        tokens,
        destination_rows,
        source.stride(0),
        source.stride(1),
        destination.stride(0),
        destination.stride(1),
        BLOCK_M=block_m,
        ALIGNED_BLOCK_M=((block_m + 127) // 128) * 128,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )


class SharedFp8ScaleGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This target requires the remote L20D_DEV GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(419)

    def test_dynamic_batches_and_graph_replay_match_full_staging(self):
        # Covers draft, target-verify, non-capture-aligned batches, and repeated
        # growth/shrink. Poisoning the unused tail catches accidental clearing;
        # the active prefix must exactly match the previous full-buffer kernel.
        sizes = (1, 16, 32, 48, 64, 8, 47, 2, 63, 16, 256, 192, 193, 384, 513, 1)
        checks = 0
        for block_m in (32, 64, 128, 192):
            for column_major in (False, True):
                source = (torch.empty_strided((768, 48), (1, 768), device="cuda", dtype=torch.int32)
                          if column_major else torch.empty((768, 48), device="cuda", dtype=torch.int32))
                destination = torch.empty_strided((4096, 48), (1, 4096), device="cuda", dtype=torch.int32)
                reference = torch.empty_strided((4096, 48), (1, 4096), device="cuda", dtype=torch.int32)
                destination.fill_(123456789)
                graphs = {}
                for tokens in sizes:
                    source.random_(0, 2**30)
                    active_rows = ((tokens + block_m - 1) // block_m) * ((block_m + 127) // 128 * 128)
                    tail_before = destination[active_rows:].clone()
                    if tokens not in graphs:
                        # Compile before capture. Capture runs the same code
                        # again, then later replays use fresh source contents.
                        scales.stage_shared_fp8_input_scales(source, destination, tokens, block_m)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            scales.stage_shared_fp8_input_scales(source, destination, tokens, block_m)
                        graphs[tokens] = graph
                    source.random_(0, 2**30)
                    graphs[tokens].replay()
                    stage_full_buffer_reference(source, reference, tokens, block_m)
                    torch.testing.assert_close(destination[:active_rows], reference[:active_rows], rtol=0, atol=0)
                    torch.testing.assert_close(destination[active_rows:], tail_before, rtol=0, atol=0)
                    # Independent forward permutation also verifies the input
                    # rows; padding correctness is checked against full staging.
                    rows = scales.shared_fp8_scale_row_indices(tokens, block_m, source.device)
                    torch.testing.assert_close(destination[rows], source[:tokens], rtol=0, atol=0)
                    checks += 1
                scales.stage_shared_fp8_input_scales(source, destination, 0, block_m)
                self.assertEqual(torch.count_nonzero(destination).item(), 0)
        print(f"Shared scale bitwise checks passed: {checks} dynamic/captured cases", flush=True)

    def test_large_capacity_gpu_latency(self):
        # Exact capacity/width observed in the GLM5.3 DP8 timeline. Graph timing
        # excludes Python and CUPTI and retains the legacy padding behavior in
        # the control graph. Both graphs read the same packed input scales.
        source = torch.zeros((768, 48), device="cuda", dtype=torch.int32)
        destination = torch.empty_strided((264192, 48), (1, 264192), device="cuda", dtype=torch.int32)
        graphs = {}
        for active, stage in (
            (False, stage_full_buffer_reference),
            (True, scales.stage_shared_fp8_input_scales),
        ):
            for _ in range(3):
                stage(source, destination, 192, 64)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32):
                    stage(source, destination, 192, 64)
            graphs[active] = graph
        results = {False: [], True: []}
        for _ in range(6):
            for active in (False, True):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    graphs[active].replay()
                end.record()
                end.synchronize()
                results[active].append(start.elapsed_time(end) * 1000 / (32 * 8))
        summary = {"full_buffer_us": results[False], "active_prefix_us": results[True],
                   "speedup": statistics.mean(results[False]) / statistics.mean(results[True])}
        Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "shared_scale_gpu_timing.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        self.assertGreater(summary["speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
