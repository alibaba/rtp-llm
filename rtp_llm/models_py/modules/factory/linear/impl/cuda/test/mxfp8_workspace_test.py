import json
import os
import site
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import mxfp8_ops as ops
from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
    CudaMxfp8Linear,
)
from rtp_llm.models_py.modules.hybrid import dense_mlp

os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "deep_gemm_cache"),
)


class Mxfp8WorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest("MXFP8 workspace requires SM100+")
        import nvidia_cutlass_dsl

        site.addsitedir(str(Path(nvidia_cutlass_dsl.__path__[0]).parent))

    def _linear(self, n: int, k: int) -> CudaMxfp8Linear:
        import flashinfer

        weights = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        q, scales = flashinfer.mxfp8_quantize(
            weights, is_sf_swizzled_layout=False, alignment=32, backend="cute-dsl"
        )
        fp32_scales = torch.exp2(scales.float() - 127).reshape(n, k // 32)
        return CudaMxfp8Linear(q, fp32_scales)

    def _mlp(self, k: int, hidden: int) -> dense_mlp.DenseMLP:
        mlp = dense_mlp.DenseMLP.__new__(dense_mlp.DenseMLP)
        torch.nn.Module.__init__(mlp)
        mlp.up_proj = self._linear(2 * hidden, k)
        mlp.down_proj = self._linear(128, hidden)
        mlp.is_gated = True
        mlp._fuse_silu_quant = False
        mlp._fuse_silu_mxfp8_quant = True
        return mlp

    def _fresh(self, mlp, x):
        with patch.object(dense_mlp, "Mxfp8QuantWorkspace", return_value=None):
            return mlp(x, skip_allreduce=True)

    def test_production_mlp_reuses_scratch_and_allocates_less(self):
        records = []
        for m in (32, 82):
            with self.subTest(m=m):
                mlp = self._mlp(6144, 2048)
                x = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16)
                reference = self._fresh(mlp, x)
                mlp(x, skip_allreduce=True)
                observed = []
                original = dense_mlp.silu_mul_mxfp8_quant_act_packed_fused

                def observe(up, workspace=None):
                    self.assertIsNotNone(workspace)
                    before = (workspace._q.data_ptr(), workspace._scale.data_ptr())
                    result = original(up, workspace=workspace)
                    after = (result[0].data_ptr(), result[1].data_ptr())
                    observed.append((before, after))
                    return result

                with patch.object(
                    dense_mlp, "silu_mul_mxfp8_quant_act_packed_fused", observe
                ):
                    actual = mlp(x, skip_allreduce=True)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, reference))
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0][0], observed[0][1])

                counts = []
                for use_workspace in (False, True):
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as profiler:
                        if use_workspace:
                            mlp(x, skip_allreduce=True)
                        else:
                            self._fresh(mlp, x)
                    counts.append(
                        sum(
                            event.count
                            for event in profiler.key_averages()
                            if event.key
                            in (
                                "aten::empty",
                                "aten::empty_like",
                                "aten::empty_strided",
                            )
                        )
                    )
                self.assertLess(counts[1], counts[0])
                timings = {}
                for use_workspace in (False, True):
                    scope = (
                        patch.object(
                            dense_mlp, "Mxfp8QuantWorkspace", ops.Mxfp8QuantWorkspace
                        )
                        if use_workspace
                        else patch.object(
                            dense_mlp, "Mxfp8QuantWorkspace", return_value=None
                        )
                    )
                    with scope:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(30):
                            mlp(x, skip_allreduce=True)
                        submitted = time.perf_counter()
                        torch.cuda.synchronize()
                        finished = time.perf_counter()
                    timings[str(use_workspace)] = {
                        "submit_us": (submitted - start) * 1e6 / 30,
                        "wall_us": (finished - start) * 1e6 / 30,
                    }
                records.append(
                    {
                        "m": m,
                        "fresh_allocations": counts[0],
                        "workspace_allocations": counts[1],
                        "same_q_and_scale_pointer": True,
                        "output_exact": True,
                        "timings": timings,
                    }
                )
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        (output / "mxfp8_workspace_allocations.json").write_text(
            json.dumps(records, indent=2)
        )

    def test_dynamic_rows_and_widths_preserve_packed_layout(self):
        workspace = ops.Mxfp8QuantWorkspace(
            torch.device("cuda", torch.cuda.current_device())
        )
        for m, k in ((3, 512), (5, 128), (1, 2048), (3, 512), (0, 128)):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            expected_q, expected_scale = ops.mxfp8_quant_act_packed_fused(x)
            q, scale = ops.mxfp8_quant_act_packed_fused(x, workspace)
            self.assertTrue(
                torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
            )
            self.assertEqual(scale.stride(), expected_scale.stride())
            self.assertTrue(torch.equal(scale, expected_scale))

    def test_output_ownership_and_capacity_in_both_directions(self):
        for k, hidden in ((512, 1024), (2048, 512)):
            mlp = self._mlp(k, hidden)
            retained = []
            for m in (1, 3, 32, 82, 32, 3, 1):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                expected = self._fresh(mlp, x)
                actual = mlp(x, skip_allreduce=True)
                retained.append((actual, expected))
            torch.cuda.synchronize()
            for actual, expected in retained:
                self.assertTrue(torch.equal(actual, expected))

    def test_workspace_rejects_other_stream(self):
        workspace = ops.Mxfp8QuantWorkspace(
            torch.device("cuda", torch.cuda.current_device())
        )
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "cannot move"):
                workspace.quantized(3, 128)

    def test_dual_stream_and_graph_outputs_survive_interleaved_forwards(self):
        mlp = self._mlp(512, 256)
        xs = [
            torch.randn(32, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        ]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    mlp(xs[0], skip_allreduce=True)
        torch.cuda.synchronize()
        outputs = []
        for stream, x in zip(streams, xs):
            with torch.cuda.stream(stream):
                outputs.append(mlp(x, skip_allreduce=True))
        torch.cuda.synchronize()
        for output, x in zip(outputs, xs):
            self.assertTrue(torch.equal(output, self._fresh(mlp, x)))

        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        captured = []
        for graph, stream, x in zip(graphs, streams, xs):
            with torch.cuda.graph(graph, stream=stream):
                captured.append(mlp(x, skip_allreduce=True))
        for _ in range(3):
            for x in xs:
                x.neg_()
            torch.cuda.synchronize()
            for graph, stream in zip(graphs, streams):
                with torch.cuda.stream(stream):
                    graph.replay()
            eager = mlp(xs[0] * 2, skip_allreduce=True)
            torch.cuda.synchronize()
            for output, x in zip(captured, xs):
                self.assertTrue(torch.equal(output, self._fresh(mlp, x)))
            self.assertTrue(torch.equal(eager, self._fresh(mlp, xs[0] * 2)))


if __name__ == "__main__":
    unittest.main()
