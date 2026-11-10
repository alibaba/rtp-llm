import json
import os
import site
import time
import unittest
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import mxfp8_ops as ops
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.mxfp8_prefill_quant_test import (
    _reference,
)
from rtp_llm.models_py.triton_kernels.common import fast_launch

_RUNNERS = {
    "auto": ops.mxfp8_quant_act_packed,
    "explicit_fused": ops.mxfp8_quant_act_packed_fused,
    "silu_fused": ops.silu_mul_mxfp8_quant_act_packed_fused,
}


class Mxfp8LaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest("native MXFP8 quantization requires SM100")
        import nvidia_cutlass_dsl

        site.addsitedir(str(Path(nvidia_cutlass_dsl.__path__[0]).parent))
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(fast_launch, "_cache", OrderedDict()))
        self.patches.enter_context(patch.object(fast_launch, "TRACE", True))
        self.patches.enter_context(
            patch.dict(os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": "auto"})
        )
        self.patches.enter_context(
            patch.dict(fast_launch.counts, raw=0, compiled_fallback=0, jit_miss=0)
        )

    def _expected(
        self, x: torch.Tensor, entry: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if entry == "silu_fused":
            gate, up = x.chunk(2, dim=-1)
            x = torch.nn.functional.silu(gate) * up
        return _reference(x)

    def _run(self, x: torch.Tensor, entry: str) -> tuple[torch.Tensor, torch.Tensor]:
        return _RUNNERS[entry](x)

    def _random_input(self, m: int, entry: str) -> torch.Tensor:
        width = 4096 if entry == "silu_fused" else 2048
        return torch.randn(m, width, dtype=torch.bfloat16, device="cuda")

    def _assert_exact(
        self,
        actual: tuple[torch.Tensor, torch.Tensor],
        expected: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        q, scale = actual
        expected_q, expected_scale = expected
        self.assertTrue(torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8)))
        self.assertEqual(scale.stride(), expected_scale.stride())
        self.assertTrue(torch.equal(scale, expected_scale))

    def test_standard_and_guarded_launch_match_and_reuse_cache(self) -> None:
        records = []
        for m in (32, 82, 1368):
            for entry in _RUNNERS:
                x = self._random_input(m, entry)
                expected = self._expected(x, entry)
                for enabled in (False, True):
                    with self.subTest(m=m, entry=entry, enabled=enabled), patch.object(
                        fast_launch, "ENABLED", enabled
                    ), patch.object(fast_launch, "_cache", OrderedDict()), patch.dict(
                        fast_launch.counts, raw=0, compiled_fallback=0, jit_miss=0
                    ):
                        for _ in range(3):
                            actual = self._run(x, entry)
                            self._assert_exact(actual, expected)
                        self.assertEqual(
                            fast_launch.counts["jit_miss"], 1 if enabled else 0
                        )
                        self.assertEqual(fast_launch.counts["raw"], 2 if enabled else 0)
                        records.append(
                            {
                                "m": m,
                                "entry": entry,
                                "enabled": enabled,
                                **fast_launch.counts,
                            }
                        )
        payload = json.dumps(records, separators=(",", ":"))
        (self.output_dir / "mxfp8_launcher_counts.json").write_text(payload)
        print("mxfp8_launcher_counts.json", payload, flush=True)

    def test_dynamic_rows_reuse_compiled_silu_binary(self) -> None:
        kernel = ops._silu_mul_mxfp8_quant_act_packed_kernel
        original_run = kernel.run
        compiled = []
        row_counts = (1, 2, 3, 16, 32, 64, 82, 1368, 4906)

        def record_run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            compiled.append(result)
            return result

        with patch.object(kernel, "run", record_run), patch.object(
            fast_launch, "ENABLED", False
        ):
            for m in row_counts:
                with self.subTest(m=m):
                    torch.manual_seed(20260918 + m)
                    x = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
                    self._assert_exact(
                        self._run(x, "silu_fused"), self._expected(x, "silu_fused")
                    )
            torch.cuda.synchronize()
        self.assertEqual(len(compiled), len(row_counts))
        self.assertIsNotNone(compiled[0])
        self.assertTrue(
            all(binary is compiled[0] for binary in compiled),
            "changing M compiled another SiLU binary",
        )
        payload = json.dumps(
            {
                "kernel": kernel.__name__,
                "dtype": str(torch.bfloat16),
                "rows": row_counts,
                "input_columns": 4096,
                "launches": len(compiled),
                "unique_binaries": len({id(binary) for binary in compiled}),
            },
            separators=(",", ":"),
        )
        (self.output_dir / "mxfp8_silu_binary_reuse.json").write_text(payload)
        print("mxfp8_silu_binary_reuse.json", payload, flush=True)

    def test_nondefault_stream_and_changed_input_graph(self) -> None:
        with patch.object(fast_launch, "ENABLED", True):
            for m in (32, 82, 1368):
                for entry in _RUNNERS:
                    with self.subTest(m=m, entry=entry):
                        x = torch.zeros(
                            m,
                            4096 if entry == "silu_fused" else 2048,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        replacement = torch.randn_like(x)
                        expected = self._expected(replacement, entry)
                        for _ in range(3):
                            self._run(x, entry)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            x.copy_(replacement)
                            actual = self._run(x, entry)
                        stream.synchronize()
                        self._assert_exact(actual, expected)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            captured = self._run(x, entry)
                        replacement.neg_()
                        x.copy_(replacement)
                        graph.replay()
                        torch.cuda.synchronize()
                        self._assert_exact(captured, self._expected(replacement, entry))

    def test_active_triton_hook_uses_compiled_fallback(self) -> None:
        self.assertIsNotNone(fast_launch.knobs)
        for entry, m in (
            ("explicit_fused", 32),
            ("silu_fused", 32),
            ("auto", 82),
            ("auto", 1368),
        ):
            with self.subTest(entry=entry, m=m), patch.object(
                fast_launch, "ENABLED", True
            ):
                x = self._random_input(m, entry)
                expected = self._expected(x, entry)
                for _ in range(3):
                    self._run(x, entry)
                calls = []

                def entered(*args) -> None:
                    calls.append("enter")

                before = dict(fast_launch.counts)
                with patch.object(
                    fast_launch.knobs.runtime, "launch_enter_hook", entered
                ):
                    actual = self._run(x, entry)
                    torch.cuda.synchronize()
                self._assert_exact(actual, expected)
                self.assertEqual(calls, ["enter"])
                self.assertEqual(fast_launch.counts["raw"], before["raw"])
                self.assertEqual(
                    fast_launch.counts["compiled_fallback"],
                    before["compiled_fallback"] + 1,
                )

    def test_host_launch_benchmark(self) -> None:
        records = []
        repetitions = 50
        with patch.object(fast_launch, "TRACE", False):
            for name, run in _RUNNERS.items():
                for m in (32, 82, 1368):
                    x = self._random_input(m, name)
                    for enabled in (False, True):
                        with patch.object(fast_launch, "ENABLED", enabled):
                            for _ in range(10):
                                run(x)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(repetitions):
                                    run(x)
                            for sample in range(3):
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                graph.replay()
                                end.record()
                                end.synchronize()
                                gpu_us = start.elapsed_time(end) * 1000 / repetitions
                                wall_start = time.perf_counter()
                                for _ in range(repetitions):
                                    run(x)
                                submit_end = time.perf_counter()
                                torch.cuda.synchronize()
                                wall_end = time.perf_counter()
                                records.append(
                                    {
                                        "entry": name,
                                        "m": m,
                                        "enabled": enabled,
                                        "sample": sample,
                                        "repetitions": repetitions,
                                        "gpu_graph_us": gpu_us,
                                        "submit_us": (submit_end - wall_start)
                                        * 1e6
                                        / repetitions,
                                        "wall_us": (wall_end - wall_start)
                                        * 1e6
                                        / repetitions,
                                    }
                                )
        payload = json.dumps(records, separators=(",", ":"))
        (self.output_dir / "mxfp8_host_launch_benchmark.json").write_text(payload)
        print("mxfp8_host_launch_benchmark.json", payload, flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
