import json
import os
import site
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import mxfp8_ops as ops
from rtp_llm.models_py.triton_kernels.common import fast_launch

os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "deep_gemm_cache"),
)


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm
    import flashinfer

    q, scales = flashinfer.mxfp8_quantize(
        x, is_sf_swizzled_layout=False, alignment=32, backend="cute-dsl"
    )
    m, k = x.shape
    scale_bytes = scales.reshape(m, k // 128, 4).to(torch.int64)
    shifts = torch.arange(4, device=x.device) * 8
    packed = (scale_bytes << shifts).sum(dim=-1).to(torch.int32)
    storage = torch.empty(
        (k // 128, deep_gemm.get_tma_aligned_size(m, 4)),
        device=x.device,
        dtype=torch.int32,
    )
    result = storage.t()[:m]
    result.copy_(packed)
    return q, result


class Mxfp8PrefillQuantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest("native MXFP8 quantization requires SM100")
        import nvidia_cutlass_dsl

        site.addsitedir(str(Path(nvidia_cutlass_dsl.__path__[0]).parent))
        cls.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])

    def _assert_exact(
        self,
        actual: tuple[torch.Tensor, torch.Tensor],
        expected: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        actual_q, actual_scale = actual
        ref_q, ref_scale = expected
        self.assertTrue(
            torch.equal(actual_q.view(torch.uint8), ref_q.view(torch.uint8))
        )
        self.assertEqual(actual_scale.dtype, torch.int32)
        self.assertEqual(actual_scale.shape, ref_scale.shape)
        self.assertEqual(actual_scale.stride(), ref_scale.stride())
        self.assertTrue(torch.equal(actual_scale, ref_scale))

    def test_prefill_random_and_boundaries_match_flashinfer_bytes(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            for m, k in (
                (83, 384),
                (82, 2048),
                (1368, 6144),
                (4906, 2048),
                (29768, 6144),
            ):
                with self.subTest(dtype=dtype, shape=(m, k)):
                    torch.manual_seed(20261109 + m)
                    x = torch.randn(m, k, dtype=dtype, device="cuda")
                    values = torch.tensor(
                        [0.0, 2.0**-20, 2.0**-8, 1.0, 7.5, 448.0, 896.0, 4096.0],
                        dtype=dtype,
                        device="cuda",
                    ).repeat((k // 32 + 7) // 8)[: k // 32]
                    x[-1].zero_()
                    x[-1, ::32] = values
                    x[-1, 1::32] = -values
                    expected = _reference(x)
                    for mode in ("auto", "0"):
                        with self.subTest(mode=mode), patch.dict(
                            os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": mode}
                        ):
                            self._assert_exact(ops.mxfp8_quant_act_packed(x), expected)

    def test_dynamic_rows_reuse_compiled_pack_and_fused_binaries(self) -> None:
        records = []
        row_counts = (1, 3, 16, 32, 64, 82, 1368, 4906, 20280, 29768)
        for kernel, fused in (
            (ops._pack_flashinfer_mxfp8_scale_kernel, False),
            (ops._mxfp8_quant_act_packed_kernel, True),
        ):
            dtypes = (torch.bfloat16, torch.float16) if fused else (torch.uint8,)
            for dtype in dtypes:
                compiled = []
                original_run = kernel.run

                def record_run(*args, **kwargs):
                    result = original_run(*args, **kwargs)
                    compiled.append(result)
                    return result

                with self.subTest(kernel=kernel.__name__, dtype=dtype), patch.object(
                    kernel, "run", record_run
                ), patch.object(fast_launch, "ENABLED", False):
                    for m in row_counts:
                        if fused:
                            x = torch.randn(m, 6144, dtype=dtype, device="cuda")
                            self._assert_exact(
                                ops.mxfp8_quant_act_packed_fused(x), _reference(x)
                            )
                        else:
                            scales = torch.randint(
                                0, 255, (m, 192), dtype=torch.uint8, device="cuda"
                            )
                            actual = ops._pack_flashinfer_mxfp8_scale(scales, m, 6144)
                            unpacked = torch.stack(
                                [(actual >> shift) & 255 for shift in (0, 8, 16, 24)],
                                dim=-1,
                            ).reshape(m, 192)
                            self.assertTrue(
                                torch.equal(unpacked, scales.to(torch.int32))
                            )
                    torch.cuda.synchronize()
                    self.assertEqual(len(compiled), len(row_counts))
                    self.assertTrue(
                        all(binary is compiled[0] for binary in compiled),
                        "changing M compiled another binary",
                    )
                    records.append(
                        {
                            "kernel": kernel.__name__,
                            "dtype": str(dtype),
                            "rows": row_counts,
                            "launches": len(compiled),
                            "unique_binaries": len({id(b) for b in compiled}),
                        }
                    )
        payload = json.dumps(records, separators=(",", ":"))
        (self.output_dir / "mxfp8_binary_reuse.json").write_text(payload)
        print("mxfp8_binary_reuse.json", payload, flush=True)

    def test_default_keeps_large_prefill_split_and_honors_override(self) -> None:
        records = []
        for m in (32, 82, 1368):
            x = torch.randn(m, 6144, dtype=torch.bfloat16, device="cuda")
            expected = _reference(x)
            for mode in ("0", "unset", "1"):
                with self.subTest(m=m, mode=mode), patch.dict(
                    os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": mode}
                ):
                    if mode == "unset":
                        os.environ.pop("RTP_LLM_MXFP8_FUSED_QUANT")
                    ops.mxfp8_quant_act_packed(x)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profile:
                        actual = ops.mxfp8_quant_act_packed(x)
                        torch.cuda.synchronize()
                    names = [
                        event.name
                        for event in profile.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    ]
                    records.append({"m": m, "mode": mode, "gpu_events": names})
                    self._assert_exact(actual, expected)
                    self.assertEqual(
                        len(names),
                        1 if mode == "1" or (mode == "unset" and m <= 64) else 2,
                    )
                    if os.environ.get("RTP_MXFP8_TEST_TIMELINE") == "1":
                        profile.export_chrome_trace(
                            str(self.output_dir / f"mxfp8_m{m}_{mode}.json")
                        )
        payload = json.dumps(records, separators=(",", ":"))
        (self.output_dir / "mxfp8_launches.json").write_text(payload)
        print("mxfp8_launches.json", payload, flush=True)

    def test_prefill_graph_replay_reads_changed_input(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype), patch.dict(
                os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": "auto"}
            ):
                x = torch.randn(1369, 2048, dtype=dtype, device="cuda")
                replacement = torch.randn_like(x)
                expected = _reference(replacement)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        ops.mxfp8_quant_act_packed(x)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = ops.mxfp8_quant_act_packed(x)
                x.copy_(replacement)
                graph.replay()
                torch.cuda.synchronize()
                self._assert_exact(actual, expected)

    def test_prefill_benchmark(self) -> None:
        records = []
        repetitions = 30
        for m in (32, 82, 1368, 4906, 20280, 29768):
            for k in (2048, 6144):
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                for mode in ("0", "auto"):
                    with patch.dict(os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": mode}):
                        for _ in range(5):
                            ops.mxfp8_quant_act_packed(x)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(repetitions):
                                ops.mxfp8_quant_act_packed(x)
                        for sample in range(3):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            wall_start = time.perf_counter()
                            for _ in range(repetitions):
                                ops.mxfp8_quant_act_packed(x)
                            submit_end = time.perf_counter()
                            end.record()
                            end.synchronize()
                            wall_end = time.perf_counter()
                            eager_us = start.elapsed_time(end) * 1000 / repetitions
                            start.record()
                            graph.replay()
                            end.record()
                            end.synchronize()
                            gpu_us = start.elapsed_time(end) * 1000 / repetitions
                            records.append(
                                {
                                    "m": m,
                                    "k": k,
                                    "mode": mode,
                                    "sample": sample,
                                    "repetitions": repetitions,
                                    "gpu_graph_us": gpu_us,
                                    "eager_event_us": eager_us,
                                    "submit_us": (submit_end - wall_start)
                                    * 1e6
                                    / repetitions,
                                    "wall_us": (wall_end - wall_start)
                                    * 1e6
                                    / repetitions,
                                }
                            )
        artifact = self.output_dir / "mxfp8_prefill_benchmark.json"
        artifact.write_text(json.dumps(records, indent=2))
        print(json.dumps({"benchmark": str(artifact), "measurements": records}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
