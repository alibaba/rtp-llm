import json
import os
import statistics
import time
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import mxfp8_ops as ops
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test import (
    mxfp8_prefill_quant_test as reference,
)


class Mxfp8DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reference.Mxfp8PrefillQuantTest.setUpClass()
        cls.output_dir = reference.Mxfp8PrefillQuantTest.output_dir

    def test_bounded_dispatch_candidates(self) -> None:
        # Given: threshold-adjacent shapes and an already rejected large sentinel.
        shapes = [
            (m, k) for m in (65, 96, 128, 256) for k in (128, 512, 1024, 2048, 6144)
        ]
        shapes += [(32, 6144), (82, 6144), (29768, 6144)]
        records = []
        repetitions = 40
        for dtype in (torch.bfloat16, torch.float16):
            for m, k in shapes:
                with self.subTest(dtype=dtype, m=m, k=k):
                    torch.manual_seed(20261109 + m + k)
                    x = torch.randn((m, k), device="cuda", dtype=dtype)
                    boundary = torch.tensor(
                        [0.0, 2.0**-20, 2.0**-8, 1.0, 7.5, 448.0, 896.0, 4096.0],
                        device="cuda",
                        dtype=dtype,
                    )
                    x[-1].zero_()
                    x[-1, :8] = boundary
                    x[-1, 8:16] = -boundary
                    expected_q, expected_scale = reference._reference(x)
                    graphs = {}
                    outputs = {}
                    for mode in ("0", "1"):
                        with patch.dict(
                            os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": mode}
                        ):
                            # When: use the existing split or fused implementation.
                            actual_q, actual_scale = ops.mxfp8_quant_act_packed(x)
                            torch.cuda.synchronize()
                            # Then: preserve quantized bytes, packed exponents and layout.
                            self.assertTrue(
                                torch.equal(
                                    actual_q.view(torch.uint8),
                                    expected_q.view(torch.uint8),
                                )
                            )
                            self.assertTrue(torch.equal(actual_scale, expected_scale))
                            self.assertEqual(
                                actual_scale.stride(), expected_scale.stride()
                            )
                            for _ in range(3):
                                ops.mxfp8_quant_act_packed(x)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(repetitions):
                                    outputs[mode] = ops.mxfp8_quant_act_packed(x)
                            graphs[mode] = graph
                    samples = {mode: {"host_us": [], "gpu_us": []} for mode in graphs}
                    for sample in range(7):
                        for mode in (("0", "1") if sample % 2 == 0 else ("1", "0")):
                            with patch.dict(
                                os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": mode}
                            ):
                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                for _ in range(repetitions):
                                    ops.mxfp8_quant_act_packed(x)
                                samples[mode]["host_us"].append(
                                    (time.perf_counter() - start_time)
                                    * 1e6
                                    / repetitions
                                )
                                torch.cuda.synchronize()
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(10):
                                    graphs[mode].replay()
                                end.record()
                                end.synchronize()
                                samples[mode]["gpu_us"].append(
                                    start.elapsed_time(end) * 1000 / (10 * repetitions)
                                )
                    for mode, values in samples.items():
                        records.append(
                            {
                                "m": m,
                                "k": k,
                                "dtype": str(dtype),
                                "mode": mode,
                                "host_us": values["host_us"],
                                "gpu_us": values["gpu_us"],
                                "host_median_us": statistics.median(values["host_us"]),
                                "gpu_median_us": statistics.median(values["gpu_us"]),
                            }
                        )
        payload = json.dumps(records, separators=(",", ":"))
        (self.output_dir / "mxfp8_dispatch_sweep.json").write_text(payload)
        print("mxfp8_dispatch_sweep.json", payload, flush=True)


if __name__ == "__main__":
    unittest.main()
