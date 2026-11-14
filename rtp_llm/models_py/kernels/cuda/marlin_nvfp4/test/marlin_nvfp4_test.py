import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import (
    Nvfp4MarlinMoE,
    align,
)
from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.test.marlin_fixture import (
    MarlinNvfp4TestFixture,
)


@unittest.skipUnless(
    Nvfp4MarlinMoE.is_supported(),
    "requires a supported NVFP4 Marlin runtime and GPU",
)
class MarlinNvfp4Test(MarlinNvfp4TestFixture, unittest.TestCase):

    def test_scale_range_zero_expert_and_fp32_weight_reference(self):
        for rank in (0, 1):
            args = self.inputs(3, rank, [rank * 4, rank * 4 + 1, rank * 4 + 3])
            self.assert_reference(self.models[rank](*args), self.reference(*args, rank))

    def test_precompiled_library_in_fresh_process_without_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "unused_extensions"
            env = {
                **os.environ,
                "TORCH_EXTENSIONS_DIR": str(cache),
                "CUDA_HOME": str(Path(directory) / "no_cuda"),
                "CXX": str(Path(directory) / "no_cxx"),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
from pathlib import Path
from unittest.mock import patch

import torch

with patch("torch.utils.cpp_extension.load", side_effect=AssertionError("runtime JIT")):
    from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import load_library
    library = load_library()
    assert str(Path(library).resolve()) in torch.ops.loaded_libraries
    q = torch.zeros((16, 64), dtype=torch.int32, device="cuda")
    packed = torch.ops.rtp_qwen4_marlin.gptq_marlin_repack(q, 128, 64, 4, False)
    assert packed.numel() == q.numel()
    assert torch.count_nonzero(packed).item() == 0
    torch.cuda.synchronize()
""",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(cache.exists())

    def test_ep_global_ids_and_all_nonlocal_routes(self):
        for rank in (0, 1):
            for rows in (1, 4, 8, 32, 64):
                args = self.inputs(rows, rank)
                self.assert_reference(
                    self.models[rank](*args), self.reference(*args, rank)
                )
            args = self.inputs(4, rank, [4, 5, 6] if rank == 0 else [0, 1, 2])
            self.assertTrue(bool((self.models[rank](*args) == 0).all()))

    def test_graph_refreshes_values_ids_weights(self):
        for rank in (0, 1):
            for rows in (1, 4, 8):
                x, ids, weights = self.inputs(rows, rank)
                model = self.models[rank]
                model(x, ids, weights)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = model(x, ids, weights)
                for _ in range(3):
                    new = self.inputs(rows, rank)
                    for dst, src in zip((x, ids, weights), new):
                        dst.copy_(src)
                    expected = model(x, ids, weights)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_alignment_stable_order_padding_and_nonlocal_routes(self):
        for rows in (1, 4, 8, 9, 32):
            for first in (0, 256):
                for mode in ("random", "same", "nonlocal", "boundary"):
                    ids = torch.randint(
                        0, 512, (rows, 10), device="cuda", dtype=torch.int32
                    )
                    if mode == "same":
                        ids.fill_(first + 3)
                    elif mode == "nonlocal":
                        ids.fill_(256 - first)
                    elif mode == "boundary":
                        ids.flatten()[::2] = first
                        ids.flatten()[1::2] = first + 255
                    routes = ids.numel()
                    active = min(routes, 256)
                    capacity = active + (routes - active) // 8
                    flat = ids.flatten().tolist()
                    expected_ids, expected_experts = [], []
                    for expert in range(256):
                        selected = [
                            i for i, value in enumerate(flat) if value == first + expert
                        ]
                        count = (len(selected) + 7) // 8
                        expected_ids.extend(
                            selected + [routes] * (count * 8 - len(selected))
                        )
                        expected_experts.extend([expert] * count)
                    padded = len(expected_ids)
                    expected_ids.extend([routes] * (capacity * 8 - padded))
                    expected_experts.extend([0] * (capacity - len(expected_experts)))
                    actual = align(ids, 256, first)
                    for tensor, expected in zip(
                        actual, (expected_ids, expected_experts, [padded])
                    ):
                        self.assertEqual(tensor.tolist(), expected)

    def test_concurrent_stream_workspaces(self):
        model = self.models[0]
        args = [self.inputs(8) for _ in range(2)]
        expected = [model(*a) for a in args]
        streams = [torch.cuda.Stream() for _ in range(2)]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        actual = []
        for stream, a in zip(streams, args):
            with torch.cuda.stream(stream):
                actual.append(model(*a))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
