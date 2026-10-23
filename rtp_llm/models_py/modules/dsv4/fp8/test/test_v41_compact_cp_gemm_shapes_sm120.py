"""Diagnose GEMM row-shape drift without relaxing exact producer arithmetic."""

import json
import os
import unittest
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32


class CompactCPGemmShapes(unittest.TestCase):
    def test_owner_shape_search(self):
        root = Path(os.environ["DSV41_CP_FIXTURES"])
        fixtures = sorted(root.glob("layer*-chunk*.pt"))
        self.assertEqual(len(fixtures), 8)
        report = {"cases": [], "exact_shapes": []}
        sizes = (512, 1024, 1536, 2048, 3072, 4096)
        exact = {size: True for size in sizes}
        for path in fixtures:
            fixture = torch.load(path, map_location="cuda", weights_only=True)
            x = fixture["tensors"]["x"]
            weights = torch.load(
                root / f'weights-layer{fixture["layer"]}.pt',
                map_location="cuda",
                weights_only=True,
            )
            for name in ("global_wkv", "global_wgate"):
                if name not in weights:
                    continue
                weight = weights[name]
                expected = _linear_bf16_bf16_fp32(x, weight)
                for size in sizes:
                    outputs = []
                    for rank in range(4):
                        local = torch.cat(
                            (
                                x[rank * 512 : (rank + 1) * 512],
                                x[(7 - rank) * 512 : (8 - rank) * 512],
                            )
                        )
                        if size == 512:
                            outputs.append(
                                torch.cat(
                                    [
                                        _linear_bf16_bf16_fp32(part, weight)
                                        for part in local.split(512)
                                    ]
                                )
                            )
                        else:
                            padded = local.new_zeros((size, x.shape[1]))
                            padded[:1024].copy_(local)
                            outputs.append(
                                _linear_bf16_bf16_fp32(padded, weight)[:1024]
                            )
                    restored = torch.cat(
                        [outputs[r][:512] for r in range(4)]
                        + [outputs[r][512:] for r in reversed(range(4))]
                    )
                    same = torch.equal(expected, restored)
                    exact[size] &= same
                    entry = {
                        "fixture": path.name,
                        "weight": name,
                        "gemm_rows": size,
                        "exact": same,
                        "max_abs_difference": float((expected - restored).abs().max()),
                    }
                    if path == fixtures[0] and name == "global_wkv":
                        sample = x[:size].contiguous()
                        for _ in range(5):
                            _linear_bf16_bf16_fp32(sample, weight)
                        torch.cuda.synchronize()
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        ) as prof:
                            _linear_bf16_bf16_fp32(sample, weight)
                        entry["kernels"] = [
                            event.name
                            for event in prof.events()
                            if event.device_type == torch.autograd.DeviceType.CUDA
                        ]
                    report["cases"].append(entry)
        report["exact_shapes"] = [size for size in sizes if exact[size]]
        Path(os.environ["DSV41_CP_PROBE_OUTPUT"]).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        self.assertTrue(
            report["exact_shapes"], "No tested owner shape preserves the baseline GEMM"
        )


if __name__ == "__main__":
    unittest.main()
