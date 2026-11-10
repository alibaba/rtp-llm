import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

if "triton" in sys.modules:
    raise RuntimeError("fresh cache must be configured before importing Triton")
_CACHE_DIR = tempfile.mkdtemp(prefix="mxfp8-cold-", dir=os.environ["TEST_TMPDIR"])
os.environ["TRITON_CACHE_DIR"] = _CACHE_DIR

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from rtp_llm.models_py.kernels.cuda import mxfp8_ops as ops  # noqa: E402
from rtp_llm.models_py.triton_kernels.common import fast_launch  # noqa: E402

MX_BLOCK = 32
BASELINE_COMMIT = "03fc7e0c855263bedc00ca4cffd1867bf5a53cfc"


@triton.jit
def _pack_flashinfer_mxfp8_scale_kernel(
    scale_u8_ptr,
    packed_ptr,
    M: tl.constexpr,
    K_GROUPS: tl.constexpr,
    K_PACKED: tl.constexpr,
    ALIGNED_MN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kp = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
    shifts = tl.arange(0, 4) * 8
    offs_g = offs_kp[:, None] * 4 + tl.arange(0, 4)[None, :]
    mask = (offs_m[:, None, None] < M) & (offs_g[None, :, :] < K_GROUPS)
    vals = tl.load(
        scale_u8_ptr + offs_m[:, None, None] * K_GROUPS + offs_g[None, :, :],
        mask=mask,
        other=0,
    ).to(tl.int32)
    packed = tl.sum(vals << shifts[None, None, :], axis=2).to(tl.int32)
    tl.store(
        packed_ptr + offs_m[:, None] + offs_kp[None, :] * ALIGNED_MN,
        packed,
        mask=(offs_m[:, None] < M) & (offs_kp[None, :] < K_PACKED),
    )


def _pack_flashinfer_mxfp8_scale(
    scale_u8: torch.Tensor, m: int, k: int
) -> torch.Tensor:
    """Pack FlashInfer uint8 UE8M0 scales into DeepGEMM's int32 TMA layout."""
    assert scale_u8.dtype == torch.uint8
    assert scale_u8.numel() == m * (k // MX_BLOCK)
    import deep_gemm

    k_groups = k // MX_BLOCK
    assert k_groups % 4 == 0
    k_packed = k_groups // 4
    aligned_mn = deep_gemm.get_tma_aligned_size(m, 4)
    storage = torch.empty(
        (k_packed, aligned_mn), device=scale_u8.device, dtype=torch.int32
    )
    packed = storage.transpose(0, 1)
    grid = (triton.cdiv(m, 64), triton.cdiv(k_packed, 32))
    with torch.cuda.device(scale_u8.device):
        _pack_flashinfer_mxfp8_scale_kernel[grid](
            scale_u8,
            packed,
            M=m,
            K_GROUPS=k_groups,
            K_PACKED=k_packed,
            ALIGNED_MN=aligned_mn,
            BLOCK_M=64,
            BLOCK_K_PACKED=32,
            num_warps=8,
        )
    return packed[:m, :]


class Mxfp8ColdTransitionTest(unittest.TestCase):
    def test_cold_shape_transitions(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "CUDA required")
        self.assertGreaterEqual(torch.cuda.get_device_capability()[0], 10)
        self.assertFalse(list(Path(_CACHE_DIR).rglob("*.cubin")))
        torch.manual_seed(20261109)
        records = []
        binaries = {"old_static": [], "new_dynamic": []}
        entries = (
            (
                "old_static",
                _pack_flashinfer_mxfp8_scale_kernel,
                _pack_flashinfer_mxfp8_scale,
            ),
            (
                "new_dynamic",
                ops._pack_flashinfer_mxfp8_scale_kernel,
                ops._pack_flashinfer_mxfp8_scale,
            ),
        )
        row_counts = (1, 3, 16, 64, 82, 1368, 4906, 20280, 29768)
        with patch.object(fast_launch, "ENABLED", False):
            for step, m in enumerate(row_counts):
                scales = torch.randint(
                    0, 256, (m, 192), dtype=torch.uint8, device="cuda"
                )
                values = scales.to(torch.int32).reshape(m, 48, 4)
                shifts = torch.tensor((0, 8, 16, 24), device="cuda")
                expected = (values << shifts).sum(dim=-1).to(torch.int32)
                outputs = {}
                for name, kernel, run in entries[:: 1 if step % 2 == 0 else -1]:
                    launched = []
                    original_run = kernel.run

                    def record_run(*args, **kwargs):
                        binary = original_run(*args, **kwargs)
                        launched.append(binary)
                        return binary

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with patch.object(kernel, "run", record_run):
                        torch.cuda.synchronize()
                        start.record()
                        begin = time.perf_counter_ns()
                        actual = run(scales, m, 6144)
                        host_us = (time.perf_counter_ns() - begin) / 1000
                        end.record()
                        end.synchronize()
                    self.assertEqual(len(launched), 1)
                    self.assertIsNotNone(launched[0])
                    binaries[name].append(launched[0])
                    self.assertEqual(actual.stride(), (1, triton.cdiv(m, 4) * 4))
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(
                        torch.equal(
                            actual.contiguous().view(torch.uint8),
                            expected.view(torch.uint8),
                        )
                    )
                    outputs[name] = actual
                    records.append(
                        {
                            "entry": name,
                            "m": m,
                            "k": 6144,
                            "phase": (
                                "initial_compile"
                                if step == 0
                                else "cold_shape_transition"
                            ),
                            "host_submit_us": host_us,
                            "cold_cuda_interval_us_includes_host_idle": start.elapsed_time(
                                end
                            )
                            * 1000,
                            "binary_id": id(launched[0]),
                            "binary_hash": launched[0].hash,
                            "unique_binaries_so_far": len(
                                {id(b) for b in binaries[name]}
                            ),
                            "stride": actual.stride(),
                            "exact_bytes": True,
                        }
                    )
                self.assertTrue(
                    torch.equal(outputs["old_static"], outputs["new_dynamic"])
                )
                for name, _, run in entries:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(32):
                            run(scales, m, 6144)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    record = next(r for r in reversed(records) if r["entry"] == name)
                    record["gpu_graph_us"] = start.elapsed_time(end) * 1000 / 32
                    record["gpu_graph_kernel_count"] = 32
        payload = json.dumps(
            {
                "baseline_commit": BASELINE_COMMIT,
                "triton_cache_dir": _CACHE_DIR,
                "gpu": torch.cuda.get_device_name(),
                "triton_version": triton.__version__,
                "scope": "pack operator cold-shape host submission; not model idle gaps",
                "records": records,
            },
            separators=(",", ":"),
        )
        (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            / "mxfp8_cold_transition.json"
        ).write_text(payload)
        print("mxfp8_cold_transition.json", payload, flush=True)
        self.assertTrue(
            all(b is binaries["new_dynamic"][0] for b in binaries["new_dynamic"])
        )
        self.assertEqual(len({id(b) for b in binaries["old_static"]}), len(row_counts))
        self.assertTrue(list(Path(_CACHE_DIR).rglob("*.cubin")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
