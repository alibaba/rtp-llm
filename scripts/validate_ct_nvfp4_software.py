"""Software-only NVFP4 adapter check. No native FP4 kernels or full model."""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "rtp_llm/config/test/fixtures/nvfp4"
spec = importlib.util.spec_from_file_location(
    "adapter_scales", ROOT / "rtp_llm/config/nvfp4_utils.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def scalar_e4m3(byte):
    sign = -1 if byte & 128 else 1
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (
        mantissa / 512 if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    )


def scalar_e2m1(code):
    exponent = (code >> 1) & 3
    mantissa = code & 1
    magnitude = (
        mantissa / 2 if exponent == 0 else math.ldexp(1 + mantissa / 2, exponent - 1)
    )
    return -magnitude if code & 8 else magnitude


def unpack(packed):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    return lut[codes & 7] * torch.where((codes & 8) != 0, -1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    started = time.time()
    report = {
        "mode": "software_reference_only",
        "native_nvfp4_tested": False,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checks": {},
    }
    checks = report["checks"]
    assert torch.cuda.is_available(), "CUDA required for the GPU software check"
    torch.cuda.set_device(0)
    report.update(
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tile = json.loads((FIXTURES / "q_proj_tile.json").read_text(encoding="utf-8"))
    report["checkpoint"] = {
        k: tile[k] for k in ("model", "revision", "module", "slice")
    }
    n, k = tile["N"], tile["K"]
    blobs = {}
    for key in ("weight_packed", "weight_scale"):
        blobs[key] = base64.b64decode(tile[key])
        assert hashlib.sha256(blobs[key]).hexdigest() == tile[key + "_sha256"]
    checks["real_payload_hashes"] = True

    # Independent scalar reference: float64, explicit IEEE-like bit decoding,
    # and checkpoint divisors. Does not call the adapter or the vector LUT.
    w_ref = torch.tensor(
        [
            scalar_e2m1(
                (blobs["weight_packed"][row * (k // 2) + col // 2] >> (4 * (col % 2)))
                & 15
            )
            * scalar_e4m3(blobs["weight_scale"][row * (k // 16) + col // 16])
            / tile["weight_global_scale"]
            for row in range(n)
            for col in range(k)
        ],
        dtype=torch.float64,
    ).reshape(n, k)

    packed = torch.tensor(
        list(blobs["weight_packed"]), dtype=torch.uint8, device="cuda"
    ).reshape(n, k // 2)
    scales = (
        torch.tensor(list(blobs["weight_scale"]), dtype=torch.uint8, device="cuda")
        .reshape(n, k // 16)
        .view(torch.float8_e4m3fn)
        .float()
    )
    outer = adapter.reciprocal_global_scale(
        [torch.tensor([tile["weight_global_scale"]], device="cuda")]
    )
    input_outer = adapter.reciprocal_global_scale(
        [torch.tensor([tile["input_global_scale"]], device="cuda")]
    )
    w_gpu = unpack(packed) * scales.repeat_interleave(16, -1) * outer
    torch.testing.assert_close(w_gpu.cpu().double(), w_ref, rtol=2e-7, atol=1e-10)
    checks["real_weight_reconstruction"] = {
        "max_abs_error": (w_gpu.cpu().double() - w_ref).abs().max().item()
    }

    all_bytes = torch.arange(256, device="cuda").to(torch.uint8).reshape(1, 256)
    expected = torch.tensor(
        [scalar_e2m1(code) for b in range(256) for code in (b & 15, b >> 4)]
    )
    torch.testing.assert_close(
        unpack(all_bytes).cpu().flatten(), expected, rtol=0, atol=0
    )
    checks["all_256_packed_byte_patterns"] = True
    finite_codes = [i for i in range(256) if (i & 127) != 127]
    e4 = (
        torch.tensor(finite_codes, dtype=torch.uint8, device="cuda")
        .view(torch.float8_e4m3fn)
        .float()
        .cpu()
    )
    torch.testing.assert_close(
        e4, torch.tensor([scalar_e4m3(i) for i in finite_codes]), rtol=0, atol=0
    )
    checks["all_254_finite_e4m3_codes"] = True

    # Uniform synthetic activation blocks avoid extra activation-rounding
    # uncertainty. Values are exact E2M1 and scales are exactly E4M3.
    generator = torch.Generator().manual_seed(1729)
    a_codes = torch.randint(0, 16, (32, k), generator=generator)
    a_ref = torch.tensor(
        [scalar_e2m1(c) for c in a_codes.flatten().tolist()], dtype=torch.float64
    ).reshape(32, k)
    a_local = (
        torch.tensor([0.5, 1, 2, 4, 0.25, 8, 1.5, 3], dtype=torch.float64)
        .reshape(1, 8)
        .expand(32, 8)
    )
    a_ref *= a_local.repeat_interleave(16, -1) / tile["input_global_scale"]
    a_vals = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])[a_codes & 7] * torch.where(
        (a_codes & 8) != 0, -1, 1
    )
    a_unscaled = (a_vals * a_local.float().repeat_interleave(16, -1)).cuda()
    a_gpu = a_unscaled * input_outer
    y_ref = a_ref @ w_ref.T
    y_gpu = a_gpu @ w_gpu.T
    y_alpha = (a_unscaled @ (unpack(packed) * scales.repeat_interleave(16, -1)).T) * (
        outer * input_outer
    )
    for name, output in (("gpu_fp32_gemm", y_gpu), ("gemm_outer_alpha", y_alpha)):
        torch.testing.assert_close(output.cpu().double(), y_ref, rtol=2e-5, atol=2e-10)
        checks[name] = {
            "max_abs_error": (output.cpu().double() - y_ref).abs().max().item(),
            "relative_l2_error": (
                (output.cpu().double() - y_ref).norm() / y_ref.norm()
            ).item(),
        }

    # Row and K partition algebra only, not production TP/EP integration.
    split = a_gpu[:, :64] @ w_gpu[:, :64].T + a_gpu[:, 64:] @ w_gpu[:, 64:].T
    torch.testing.assert_close(split.cpu().double(), y_ref, rtol=2e-5, atol=2e-10)
    row_split = torch.cat([a_gpu @ part.T for part in w_gpu.chunk(2)], dim=1)
    torch.testing.assert_close(row_split.cpu().double(), y_ref, rtol=2e-5, atol=2e-10)
    checks["group_aligned_k_and_row_partition_algebra"] = True
    padded_p = torch.cat(
        (packed, torch.zeros(n, 8, dtype=torch.uint8, device="cuda")), -1
    )
    padded_s = torch.cat((scales, torch.zeros(n, 1, device="cuda")), -1)
    padded_w = unpack(padded_p) * padded_s.repeat_interleave(16, -1) * outer
    assert (
        torch.equal(padded_w[:, :k], w_gpu)
        and torch.count_nonzero(padded_w[:, k:]).item() == 0
    )
    checks["zero_padding_algebra"] = True

    for name, wrong in (
        (
            "missing_global_reciprocal",
            unpack(packed)
            * scales.repeat_interleave(16, -1)
            * tile["weight_global_scale"],
        ),
        (
            "swapped_nibbles",
            unpack((packed >> 4) | ((packed & 15) << 4))
            * scales.repeat_interleave(16, -1)
            * outer,
        ),
        (
            "wrong_group_scales",
            unpack(packed) * scales.roll(1, -1).repeat_interleave(16, -1) * outer,
        ),
    ):
        relative_error = ((wrong.cpu().double() - w_ref).norm() / w_ref.norm()).item()
        assert relative_error > 0.01, (name, relative_error)
        checks["negative_control_" + name] = {
            "detected": True,
            "relative_l2_error": relative_error,
        }
    try:
        adapter.reciprocal_global_scale([torch.tensor([2.0]), torch.tensor([3.0])])
    except ValueError:
        checks["unequal_fused_globals_rejected"] = True
    else:
        raise AssertionError("unequal fused scales accepted")
    torch.cuda.synchronize()
    report.update(
        status="PASS",
        elapsed_seconds=time.time() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitations=[
            "No native NVFP4 kernel",
            "No production loader/TP/EP execution",
            "No whole model",
            "No performance claim",
        ],
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
