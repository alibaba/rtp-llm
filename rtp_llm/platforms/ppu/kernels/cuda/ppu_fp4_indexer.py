"""SGLang's FP4 Indexer kernels with RTP-owned launch and cache adapters."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def _module(kind, arch, parameter=0):
    from tvm_ffi.cpp import load_inline

    root = Path(__file__).with_name("dsv4_ppu") / "fp4_indexer"
    manifest = json.loads((root / "source-manifest.json").read_text())
    digest = hashlib.sha256()
    for name, expected in sorted(manifest.items()):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Modified SGLang FP4 source: {name}")
        digest.update(name.encode())
        digest.update(data)
    for header in sorted((root / "compat").rglob("*")):
        if header.is_file():
            digest.update(header.read_bytes())
    pdl = "true" if arch[0] >= 9 else "false"
    choices = {
        "q": (
            "main_norm_rope.cuh",
            f"FusedQIndexerRopeHadamardFp4QuantKernel<bf16_t,{pdl}>::forward",
        ),
        "compress": (
            "c4_v2.cuh",
            f"FlashCompress4Kernel<128,fp32_t,fp32_t,{pdl}>::run_prefill",
        ),
        "store": (
            "fused_norm_rope_v2.cuh",
            f"FusedNormRopeKernel<fp32_t,128,64,{parameter},{pdl}>::forward_fp4",
        ),
        "topk": ("topk_prefill_bf16.cuh", "TopKPrefillBF16Kernel::transform"),
    }
    filename, symbol = choices[kind]
    source = root / "csrc" / "deepseek_v4" / filename
    if kind in ("store", "topk"):
        source = root / "compat" / filename
    flags = [
        f"-DSGL_CUDA_ARCH={arch[0] * 100 + arch[1] * 10}",
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
    ]
    if kind == "compress":
        flags += ["-use_fast_math"]
    if kind == "topk":
        flags += [f"-DSGL_TOPK={parameter}"]
    return load_inline(
        f"rtp_sg_fp4_{kind}_{parameter}_{arch[0]}{arch[1]}_{digest.hexdigest()[:16]}",
        cpp_sources=[],
        cuda_sources=[
            f'#include "{source}"',
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, ({symbol}));",
        ],
        extra_cflags=["-std=c++20", "-O3"],
        extra_cuda_cflags=flags,
        extra_include_paths=[str(root / "compat"), str(root / "include")],
    )


def quantize_q(q, weights, weight_scale, freqs, positions):
    """BF16 Q/projection -> packed E2M1, four UE8M0 bytes, FP32 weights."""
    m, h, d = q.shape
    if d != 128 or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("FP4 Indexer Q requires contiguous BF16 [M,H,128]")
    packed = torch.empty((m, h, 64), dtype=torch.int8, device=q.device)
    scales = torch.empty((m, h), dtype=torch.int32, device=q.device)
    scaled_weights = torch.empty((m, h, 1), dtype=torch.float32, device=q.device)
    _module("q", torch.cuda.get_device_capability(q.device)).forward(
        q, packed, scales, weights, scaled_weights, weight_scale, freqs, positions
    )
    return packed, scales.unsqueeze(-1), scaled_weights.squeeze(-1)


def compress4(state, kv_score, ape, plan_c, plan_w):
    output = torch.empty(
        (plan_c.shape[0], 128), dtype=torch.float32, device=kv_score.device
    )
    _module("compress", torch.cuda.get_device_capability(kv_score.device)).forward(
        state.view(-1, 4, 512), kv_score, output, ape, plan_c, plan_w
    )
    return output


def norm_rope_store(
    compressed, plan_c, norm_weight, eps, freqs, slots, cache, page_size
):
    _module("store", torch.cuda.get_device_capability(cache.device), page_size).forward(
        compressed,
        plan_c,
        norm_weight,
        eps,
        freqs,
        slots,
        cache.view(-1, page_size * 68),
        False,
        4,
    )


@lru_cache(maxsize=None)
def _identity_page(device):
    return torch.zeros((1, 1), dtype=torch.int32, device=device)


def topk_bf16(scores, starts, ends, out):
    if scores.dtype != torch.bfloat16 or scores.shape[1] >= 2**31:
        raise ValueError("BF16 TopK requires BF16 scores with fewer than 2**31 columns")
    # Both frozen SG and this port fail in the PPU >16K two-pass kernel.
    if scores.shape[1] > 16384:
        raise ValueError(
            "PPU SG BF16 TopK above 16384 candidate columns is not qualified"
        )
    if not scores.shape[0]:
        return out
    pages = _identity_page(scores.device).expand(scores.shape[0], 1)
    _module(
        "topk", torch.cuda.get_device_capability(scores.device), out.shape[1]
    ).forward(scores, starts, ends, pages, out, 2**31, None)
    return out
