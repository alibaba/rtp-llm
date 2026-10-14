"""Compare the SM100/SM103 native inverse-RoPE quant op with Triton."""

from __future__ import annotations

import torch
from rtp_kernel.dsv4_mega import mla_o_inv_rope_quant

from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)


def _cuda_workspace(m: int, groups: int, heads_per_group: int, head_dim: int):
    aligned_m = ((m + 3) // 4) * 4
    group_dim = heads_per_group * head_dim
    fp8 = torch.empty(
        groups,
        m,
        group_dim,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    ).transpose(0, 1)
    scale = (
        torch.empty(
            groups * heads_per_group * aligned_m,
            dtype=torch.int32,
            device="cuda",
        )
        .as_strided(
            (groups, m, heads_per_group),
            (heads_per_group * aligned_m, 1, aligned_m),
        )
        .transpose(0, 1)
    )
    return fp8, scale


def _dequant_grouped(fp8: torch.Tensor, packed_scale: torch.Tensor):
    m, groups, group_dim = fp8.shape
    heads_per_group = packed_scale.shape[-1]
    head_dim = group_dim // heads_per_group
    chunks = head_dim // 128
    exponent = (
        packed_scale.contiguous()
        .view(torch.uint8)
        .view(m, groups, heads_per_group, 4)[..., :chunks]
        .float()
    )
    scale = torch.exp2(exponent - 127.0)
    return fp8.float().view(m, groups, heads_per_group, chunks, 128) * scale.unsqueeze(
        -1
    )


def test_cuda_matches_triton_at_flash_geometry():
    torch.manual_seed(20260822)
    n_heads, head_dim, rope_dim = 64, 512, 64
    nope_dim = head_dim - rope_dim
    n_groups = 8
    heads_per_group = n_heads // n_groups
    table_size = 257
    table_angles = torch.rand(table_size, rope_dim // 2, device="cuda") * 6.28
    table_freqs = torch.polar(torch.ones_like(table_angles), table_angles).to(
        torch.complex64
    )
    rope_cos = table_freqs.real.contiguous()
    rope_sin = table_freqs.imag.contiguous()

    for batch in (1, 3, 8, 17, 32, 64, 96, 128):
        value = (
            torch.randn(
                batch,
                1,
                n_heads,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.3
        )
        positions = torch.randint(
            0, table_size, (batch,), device="cuda", dtype=torch.int64
        )
        freqs = table_freqs.index_select(0, positions).contiguous()
        triton_fp8, triton_scale = fused_inv_rope_fp8_quant(
            value,
            freqs,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_head_dim=rope_dim,
        )
        cuda_fp8, cuda_scale = _cuda_workspace(
            batch, n_groups, heads_per_group, head_dim
        )
        mla_o_inv_rope_quant(
            value.view(batch, n_heads, head_dim),
            positions,
            rope_cos,
            rope_sin,
            cuda_fp8,
            cuda_scale,
        )
        torch.cuda.synchronize()

        fp8_delta = (
            triton_fp8.contiguous().view(torch.uint8).to(torch.int16)
            - cuda_fp8.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        scale_delta = (
            triton_scale.contiguous().view(torch.uint8).to(torch.int16)
            - cuda_scale.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        fp8_exact = (fp8_delta == 0).float().mean().item()
        scale_exact = (scale_delta == 0).float().mean().item()
        triton_deq = _dequant_grouped(triton_fp8, triton_scale)
        cuda_deq = _dequant_grouped(cuda_fp8, cuda_scale)
        deq_delta = (triton_deq - cuda_deq).abs()
        calc_diff = (
            (triton_deq - cuda_deq).square().sum()
            / (triton_deq.square().sum() + 1.0e-12)
        ).item()
        print(
            f"  [B={batch}] CUDA vs Triton: fp8_exact={fp8_exact * 100:.4f}% "
            f"max_code_delta={int(fp8_delta.max())} "
            f"scale_exact={scale_exact * 100:.4f}% "
            f"max_scale_byte_delta={int(scale_delta.max())} "
            f"deq_max={deq_delta.max().item():.4e} calc_diff={calc_diff:.4e}"
        )

        assert fp8_exact >= 0.95
        assert scale_exact >= 0.99
        assert int(scale_delta.max()) <= 1
        assert calc_diff <= 1.0e-4


if __name__ == "__main__":
    test_cuda_matches_triton_at_flash_geometry()
    print("OK")
