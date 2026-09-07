"""Masked SiLU/FP8 quantization producing MN-major UE8M0 scales on SM120.

Each program owns complete packed words (four block-128 scales). Padding
scale words, including TMA alignment rows and incomplete K packs, are written
on every invocation. FP8 padding rows remain unspecified and consumers must
respect masked_m. The FP32 arithmetic follows the ordinary masked SiLU kernel,
not the intermediate BF16 rounding used by the SM100 packed implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_masked_packed(
    x,
    out,
    scales,
    counts,
    M: tl.constexpr,
    H: tl.constexpr,
    ALIGNED_M: tl.constexpr,
    PACKED_G: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
):
    pack_id = tl.program_id(0)
    worker = tl.program_id(1)
    expert = tl.program_id(2)
    workers = tl.num_programs(1)
    count = tl.load(counts + expert)
    tl.device_assert((count >= 0) & (count <= M), "masked_m outside capacity")
    groups = pack_id * 4 + tl.arange(0, 4)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    x_base = x + expert.to(tl.int64) * M * (2 * H)
    out_base = out + expert.to(tl.int64) * M * H
    scale_base = scales + (expert.to(tl.int64) * PACKED_G + pack_id) * ALIGNED_M

    for start in range(worker * BLOCK_T, count, workers * BLOCK_T):
        rows = start + tl.arange(0, BLOCK_T)
        valid = (rows[:, None, None] < count) & (cols[None, :, :] < H)
        offsets = rows[:, None, None].to(tl.int64) * (2 * H) + cols[None, :, :]
        up = tl.load(x_base + offsets, mask=valid, other=0).to(tl.float32)
        gate = tl.load(x_base + offsets + H, mask=valid, other=0).to(tl.float32)
        gate = gate / (1 + tl.exp(-gate))
        values = up * gate
        amax = tl.maximum(tl.max(tl.abs(values), axis=2), 1e-10)
        scale = amax / 448.0
        scale = tl.exp2(tl.ceil(tl.log2(tl.abs(scale))))
        quant = tl.minimum(tl.maximum(values / scale[:, :, None], -448.0), 448.0)
        tl.store(
            out_base + rows[:, None, None].to(tl.int64) * H + cols[None, :, :],
            quant.to(out.dtype.element_ty),
            mask=valid,
        )
        exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
        exponent = tl.where(groups[None, :] < H // 128, exponent, 0)
        packed = tl.sum(exponent << (tl.arange(0, 4)[None, :] * 8), axis=1).to(tl.int32)
        tl.store(scale_base + rows, packed, mask=rows < count)

    # Vectorized scale-only writes; no input loads or quantization for padding.
    # These rows never overlap the valid stores, even if counts change at replay.
    for start in range(count + worker * BLOCK_PAD, ALIGNED_M, workers * BLOCK_PAD):
        rows = start + tl.arange(0, BLOCK_PAD)
        tl.store(scale_base + rows, 0, mask=rows < ALIGNED_M)


def create_packed_scale_tensor(
    expert_num: int,
    token_num_padded: int,
    hidden_dim: int,
    quant_group_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate [E, M, ceil(G/4)] scales with 16-byte MN alignment.

    hidden_dim is the concatenated up/gate dimension (2H). G=H/128
    need not be divisible by four. Storage, including hidden alignment rows,
    is initialized by silu_and_mul_masked_post_quant_packed_fwd, not here.
    """
    assert quant_group_size == 128
    assert hidden_dim > 0 and hidden_dim % (2 * quant_group_size) == 0
    assert expert_num >= 0 and token_num_padded >= 0
    groups = hidden_dim // 2 // quant_group_size
    packed_groups = triton.cdiv(groups, 4)
    aligned_m = triton.cdiv(token_num_padded, 4) * 4
    storage = torch.empty(
        (expert_num, packed_groups, aligned_m), device=device, dtype=torch.int32
    )
    return storage.transpose(1, 2)[:, :token_num_padded, :]


def silu_and_mul_masked_post_quant_packed_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
) -> None:
    """Write valid FP8 rows and ALL packed scale storage without a pack pass.

    output_scale must be allocated by create_packed_scale_tensor. Counts are
    device-resident runtime inputs, so changing routes is CUDA Graph safe.
    """
    assert input.is_cuda and input.dtype == torch.bfloat16 and input.is_contiguous()
    assert input.ndim == 3 and quant_group_size == 128
    experts, capacity, hidden_dim = input.shape
    assert hidden_dim > 0 and hidden_dim % 256 == 0
    h = hidden_dim // 2
    packed_groups = triton.cdiv(h // 128, 4)
    aligned_m = triton.cdiv(capacity, 4) * 4
    assert output.shape == (experts, capacity, h)
    assert output.dtype == torch.float8_e4m3fn and output.is_contiguous()
    assert output_scale.dtype == torch.int32
    assert output_scale.shape == (experts, capacity, packed_groups)
    assert masked_m.shape == (experts,) and masked_m.dtype == torch.int32
    assert masked_m.is_contiguous()
    assert output.device == output_scale.device == masked_m.device == input.device
    if experts == 0 or capacity == 0:
        return
    assert output_scale.stride() == (packed_groups * aligned_m, 1, aligned_m)
    # A matching strided view need not own the final TMA padding rows (e.g.
    # empty_strided allocates only up to the final logical element).
    required_words = output_scale.storage_offset() + experts * packed_groups * aligned_m
    assert (
        required_words * output_scale.element_size()
        <= output_scale.untyped_storage().nbytes()
    )
    # Enough independent workers for skew/full experts, without launching a
    # CTA for every capacity row. Selected with sparse/skew/full SM120 sweeps.
    workers = min(32 if experts < 64 else 16, triton.cdiv(capacity, 4))
    _silu_mul_masked_packed[(packed_groups, workers, experts)](
        input,
        output,
        output_scale,
        masked_m,
        M=capacity,
        H=h,
        ALIGNED_M=aligned_m,
        PACKED_G=packed_groups,
        BLOCK_T=4,
        BLOCK_PAD=128,
        num_warps=4,
    )
