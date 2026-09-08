# SPDX-License-Identifier: Apache-2.0
"""Triton MRoPE kernel for the PPU Qwen3.5 attention path.

The tile layout and operation order are adapted from vLLM's MRoPE kernel. This
lives under ppu_impl because it is only used by the PPU FA3 attention MRoPE
implementation; it is intentionally not placed in the shared open-source
triton_kernels tree.
"""

import triton
import triton.language as tl


@triton.jit
def _mrope_forward_kernel(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    """Apply Qwen MRoPE in place, matching vLLM's per-token tile layout."""
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    half_offsets = tl.arange(0, pad_hd // 2)
    q_head_offsets = tl.arange(0, pad_n_qh)[:, None]
    k_head_offsets = tl.arange(0, pad_n_kh)[:, None]
    first_q_offsets = q_head_offsets * hd + half_offsets[None, :]
    first_k_offsets = k_head_offsets * hd + half_offsets[None, :]
    first_q_mask = (q_head_offsets < n_qh) & (half_offsets[None, :] < half_rd)
    first_k_mask = (k_head_offsets < n_kh) & (half_offsets[None, :] < half_rd)

    q1 = tl.load(q_ptr + first_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k1 = tl.load(k_ptr + first_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)
    second_q_offsets = first_q_offsets + half_rd
    second_k_offsets = first_k_offsets + half_rd
    q2 = tl.load(q_ptr + second_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k2 = tl.load(k_ptr + second_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    tl.store(
        q_ptr + first_q_offsets,
        q1 * cos_row - q2 * sin_row,
        mask=first_q_mask,
    )
    tl.store(
        q_ptr + second_q_offsets,
        q2 * cos_row + q1 * sin_row,
        mask=first_q_mask,
    )
    tl.store(
        k_ptr + first_k_offsets,
        k1 * cos_row - k2 * sin_row,
        mask=first_k_mask,
    )
    tl.store(
        k_ptr + second_k_offsets,
        k2 * cos_row + k1 * sin_row,
        mask=first_k_mask,
    )


def apply_mrope_triton_inplace(
    query,
    key,
    cos,
    sin,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved=True,
):
    """Apply MRoPE to contiguous [tokens, heads, head_size] Q/K tensors."""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("MRoPE Triton expects three-dimensional Q and K")
    if query.shape[0] != key.shape[0]:
        raise ValueError("MRoPE Triton expects equal Q and K token counts")
    if query.shape[-1] != head_size or key.shape[-1] != head_size:
        raise ValueError("MRoPE Triton head size does not match Q/K")
    if len(mrope_section) != 3 or sum(mrope_section) != rotary_dim // 2:
        raise ValueError("MRoPE sections must contain three rotary half-dim parts")

    num_tokens = query.shape[0]
    q_flat = query.reshape(num_tokens, -1)
    k_flat = key.reshape(num_tokens, -1)
    q_work = q_flat.contiguous()
    k_work = k_flat.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    if cos.shape != (3, num_tokens, rotary_dim // 2) or sin.shape != cos.shape:
        raise ValueError("MRoPE cos/sin must have shape [3, tokens, rotary_dim/2]")

    n_qh = q_flat.shape[1] // head_size
    n_kh = k_flat.shape[1] // head_size
    _mrope_forward_kernel[(num_tokens,)](
        q_work,
        k_work,
        cos,
        sin,
        num_tokens,
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        triton.next_power_of_2(n_qh),
        triton.next_power_of_2(n_kh),
        triton.next_power_of_2(head_size),
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
    )
    query.copy_(q_work.reshape_as(query))
    key.copy_(k_work.reshape_as(key))
