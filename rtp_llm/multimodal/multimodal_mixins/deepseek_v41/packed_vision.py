"""Packed V4.1 vision using existing PyTorch operations and rotary kernels.

This module owns packed execution, not weights, scheduler admission, or graph
policy. Optional layout/fusion paths retain separate enable flags and gates.
Exact-byte agreement with the sequential implementation is a separate
GPU acceptance gate: changing GEMM row counts or SDPA batch shape can change
floating-point results even though the mathematical operation is unchanged.

Returned aligned tensors are disjoint row views of one batch allocation. Final
request assembly owns its output; clone explicitly if separate aligned storage
is required. This module never substitutes an eager fallback after a failure.
"""

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    _vision_cos_sin,
)

# The unchanged V4.1 rotary Triton kernel launches eight rows per grid.y
# block. CUDA's grid.y limit is 65535; packed rows can exceed it even when
# each original image is valid. Keep the original kernel and its arithmetic.
_MAX_ROPE_ROWS_PER_LAUNCH = 65535 * 8


def _packed_rotary(attention, q, k, cos, sin):
    rows = q.shape[0]
    if rows <= _MAX_ROPE_ROWS_PER_LAUNCH:
        # Preserve the exact original path, allocations and launch count for
        # all previously supported packed shapes.
        return attention._apply_rotary(q, k, cos, sin)
    packed_rotary = getattr(attention, "_apply_packed_rotary", None)
    if q.is_cuda and not torch.is_grad_enabled() and packed_rotary is not None:
        result = packed_rotary(q, k, cos, sin)
        if result is not None:
            return result
    q_parts, k_parts = [], []
    for start in range(0, rows, _MAX_ROPE_ROWS_PER_LAUNCH):
        stop = min(start + _MAX_ROPE_ROWS_PER_LAUNCH, rows)
        # Row views retain the Q/K shared storage, strides and relative offset
        # required by the original specialization. Slice positional tables by
        # the same absolute rows; never restart image positions at a boundary.
        rotated_q, rotated_k = attention._apply_rotary(
            q[start:stop], k[start:stop], cos[start:stop], sin[start:stop]
        )
        q_parts.append(rotated_q)
        k_parts.append(rotated_k)
    # Only rotary launches are partitioned. V, packed GEMMs and per-image
    # attention grouping remain untouched. Failures propagate without retry.
    return torch.cat(q_parts, dim=0), torch.cat(k_parts, dim=0)


def _packed_attention(q, k, v, grids, lengths, *, qk_bhsd=None):
    # Group by full geometry, not just token count: rotary positions already
    # differ per image, and this makes the intended attention topology explicit.
    groups = OrderedDict()
    for index, grid in enumerate(grids):
        groups.setdefault(tuple(grid), []).append(index)
    if len(groups) == 1:
        batch_size, sequence_length = len(lengths), lengths[0]
        # This produces the same contiguous [B, H, S, D] input as stacking
        # transposed image slices, with one whole-batch copy per tensor. The
        # final view also canonicalizes strides on singleton dimensions.
        tensors = [
            tensor.view(batch_size, sequence_length, *tensor.shape[1:])
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, tensor.shape[1], sequence_length, tensor.shape[2])
            for tensor in ((v,) if qk_bhsd is not None else (q, k, v))
        ]
        if qk_bhsd is not None:
            # The opt-in rotary producer wrote the exact same BHSD descriptor
            # and values that the old Q/K contiguous copies produced. Keep V,
            # SDPA/backend selection and the output/GEMM boundary unchanged.
            tensors = [*qk_bhsd, *tensors]
        attended = F.scaled_dot_product_attention(*tensors)
        return attended.transpose(1, 2).contiguous().view(q.shape[0], -1)
    if qk_bhsd is not None:
        raise ValueError("BHSD rotary inputs require one homogeneous grid group")
    q_parts, k_parts, v_parts = (tensor.split(lengths, dim=0) for tensor in (q, k, v))
    outputs = [None] * len(lengths)
    for indices in groups.values():
        # Each independent image occupies a batch axis; no concatenated token
        # axis and no mask trick can allow one image to attend to another.
        tensors = [
            torch.stack([parts[i].transpose(0, 1) for i in indices], dim=0)
            for parts in (q_parts, k_parts, v_parts)
        ]
        attended = F.scaled_dot_product_attention(*tensors)
        for row, image_index in enumerate(indices):
            outputs[image_index] = (
                attended[row].transpose(0, 1).reshape(lengths[image_index], -1)
            )
    return torch.cat(outputs, dim=0)


def forward_packed(vision, aligner, patches, grids, *, rope_bhsd=None):
    """Return aligned tensors in original image order, without token assembly.

    Concatenated rows share patch projection, QKV/WO, norms, and MLP GEMMs.
    Attention is isolated by image, grouped only for compatible grid shapes.
    Empty input returns []; a singleton uses this same packed implementation.
    No retry after execution failure or precision tolerance is applied.

    ``rope_bhsd`` explicitly opts into a homogeneous-grid rotary writer. None
    reads V41_PACKED_ROPE_BHSD (default 0) once per forward. Unsupported rotary
    contracts retain the original path, while supported launch errors surface.
    """
    # The production adapter supplies prepared patches and planned grids.
    # Only derive split offsets here; do not rescan model weights or metadata
    # for every forward.
    lengths = [height * width for height, width in grids]
    if not lengths:
        return []
    if rope_bhsd is None:
        rope_bhsd = os.environ.get("V41_PACKED_ROPE_BHSD", "0") == "1"
    use_bhsd = rope_bhsd and all(tuple(grid) == tuple(grids[0]) for grid in grids)
    if use_bhsd:
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_rotary import (
            apply_packed_vision_qk_rope_bhsd,
        )
    x = vision.patch_embed(torch.cat([part.flatten(1) for part in patches], dim=0))
    rotary = [
        _vision_cos_sin(h, w, vision.rope_dim, vision.rope_theta, str(x.device))
        for h, w in grids
    ]
    cos = torch.cat([item[0] for item in rotary], dim=0)
    sin = torch.cat([item[1] for item in rotary], dim=0)
    for block in vision.blocks:
        normed = block.norm1(x)
        qkv = block.attn.wqkv(normed)
        q, k, v = [
            part.view(x.shape[0], block.attn.n_heads, block.attn.head_dim)
            for part in qkv.chunk(3, dim=-1)
        ]
        qk_bhsd = None
        if use_bhsd:
            qk_bhsd = apply_packed_vision_qk_rope_bhsd(
                q, k, cos, sin, len(lengths), lengths[0]
            )
        if qk_bhsd is None:
            q, k = _packed_rotary(block.attn, q, k, cos, sin)
            attention = _packed_attention(q, k, v, grids, lengths)
        else:
            attention = _packed_attention(q, k, v, grids, lengths, qk_bhsd=qk_bhsd)
        attention_output = block.attn.wo(attention)
        x = x + attention_output
        normed = block.norm2(x)
        mlp = block.mlp(normed)
        x = x + mlp
    x = vision.norm(x)
    ratio = aligner.downsample_ratio
    columns, output_lengths = [], []
    for part, (height, width) in zip(x.split(lengths, dim=0), grids):
        image = part.view(height, width, -1).permute(2, 0, 1)
        image = F.pad(image, (0, -width % ratio, 0, -height % ratio))
        unfolded = F.unfold(image.unsqueeze(0), ratio, stride=ratio).squeeze(0)
        columns.append(unfolded)
        output_lengths.append(unfolded.shape[1])
    # Preserve each image's original w1 GEMM
    # dimensions AND column-major leading dimension. Concatenating columns
    # before w1 changes its kernel/accumulation and fails the real-weight
    # byte-exact gate even when every preceding ViT boundary is identical.
    per_image_hidden = []
    for column in columns:
        unfolded = column.transpose(0, 1)
        per_image_hidden.append(aligner.w1(unfolded))
    hidden = torch.cat(per_image_hidden, dim=0)
    hidden = F.gelu(hidden)
    aligned = aligner.w2(hidden)
    return list(aligned.split(output_lengths, dim=0))
