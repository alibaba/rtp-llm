"""Checkpoint layouts for the DeepSeek V4 grouped FP8 output projection."""

import torch


def prepare_wo_a_stacked(weight_fp8, scale_raw, G, R, K):
    """Return [G,R,K] weights and MN-major, int32-packed UE8M0 scales."""
    if G <= 0 or R <= 0 or K <= 0 or R % 128 or K % 128:
        raise ValueError(f"Invalid V4 wo_a block geometry: G={G}, R={R}, K={K}")
    if scale_raw.device != weight_fp8.device:
        raise ValueError("V4 wo_a weight and scale must be on the same device")
    weight = weight_fp8.view(G, R, K).contiguous()
    blocks = K // 128
    words = (blocks + 3) // 4
    if scale_raw.dtype == torch.int32:
        # Descriptor-based checkpoints may already contain one packed row per
        # weight row. Only restore the grouped MN-major storage order.
        if scale_raw.shape != (G * R, words):
            raise ValueError(f"Invalid packed V4 wo_a scale shape {scale_raw.shape}")
        dst = torch.empty((G, words, R), dtype=torch.int32, device=weight.device)
        dst.copy_(scale_raw.reshape(G, R, words).transpose(1, 2))
        return weight, dst.transpose(1, 2)

    if scale_raw.shape != (G * R // 128, blocks):
        raise ValueError(f"Invalid blockwise V4 wo_a scale shape {scale_raw.shape}")
    if scale_raw.dtype == torch.float8_e8m0fnu:
        # The checkpoint already contains the exponent bytes consumed by
        # fp8_einsum. No floating-point conversion or DeepGEMM pack kernel
        # is needed. Copy the *compact* block grid first: an otherwise
        # contiguous input slice need not be aligned for an int32 view.
        raw = torch.empty(
            (G, R // 128, words * 4), dtype=torch.uint8, device=weight.device
        )
        raw[..., :blocks].copy_(
            scale_raw.view(torch.uint8).reshape(G, R // 128, blocks)
        )
        if blocks % 4:
            raw[..., blocks:].zero_()
        block_words = raw.view(torch.int32).transpose(1, 2)
        # R is a multiple of 128, hence naturally TMA-aligned. Broadcast
        # each scale block to 128 rows directly into the final storage.
        dst = torch.empty((G, words, R), dtype=torch.int32, device=weight.device)
        dst.view(G, words, R // 128, 128).copy_(block_words.unsqueeze(-1))
        return weight, dst.transpose(1, 2)

    # A numeric block scale is not an exponent byte. Preserve the previous
    # conversion for callers providing floating-point reference weights.
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    scale_fp32 = scale_raw.float().reshape(G, R // 128, blocks)
    idx = torch.arange(R, device=scale_raw.device) // 128
    return weight, get_mn_major_tma_aligned_packed_ue8m0_tensor(
        scale_fp32.index_select(-2, idx).contiguous()
    )
