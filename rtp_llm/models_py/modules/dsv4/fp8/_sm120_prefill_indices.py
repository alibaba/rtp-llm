"""Split combined prefill indices into SM120 SWA and compressed-cache tables."""

from typing import Tuple

import torch


def split_sm120_combined_tables(
    combined_indices: torch.Tensor,  # [rows, 1, W] | [rows, W] int32, from combine_*
    combined_lens: torch.Tensor,  # [rows] int32
    *,
    M: int,  # per-request workspace stride
    N: int,  # compressed-region size within the stride
    window_size: int,
    extra_width: int,  # max(cmp_topk width, 1)
    ratio: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split producer-combined tables into SM120 SWA/extra consumer tables.

    This is the exact production split (extracted verbatim from
    ``attention.py``'s workspace prefill path so tests exercise the same
    code, not a copy). ``M``/``N`` are the workspace-stride geometry:
    slots < N of each request's stride belong to the compressed (extra)
    region, the rest to SWA.

    Returns ``(swa_indices, swa_lens, extra_indices, extra_lens,
    extra_page_size)``.
    """

    from rtp_llm.models_py.modules.dsv4.const_cache import cached_arange

    gather_width = M - N
    combined_2d = combined_indices.squeeze(1).to(torch.int64)
    valid = combined_2d >= 0
    request_ids = torch.div(combined_2d.clamp_min(0), M, rounding_mode="floor")
    local_slots = torch.remainder(combined_2d.clamp_min(0), M)
    is_extra = valid & (local_slots < N)
    extra_lens = is_extra.sum(dim=1, dtype=torch.int32)
    swa_lens = combined_lens.to(torch.int32) - extra_lens
    extra_cols = cached_arange(extra_width, dtype=torch.int64, device=device).unsqueeze(
        0
    )
    extra_src = combined_2d[:, :extra_width]
    extra_req = request_ids[:, :extra_width]
    extra_local = local_slots[:, :extra_width]
    extra_indices = (extra_req * N + extra_local).to(torch.int32)
    extra_indices.masked_fill_(extra_cols >= extra_lens.to(torch.int64).unsqueeze(1), 0)
    extra_indices.masked_fill_(extra_src < 0, 0)
    aligned_extra_width = (extra_width + 63) // 64 * 64
    if aligned_extra_width != extra_width:
        padded_extra = torch.empty(
            (combined_2d.shape[0], aligned_extra_width),
            dtype=torch.int32,
            device=device,
        )
        padded_extra[:, :extra_width] = extra_indices
        padded_extra[:, extra_width:].zero_()
        extra_indices = padded_extra
    swa_cols = cached_arange(window_size, dtype=torch.int64, device=device).unsqueeze(0)
    swa_src_cols = extra_lens.to(torch.int64).unsqueeze(1) + swa_cols
    safe_cols = swa_src_cols.clamp_max(int(combined_2d.shape[1]) - 1)
    swa_src = combined_2d.gather(1, safe_cols)
    swa_req = torch.div(swa_src.clamp_min(0), M, rounding_mode="floor")
    swa_local = torch.remainder(swa_src.clamp_min(0), M) - N
    swa_indices = (swa_req * gather_width + swa_local).to(torch.int32)
    swa_indices.masked_fill_(swa_cols >= swa_lens.to(torch.int64).unsqueeze(1), 0)
    swa_indices.masked_fill_(swa_src < 0, 0)
    extra_page_size = 64 if ratio == 4 else 2
    return swa_indices, swa_lens, extra_indices, extra_lens, extra_page_size
