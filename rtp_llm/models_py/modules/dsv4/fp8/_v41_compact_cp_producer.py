"""Owner-only production with the qualified SM120 arithmetic and cache codecs."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import _v41_fp4_triton as codec


@dataclass(frozen=True)
class OwnerRows:
    main: torch.Tensor
    index_payload: torch.Tensor
    index_scales: torch.Tensor
    state_tail: torch.Tensor | None


def project_owner(projection, x, owner, ratio):
    if ratio not in (1, 2):
        raise ValueError("Compact CP requires ratio one or two")
    values = projection.forward(x, owner.global_wkv)
    scores = projection.forward(x, owner.global_wgate) if ratio == 2 else None
    return values, scores


def produce_owner_rows(
    values, scores, owner, eps, frequencies, rope_dim, ratio, *, keep_tail
):
    from .attention_v41 import compress_pairs, rms_norm, rope_only

    if (
        ratio not in (1, 2)
        or values.shape != (1024, 512)
        or values.dtype != torch.float32
        or not values.is_cuda
        or frequencies.shape != (1024 // ratio, rope_dim // 2)
        or frequencies.dtype != torch.complex64
        or frequencies.device != values.device
        or (ratio == 1 and (scores is not None or keep_tail))
        or (
            ratio == 2
            and (
                scores is None
                or scores.shape != values.shape
                or scores.dtype != values.dtype
                or scores.device != values.device
            )
        )
    ):
        raise ValueError("Invalid compact CP owner projection")
    main_parts, payload_parts, scale_parts = [], [], []
    rows = 512 // ratio
    main_entries = 256 // ratio
    slots = torch.arange(rows, device=values.device, dtype=torch.int64)
    for first in (0, 512):
        part = values[first : first + 512]
        if ratio == 2:
            latent = compress_pairs(
                part.reshape(-1, 2, 512),
                scores[first : first + 512].reshape(-1, 2, 512),
                owner.global_norm,
                eps,
            )
        else:
            latent = rms_norm(part, owner.global_norm, eps).bfloat16()
        # Preserve the separately qualified GEMM shape for each disjoint half.
        projected_index = F.linear(latent, owner.index_wk)
        freqs = frequencies[first // ratio : (first + 512) // ratio]
        main = rope_only(latent.clone(), freqs, rope_dim)
        index = rope_only(
            rms_norm(projected_index, owner.index_k_norm, eps), freqs, rope_dim
        )
        main_pool = torch.empty(
            (rows // main_entries, main_entries, 288),
            device=values.device,
            dtype=torch.uint8,
        )
        index_pool = torch.empty(
            (rows // 256, 256, 68), device=values.device, dtype=torch.uint8
        )
        codec.quantize_and_insert_k_cache_fp4(main.contiguous(), main_pool, slots)
        codec.quantize_indexer_k_fp4(index.contiguous(), slots, index_pool)
        main_parts.append(codec.gather_k_cache_bytes_fp4(main_pool, slots))
        payload, scales = codec.gather_indexer_k_fp4(index_pool, slots)
        payload_parts.append(payload)
        scale_parts.append(scales)
    tail = torch.cat((values[-2:], scores[-2:]), dim=1) if keep_tail else None
    return OwnerRows(
        torch.cat(main_parts), torch.cat(payload_parts), torch.cat(scale_parts), tail
    )
