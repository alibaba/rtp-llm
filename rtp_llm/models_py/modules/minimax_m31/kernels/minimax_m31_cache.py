"""Model-local views of the two MiniMax-M3.1 NVFP4 cache groups."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class M31CacheViews:
    main_k: torch.Tensor
    main_v: torch.Tensor
    main_k_scale: torch.Tensor
    main_v_scale: torch.Tensor
    index_k: torch.Tensor
    index_k_scale: torch.Tensor


def m31_cache_views(
    main_pool: torch.Tensor,
    index_pool: torch.Tensor,
    kv_heads: int,
    head_dim: int,
    index_dim: int,
    page_size: int,
) -> M31CacheViews:
    if page_size != 128:
        raise ValueError("MiniMax-M3.1 NVFP4 cache requires 128-token pages")
    if head_dim % 16 or index_dim % 16 or kv_heads <= 0:
        raise ValueError("MiniMax-M3.1 NVFP4 cache dimensions are invalid")
    if main_pool.ndim != 2 or index_pool.ndim != 2:
        raise ValueError("MiniMax-M3.1 cache groups must be 2-D byte pools")
    if main_pool.dtype != torch.uint8 or index_pool.dtype != torch.uint8:
        raise ValueError("MiniMax-M3.1 cache groups must use uint8 storage")
    if main_pool.device != index_pool.device:
        raise ValueError("MiniMax-M3.1 cache groups must share a device")

    plane_bytes = kv_heads * page_size * head_dim // 2
    scale_bytes = kv_heads * page_size * head_dim // 16
    index_bytes = page_size * index_dim // 2
    index_scale_bytes = page_size * index_dim // 16
    if main_pool.shape[1] != 2 * (plane_bytes + scale_bytes):
        raise ValueError("MiniMax-M3.1 main KV group has incorrect page bytes")
    if index_pool.shape[1] != index_bytes + index_scale_bytes:
        raise ValueError("MiniMax-M3.1 indexer group has incorrect page bytes")

    return M31CacheViews(
        main_k=main_pool[:, :plane_bytes],
        main_v=main_pool[:, plane_bytes : 2 * plane_bytes],
        main_k_scale=main_pool[:, 2 * plane_bytes : 2 * plane_bytes + scale_bytes].view(
            torch.float8_e4m3fn
        ),
        main_v_scale=main_pool[:, 2 * plane_bytes + scale_bytes :].view(
            torch.float8_e4m3fn
        ),
        index_k=index_pool[:, :index_bytes],
        index_k_scale=index_pool[:, index_bytes:].view(torch.float8_e4m3fn),
    )
