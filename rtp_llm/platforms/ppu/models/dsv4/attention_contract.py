"""PPU FlashMLA pool validation and scheduler identity."""

import torch

from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.decode_attn_metadata import RICH_CSA_INDEX_WIDTH, RICH_HCA_INDEX_WIDTH, RICH_SWA_INDEX_WIDTH, RICH_SWA_PAGE, RICH_SCHED_BUCKETS, RICH_CSA_BUCKET, RICH_HCA_BUCKET


def _validate_model1_cache_tensor(pool, *, block_size):
    if pool.dtype != torch.uint8 or pool.ndim != 3 or tuple(pool.shape[1:]) != (block_size, 584):
        raise ValueError("MODEL1 cache must be uint8 [blocks, block_size, 584]")
    if pool.stride(2) != 1 or pool.stride(1) != 584 or pool.stride(0) < block_size * 584:
        raise ValueError("MODEL1 cache requires dense token bytes and a non-overlapping block stride")


def _validate_rich_pool(
    *,
    q: torch.Tensor,
    pool: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    page: int,
    index_width: int,
    name: str,
    allowed_widths=None,
    multiple_of=None,
    allow_ring_extension=False,
) -> None:
    """Validate the declared cache geometry and sparse attention ABI."""
    if q.ndim != 4 or int(q.shape[1]) < 1 or int(q.shape[-1]) != 512:
        raise ValueError(f"{name}: q must be [B, q_len, H, D], q_len must be positive")
    if pool.device != q.device:
        raise ValueError(f"{name}: pool must share q device")
    actual_page = int(pool.shape[1]) if pool.ndim >= 2 else 0
    _validate_rich_pool_page(
        actual_page=actual_page,
        expected_page=page,
        allow_ring_extension=allow_ring_extension,
        name=name,
    )
    _validate_model1_cache_tensor(pool, block_size=actual_page)
    if int(pool.stride(0)) % 576:
        raise ValueError(f"{name}: physical block stride must be 576-byte aligned")
    if indices.device != q.device:
        raise ValueError(f"{name}: indices must share q device")
    if indices.ndim != 3 or tuple(indices.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError(f"{name}: indices must align with [B,q_len]")
    actual_width = int(indices.shape[-1])
    if allowed_widths is not None:
        width_ok = actual_width in allowed_widths
    elif multiple_of is not None:
        width_ok = actual_width > 0 and actual_width % multiple_of == 0
    else:
        width_ok = actual_width == index_width
    if not width_ok or indices.dtype != torch.int32:
        raise ValueError(f"{name}: indices have unsupported width {actual_width}")
    if not indices.is_contiguous():
        raise ValueError(f"{name}: indices must be contiguous")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"{name}: q must be BF16")
    if sink.device != q.device:
        raise ValueError(f"{name}: attn_sink must share q device")
    if sink.ndim != 1 or sink.shape[0] != q.shape[2]:
        raise ValueError(f"{name}: attn_sink must be [num_heads]")
    if sink.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"{name}: attn_sink must be FP32 or BF16")


def _validate_rich_pool_page(
    *, actual_page: int, expected_page: int, allow_ring_extension: bool, name: str
) -> None:
    """Validate a rich-pool page while preserving MTP state-ring slack.

    The SWA opaque cache stores ``window + gamma`` entries and rounds that
    count up to an even number.  FlashMLA addresses it through the runtime
    tensor shape, so the extra ring entries are physical capacity rather than
    a change to the 128-token attention window.  Compressed pools retain the
    exact-page contract.
    """
    actual_page = int(actual_page)
    expected_page = int(expected_page)
    if actual_page == expected_page:
        return
    if (
        allow_ring_extension
        and actual_page > expected_page
        and actual_page <= 2 * expected_page
        and actual_page % 2 == 0
    ):
        return
    raise ValueError(
        f"{name}: unsupported MODEL1 page {actual_page}; " f"expected {expected_page}"
    )


def _validate_lengths(length, *, q: torch.Tensor, name: str) -> None:
    if length is None:
        return
    if length.ndim != 1 or int(length.shape[0]) != int(q.shape[0]):
        raise ValueError(f"{name}: effective lengths must have shape [B]")
    if length.dtype != torch.int32:
        raise TypeError(f"{name}: effective lengths must be int32")
    if length.device != q.device:
        raise ValueError(f"{name}: effective lengths must share q device")


def validate_swa_pool(*, q, swa_pool_3d, swa_topk_3d, attn_sink, topk_length):
    _validate_rich_pool(
        q=q,
        pool=swa_pool_3d,
        indices=swa_topk_3d,
        sink=attn_sink,
        page=RICH_SWA_PAGE,
        index_width=RICH_SWA_INDEX_WIDTH,
        name="swa",
        allowed_widths=(RICH_SWA_INDEX_WIDTH,),
        allow_ring_extension=True,
    )
    _validate_lengths(topk_length, q=q, name="swa.topk_length")


def validate_dual_pool(*, q, swa_pool_3d, swa_topk_3d, cmp_pool_3d, cmp_topk_3d, attn_sink, topk_length, extra_topk_length, compress_ratio, cmp_tokens_per_block):
    _validate_rich_pool(
        q=q,
        pool=swa_pool_3d,
        indices=swa_topk_3d,
        sink=attn_sink,
        page=RICH_SWA_PAGE,
        index_width=RICH_SWA_INDEX_WIDTH,
        name="dual.swa",
        allow_ring_extension=True,
    )
    _validate_lengths(topk_length, q=q, name="dual.topk_length")
    _validate_lengths(extra_topk_length, q=q, name="dual.extra_topk_length")
    # Pool identity comes from the layer contract, not its physical page size.
    # CUDA block128 has CSA32/HCA1; block256 has CSA64/HCA2.
    if compress_ratio not in (4, 128):
        raise ValueError("Compressed attention requires CSA ratio 4 or HCA ratio 128")
    if cmp_tokens_per_block <= 0 or cmp_tokens_per_block % compress_ratio:
        raise ValueError("Compressed cache block size must be divisible by its ratio")
    extra_page = cmp_tokens_per_block // compress_ratio
    is_csa = compress_ratio == 4
    extra_width = RICH_CSA_INDEX_WIDTH if is_csa else RICH_HCA_INDEX_WIDTH
    _validate_rich_pool(
        q=q,
        pool=cmp_pool_3d,
        indices=cmp_topk_3d,
        sink=attn_sink,
        page=extra_page,
        index_width=extra_width,
        name="dual.extra",
        allowed_widths=(512, 1024) if is_csa else None,
        multiple_of=64 if not is_csa else None,
    )


def scheduler_key(batch_size, q_len, topk, extra_attn_type, extra_index_width):
    if extra_attn_type not in RICH_SCHED_BUCKETS:
        raise ValueError(
            f"unsupported FlashMLA rich scheduler bucket: {extra_attn_type!r}"
        )
    if int(q_len) < 1:
        raise ValueError(f"FlashMLA rich q_len must be positive: {q_len}")
    if int(topk) <= 0 or int(topk) % 64:
        raise ValueError(
            "FlashMLA scheduler primary width must be positive and "
            f"64-aligned: {topk}"
        )

    if extra_attn_type is None:
        if extra_index_width not in (None, 0):
            raise ValueError("SWA-only scheduler cannot carry extra index width")
    elif extra_index_width is None or int(extra_index_width) <= 0:
        raise ValueError("dual scheduler requires a positive extra index width")
    elif extra_attn_type == RICH_CSA_BUCKET and int(extra_index_width) not in (
        512,
        1024,
    ):
        raise ValueError("CSA scheduler extra index width must be 512 or 1024")
    elif extra_attn_type == RICH_HCA_BUCKET and int(extra_index_width) % 64:
        raise ValueError("HCA scheduler extra index width must be 64-byte aligned")

    return (batch_size, q_len, int(topk), extra_attn_type, int(extra_index_width or 0))
