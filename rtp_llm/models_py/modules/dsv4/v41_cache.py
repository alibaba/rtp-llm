"""V4.1 planar cache views and explicit eager codec reference.

The default execution backend is Triton. ``backend="eager"`` is a diagnostic
Torch path with value validation and synchronization, not a serving fallback.
Quantization follows the DeepSeek/vLLM formulas attributed in
``fp8/_v41_cache_triton.py``. Physical slots are supplied by the caller; ring
ownership must be resolved before writing, and nonnegative write slots must
be unique within a launch. Valid inputs must be finite and slots must be in
range; -1 skips a write or zero-fills a read. The Triton path checks tensor
metadata without host synchronization; value assertions require TRITON_DEBUG.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models.dsv41_kv_cache import (
    V41CacheBinding,
    V41PageLayout,
    dsv41_page_layouts,
)

_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(left.numel() and right.numel()) and (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservative byte bounds, allowing disjoint views of native pool storage."""
    if left.device != right.device or not left.numel() or not right.numel():
        return False

    def end(value):
        return value.data_ptr() + value.element_size() * (
            1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride()))
        )

    return left.data_ptr() < end(right) and right.data_ptr() < end(left)


def _codec_shape(tag: str) -> tuple[int, int]:
    if tag in ("csa_kv", "hca_kv"):
        return 512, 16
    if tag == "indexer_kv":
        return 128, 32
    if tag == "swa_kv":
        return 512, 32
    raise ValueError(f"No V4.1 quantized codec for {tag!r}")


def quantize_rows_eager(values: torch.Tensor, tag: str) -> torch.Tensor:
    """Encode floating rows as payload-plus-scale transport rows, without pages."""
    dim, group = _codec_shape(tag)
    if values.ndim != 2 or values.shape[1] != dim or values.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"V4.1 {tag} expects floating [N, {dim}] rows")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("V4.1 codec inputs must be finite")
    x = values.float().reshape(-1, dim // group, group)
    maximum = x.abs().amax(-1)
    if tag == "swa_kv":
        exponent = torch.ceil(torch.log2(maximum.clamp_min(1e-4) * (1.0 / 448.0)))
        scale_bytes = (exponent + 127).to(torch.uint8)
        payload = (x * torch.exp2(-exponent)[..., None]).clamp(-448, 448)
        payload = payload.to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.cat((payload.reshape(-1, dim), scale_bytes), -1)
    if tag in ("csa_kv", "hca_kv"):
        scale = (maximum.clamp_min(6.0 * 2.0**-9) / 6.0).clamp_max(448)
        scale_fp8 = scale.to(torch.float8_e4m3fn)
        scale_bytes, scale = scale_fp8.view(torch.uint8), scale_fp8.float()
    else:
        # Round the FP32 product upward to a power of two, as fast_round_scale does.
        bits = (maximum.clamp_min(6.0 * 2.0**-126) * (1.0 / 6.0)).view(torch.int32)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
        scale_bytes = exponent.to(torch.uint8)
        scale = (exponent << 23).view(torch.float32)
    normalized = (x / scale[..., None]).clamp(-6, 6)
    magnitude = normalized.abs()
    codes = (magnitude > 0.25).to(torch.uint8)
    for threshold, inclusive in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        codes += (magnitude >= threshold if inclusive else magnitude > threshold).to(
            torch.uint8
        )
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    codes = codes.reshape(-1, dim)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat((payload, scale_bytes), -1)


def dequantize_rows_eager(
    rows: torch.Tensor, tag: str, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    dim, group = _codec_shape(tag)
    payload_bytes = dim if tag == "swa_kv" else dim // 2
    if (
        rows.ndim != 2
        or rows.shape[1] != payload_bytes + dim // group
        or rows.dtype != torch.uint8
    ):
        raise ValueError(f"Invalid V4.1 {tag} transport rows")
    if out_dtype not in _FLOAT_DTYPES:
        raise ValueError("V4.1 dequantization requires a floating output dtype")
    payload = rows[:, :payload_bytes].contiguous()
    scale_bytes = rows[:, payload_bytes:].contiguous()
    if tag in ("csa_kv", "hca_kv"):
        scales = scale_bytes.view(torch.float8_e4m3fn).float()
    else:
        scales = torch.exp2(scale_bytes.float() - 127)
        scales = torch.where(scale_bytes == 255, float("nan"), scales)
    if tag == "swa_kv":
        values = payload.view(torch.float8_e4m3fn).float()
    else:
        codes = torch.stack((payload & 15, payload >> 4), -1).reshape(-1, dim)
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=rows.device)
        values = levels[(codes & 7).long()]
        values = torch.where(codes >= 8, -values, values)
    return (values * scales.repeat_interleave(group, -1)).to(out_dtype)


@dataclass(frozen=True)
class V41PagedCache:
    raw: torch.Tensor
    tag: str
    gen_num_per_cycle: int = 1

    def __post_init__(self):
        _codec_shape(self.tag)
        if (
            self.raw.dtype != torch.uint8
            or self.raw.ndim != 2
            or self.raw.shape[1] != self.layout.stride_bytes
            or self.raw.stride(1) != 1
            or self.raw.stride(0) < self.layout.stride_bytes
        ):
            raise ValueError("V4.1 cache requires uint8 [pages, page_stride] storage")

    @property
    def layout(self) -> V41PageLayout:
        return dsv41_page_layouts(self.gen_num_per_cycle)[self.tag]

    @classmethod
    def from_layer_cache(
        cls, layer_cache, binding: V41CacheBinding, gen_num_per_cycle: int = 1
    ):
        expected_source = binding.source_layer_id if binding.is_read_replica else None
        if (
            layer_cache.tag != binding.tag
            or layer_cache.layer_id != binding.storage_layer_id
            or layer_cache.read_replica_source_layer_id != expected_source
            or layer_cache.seq_size_per_block != 256
        ):
            raise ValueError("V4.1 cache view does not match its source binding")
        scales = layer_cache.kv_scale_base
        if scales is not None and scales.numel():
            raise ValueError("V4.1 scale plane must reside inside the opaque cache")
        return cls(layer_cache.kv_cache_base, binding.tag, gen_num_per_cycle)

    def _check_slots(self, slots: torch.Tensor, backend: str, *, writing: bool = False):
        if backend not in ("eager", "triton"):
            raise ValueError(f"Unknown V4.1 codec backend {backend!r}")
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.device != self.raw.device
            or not slots.is_contiguous()
            or _shares_storage(slots, self.raw)
        ):
            raise ValueError("V4.1 slots must be a contiguous device integer vector")
        if backend == "triton" and not self.raw.is_cuda:
            raise ValueError("V4.1 Triton codecs require CUDA storage")
        if backend == "eager":
            if bool(
                (
                    (slots < -1) | (slots >= self.raw.shape[0] * self.layout.entries)
                ).any()
            ):
                raise ValueError("V4.1 slot is outside the allocated cache")
            valid = slots[slots >= 0]
            if writing and valid.unique().numel() != valid.numel():
                raise ValueError("V4.1 cache writes require unique slots")

    def _check_rows(
        self, rows: torch.Tensor, slots: torch.Tensor, width: int, *, floating: bool
    ):
        if (
            rows.shape != (slots.numel(), width)
            or rows.device != self.raw.device
            or rows.dtype not in (_FLOAT_DTYPES if floating else (torch.uint8,))
            or not rows.is_contiguous()
            or _shares_storage(rows, self.raw)
            or _shares_storage(rows, slots)
        ):
            raise ValueError(
                "V4.1 rows must be contiguous, disjoint from cache/slots, and match device/shape/dtype"
            )

    def _addresses(self, slots: torch.Tensor):
        layout = self.layout
        valid = slots >= 0
        selected = slots[valid].long()
        pages, entries = selected // layout.entries, selected % layout.entries
        payload = entries[:, None] * layout.payload_bytes_per_entry + torch.arange(
            layout.payload_bytes_per_entry, device=slots.device
        )
        scales = (
            layout.scale_plane_offset
            + entries[:, None] * layout.scale_bytes_per_entry
            + torch.arange(layout.scale_bytes_per_entry, device=slots.device)
        )
        return valid, pages[:, None], torch.cat((payload, scales), -1)

    def gather(
        self,
        slots: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        backend: str = "triton",
    ) -> torch.Tensor:
        """Serialize slots to payload-plus-scale rows, zeroing -1 slots."""
        self._check_slots(slots, backend)
        width = self.layout.payload_bytes_per_entry + self.layout.scale_bytes_per_entry
        if out is None:
            out = torch.empty(
                (slots.numel(), width), dtype=torch.uint8, device=self.raw.device
            )
        self._check_rows(out, slots, width, floating=False)
        if backend == "eager":
            valid, pages, columns = self._addresses(slots)
            out.zero_()
            out[valid] = self.raw[pages, columns]
        elif slots.numel():
            from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import (
                transfer_rows,
            )

            transfer_rows(self, slots, out, scatter=False)
        return out

    def scatter(
        self, rows: torch.Tensor, slots: torch.Tensor, *, backend: str = "triton"
    ) -> None:
        """Import packed rows without requantization, preserving other slots/padding."""
        self._check_slots(slots, backend, writing=True)
        width = self.layout.payload_bytes_per_entry + self.layout.scale_bytes_per_entry
        self._check_rows(rows, slots, width, floating=False)
        if backend == "eager":
            valid, pages, columns = self._addresses(slots)
            self.raw[pages, columns] = rows[valid]
        elif slots.numel():
            from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import (
                transfer_rows,
            )

            transfer_rows(self, slots, rows, scatter=True)

    def write(
        self, values: torch.Tensor, slots: torch.Tensor, *, backend: str = "triton"
    ) -> None:
        self._check_slots(slots, backend, writing=True)
        dim, _ = _codec_shape(self.tag)
        self._check_rows(values, slots, dim, floating=True)
        if backend == "eager":
            valid = slots >= 0
            self.scatter(
                quantize_rows_eager(values[valid], self.tag),
                slots[valid],
                backend=backend,
            )
        elif slots.numel():
            from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import encode_rows

            encode_rows(self, slots, values)

    def read(
        self,
        slots: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        out_dtype: torch.dtype = torch.bfloat16,
        backend: str = "triton",
    ) -> torch.Tensor:
        self._check_slots(slots, backend)
        dim, _ = _codec_shape(self.tag)
        if out is None:
            if out_dtype not in _FLOAT_DTYPES:
                raise ValueError("V4.1 dequantization requires a floating output dtype")
            out = torch.empty(
                (slots.numel(), dim), dtype=out_dtype, device=self.raw.device
            )
        self._check_rows(out, slots, dim, floating=True)
        if backend == "eager":
            out.copy_(
                dequantize_rows_eager(
                    self.gather(slots, backend=backend), self.tag, out.dtype
                )
            )
        elif slots.numel():
            from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import decode_rows

            decode_rows(self, slots, out)
        return out


def logical_to_physical_slots(
    blocks: torch.Tensor, indices: torch.Tensor, tag: str, compression_ratio: int
) -> torch.Tensor:
    """Eager request-local planner for GLOBAL/index rows, including ratio-2 index pages.

    ``indices`` counts compressed rows, not raw tokens. Each block-table entry
    covers 256 raw tokens. Indexer pages retain 256 allocated entries at ratio 2.
    This validating helper is not a CUDA graph planner.
    """
    if (tag, compression_ratio) not in (
        ("csa_kv", 2),
        ("hca_kv", 1),
        ("indexer_kv", 1),
        ("indexer_kv", 2),
    ):
        raise ValueError("Invalid V4.1 GLOBAL/index compression ratio")
    if (
        blocks.ndim != 1
        or indices.ndim != 1
        or blocks.device != indices.device
        or any(t.dtype not in (torch.int32, torch.int64) for t in (blocks, indices))
    ):
        raise ValueError("V4.1 block table and indices must be device integer vectors")
    used_entries = 256 // compression_ratio
    if bool(((indices < -1) | (indices >= blocks.numel() * used_entries)).any()):
        raise ValueError("V4.1 logical index is outside the block table")
    result = torch.full_like(indices, -1, dtype=torch.int64)
    valid = indices >= 0
    selected = indices[valid].long()
    pages = blocks[selected // used_entries].long()
    if bool((pages < 0).any()):
        raise ValueError("V4.1 logical index refers to an unallocated page")
    physical_entries = dsv41_page_layouts(0)[tag].entries
    result[valid] = pages * physical_entries + selected % used_entries
    return result
