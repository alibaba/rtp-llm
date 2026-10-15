"""Strict zero-copy cache payload views shared by model-specific adapters."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CachePayloadGeometry:
    storage_dtype: torch.dtype
    payload_dtype: torch.dtype
    entries: int
    width: int

    @property
    def row_bytes(self) -> int:
        if self.entries <= 0 or self.width <= 0:
            raise RuntimeError(
                f"invalid cache geometry entries={self.entries}, width={self.width}"
            )
        return (
            self.entries
            * self.width
            * torch.empty((), dtype=self.payload_dtype).element_size()
        )


def payload_view(base, geometry: CachePayloadGeometry, label: str) -> torch.Tensor:
    if base is None or base.dim() != 2 or not base.is_contiguous():
        raise RuntimeError(f"{label} must be a contiguous 2-D pool")
    if base.dtype != geometry.storage_dtype:
        raise RuntimeError(
            f"{label} has dtype {base.dtype}, expected {geometry.storage_dtype}"
        )
    raw = base.view(torch.uint8)
    expected_bytes = geometry.row_bytes
    if int(raw.shape[1]) != expected_bytes:
        raise RuntimeError(
            f"{label} row has {raw.shape[1]} bytes, expected {expected_bytes}"
        )
    return raw.view(geometry.payload_dtype).view(
        int(raw.shape[0]), geometry.entries, geometry.width
    )


def same_tensor_metadata(left, right) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return left is right
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return left == right
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    if left.stride() == right.stride() and left.data_ptr() == right.data_ptr():
        return True
    return bool(torch.equal(left, right))
