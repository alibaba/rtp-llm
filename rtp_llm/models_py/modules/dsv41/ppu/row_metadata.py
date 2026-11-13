"""Metadata-only descriptions of the existing per-forward row ABIs.

Descriptions retain their sources, but neither cache payloads nor certify an
optimized path. Inference tensors follow the existing read-only-within-forward
contract: without a version counter, in-place value changes are not detectable.
CP8 physical cache ownership remains the native cache adapter's responsibility.
"""

from dataclasses import dataclass
from enum import Enum

import torch


def tensor_metadata_key(tensor):
    if tensor is None:
        return None
    # Native bindings may return another Python wrapper for one TensorImpl.
    version = None if tensor.is_inference() else tensor._version
    return (
        tensor._cdata,
        tensor.data_ptr(),
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
        version,
    )


def is_plain_tensor(tensor):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and not tensor.is_neg()
        and not tensor.is_conj()
    )


class RowLayout(Enum):
    LOCAL_QUERY = "local_query"
    CP_QUERY = "cp_query"
    COMPACT_QUERY = "compact_query"
    FULL = "full_request_major"
    SELECTED = "selected_request_major"


class SelectorKind(Enum):
    ALL = "all"
    RANGE = "range"
    MASK = "mask"
    INDICES = "indices"


@dataclass(frozen=True, eq=False)
class RowScope:
    """Strong owners distinguish forwards even when storage is reused."""

    forward: object
    context: object = None
    plan: object = None
    proof: object = None
    lengths: tuple = ()
    prefixes: tuple = ()
    chunks: tuple = ()

    @property
    def key(self):
        return (
            id(self.forward),
            id(self.context),
            id(self.plan),
            id(self.proof),
            self.lengths,
            self.prefixes,
            self.chunks,
        )


def _value_key(value):
    if isinstance(value, torch.Tensor):
        return tensor_metadata_key(value)
    if isinstance(value, slice):
        return slice, value.start, value.stop, value.step
    return type(value), id(value)


@dataclass(frozen=True, eq=False)
class RowView:
    """Wrap one legacy tuple without converting, copying or validating it.

    Keys and structural validation are opt-in. Compatibility callers therefore
    keep their original errors and perform no additional tensor metadata reads.
    A fresh view is required to describe rebound state/context sources.
    """

    legacy: tuple
    layout: RowLayout
    scope: RowScope
    query: bool
    sources: tuple = ()

    @property
    def positions(self):
        return self.legacy[0]

    @property
    def request_ids(self):
        return self.legacy[1]

    @property
    def cu_seqlens(self):
        return self.as_rows()[2]

    @property
    def valid_mask(self):
        return self.as_rows()[3]

    def as_rows(self):
        if not self.query:
            raise ValueError("full-row view has no query cu_seqlens")
        return self.legacy

    def as_full_rows(self):
        if self.query:
            raise ValueError("query-row view uses the four-field rows ABI")
        return self.legacy

    @property
    def signature(self):
        return (
            self.layout,
            self.scope.key,
            self.query,
            tuple(_value_key(value) for value in self.legacy),
            tuple(_value_key(value) for value in self.sources),
        )

    @property
    def selector(self):
        return RowSelector(self.legacy[3 if self.query else 2], self, self.signature)

    def validate(self):
        """Check shape/dtype/device only; never inspect device tensor values."""
        if not isinstance(self.legacy, tuple) or len(self.legacy) != (
            4 if self.query else 3
        ):
            raise ValueError("invalid row tuple ABI")
        positions, requests = self.legacy[:2]
        if (
            not is_plain_tensor(positions)
            or not is_plain_tensor(requests)
            or positions.ndim != 1
            or requests.shape != positions.shape
            or positions.dtype != torch.int64
            or requests.dtype != torch.int64
            or positions.device != requests.device
        ):
            raise ValueError("positions/request_ids must be matching int64 vectors")
        if self.query:
            cu = self.cu_seqlens
            if (
                not is_plain_tensor(cu)
                or cu.ndim != 1
                or cu.dtype not in (torch.int32, torch.int64)
                or cu.device != positions.device
            ):
                raise ValueError(
                    "cu_seqlens must be an integer vector on the row device"
                )
            if self.selector.kind is not SelectorKind.MASK:
                raise ValueError("query rows require their original boolean mask")
        self.selector.validate()
        return self


@dataclass(frozen=True, eq=False)
class RowSelector:
    """A selector in one described domain, never a kernel eligibility proof."""

    value: object
    owner: RowView
    domain: tuple

    @property
    def kind(self):
        if isinstance(self.value, slice):
            return SelectorKind.ALL if self.value == slice(None) else SelectorKind.RANGE
        if isinstance(self.value, torch.Tensor):
            if self.value.dtype == torch.bool:
                return SelectorKind.MASK
            if self.value.dtype in (torch.int32, torch.int64):
                return SelectorKind.INDICES
        raise ValueError("unsupported row selector")

    def belongs_to(self, view):
        return isinstance(view, RowView) and self.domain == view.signature

    def validate(self):
        kind = self.kind
        if kind in (SelectorKind.MASK, SelectorKind.INDICES):
            if (
                not is_plain_tensor(self.value)
                or self.value.ndim != 1
                or self.value.device != self.owner.positions.device
            ):
                raise ValueError("selector must be a vector on the row device")
            if (
                kind is SelectorKind.MASK
                and self.value.shape != self.owner.positions.shape
            ):
                raise ValueError("mask must match the described rows")
        # Index bounds, mask values and slice normalization are deliberately
        # left to the existing consumer; checking them would add device work.
        return self
