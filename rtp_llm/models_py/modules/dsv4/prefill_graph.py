"""Utilities for a production-safe DSV4 prefill CUDA graph path.

This module provides the request-metadata building blocks the DSV4 prefill graph
path needs: recursively mirror nested metadata objects into stable tensors whose
contents can be updated per request, and track graph-owned layer-loop state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


_SCALAR_TYPES = (type(None), bool, int, float, str, torch.device, torch.dtype)
_MATERIALIZE_DROP = object()


@dataclasses.dataclass(frozen=True)
class PrefillGraphKey:
    """Shape/mode identity for a DSV4 prefill graph bucket."""

    token_bucket: int
    batch_bucket: int
    cp_size: int
    prefix_bucket: int = 0
    reuse_bucket: int = 0
    local_token_bucket: int = 0
    hidden_shape_tail: tuple[int, ...] = ()
    hidden_dtype: str = ""
    block_cap: int = 0
    block_table_keys: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class PrefillGraphRequest:
    """Runtime facts needed before deciding graph replay vs eager fallback."""

    token_count: int
    batch_size: int
    cp_size: int
    prefix_length: int = 0
    max_prefix_length: int = 0
    prefix_unknown: bool = False
    prepare_hidden: bool = False
    cache_store: bool = False
    mtp_hidden: bool = False


@dataclasses.dataclass(frozen=True)
class PrefillGraphDecision:
    enabled: bool
    key: PrefillGraphKey | None
    reason: str


@dataclasses.dataclass(frozen=True)
class GraphTensorPointer:
    name: str
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


@dataclasses.dataclass(frozen=True)
class GraphCaptureSurfaceReport:
    static_bound: bool
    live_tensor_count: int
    static_bound_count: int
    live_not_static: tuple[str, ...]
    missing_static: tuple[str, ...]
    skipped_critical: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class StaticPrefillLayerLoopArgs:
    """Exact-shape static tensors that can replace live layer-loop arguments."""

    input_ids: torch.Tensor
    hidden: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    block_tables_by_type: dict[int, torch.Tensor] | None


def _parse_positive_int_list(value: str | Sequence[int]) -> tuple[int, ...]:
    return _parse_int_list(value, min_value=1)


def _parse_nonnegative_int_list(value: str | Sequence[int]) -> tuple[int, ...]:
    return _parse_int_list(value, min_value=0)


def _parse_int_list(value: str | Sequence[int], *, min_value: int) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
        items = [int(item) for item in raw_items]
    else:
        items = [int(item) for item in value]
    buckets = tuple(sorted(set(item for item in items if item >= min_value)))
    if not buckets:
        raise ValueError(f"at least one graph bucket >= {min_value} is required")
    return buckets


def _pick_bucket(value: int, buckets: Sequence[int]) -> int | None:
    for bucket in buckets:
        if int(value) <= int(bucket):
            return int(bucket)
    return None


def select_prefill_graph_key(
    request: PrefillGraphRequest,
    *,
    enabled: bool,
    token_buckets: str | Sequence[int],
    batch_buckets: str | Sequence[int] = (1,),
    prefix_buckets: str | Sequence[int] = (0,),
    reuse_buckets: str | Sequence[int] = (0,),
    fixed_cp_size: int | None = 8,
    allow_prefix_reuse: bool = False,
) -> PrefillGraphDecision:
    """Return the graph key for a request, or an eager-fallback reason.

    This is intentionally conservative.  It only decides whether graph replay is
    *eligible*; it does not prove static metadata has been populated or launch a
    graph.  Production replay must still verify its static buffers before use.
    """

    if not enabled:
        return PrefillGraphDecision(False, None, "disabled")
    if request.token_count <= 0:
        return PrefillGraphDecision(False, None, "empty_tokens")
    if request.batch_size <= 0:
        return PrefillGraphDecision(False, None, "empty_batch")
    if request.prepare_hidden:
        return PrefillGraphDecision(False, None, "prepare_hidden")
    if request.cache_store:
        return PrefillGraphDecision(False, None, "cache_store")
    if request.mtp_hidden:
        return PrefillGraphDecision(False, None, "mtp_hidden")
    if request.prefix_unknown:
        return PrefillGraphDecision(False, None, "prefix_unknown")
    if fixed_cp_size is not None and int(request.cp_size) != int(fixed_cp_size):
        return PrefillGraphDecision(False, None, "cp_size")
    if not allow_prefix_reuse and (
        int(request.prefix_length) != 0 or int(request.max_prefix_length) != 0
    ):
        return PrefillGraphDecision(False, None, "prefix_reuse")

    try:
        parsed_token_buckets = _parse_positive_int_list(token_buckets)
        parsed_batch_buckets = _parse_positive_int_list(batch_buckets)
        parsed_prefix_buckets = _parse_nonnegative_int_list(prefix_buckets)
        parsed_reuse_buckets = _parse_nonnegative_int_list(reuse_buckets)
    except (TypeError, ValueError):
        return PrefillGraphDecision(False, None, "invalid_buckets")

    token_bucket = _pick_bucket(int(request.token_count), parsed_token_buckets)
    if token_bucket is None:
        return PrefillGraphDecision(False, None, "token_overflow")
    batch_bucket = _pick_bucket(int(request.batch_size), parsed_batch_buckets)
    if batch_bucket is None:
        return PrefillGraphDecision(False, None, "batch_overflow")
    prefix_bucket = _pick_bucket(int(request.max_prefix_length), parsed_prefix_buckets)
    if prefix_bucket is None:
        return PrefillGraphDecision(False, None, "prefix_overflow")
    reuse_bucket = _pick_bucket(int(request.prefix_length), parsed_reuse_buckets)
    if reuse_bucket is None:
        return PrefillGraphDecision(False, None, "reuse_overflow")

    key = PrefillGraphKey(
        token_bucket=token_bucket,
        batch_bucket=batch_bucket,
        cp_size=int(request.cp_size),
        prefix_bucket=prefix_bucket,
        reuse_bucket=reuse_bucket,
    )
    return PrefillGraphDecision(True, key, "ok")


def with_static_state_invariants(
    key: PrefillGraphKey,
    *,
    local_token_bucket: int,
    hidden_shape_tail: Sequence[int],
    hidden_dtype: torch.dtype,
    block_cap: int,
    block_table_keys: Sequence[int],
) -> PrefillGraphKey:
    return dataclasses.replace(
        key,
        local_token_bucket=int(local_token_bucket),
        hidden_shape_tail=tuple(int(dim) for dim in hidden_shape_tail),
        hidden_dtype=str(hidden_dtype),
        block_cap=int(block_cap),
        block_table_keys=tuple(sorted(int(key) for key in block_table_keys)),
    )


def _is_namedtuple_instance(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


class StaticTensorMirror:
    """Keep tensor addresses stable while refreshing their contents.

    ``update(obj)`` returns an object with the same nested structure as ``obj``.
    Tensor leaves are persistent buffers; subsequent same-shape updates copy new
    values into those buffers and return the same object graph. Shape/dtype/
    device changes, or scalar metadata changes, rebuild the mirror.

    Unknown object leaves are rejected by default.  A CUDA graph metadata buffer
    must not key opaque Python objects by ``id()`` because object addresses can
    be reused and the graph may silently keep stale owners alive.
    """

    def __init__(self, *, allow_opaque_objects: bool = False) -> None:
        self._static: Any = None
        self._signature: Any = None
        self._allow_opaque_objects = bool(allow_opaque_objects)

    def update(self, value: Any) -> Any:
        signature = self._signature_of(value)
        if self._static is None or signature != self._signature:
            self._static = self._allocate(value)
            self._signature = signature
        else:
            self._copy_into(self._static, value)
        return self._static

    @property
    def signature(self) -> Any:
        return self._signature

    def _signature_of(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                tuple(value.stride()),
            )
        if isinstance(value, _SCALAR_TYPES):
            return ("scalar", value)
        if _is_namedtuple_instance(value):
            return (
                "namedtuple",
                type(value),
                tuple(
                    (name, self._signature_of(getattr(value, name)))
                    for name in value._fields
                ),
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value),
                tuple(
                    (field.name, self._signature_of(getattr(value, field.name)))
                    for field in dataclasses.fields(value)
                ),
            )
        if isinstance(value, Mapping):
            return (
                "mapping",
                type(value),
                tuple(
                    (key, self._signature_of(value[key]))
                    for key in sorted(value.keys())
                ),
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return (
                "sequence",
                type(value),
                tuple(self._signature_of(item) for item in value),
            )
        if self._allow_opaque_objects:
            return ("object", type(value), id(value))
        raise TypeError(
            "StaticTensorMirror cannot mirror opaque object "
            f"{type(value).__name__}; convert replay-varying metadata to tensors "
            "or pass allow_opaque_objects=True for non-graph debug use."
        )

    def _allocate(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, _SCALAR_TYPES):
            return value
        if _is_namedtuple_instance(value):
            return type(value)(
                *(self._allocate(getattr(value, name)) for name in value._fields)
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.replace(
                value,
                **{
                    field.name: self._allocate(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                },
            )
        if isinstance(value, Mapping):
            return type(value)(
                (key, self._allocate(value[key])) for key in value.keys()
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return type(value)(self._allocate(item) for item in value)
        if self._allow_opaque_objects:
            return value
        raise TypeError(f"cannot allocate opaque object {type(value).__name__}")

    def _copy_into(self, dst: Any, src: Any) -> None:
        if isinstance(dst, torch.Tensor):
            dst.copy_(src, non_blocking=True)
            return
        if isinstance(src, _SCALAR_TYPES):
            return
        if _is_namedtuple_instance(src):
            for name in src._fields:
                self._copy_into(getattr(dst, name), getattr(src, name))
            return
        if dataclasses.is_dataclass(src) and not isinstance(src, type):
            for field in dataclasses.fields(src):
                self._copy_into(getattr(dst, field.name), getattr(src, field.name))
            return
        if isinstance(src, Mapping):
            for key in src.keys():
                self._copy_into(dst[key], src[key])
            return
        if isinstance(src, Sequence) and not isinstance(src, (str, bytes, bytearray)):
            for dst_item, src_item in zip(dst, src):
                self._copy_into(dst_item, src_item)
            return
        if self._allow_opaque_objects:
            return
        raise TypeError(f"cannot copy opaque object {type(src).__name__}")


class StaticPrefillGraphInputs:
    """Fixed-capacity request metadata buffers for future prefill graph replay.

    This class stores replay-varying scalar values as device tensors and mirrors
    block tables into stable fixed-capacity tensors.  It is deliberately small:
    it does not attempt to mirror full ``PrefillMeta`` yet, and it is not wired
    into model forward.
    """

    def __init__(
        self,
        *,
        token_cap: int,
        batch_cap: int,
        block_cap: int,
        device: torch.device | str,
        block_table_keys: Sequence[int],
    ) -> None:
        self.token_cap = int(token_cap)
        self.batch_cap = int(batch_cap)
        self.block_cap = int(block_cap)
        self.device = torch.device(device)
        self.position_ids = torch.empty(
            self.token_cap, dtype=torch.int64, device=self.device
        )
        self.req_id_per_token = torch.empty(
            self.token_cap, dtype=torch.int32, device=self.device
        )
        self.cu_seqlens = torch.empty(
            self.batch_cap + 1, dtype=torch.int64, device=self.device
        )
        self.input_lengths = torch.empty(
            self.batch_cap, dtype=torch.int32, device=self.device
        )
        self.prefix_lengths = torch.empty(
            self.batch_cap, dtype=torch.int32, device=self.device
        )
        self.scalar_i64 = torch.zeros(4, dtype=torch.int64, device=self.device)
        self.block_tables_by_type = {
            int(key): torch.empty(
                self.batch_cap,
                self.block_cap,
                dtype=torch.int32,
                device=self.device,
            )
            for key in block_table_keys
        }

    def update(
        self,
        *,
        position_ids: torch.Tensor,
        req_id_per_token: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        block_tables_by_type: Mapping[int, torch.Tensor],
        seq_len_full: int,
        prefix_length: int,
    ) -> None:
        token_count, batch_size = self._validate_update(
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            cu_seqlens=cu_seqlens,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            block_tables_by_type=block_tables_by_type,
            seq_len_full=seq_len_full,
        )

        self._copy_prefix(self.position_ids, position_ids.to(torch.int64), token_count)
        self._copy_prefix(
            self.req_id_per_token, req_id_per_token.to(torch.int32), token_count
        )
        self._copy_prefix(self.cu_seqlens, cu_seqlens.to(torch.int64), batch_size + 1)
        self._copy_prefix(self.input_lengths, input_lengths.to(torch.int32), batch_size)
        self._copy_prefix(
            self.prefix_lengths, prefix_lengths.to(torch.int32), batch_size
        )
        self.scalar_i64[0] = token_count
        self.scalar_i64[1] = batch_size
        self.scalar_i64[2] = int(seq_len_full)
        self.scalar_i64[3] = int(prefix_length)

        for key, dst in self.block_tables_by_type.items():
            src = block_tables_by_type.get(key)
            if src is None:
                dst.zero_()
                continue
            rows = int(src.size(0))
            cols = int(src.size(1))
            dst.zero_()
            dst[:rows, :cols].copy_(
                src.to(device=self.device, dtype=torch.int32), non_blocking=True
            )

    def _validate_update(
        self,
        *,
        position_ids: torch.Tensor,
        req_id_per_token: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        block_tables_by_type: Mapping[int, torch.Tensor],
        seq_len_full: int,
    ) -> tuple[int, int]:
        token_count = int(position_ids.numel())
        batch_size = int(input_lengths.numel())
        if token_count > self.token_cap:
            raise ValueError(f"token_count={token_count} exceeds cap={self.token_cap}")
        if batch_size > self.batch_cap:
            raise ValueError(f"batch_size={batch_size} exceeds cap={self.batch_cap}")
        if int(cu_seqlens.numel()) != batch_size + 1:
            raise ValueError("cu_seqlens must have batch_size + 1 elements")
        if int(req_id_per_token.numel()) != token_count:
            raise ValueError("req_id_per_token must match position_ids length")
        if int(prefix_lengths.numel()) != batch_size:
            raise ValueError("prefix_lengths must match input_lengths length")
        # These host checks are intentionally outside graph capture/replay.  The
        # production replay path must update equivalent validity state before
        # entering capture-safe work.
        cu_seqlens_cpu = cu_seqlens.detach().cpu().to(torch.int64)
        input_lengths_cpu = input_lengths.detach().cpu().to(torch.int64)
        req_id_cpu = req_id_per_token.detach().cpu().to(torch.int64)
        if int(cu_seqlens_cpu[0].item()) != 0:
            raise ValueError("cu_seqlens[0] must be 0")
        diffs = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
        if bool((diffs < 0).any().item()):
            raise ValueError("cu_seqlens must be monotonic")
        if not torch.equal(diffs, input_lengths_cpu):
            raise ValueError("cu_seqlens diffs must match input_lengths")
        if int(cu_seqlens_cpu[-1].item()) != token_count:
            raise ValueError("cu_seqlens[-1] must match token_count")
        if int(input_lengths_cpu.sum().item()) != token_count:
            raise ValueError("input_lengths sum must match token_count")
        if token_count:
            if bool((req_id_cpu < 0).any().item()) or bool(
                (req_id_cpu >= batch_size).any().item()
            ):
                raise ValueError("req_id_per_token values must be in batch range")
            expected_req_ids = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.int64), input_lengths_cpu
            )
            if not torch.equal(req_id_cpu, expected_req_ids):
                raise ValueError("req_id_per_token must match cu_seqlens boundaries")
        if int(seq_len_full) < token_count:
            raise ValueError("seq_len_full must be >= token_count")
        extra_keys = set(int(key) for key in block_tables_by_type.keys()) - set(
            self.block_tables_by_type.keys()
        )
        if extra_keys:
            raise ValueError(f"unexpected block table keys: {sorted(extra_keys)}")
        for key, src in block_tables_by_type.items():
            if src.dim() != 2:
                raise ValueError(
                    f"block table {key} must be 2D, got {tuple(src.shape)}"
                )
            rows = int(src.size(0))
            cols = int(src.size(1))
            if rows != batch_size:
                raise ValueError(
                    f"block table {key} rows={rows} must match batch_size={batch_size}"
                )
            if rows > self.batch_cap or cols > self.block_cap:
                raise ValueError(
                    f"block table {key} shape={tuple(src.shape)} exceeds "
                    f"cap=({self.batch_cap}, {self.block_cap})"
                )
        return token_count, batch_size

    @staticmethod
    def _copy_prefix(dst: torch.Tensor, src: torch.Tensor, count: int) -> None:
        dst.zero_()
        if count:
            dst[:count].copy_(
                src.to(device=dst.device, dtype=dst.dtype), non_blocking=True
            )


class StaticMetadataBuffers:
    """Stable tensor buffers for nested prefill metadata leaves.

    Tensor leaves are stored in ``tensors[path]``. Integer/bool scalar leaves
    are stored as 0-D int64 tensors in ``scalar_i64[path]``. Subsequent updates
    require the same graph-relevant structure and tensor shapes; changes raise
    ``ValueError`` so the caller can recapture/reallocate instead of replaying
    with stale metadata.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        skip_path: Callable[[str, Any], bool] | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self._skip_path = skip_path
        self.tensors: dict[str, torch.Tensor] = {}
        self.scalar_i64: dict[str, torch.Tensor] = {}
        self._tensor_sig: dict[str, tuple[Any, ...]] | None = None
        self._scalar_sig: dict[str, int] | None = None

    def update(self, metadata: Any) -> None:
        tensor_leaves: dict[str, torch.Tensor] = {}
        scalar_leaves: dict[str, int] = {}
        self._collect(metadata, "", tensor_leaves, scalar_leaves)

        tensor_sig = {
            path: (
                tuple(t.shape),
                str(t.dtype),
                str(self.device or t.device),
            )
            for path, t in tensor_leaves.items()
        }
        scalar_sig = dict(scalar_leaves)
        if self._tensor_sig is None:
            self._allocate(tensor_leaves, scalar_leaves, tensor_sig, scalar_sig)
            return
        if tensor_sig != self._tensor_sig:
            raise ValueError("metadata tensor structure changed; recapture required")
        if scalar_sig != self._scalar_sig:
            raise ValueError("metadata scalar values changed; recapture required")
        for path, src in tensor_leaves.items():
            dst = self.tensors[path]
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype), non_blocking=True)
        for path, value in scalar_leaves.items():
            self.scalar_i64[path].fill_(int(value))

    def materialize(self, metadata: Any, *, workspace: Any = None) -> Any:
        """Rebuild ``metadata`` with mirrored tensor leaves.

        Skipped critical owner objects are intentionally returned live.  They are
        still reported by capture-surface analysis and keep the graph path from
        being considered fully static-bound until typed adapters exist.
        """

        if self._tensor_sig is None:
            raise ValueError("static metadata buffers have not been updated")
        return self._materialize(metadata, "", workspace=workspace)

    def _allocate(
        self,
        tensor_leaves: Mapping[str, torch.Tensor],
        scalar_leaves: Mapping[str, int],
        tensor_sig: Mapping[str, tuple[Any, ...]],
        scalar_sig: Mapping[str, int],
    ) -> None:
        self.tensors = {}
        for path, tensor in tensor_leaves.items():
            dst_device = self.device or tensor.device
            self.tensors[path] = tensor.detach().to(device=dst_device).clone()
        self.scalar_i64 = {}
        scalar_device = self.device
        if scalar_device is None and self.tensors:
            scalar_device = next(iter(self.tensors.values())).device
        scalar_device = scalar_device or torch.device("cpu")
        for path, value in scalar_leaves.items():
            self.scalar_i64[path] = torch.tensor(
                int(value), dtype=torch.int64, device=scalar_device
            )
        self._tensor_sig = dict(tensor_sig)
        self._scalar_sig = dict(scalar_sig)

    def _collect(
        self,
        value: Any,
        path: str,
        tensor_leaves: dict[str, torch.Tensor],
        scalar_leaves: dict[str, int],
    ) -> None:
        if self._skip_path is not None and self._skip_path(path, value):
            return
        if isinstance(value, torch.Tensor):
            tensor_leaves[path] = value
            return
        if isinstance(value, bool):
            scalar_leaves[path] = int(value)
            return
        if isinstance(value, int):
            scalar_leaves[path] = int(value)
            return
        if value is None or isinstance(value, (torch.device, torch.dtype)):
            return
        if isinstance(value, (float, str)):
            raise TypeError(
                f"StaticMetadataBuffers cannot collect replay-varying "
                f"{type(value).__name__} at {path}"
            )
        if _is_namedtuple_instance(value):
            for name in value._fields:
                self._collect(
                    getattr(value, name),
                    f"{path}.{name}" if path else name,
                    tensor_leaves,
                    scalar_leaves,
                )
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                self._collect(
                    getattr(value, field.name),
                    f"{path}.{field.name}" if path else field.name,
                    tensor_leaves,
                    scalar_leaves,
                )
            return
        if isinstance(value, Mapping):
            for key in sorted(value.keys()):
                self._collect(
                    value[key],
                    f"{path}.{key}" if path else str(key),
                    tensor_leaves,
                    scalar_leaves,
                )
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for idx, item in enumerate(value):
                self._collect(
                    item,
                    f"{path}.{idx}" if path else str(idx),
                    tensor_leaves,
                    scalar_leaves,
                )
            return
        raise TypeError(
            f"StaticMetadataBuffers cannot collect opaque object at {path}: "
            f"{type(value).__name__}"
        )

    def _materialize(self, value: Any, path: str, *, workspace: Any = None) -> Any:
        if path == "workspace" and workspace is not None:
            return workspace
        replaced, replacement = _materialize_static_safe_prefill_owner(path, value)
        if replaced:
            return replacement
        if self._skip_path is not None and self._skip_path(path, value):
            return value
        if isinstance(value, torch.Tensor):
            return self.tensors[path]
        if isinstance(value, (bool, int)):
            return value
        if value is None or isinstance(value, (torch.device, torch.dtype)):
            return value
        if isinstance(value, (float, str)):
            raise TypeError(
                f"StaticMetadataBuffers cannot materialize replay-varying "
                f"{type(value).__name__} at {path}"
            )
        if _is_namedtuple_instance(value):
            return type(value)(
                *(
                    self._materialize(
                        getattr(value, name),
                        f"{path}.{name}" if path else name,
                        workspace=workspace,
                    )
                    for name in value._fields
                )
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.replace(
                value,
                **{
                    field.name: self._materialize(
                        getattr(value, field.name),
                        f"{path}.{field.name}" if path else field.name,
                        workspace=workspace,
                    )
                    for field in dataclasses.fields(value)
                },
            )
        if isinstance(value, Mapping):
            return type(value)(
                (
                    key,
                    self._materialize(
                        value[key],
                        f"{path}.{key}" if path else str(key),
                        workspace=workspace,
                    ),
                )
                for key in value.keys()
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return type(value)(
                self._materialize(
                    item,
                    f"{path}.{idx}" if path else str(idx),
                    workspace=workspace,
                )
                for idx, item in enumerate(value)
            )
        raise TypeError(
            f"StaticMetadataBuffers cannot materialize opaque object at {path}: "
            f"{type(value).__name__}"
        )


def _skip_prefill_meta_owner(path: str, value: Any) -> bool:
    if path == "":
        return False
    # These are Python owner/helper objects. A production graph manager needs
    # typed handling for them instead of recursively mirroring object identity.
    if path == "workspace" or path.startswith("workspace."):
        if getattr(value, "_dsv4_static_graph_workspace", False) and hasattr(
            value, "_union"
        ):
            return False
        return True
    if value is not None and (
        path == "cp_ctx.cp_info" or path.startswith("cp_ctx.cp_info.")
    ):
        return True
    if value is not None and (
        path == "cmp_reader" or path.endswith(".cmp_reader") or ".cmp_reader." in path
    ):
        return True
    return False


def _materialize_static_safe_prefill_owner(path: str, value: Any) -> tuple[bool, Any]:
    """Return replacements for opaque owners that are safe in static metadata.

    ``CPContext.cp_info`` is used only to build ``CPContext`` before prefill meta
    exists.  The layer loop reads derived CPContext tensor/scalar fields instead,
    so static meta can drop the raw owner.

    ``LocalPoolReader`` is stateless and semantically equivalent to
    ``WorkspaceMeta.cmp_reader=None`` because the workspace path creates a
    LocalPoolReader when the field is absent.  CP-sharded readers are stateful and
    intentionally remain critical/live.
    """

    if path == "cp_ctx.cp_info":
        return True, None
    if path == "cmp_reader" or path.endswith(".cmp_reader"):
        if value is None:
            return True, None
        if type(value).__name__ == "LocalPoolReader":
            return True, None
    return False, value


class StaticPrefillMetaBuckets:
    """Static metadata leaves grouped by compress-ratio bucket."""

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else None
        self.by_ratio: dict[int, StaticMetadataBuffers] = {}
        self.ratios: tuple[int, ...] = ()

    def update(self, meta_by_ratio: Mapping[int, Any]) -> None:
        ratios = tuple(sorted(int(r) for r in meta_by_ratio.keys()))
        if self.ratios and ratios != self.ratios:
            raise ValueError(
                f"prefill meta ratio set changed: old={self.ratios} new={ratios}"
            )
        self.ratios = ratios
        for ratio in ratios:
            bucket = self.by_ratio.get(ratio)
            if bucket is None:
                bucket = StaticMetadataBuffers(
                    device=self.device,
                    skip_path=_skip_prefill_meta_owner,
                )
            self.by_ratio[ratio] = bucket
            bucket.update(meta_by_ratio[ratio])

    def materialize(
        self, meta_by_ratio: Mapping[int, Any], *, workspace: Any = None
    ) -> dict[int, Any]:
        ratios = tuple(sorted(int(r) for r in meta_by_ratio.keys()))
        if ratios != self.ratios:
            raise ValueError(
                f"prefill meta ratio set changed: old={self.ratios} new={ratios}"
            )
        return {
            ratio: self.by_ratio[ratio].materialize(
                meta_by_ratio[ratio], workspace=workspace
            )
            for ratio in ratios
        }


class StaticPrefillGraphState:
    """Graph-owned static buffers for one prefill graph key.

    This is still capture-agnostic.  It gives the future graph replay code one
    stable owner for request inputs, hidden tensors, block tables, and per-ratio
    metadata leaves.
    """

    def __init__(
        self,
        *,
        key: PrefillGraphKey,
        device: torch.device | str,
        hidden_shape_tail: Sequence[int],
        hidden_dtype: torch.dtype,
        block_cap: int,
        block_table_keys: Sequence[int],
    ) -> None:
        self.key = key
        self.device = torch.device(device)
        self.hidden_shape_tail = tuple(int(dim) for dim in hidden_shape_tail)
        self.hidden_dtype = hidden_dtype
        token_cap = int(key.local_token_bucket or key.token_bucket)
        self.input_ids = torch.empty(token_cap, dtype=torch.int64, device=self.device)
        hidden_shape = (token_cap,) + self.hidden_shape_tail
        self.hidden = torch.empty(hidden_shape, dtype=hidden_dtype, device=self.device)
        self.output_hidden = torch.empty(
            hidden_shape, dtype=hidden_dtype, device=self.device
        )
        self.request = StaticPrefillGraphInputs(
            token_cap=token_cap,
            batch_cap=key.batch_bucket,
            block_cap=block_cap,
            device=self.device,
            block_table_keys=block_table_keys,
        )
        self.meta = StaticPrefillMetaBuckets(device=self.device)
        self.workspace: Any = None
        self._workspace_config: tuple[tuple[str, Any], ...] | None = None
        self.valid = False
        self.pointer_stable = False
        self._last_pointer_signature: tuple[tuple[Any, ...], ...] | None = None
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.graph_capture_error: str | None = None
        self.graph_capture_count = 0
        self.graph_replay_count = 0
        self.graph_kv_cache: Any = None
        self.graph_kv_block_cap = 0
        self._graph_kv_signature: tuple[Any, ...] | None = None

    def update(
        self,
        *,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        req_id_per_token: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        block_tables_by_type: Mapping[int, torch.Tensor],
        seq_len_full: int,
        prefix_length: int,
        meta_by_ratio: Mapping[int, Any] | None = None,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.valid = False
        token_count = int(input_ids.numel())
        if token_count > self.key.token_bucket:
            raise ValueError(
                f"input_ids token_count={token_count} exceeds "
                f"bucket={self.key.token_bucket}"
            )
        if int(hidden.size(0)) != token_count:
            raise ValueError("hidden first dimension must match input_ids length")
        if int(position_ids.numel()) != token_count:
            raise ValueError("position_ids length must match input_ids length")
        if tuple(hidden.shape[1:]) != self.hidden_shape_tail:
            raise ValueError(
                f"hidden tail shape must be {self.hidden_shape_tail}, "
                f"got {tuple(hidden.shape[1:])}"
            )
        self._copy_prefix(self.input_ids, input_ids.to(torch.int64), token_count)
        self._copy_hidden_prefix(hidden, token_count)
        self.request.update(
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            cu_seqlens=cu_seqlens,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            block_tables_by_type=block_tables_by_type,
            seq_len_full=seq_len_full,
            prefix_length=prefix_length,
        )
        if meta_by_ratio is not None:
            self.meta.update(meta_by_ratio)
        if workspace_config is not None:
            self.ensure_workspace(workspace_config)
        signature = self.pointer_signature()
        self.pointer_stable = (
            self._last_pointer_signature is None
            or self._last_pointer_signature == signature
        )
        self._last_pointer_signature = signature
        self.valid = self.pointer_stable

    def reset_cuda_graph(self, reason: str | None = None) -> None:
        self.cuda_graph = None
        self.graph_capture_error = reason
        self.graph_capture_count = 0
        self.graph_replay_count = 0

    def mark_cuda_graph_captured(self, graph: torch.cuda.CUDAGraph) -> None:
        self.cuda_graph = graph
        self.graph_capture_error = None
        self.graph_capture_count += 1

    @property
    def cuda_graph_ready(self) -> bool:
        return self.valid and self.cuda_graph is not None

    def ensure_graph_kv_cache(
        self,
        live_kv_cache: Any,
        block_tables_by_type: Mapping[int, torch.Tensor] | None,
        *,
        min_block_cap: int = 0,
    ) -> Any:
        """Allocate graph-owned KV pools matching the live DSV4 KV structure."""

        block_cap = max(
            int(min_block_cap),
            self._max_block_id(block_tables_by_type or {}) + 1,
            1,
        )
        signature = self._graph_kv_structure_signature(live_kv_cache, block_cap)
        if self.graph_kv_cache is not None:
            if signature != self._graph_kv_signature:
                self.reset_cuda_graph("graph_kv_structure_changed")
                self.graph_kv_cache = None
            else:
                return self.graph_kv_cache

        from rtp_llm.ops.compute_ops import KVCache

        graph_kv = KVCache()
        for name in (
            "seq_size_per_block",
            "kernel_seq_size_per_block",
            "num_kv_heads",
            "head_dim",
            "use_mla",
            "kv_lora_rank",
            "rope_head_dim",
            "layer_group_types",
            "group_region_names",
            "group_seq_size_per_block",
            "layer_region_to_group_id",
        ):
            setattr(graph_kv, name, getattr(live_kv_cache, name))

        live_by_region = getattr(live_kv_cache, "kv_cache_base_by_layer_region", [])
        live_scale_by_region = getattr(
            live_kv_cache, "kv_scale_base_by_layer_region", []
        )
        layer_count = len(live_by_region)
        region_count = 8
        base_by_region: list[list[torch.Tensor]] = []
        scale_by_region: list[list[torch.Tensor]] = []
        for layer_idx in range(layer_count):
            base_row: list[torch.Tensor] = []
            scale_row: list[torch.Tensor] = []
            for region_idx in range(region_count):
                live_base = live_by_region[layer_idx][region_idx]
                if isinstance(live_base, torch.Tensor) and live_base.numel() > 0:
                    shape = (block_cap,) + tuple(int(dim) for dim in live_base.shape[1:])
                    base_row.append(
                        torch.empty(
                            shape,
                            dtype=live_base.dtype,
                            device=self.device,
                        )
                    )
                else:
                    base_row.append(torch.empty((0,), dtype=torch.uint8, device=self.device))
                live_scale = None
                if (
                    live_scale_by_region
                    and layer_idx < len(live_scale_by_region)
                    and region_idx < len(live_scale_by_region[layer_idx])
                ):
                    live_scale = live_scale_by_region[layer_idx][region_idx]
                if isinstance(live_scale, torch.Tensor) and live_scale.numel() > 0:
                    shape = (block_cap,) + tuple(int(dim) for dim in live_scale.shape[1:])
                    scale_row.append(
                        torch.empty(
                            shape,
                            dtype=live_scale.dtype,
                            device=self.device,
                        )
                    )
                else:
                    scale_row.append(torch.empty((0,), dtype=torch.float32, device=self.device))
            base_by_region.append(base_row)
            scale_by_region.append(scale_row)
        graph_kv.kv_cache_base_by_layer_region = base_by_region
        graph_kv.kv_scale_base_by_layer_region = scale_by_region
        graph_kv.kv_cache_base_by_layer_region_flat = [
            tensor for row in base_by_region for tensor in row
        ]
        graph_kv.kv_cache_base_by_layer = [
            row[0] for row in base_by_region
        ]
        graph_kv.kv_scale_base_by_layer = [
            row[0] for row in scale_by_region
        ]
        self.graph_kv_cache = graph_kv
        self.graph_kv_block_cap = block_cap
        self._graph_kv_signature = signature
        return graph_kv

    def graph_kv_fits(
        self, block_tables_by_type: Mapping[int, torch.Tensor] | None
    ) -> bool:
        return self._max_block_id(block_tables_by_type or {}) < int(self.graph_kv_block_cap)

    def copy_graph_kv_to_live(
        self,
        live_kv_cache: Any,
        block_tables_by_type: Mapping[int, torch.Tensor] | None,
    ) -> int:
        """Copy graph-owned KV/cache blocks back to the serving KV cache.

        Graph replay must not capture live serving KV pointers, but decode still
        has to read the prefill writes from the real cache.  This method copies
        only block-table-referenced rows for every layer/region.  It is called
        outside CUDA graph replay, so normal D2D copies are acceptable for the
        first correctness-safe graph path.

        Returns the number of tensor row-copy operations issued.
        """

        if self.graph_kv_cache is None:
            raise ValueError("graph KV cache has not been allocated")
        block_tables = block_tables_by_type or {}
        if not self.graph_kv_fits(block_tables):
            raise ValueError("graph KV cache block capacity exceeded")

        graph_by_region = getattr(
            self.graph_kv_cache, "kv_cache_base_by_layer_region", []
        )
        live_by_region = getattr(live_kv_cache, "kv_cache_base_by_layer_region", [])
        graph_scale_by_region = getattr(
            self.graph_kv_cache, "kv_scale_base_by_layer_region", []
        )
        live_scale_by_region = getattr(
            live_kv_cache, "kv_scale_base_by_layer_region", []
        )

        copied = 0
        for region_id, table in sorted(block_tables.items()):
            region_id = int(region_id)
            block_ids = self._valid_block_ids_for_copy(table)
            if block_ids.numel() == 0:
                continue
            region_base_copied = 0
            region_owner_count = 0
            for layer_idx, graph_row in enumerate(graph_by_region):
                if not self._layer_owns_region(live_kv_cache, layer_idx, region_id):
                    continue
                region_owner_count += 1
                live_row = live_by_region[layer_idx] if layer_idx < len(live_by_region) else []
                graph_tensor = graph_row[region_id] if region_id < len(graph_row) else None
                live_tensor = live_row[region_id] if region_id < len(live_row) else None
                base_copied = self._copy_kv_rows(
                    graph_tensor,
                    live_tensor,
                    block_ids,
                    name=f"layer{layer_idx}.region{region_id}.base",
                )
                copied += base_copied
                region_base_copied += base_copied
                graph_scale_row = (
                    graph_scale_by_region[layer_idx]
                    if layer_idx < len(graph_scale_by_region)
                    else []
                )
                live_scale_row = (
                    live_scale_by_region[layer_idx]
                    if layer_idx < len(live_scale_by_region)
                    else []
                )
                graph_scale = (
                    graph_scale_row[region_id]
                    if region_id < len(graph_scale_row)
                    else None
                )
                live_scale = (
                    live_scale_row[region_id]
                    if region_id < len(live_scale_row)
                    else None
                )
                copied += self._copy_kv_rows(
                    graph_scale,
                    live_scale,
                    block_ids,
                    name=f"layer{layer_idx}.region{region_id}.scale",
                    allow_both_empty=True,
                )
            if region_owner_count == 0:
                raise ValueError(
                    f"graph/live KV copy found no owning layers for region {region_id}"
                )
            if region_base_copied == 0:
                raise ValueError(
                    f"graph/live KV copy found no base tensors for region {region_id}"
                )
        return copied

    @staticmethod
    def _layer_owns_region(live_kv_cache: Any, layer_idx: int, region_id: int) -> bool:
        mapping = getattr(live_kv_cache, "layer_region_to_group_id", None)
        if not mapping:
            return True
        if layer_idx >= len(mapping):
            return False
        row = mapping[layer_idx]
        if region_id >= len(row):
            return False
        return int(row[region_id]) >= 0

    @staticmethod
    def _valid_block_ids_for_copy(table: torch.Tensor) -> torch.Tensor:
        if not isinstance(table, torch.Tensor) or table.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        block_ids = torch.unique(table.detach().reshape(-1).to(torch.long))
        return block_ids[block_ids > 0].contiguous()

    @staticmethod
    def _copy_kv_rows(
        graph_tensor: torch.Tensor,
        live_tensor: torch.Tensor,
        block_ids: torch.Tensor,
        *,
        name: str,
        allow_both_empty: bool = False,
    ) -> int:
        graph_is_tensor = isinstance(graph_tensor, torch.Tensor)
        live_is_tensor = isinstance(live_tensor, torch.Tensor)
        graph_empty = (not graph_is_tensor) or graph_tensor.numel() == 0
        live_empty = (not live_is_tensor) or live_tensor.numel() == 0
        if graph_empty and live_empty and allow_both_empty:
            return 0
        if graph_empty != live_empty:
            raise ValueError(f"graph/live KV copy missing tensor for {name}")
        if graph_empty and live_empty:
            return 0
        if not (
            isinstance(graph_tensor, torch.Tensor)
            and isinstance(live_tensor, torch.Tensor)
        ):
            raise ValueError(f"graph/live KV copy expected tensors for {name}")
        if graph_tensor.dim() < 1 or live_tensor.dim() < 1:
            raise ValueError(f"graph/live KV copy invalid rank for {name}")
        if tuple(graph_tensor.shape[1:]) != tuple(live_tensor.shape[1:]):
            raise ValueError(
                f"graph/live KV copy trailing shape mismatch for {name}: "
                f"{tuple(graph_tensor.shape[1:])} vs {tuple(live_tensor.shape[1:])}"
            )
        max_rows = min(int(graph_tensor.shape[0]), int(live_tensor.shape[0]))
        ids = block_ids.to(device=graph_tensor.device, dtype=torch.long)
        if ids.numel() > 0 and bool((ids >= max_rows).any().item()):
            raise ValueError(
                f"graph/live KV copy block id exceeds rows={max_rows}"
            )
        if ids.numel() == 0:
            return 0
        rows = graph_tensor.index_select(0, ids)
        live_tensor.index_copy_(
            0,
            ids.to(device=live_tensor.device),
            rows.to(device=live_tensor.device, dtype=live_tensor.dtype),
        )
        return 1

    @staticmethod
    def _max_block_id(block_tables_by_type: Mapping[int, torch.Tensor]) -> int:
        max_id = 0
        for table in block_tables_by_type.values():
            if isinstance(table, torch.Tensor) and table.numel() > 0:
                max_id = max(max_id, int(table.max().item()))
        return max_id

    def _graph_kv_structure_signature(
        self, live_kv_cache: Any, block_cap: int
    ) -> tuple[Any, ...]:
        live_by_region = getattr(live_kv_cache, "kv_cache_base_by_layer_region", [])
        live_scale_by_region = getattr(
            live_kv_cache, "kv_scale_base_by_layer_region", []
        )
        base_sig = tuple(
            tuple(
                (
                    tuple(t.shape[1:]) if isinstance(t, torch.Tensor) and t.dim() >= 1 else (),
                    str(t.dtype) if isinstance(t, torch.Tensor) else "",
                )
                for t in row
            )
            for row in live_by_region
        )
        scale_sig = tuple(
            tuple(
                (
                    tuple(t.shape[1:]) if isinstance(t, torch.Tensor) and t.dim() >= 1 else (),
                    str(t.dtype) if isinstance(t, torch.Tensor) else "",
                )
                for t in row
            )
            for row in live_scale_by_region
        )
        return (
            int(block_cap),
            int(getattr(live_kv_cache, "seq_size_per_block", 0)),
            int(getattr(live_kv_cache, "kernel_seq_size_per_block", 0)),
            tuple(int(x) for x in getattr(live_kv_cache, "group_region_names", []) or []),
            tuple(int(x) for x in getattr(live_kv_cache, "group_seq_size_per_block", []) or []),
            tuple(
                tuple(int(x) for x in row)
                for row in getattr(live_kv_cache, "layer_region_to_group_id", []) or []
            ),
            base_sig,
            scale_sig,
        )

    @staticmethod
    def _copy_prefix(dst: torch.Tensor, src: torch.Tensor, count: int) -> None:
        dst.zero_()
        if count:
            dst[:count].copy_(
                src.to(device=dst.device, dtype=dst.dtype), non_blocking=True
            )

    def _copy_hidden_prefix(self, hidden: torch.Tensor, token_count: int) -> None:
        self.hidden.zero_()
        if token_count:
            self.hidden[:token_count].copy_(
                hidden.to(device=self.device, dtype=self.hidden_dtype),
                non_blocking=True,
            )

    def ensure_workspace(self, config: Mapping[str, Any]) -> Any:
        normalized = tuple(sorted((str(key), value) for key, value in config.items()))
        if self.workspace is not None:
            if self._workspace_config != normalized:
                raise ValueError("prefill workspace config changed; recapture required")
            return self.workspace

        from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace

        ws = PrefillWorkspace(self.device, **dict(config))
        setattr(ws, "_dsv4_static_graph_workspace", True)
        self.workspace = ws
        self._workspace_config = normalized
        return ws

    def pointer_inventory(self) -> tuple[GraphTensorPointer, ...]:
        items: list[GraphTensorPointer] = []

        def add(name: str, tensor: torch.Tensor) -> None:
            items.append(
                GraphTensorPointer(
                    name=name,
                    data_ptr=int(tensor.data_ptr()),
                    shape=tuple(int(dim) for dim in tensor.shape),
                    stride=tuple(int(step) for step in tensor.stride()),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                )
            )

        add("input_ids", self.input_ids)
        add("hidden", self.hidden)
        add("output_hidden", self.output_hidden)
        add("request.position_ids", self.request.position_ids)
        add("request.req_id_per_token", self.request.req_id_per_token)
        add("request.cu_seqlens", self.request.cu_seqlens)
        add("request.input_lengths", self.request.input_lengths)
        add("request.prefix_lengths", self.request.prefix_lengths)
        add("request.scalar_i64", self.request.scalar_i64)
        for key, tensor in sorted(self.request.block_tables_by_type.items()):
            add(f"request.block_tables_by_type.{key}", tensor)
        for ratio, bucket in sorted(self.meta.by_ratio.items()):
            for path, tensor in sorted(bucket.tensors.items()):
                add(f"meta.{ratio}.tensor.{path}", tensor)
            for path, tensor in sorted(bucket.scalar_i64.items()):
                add(f"meta.{ratio}.scalar_i64.{path}", tensor)
        if self.workspace is not None:
            union = getattr(self.workspace, "_union", None)
            if isinstance(union, torch.Tensor):
                for ratio in self.meta.ratios:
                    add(f"meta.{int(ratio)}.tensor.workspace._union", union)
        return tuple(sorted(items, key=lambda item: item.name))

    def pointer_signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                item.name,
                item.data_ptr,
                item.shape,
                item.stride,
                item.dtype,
                item.device,
            )
            for item in self.pointer_inventory()
        )


class StaticPrefillGraphStateManager:
    """Own static graph states by graph key."""

    def __init__(self, *, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.states: dict[PrefillGraphKey, StaticPrefillGraphState] = {}

    def get_or_create(
        self,
        key: PrefillGraphKey,
        *,
        hidden_shape_tail: Sequence[int],
        hidden_dtype: torch.dtype,
        block_cap: int,
        block_table_keys: Sequence[int],
    ) -> StaticPrefillGraphState:
        state = self.states.get(key)
        if state is not None:
            if state.hidden_shape_tail != tuple(int(dim) for dim in hidden_shape_tail):
                raise ValueError("hidden_shape_tail changed for existing graph key")
            if state.hidden_dtype != hidden_dtype:
                raise ValueError("hidden_dtype changed for existing graph key")
            if state.request.block_cap != int(block_cap):
                raise ValueError("block_cap changed for existing graph key")
            old_keys = set(state.request.block_tables_by_type.keys())
            new_keys = set(int(key) for key in block_table_keys)
            if old_keys != new_keys:
                raise ValueError("block_table_keys changed for existing graph key")
            return state
        state = StaticPrefillGraphState(
            key=key,
            device=self.device,
            hidden_shape_tail=hidden_shape_tail,
            hidden_dtype=hidden_dtype,
            block_cap=block_cap,
            block_table_keys=block_table_keys,
        )
        self.states[key] = state
        return state


def exact_static_prefill_layer_loop_args(
    state: StaticPrefillGraphState,
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    block_tables_by_type: Mapping[int, torch.Tensor] | None,
) -> StaticPrefillLayerLoopArgs:
    """Return graph-owned layer-loop args when exact-shape binding is safe.

    This is a conservative bridge toward production graph replay.  It refuses
    padded/capacity views for now because several attention helpers inspect
    block-table shapes.  Future padded graph execution needs typed adapters for
    those helpers rather than silently passing larger tables.
    """

    if not state.valid:
        raise ValueError("static prefill graph state is not valid")
    token_count = int(input_ids.numel())
    if int(hidden.size(0)) != token_count:
        raise ValueError("hidden first dimension must match input_ids length")
    if int(position_ids.numel()) != token_count:
        raise ValueError("position_ids length must match input_ids length")
    if token_count != int(state.input_ids.numel()):
        raise ValueError(
            f"exact static bind requires token_count={token_count} to match "
            f"static token cap={int(state.input_ids.numel())}"
        )
    if tuple(hidden.shape) != tuple(state.hidden.shape):
        raise ValueError(
            f"exact static bind requires hidden shape {tuple(hidden.shape)} "
            f"to match static shape {tuple(state.hidden.shape)}"
        )
    if cu_seqlens is None:
        raise ValueError("exact static bind requires cu_seqlens")
    if tuple(cu_seqlens.shape) != tuple(state.request.cu_seqlens.shape):
        raise ValueError(
            f"exact static bind requires cu_seqlens shape {tuple(cu_seqlens.shape)} "
            f"to match static shape {tuple(state.request.cu_seqlens.shape)}"
        )

    static_tables: dict[int, torch.Tensor] | None = None
    live_tables = block_tables_by_type or {}
    if live_tables:
        static_tables = {}
        live_keys = set(int(key) for key in live_tables.keys())
        static_keys = set(state.request.block_tables_by_type.keys())
        if live_keys != static_keys:
            raise ValueError(
                f"exact static bind requires block table keys {sorted(live_keys)} "
                f"to match static keys {sorted(static_keys)}"
            )
        for key, live_table in live_tables.items():
            key = int(key)
            static_table = state.request.block_tables_by_type[key]
            if tuple(live_table.shape) != tuple(static_table.shape):
                raise ValueError(
                    f"exact static bind requires block table {key} shape "
                    f"{tuple(live_table.shape)} to match static shape "
                    f"{tuple(static_table.shape)}"
                )
            static_tables[key] = static_table

    return StaticPrefillLayerLoopArgs(
        input_ids=state.input_ids,
        hidden=state.hidden,
        position_ids=state.request.position_ids,
        cu_seqlens=state.request.cu_seqlens,
        block_tables_by_type=static_tables,
    )


def try_update_static_prefill_graph_state(
    v4: Any,
    decision: PrefillGraphDecision,
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    req_id_per_token: torch.Tensor,
    cu_seqlens: torch.Tensor,
    input_lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    block_tables_by_type: Mapping[int, torch.Tensor] | None,
    seq_len_full: int,
    prefix_length: int,
    meta_by_ratio: Mapping[int, Any] | None,
    block_cap: int,
    workspace_config: Mapping[str, Any] | None = None,
) -> PrefillGraphDecision:
    """Update graph-owned static state, failing closed on any mismatch."""

    if not decision.enabled or decision.key is None:
        setattr(v4, "_last_prefill_graph_state", None)
        setattr(v4, "_last_prefill_graph_state_error", decision.reason)
        return decision
    block_tables = block_tables_by_type or {}
    block_table_keys = tuple(sorted(int(key) for key in block_tables.keys()))
    state_key = with_static_state_invariants(
        decision.key,
        local_token_bucket=int(input_ids.numel()),
        hidden_shape_tail=tuple(int(dim) for dim in hidden.shape[1:]),
        hidden_dtype=hidden.dtype,
        block_cap=int(block_cap),
        block_table_keys=block_table_keys,
    )
    manager = getattr(v4, "_dsv4_static_prefill_graph_state_manager", None)
    if manager is None:
        manager = StaticPrefillGraphStateManager(device=hidden.device)
        setattr(v4, "_dsv4_static_prefill_graph_state_manager", manager)
    state = None
    try:
        state = manager.get_or_create(
            state_key,
            hidden_shape_tail=tuple(int(dim) for dim in hidden.shape[1:]),
            hidden_dtype=hidden.dtype,
            block_cap=int(block_cap),
            block_table_keys=block_table_keys,
        )
        state.update(
            input_ids=input_ids,
            hidden=hidden,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            cu_seqlens=cu_seqlens,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            block_tables_by_type=block_tables,
            seq_len_full=seq_len_full,
            prefix_length=prefix_length,
            meta_by_ratio=meta_by_ratio,
            workspace_config=workspace_config,
        )
    except Exception as exc:
        if state is not None:
            state.reset_cuda_graph("static_update_failed")
        setattr(v4, "_last_prefill_graph_state", None)
        setattr(v4, "_last_prefill_graph_state_error", str(exc))
        return PrefillGraphDecision(False, state_key, "static_update_failed")
    if not state.valid or not state.pointer_stable:
        state.reset_cuda_graph("pointer_drift")
        setattr(v4, "_last_prefill_graph_state", state)
        setattr(v4, "_last_prefill_graph_state_error", "pointer_drift")
        return PrefillGraphDecision(False, state_key, "pointer_drift")
    setattr(v4, "_last_prefill_graph_state", state)
    setattr(v4, "_last_prefill_graph_state_error", None)
    return PrefillGraphDecision(True, state_key, decision.reason)


def update_static_prefill_meta_buckets(
    v4: Any, meta_by_ratio: Mapping[int, Any]
) -> None:
    manager = getattr(v4, "_dsv4_static_prefill_meta_buckets", None)
    if manager is None:
        device = None
        for meta in meta_by_ratio.values():
            device = getattr(meta, "device", None)
            if device is not None:
                break
        manager = StaticPrefillMetaBuckets(device=device)
        setattr(v4, "_dsv4_static_prefill_meta_buckets", manager)
    manager.update(meta_by_ratio)


def analyze_prefill_capture_surface(
    state: StaticPrefillGraphState,
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    req_id_per_token: torch.Tensor,
    cu_seqlens: torch.Tensor,
    input_lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    block_tables_by_type: Mapping[int, torch.Tensor] | None,
    meta_by_ratio: Mapping[int, Any] | None,
) -> GraphCaptureSurfaceReport:
    """Compare the live layer-loop capture surface against graph-owned buffers.

    This is a dry-run diagnostic.  It does not make replay decisions by itself;
    it tells the future capture/replay code whether the objects currently passed
    into the layer loop are already graph-owned static tensors.
    """

    skipped_critical: list[str] = []
    live = _capture_surface_inventory(
        input_ids=input_ids,
        hidden=hidden,
        position_ids=position_ids,
        req_id_per_token=req_id_per_token,
        cu_seqlens=cu_seqlens,
        input_lengths=input_lengths,
        prefix_lengths=prefix_lengths,
        block_tables_by_type=block_tables_by_type or {},
        meta_by_ratio=meta_by_ratio or {},
        skipped_critical=skipped_critical,
    )
    static_by_name = {item.name: item for item in state.pointer_inventory()}
    live_not_static: list[str] = []
    missing_static: list[str] = []
    static_bound_count = 0
    for item in live:
        static_item = static_by_name.get(item.name)
        if static_item is None:
            missing_static.append(item.name)
            continue
        if (
            static_item.data_ptr == item.data_ptr
            and static_item.shape == item.shape
            and static_item.stride == item.stride
            and static_item.dtype == item.dtype
            and static_item.device == item.device
        ):
            static_bound_count += 1
        else:
            live_not_static.append(item.name)
    return GraphCaptureSurfaceReport(
        static_bound=not live_not_static
        and not missing_static
        and not skipped_critical,
        live_tensor_count=len(live),
        static_bound_count=static_bound_count,
        live_not_static=tuple(sorted(live_not_static)),
        missing_static=tuple(sorted(missing_static)),
        skipped_critical=tuple(sorted(set(skipped_critical))),
    )


def _capture_surface_inventory(
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    req_id_per_token: torch.Tensor,
    cu_seqlens: torch.Tensor,
    input_lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    block_tables_by_type: Mapping[int, torch.Tensor],
    meta_by_ratio: Mapping[int, Any],
    skipped_critical: list[str],
) -> tuple[GraphTensorPointer, ...]:
    items: list[GraphTensorPointer] = []

    def add(name: str, tensor: torch.Tensor) -> None:
        items.append(
            GraphTensorPointer(
                name=name,
                data_ptr=int(tensor.data_ptr()),
                shape=tuple(int(dim) for dim in tensor.shape),
                stride=tuple(int(step) for step in tensor.stride()),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
            )
        )

    add("input_ids", input_ids)
    add("hidden", hidden)
    add("request.position_ids", position_ids)
    add("request.req_id_per_token", req_id_per_token)
    add("request.cu_seqlens", cu_seqlens)
    add("request.input_lengths", input_lengths)
    add("request.prefix_lengths", prefix_lengths)
    for key, tensor in sorted(block_tables_by_type.items()):
        add(f"request.block_tables_by_type.{int(key)}", tensor)
    for ratio, meta in sorted(meta_by_ratio.items()):
        _collect_capture_surface_meta_tensors(
            meta,
            path=f"meta.{int(ratio)}.tensor",
            add=add,
            skipped_critical=skipped_critical,
        )
    return tuple(sorted(items, key=lambda item: item.name))


def _collect_capture_surface_meta_tensors(
    value: Any,
    *,
    path: str,
    add: Callable[[str, torch.Tensor], None],
    skipped_critical: list[str],
) -> None:
    rel_path = path.split(".tensor.", 1)[1] if ".tensor." in path else ""
    if _skip_prefill_meta_owner(rel_path, value):
        skipped_critical.append(path)
        return
    if getattr(value, "_dsv4_static_graph_workspace", False) and hasattr(
        value, "_union"
    ):
        union = getattr(value, "_union")
        if isinstance(union, torch.Tensor):
            add(f"{path}._union", union)
            return
    if isinstance(value, torch.Tensor):
        add(path, value)
        return
    if isinstance(value, _SCALAR_TYPES):
        return
    if _is_namedtuple_instance(value):
        for name in value._fields:
            _collect_capture_surface_meta_tensors(
                getattr(value, name),
                path=f"{path}.{name}",
                add=add,
                skipped_critical=skipped_critical,
            )
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _collect_capture_surface_meta_tensors(
                getattr(value, field.name),
                path=f"{path}.{field.name}",
                add=add,
                skipped_critical=skipped_critical,
            )
        return
    if isinstance(value, Mapping):
        for key in sorted(value.keys()):
            _collect_capture_surface_meta_tensors(
                value[key],
                path=f"{path}.{key}",
                add=add,
                skipped_critical=skipped_critical,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            _collect_capture_surface_meta_tensors(
                item,
                path=f"{path}.{idx}",
                add=add,
                skipped_critical=skipped_critical,
            )
        return
    skipped_critical.append(path)
