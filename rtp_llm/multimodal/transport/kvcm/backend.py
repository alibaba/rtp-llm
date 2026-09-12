import logging
import sys
import threading
import time
import uuid
from itertools import islice
from typing import Dict, Iterator, List, Protocol, Sequence, Tuple

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    TensorDataTypePB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.multimodal.transport.base import MMOutputResult, MMTransportBackend

TRANSPORT_KVCM = "kvcm"
_REMOVE_RETRY_SECONDS = 1.0
_MAX_GC_WAIT_SECONDS = 60.0
# Must not exceed the shared gRPC control client's pending-release capacity.
_MAX_OBJECTS_PER_RECEIPT = 1024
_MAX_LOGICAL_VALUES_PER_RECEIPT = 16384
_MAX_TENSOR_DIMENSIONS = 16
_MAX_KVCM_KEY_BYTES = 512
_PROTO_INT32_MAX = (1 << 31) - 1
_PROTO_INT64_MAX = (1 << 63) - 1
_KVCM_MAX_OBJECT_BYTES = 1024 * 1024 * 1024
_DTYPE_TO_PROTO = {
    torch.float32: TensorDataTypePB.RDMA_TENSOR_FLOAT32,
    torch.int32: TensorDataTypePB.RDMA_TENSOR_INT32,
    torch.float16: TensorDataTypePB.RDMA_TENSOR_FLOAT16,
    torch.bfloat16: TensorDataTypePB.RDMA_TENSOR_BFLOAT16,
}


class _KvcmWriter(Protocol):
    def save(self, keys: Sequence[str], tensors: Sequence[torch.Tensor]) -> None: ...

    def remove(self, keys: Sequence[str]) -> None: ...


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _strict_positive_int(value, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"KVCM {label} must be an integer")
    if not 0 < value <= maximum:
        raise ValueError(f"KVCM {label} must be in [1, {maximum}]")
    return value


def _bounded_logical_values(values, label: str) -> list:
    try:
        snapshot = list(islice(iter(values), _MAX_LOGICAL_VALUES_PER_RECEIPT + 1))
    except TypeError as error:
        raise ValueError(f"KVCM {label} must be iterable") from error
    if len(snapshot) > _MAX_LOGICAL_VALUES_PER_RECEIPT:
        raise RuntimeError(
            f"KVCM output contains more than "
            f"{_MAX_LOGICAL_VALUES_PER_RECEIPT} {label} logical values"
        )
    return snapshot


def _synchronize_cuda_tensors(tensors: Sequence[torch.Tensor]) -> None:
    # KVCM receives raw device pointers and its SDKs do not participate in
    # PyTorch stream dependency tracking. Make all producer work visible
    # before the first metadata mutation/data-plane read of those pointers.
    devices = {tensor.device for tensor in tensors if tensor.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)


def _chunk_count(nbytes: int, rows: int, max_object_bytes: int) -> int:
    if nbytes <= 0 or rows <= 0 or nbytes % rows != 0:
        raise ValueError("KVCM tensor rows do not have a stable byte width")
    if nbytes <= max_object_bytes:
        return 1
    row_bytes = nbytes // rows
    if row_bytes <= 0 or row_bytes > max_object_bytes:
        raise ValueError(
            f"one KVCM tensor row ({row_bytes} bytes) exceeds max_object_bytes "
            f"({max_object_bytes})"
        )
    rows_per_chunk = max_object_bytes // row_bytes
    return (rows + rows_per_chunk - 1) // rows_per_chunk


def _concatenated_layout(
    tensors: Sequence[torch.Tensor], label: str
) -> Tuple[int, int]:
    reference = tensors[0]
    trailing_shape = tuple(reference.shape[1:])
    promoted_dtype = reference.dtype
    total_elements = 0
    total_rows = 0
    for index, tensor in enumerate(tensors):
        if tensor.dim() != reference.dim() or tuple(tensor.shape[1:]) != trailing_shape:
            raise ValueError(
                f"KVCM {label}[{index}] shape is incompatible with {label}[0]"
            )
        if tensor.device != reference.device:
            raise ValueError(
                f"KVCM {label} tensors must be on the same device before concatenation"
            )
        promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)
        total_elements += tensor.numel()
        total_rows += int(tensor.shape[0])
    if promoted_dtype not in _DTYPE_TO_PROTO:
        raise ValueError(
            f"KVCM concatenated {label} dtype {promoted_dtype} is unsupported"
        )
    element_bytes = torch.empty((), dtype=promoted_dtype).element_size()
    return total_elements * element_bytes, total_rows


def _chunk_tensor(
    tensor: torch.Tensor, max_object_bytes: int
) -> Iterator[torch.Tensor]:
    tensor = tensor.contiguous()
    if tensor.dim() < 1 or tensor.dim() > _MAX_TENSOR_DIMENSIONS:
        raise ValueError(
            f"KVCM tensors must have between 1 and {_MAX_TENSOR_DIMENSIONS} dimensions"
        )
    if any(int(dimension) <= 0 for dimension in tensor.shape):
        raise ValueError("KVCM tensor dimensions must all be positive")
    nbytes = _tensor_nbytes(tensor)
    if nbytes <= 0:
        raise ValueError("KVCM cannot store an empty tensor")
    if tensor.dtype not in _DTYPE_TO_PROTO:
        raise ValueError(f"KVCM does not support tensor dtype {tensor.dtype}")
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError(f"KVCM does not support tensor device {tensor.device.type}")
    rows = int(tensor.shape[0])
    chunk_count = _chunk_count(nbytes, rows, max_object_bytes)
    if chunk_count == 1:
        yield tensor
        return
    row_bytes = nbytes // rows
    rows_per_chunk = max_object_bytes // row_bytes
    for start in range(0, rows, rows_per_chunk):
        yield tensor.narrow(0, start, min(rows_per_chunk, rows - start)).contiguous()


class KvcmOutputBackend(MMTransportBackend):
    """Store ViT outputs as isolated exact-size KVMeta objects."""

    name = TRANSPORT_KVCM

    def __init__(self, writer: _KvcmWriter, kvcm_config, *, _owns_writer=False):
        if not callable(getattr(writer, "save", None)) or not callable(
            getattr(writer, "remove", None)
        ):
            raise TypeError("KVCM writer must provide callable save and remove methods")
        self._writer = writer
        self._owns_writer = _owns_writer
        self._max_object_bytes = _strict_positive_int(
            kvcm_config.max_object_bytes,
            "max_object_bytes",
            _KVCM_MAX_OBJECT_BYTES,
        )
        self._max_receipt_bytes = _strict_positive_int(
            kvcm_config.max_receipt_bytes,
            "max_receipt_bytes",
            sys.maxsize,
        )
        gc_timeout_ms = _strict_positive_int(
            kvcm_config.object_gc_timeout_ms,
            "object_gc_timeout_ms",
            _PROTO_INT64_MAX,
        )
        if self._max_receipt_bytes < self._max_object_bytes:
            raise ValueError("KVCM max_receipt_bytes is smaller than max_object_bytes")
        self._gc_timeout_seconds = gc_timeout_ms / 1000.0
        self._pending: Dict[str, float] = {}
        self._condition = threading.Condition()
        self._active_operations = 0
        self._closing = False
        self._closed = False
        self._gc_thread = threading.Thread(
            target=self._gc_loop, name="mm-kvcm-object-gc", daemon=True
        )
        self._gc_thread.start()

    @classmethod
    def create(cls, kvcm_config) -> "KvcmOutputBackend":
        try:
            from kv_cache_manager.client import (
                KvMetaObjectClient,
                KvMetaObjectClientConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "KVCM EMB storage requires the kvcm_py_client wheel with "
                "KVMeta object support"
            ) from error

        try:
            writer = KvMetaObjectClient(
                KvMetaObjectClientConfig(
                    addresses=tuple(kvcm_config.addresses),
                    instance_id=kvcm_config.instance_id,
                    instance_group=kvcm_config.instance_group,
                    user_data=kvcm_config.user_data,
                    transfer_client_config=kvcm_config.transfer_client_config,
                    call_timeout_ms=kvcm_config.call_timeout_ms,
                    write_timeout_seconds=kvcm_config.write_timeout_seconds,
                    max_object_bytes=kvcm_config.max_object_bytes,
                )
            )
        except ImportError as error:
            raise RuntimeError(
                "KVCM EMB storage requires the kvcm_py_client wheel with "
                "KVMeta object support"
            ) from error
        try:
            return cls(writer, kvcm_config, _owns_writer=True)
        except Exception:
            try:
                writer.close()
            except Exception as close_error:  # noqa: BLE001 - preserve root cause
                logging.warning(
                    "[VIT] KVCM writer initialization rollback failed "
                    "(exception_type=%s)",
                    type(close_error).__name__,
                )
            raise

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        self._begin_operation()
        try:
            return self._transfer(request, res)
        finally:
            self._end_operation()

    def _transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        if not request.support_kvcm:
            raise RuntimeError(
                "KVCM transport was selected but the client did not advertise KVCM support"
            )
        tensors, split_size = self._prepare_tensors(res)
        objects: List[Tuple[torch.Tensor, int, int]] = []
        for tensor, role, logical_index in tensors:
            for chunk in _chunk_tensor(tensor, self._max_object_bytes):
                if len(objects) >= _MAX_OBJECTS_PER_RECEIPT:
                    raise RuntimeError(
                        f"KVCM output exceeds receipt object limit "
                        f"{_MAX_OBJECTS_PER_RECEIPT}"
                    )
                objects.append((chunk, role, logical_index))
        if not objects:
            raise RuntimeError("KVCM transport produced no exact-size objects")
        total_bytes = sum(_tensor_nbytes(tensor) for tensor, _, _ in objects)
        if total_bytes > self._max_receipt_bytes:
            raise RuntimeError(
                f"KVCM output size {total_bytes} exceeds max_receipt_bytes "
                f"{self._max_receipt_bytes}"
            )
        keys = [f"rtp-mm-{uuid.uuid4().hex}" for _ in objects]
        object_tensors = [tensor for tensor, _, _ in objects]
        _synchronize_cuda_tensors(object_tensors)
        try:
            self._writer.save(keys, object_tensors)
        except Exception:
            # save() may have committed one or more service-sized prefixes
            # before a later prefix failed.  KVMeta V1 has no object TTL, so a
            # failed rollback must enter the same retry queue as normal lease
            # cleanup instead of being abandoned after one best-effort call.
            self._remove_or_retry(keys, "save rollback")
            raise

        try:
            receipt = MultimodalOutputPB(split_size=split_size)
            role_bytes: Dict[int, int] = {}
            for key, (tensor, role, logical_index) in zip(keys, objects):
                nbytes = _tensor_nbytes(tensor)
                obj = receipt.output_kvcm_objects.add(
                    key=key,
                    value_size=nbytes,
                    role=role,
                    logical_index=logical_index,
                )
                obj.tensor.shape.extend(int(dimension) for dimension in tensor.shape)
                obj.tensor.data_type = _DTYPE_TO_PROTO[tensor.dtype]
                obj.tensor.offset = 0
                obj.tensor.nbytes = nbytes
                role_bytes[role] = role_bytes.get(role, 0) + nbytes
            result = MMOutputResult(
                receipt=receipt,
                transport=TRANSPORT_KVCM,
                payload_embedding_bytes=role_bytes.get(MMRdmaSlotPB.EMBEDDING, 0),
                payload_pos_bytes=role_bytes.get(MMRdmaSlotPB.POS_ID, 0),
                payload_extra_bytes=role_bytes.get(MMRdmaSlotPB.EXTRA_INPUT, 0),
            )
            self._track(keys)
        except Exception:
            # The values are already committed at this point.  Keep retrying
            # removal if receipt construction/publication cannot complete;
            # otherwise no consumer exists that can release these objects.
            self._remove_or_retry(keys, "receipt rollback")
            raise

        return result

    def _prepare_tensors(
        self, res: MMEmbeddingRes
    ) -> Tuple[List[Tuple[torch.Tensor, int, int]], List[int]]:
        # Snapshot the result lists once. Besides making validation errors
        # deterministic, this keeps receipt metadata tied to the exact tensors
        # handed to storage if a caller incorrectly mutates MMEmbeddingRes from
        # another thread.
        embeddings = _bounded_logical_values(res.embeddings, "embedding")
        positions = (
            []
            if res.position_ids is None
            else _bounded_logical_values(res.position_ids, "position_ids")
        )
        extras = (
            []
            if res.extra_input is None
            else _bounded_logical_values(res.extra_input, "extra_input")
        )
        if not embeddings:
            raise RuntimeError("KVCM transport received no multimodal embeddings")
        split_size: List[int] = []
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, torch.Tensor):
                raise ValueError("KVCM embeddings must be torch tensors")
            if embedding.dim() < 1 or embedding.dim() > _MAX_TENSOR_DIMENSIONS:
                raise ValueError(
                    f"KVCM embeddings must have between 1 and "
                    f"{_MAX_TENSOR_DIMENSIONS} dimensions"
                )
            if any(int(dimension) <= 0 for dimension in embedding.shape):
                raise ValueError("KVCM embedding dimensions must all be positive")
            rows = int(embedding.shape[0])
            if rows > _PROTO_INT32_MAX:
                raise ValueError(f"KVCM embeddings[{index}] rows exceed int32")
            if embedding.dtype not in _DTYPE_TO_PROTO:
                raise ValueError(
                    f"KVCM does not support embedding dtype {embedding.dtype}"
                )
            if embedding.device.type not in ("cpu", "cuda"):
                raise ValueError(
                    f"KVCM does not support embedding device {embedding.device.type}"
                )
            split_size.append(rows)

        if positions:
            if len(positions) != len(embeddings):
                raise ValueError("KVCM position_ids count must match embedding count")
            for index, (position, rows) in enumerate(zip(positions, split_size)):
                if not isinstance(position, torch.Tensor):
                    raise ValueError("KVCM position_ids must be torch tensors")
                if position.dim() < 1 or position.dim() > _MAX_TENSOR_DIMENSIONS:
                    raise ValueError(
                        f"KVCM position_ids must have between 1 and "
                        f"{_MAX_TENSOR_DIMENSIONS} dimensions"
                    )
                if any(int(dimension) <= 0 for dimension in position.shape):
                    raise ValueError(
                        "KVCM position_ids dimensions must all be positive"
                    )
                if int(position.shape[0]) != rows:
                    raise ValueError(
                        f"KVCM position_ids[{index}] rows do not match "
                        f"embeddings[{index}]"
                    )
                if position.dtype not in _DTYPE_TO_PROTO:
                    raise ValueError(
                        f"KVCM does not support position_ids dtype {position.dtype}"
                    )
                if position.device.type not in ("cpu", "cuda"):
                    raise ValueError(
                        f"KVCM does not support position_ids device {position.device.type}"
                    )

        if extras:
            if len(extras) != len(embeddings):
                raise ValueError("KVCM extra_input count must match embedding count")
            for extra in extras:
                if not isinstance(extra, torch.Tensor):
                    raise ValueError("KVCM extra_input values must be torch tensors")
                if extra.dim() != 1 or extra.shape[0] <= 0:
                    raise ValueError(
                        "KVCM extra_input must contain one non-empty flat tensor per image"
                    )
                if extra.dtype not in _DTYPE_TO_PROTO:
                    raise ValueError(
                        f"KVCM does not support extra_input dtype {extra.dtype}"
                    )
                if extra.device.type not in ("cpu", "cuda"):
                    raise ValueError(
                        f"KVCM does not support extra_input device {extra.device.type}"
                    )

        # Reject impossible receipts before torch.concat() or chunk materialization
        # can duplicate a caller-provided payload. This keeps invalid requests
        # bounded by the configured receipt/object limits and guarantees that no
        # storage mutation has started.
        embedding_bytes, embedding_rows = _concatenated_layout(embeddings, "embeddings")
        layouts = [(embedding_bytes, embedding_rows)]
        if positions:
            layouts.append(_concatenated_layout(positions, "position_ids"))
        layouts.extend((_tensor_nbytes(extra), int(extra.shape[0])) for extra in extras)
        total_bytes = sum(nbytes for nbytes, _ in layouts)
        if total_bytes > self._max_receipt_bytes:
            raise RuntimeError(
                f"KVCM output size {total_bytes} exceeds max_receipt_bytes "
                f"{self._max_receipt_bytes}"
            )
        object_count = sum(
            _chunk_count(nbytes, rows, self._max_object_bytes)
            for nbytes, rows in layouts
        )
        if object_count > _MAX_OBJECTS_PER_RECEIPT:
            raise RuntimeError(
                f"KVCM output requires {object_count} objects, exceeding receipt limit "
                f"{_MAX_OBJECTS_PER_RECEIPT}"
            )

        embedding = torch.concat(embeddings).contiguous()
        output: List[Tuple[torch.Tensor, int, int]] = [
            (embedding, MMRdmaSlotPB.EMBEDDING, 0)
        ]
        if positions:
            combined_positions = (
                torch.concat(positions).to(device=embedding.device).contiguous()
            )
            if combined_positions.shape[0] != sum(split_size):
                raise ValueError("KVCM position rows do not match embedding rows")
            output.append((combined_positions, MMRdmaSlotPB.POS_ID, 0))
        if extras:
            for index, extra in enumerate(extras):
                output.append(
                    (
                        extra.to(device=embedding.device).contiguous(),
                        MMRdmaSlotPB.EXTRA_INPUT,
                        index,
                    )
                )
        return output, split_size

    def _track(self, keys: Sequence[str]) -> None:
        deadline = time.monotonic() + self._gc_timeout_seconds
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("KVCM output backend is closed")
            for key in keys:
                self._pending[key] = deadline
            self._condition.notify_all()

    def release(self, handles: List[str]) -> None:
        if not self._try_begin_operation():
            return
        try:
            # A bare string is iterable but is not a sequence of handles.
            # Reject it explicitly so a malformed caller cannot accidentally
            # release one-character keys now or after a key-format change.
            if isinstance(handles, (str, bytes)):
                logging.warning("[VIT] ignoring scalar KVCM release handle")
                return
            try:
                snapshot = list(islice(iter(handles), _MAX_OBJECTS_PER_RECEIPT + 1))
            except TypeError:
                logging.warning("[VIT] ignoring non-iterable KVCM release handles")
                return
            if len(snapshot) > _MAX_OBJECTS_PER_RECEIPT:
                logging.warning(
                    "[VIT] ignoring KVCM release handles beyond the %d-object limit; "
                    "object GC will reclaim them",
                    _MAX_OBJECTS_PER_RECEIPT,
                )
                snapshot.pop()
            requested = []
            seen = set()
            for handle in snapshot:
                if not isinstance(handle, str) or not handle:
                    continue
                try:
                    valid_size = len(handle.encode("utf-8")) <= _MAX_KVCM_KEY_BYTES
                except UnicodeEncodeError:
                    valid_size = False
                if valid_size and handle not in seen:
                    seen.add(handle)
                    requested.append(handle)
            with self._condition:
                owned = [handle for handle in requested if handle in self._pending]
                for handle in owned:
                    self._pending.pop(handle, None)
            if owned:
                self._remove_or_retry(owned, "release")
        finally:
            self._end_operation()

    def _try_begin_operation(self) -> bool:
        with self._condition:
            if self._closing or self._closed:
                return False
            self._active_operations += 1
            return True

    def _begin_operation(self) -> None:
        if not self._try_begin_operation():
            raise RuntimeError("KVCM output backend is closed")

    def _end_operation(self) -> None:
        with self._condition:
            if self._active_operations <= 0:
                raise RuntimeError("KVCM output backend operation accounting underflow")
            self._active_operations -= 1
            if self._active_operations == 0:
                self._condition.notify_all()

    def _remove_or_retry(self, keys: Sequence[str], reason: str) -> None:
        if not keys:
            return
        with self._condition:
            if self._closed:
                return
        try:
            self._writer.remove(list(keys))
        except Exception as error:  # noqa: BLE001 - retrying GC is the backstop
            # Native errors may contain endpoints, keys or provider details.
            # Keep failure logs actionable without copying those values.
            logging.warning(
                "[VIT] KVCM %s failed; scheduling retry "
                "(object_count=%d, exception_type=%s)",
                reason,
                len(keys),
                type(error).__name__,
            )
            with self._condition:
                if not self._closed:
                    # GC lifetime and retry cadence are independent. Tying
                    # retries to a very small (misconfigured) object TTL can
                    # otherwise spin and flood logs while storage is down.
                    retry_at = time.monotonic() + _REMOVE_RETRY_SECONDS
                    for key in keys:
                        self._pending[key] = retry_at
                    self._condition.notify_all()

    def _best_effort_remove(self, keys: Sequence[str], reason: str) -> None:
        if not keys:
            return
        try:
            self._writer.remove(list(keys))
        except Exception as error:  # noqa: BLE001 - shutdown must complete
            logging.warning(
                "[VIT] KVCM %s failed; objects require later namespace cleanup "
                "(object_count=%d, exception_type=%s)",
                reason,
                len(keys),
                type(error).__name__,
            )

    def _best_effort_close_writer(self) -> None:
        if not self._owns_writer:
            return
        try:
            close = getattr(self._writer, "close", None)
            if callable(close):
                close()
        except Exception as error:  # noqa: BLE001 - shutdown must complete
            logging.warning(
                "[VIT] KVCM writer close failed (exception_type=%s)",
                type(error).__name__,
            )

    def _gc_loop(self) -> None:
        while True:
            with self._condition:
                while not self._closing and not self._closed and not self._pending:
                    self._condition.wait()
                if self._closing or self._closed:
                    return
                now = time.monotonic()
                deadline = min(self._pending.values())
                if deadline > now:
                    # threading.Condition delegates to a platform timed wait,
                    # whose accepted timeout is much smaller than int64 ms on
                    # some systems. Wake periodically for huge valid leases.
                    self._condition.wait(
                        timeout=min(deadline - now, _MAX_GC_WAIT_SECONDS)
                    )
                    continue
                expired = [
                    key
                    for key, expires_at in self._pending.items()
                    if expires_at <= now
                ]
                for key in expired:
                    self._pending.pop(key, None)
                # close() waits for this attempt before taking its final
                # snapshot. A failed removal can therefore be requeued before
                # the worker exits, rather than disappearing in a pop/close
                # race.
                self._active_operations += 1
            try:
                self._remove_or_retry(expired, "expiry cleanup")
            finally:
                self._end_operation()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._condition.wait()
                return
            self._closing = True
            self._condition.notify_all()
            while self._active_operations:
                self._condition.wait()
            remaining = list(self._pending)
            self._pending.clear()
        try:
            self._gc_thread.join()
            if remaining:
                self._best_effort_remove(remaining, "shutdown cleanup")
        finally:
            try:
                self._best_effort_close_writer()
            finally:
                with self._condition:
                    self._closed = True
                    self._closing = False
                    self._condition.notify_all()
