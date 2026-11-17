"""RTP-facing client for KVCM's exact-size embedding object service."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, TypeVar

from ._config import DEFAULT_MAX_OBJECT_BYTES, RtpKvMetaObjectClientConfig

_MISSING_WHEEL_MESSAGE = (
    "KVCM EMB storage requires the kvcm_py_client wheel with KVMeta object support"
)
_INCOMPATIBLE_WHEEL_MESSAGE = "KVCM EMB storage requires KVMeta object API version 2"
_REQUIRED_KVCM_OBJECT_API_VERSION = 2
_TensorT = TypeVar("_TensorT")


def _load_kvcm_client_types(
    *, require_address_update=False
) -> tuple[type[Any], type[Any], type[BaseException], Any]:
    """Import the optional KVCM package only when a client is constructed."""

    try:
        from kv_cache_manager.client import (
            KV_META_OBJECT_API_VERSION,
            KvMetaObjectClient,
            KvMetaObjectClientConfig,
        )
    except ImportError as error:
        raise RuntimeError(_MISSING_WHEEL_MESSAGE) from error
    # API v2 is the first capability level that includes the production
    # lifecycle/Close contract used by this facade. Fail before registration
    # rather than silently running an older, source-compatible but unsafe wheel.
    if (
        type(KV_META_OBJECT_API_VERSION) is not int
        or KV_META_OBJECT_API_VERSION != _REQUIRED_KVCM_OBJECT_API_VERSION
    ):
        raise RuntimeError(_INCOMPATIBLE_WHEEL_MESSAGE)
    try:
        from kv_cache_manager.client import KvMetaObjectClientError
        from kv_cache_manager.client.pybind import kvcm_py_client
    except ImportError as error:
        raise RuntimeError(_INCOMPATIBLE_WHEEL_MESSAGE) from error
    if require_address_update and (
        type(getattr(kvcm_py_client, "KV_META_OBJECT_ADDRESS_UPDATE_VERSION", None))
        is not int
        or kvcm_py_client.KV_META_OBJECT_ADDRESS_UPDATE_VERSION != 1
        or not callable(getattr(KvMetaObjectClient, "update_addresses", None))
    ):
        raise RuntimeError(
            "KVCM VipServer requires metadata address update API version 1"
        )
    try:
        not_found_code = kvcm_py_client.ClientErrorCode.ER_SERVICE_NOT_FOUND
    except (AttributeError, TypeError):
        raise RuntimeError(_INCOMPATIBLE_WHEEL_MESSAGE) from None
    if not isinstance(KvMetaObjectClientError, type) or not issubclass(
        KvMetaObjectClientError, BaseException
    ):
        raise RuntimeError(_INCOMPATIBLE_WHEEL_MESSAGE)
    return (
        KvMetaObjectClient,
        KvMetaObjectClientConfig,
        KvMetaObjectClientError,
        not_found_code,
    )


class RtpKvMetaObjectClient:
    """Small RTP facade over KVCM's generic exact-object Python client.

    Construct without arguments when the process already has RTP's established
    ``RECO_*`` variables, or use :meth:`from_kv_cache_config` when RTP has
    already parsed those variables and CLI overrides. Construction validates
    the complete config and registers the derived ``kve_`` instance. One
    instance is thread-safe and should be reused for the worker lifetime.
    """

    def __init__(self, *, gpu_tensors: Sequence[Any] = ()) -> None:
        self._initialize(
            RtpKvMetaObjectClientConfig.from_env(), gpu_tensors=gpu_tensors
        )

    @classmethod
    def _from_config(
        cls, config: RtpKvMetaObjectClientConfig, *, gpu_tensors=()
    ) -> RtpKvMetaObjectClient:
        instance = object.__new__(cls)
        instance._initialize(config, gpu_tensors=gpu_tensors)
        return instance

    def _initialize(
        self, config: RtpKvMetaObjectClientConfig, *, gpu_tensors=()
    ) -> None:
        if not isinstance(config, RtpKvMetaObjectClientConfig):
            raise TypeError("config must be a RtpKvMetaObjectClientConfig")
        if isinstance(gpu_tensors, (str, bytes)) or not isinstance(
            gpu_tensors, Sequence
        ):
            raise TypeError("gpu_tensors must be a sequence")
        if len(gpu_tensors) > 64:
            raise ValueError("at most 64 GPU registration tensors are allowed")
        gpu_tensors = tuple(gpu_tensors)
        (
            client_type,
            config_type,
            client_error_type,
            not_found_code,
        ) = _load_kvcm_client_types(
            require_address_update=bool(config.vipserver_domain)
        )
        generic_config = config_type(
            addresses=config.addresses,
            instance_id=config.instance_id,
            instance_group=config.instance_group,
            user_data=config.user_data,
            transfer_client_config=config.transfer_client_config,
            call_timeout_ms=config.call_timeout_ms,
            write_timeout_seconds=config.write_timeout_seconds,
            max_object_bytes=config.max_object_bytes,
        )
        client = (
            client_type(generic_config, gpu_tensors=gpu_tensors)
            if gpu_tensors
            else client_type(generic_config)
        )

        self._config = config
        self._client = client
        self._client_error_type = client_error_type
        self._not_found_code = not_found_code
        self._lookup = None
        self._lookup_lock = threading.Lock()
        self._metadata_lock = threading.Lock()
        self._metadata_addresses = tuple(config.addresses)
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        call_timeout_ms: int | None = None,
        put_timeout_ms: int | None = None,
        get_timeout_ms: int | None = None,
        gpu_tensors: Sequence[Any] = (),
    ) -> RtpKvMetaObjectClient:
        """Create from ``environ``, or the current process's ``RECO_*`` values."""

        config = RtpKvMetaObjectClientConfig.from_env(
            environ=environ,
            max_object_bytes=max_object_bytes,
            call_timeout_ms=call_timeout_ms,
            put_timeout_ms=put_timeout_ms,
            get_timeout_ms=get_timeout_ms,
        )
        return cls._from_config(config, gpu_tensors=gpu_tensors)

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: Any,
        *,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        call_timeout_ms: int | None = None,
        put_timeout_ms: int | None = None,
        get_timeout_ms: int | None = None,
        gpu_tensors: Sequence[Any] = (),
    ) -> RtpKvMetaObjectClient:
        """Create and register from RTP's parsed ``KVCacheConfig`` object."""

        config = RtpKvMetaObjectClientConfig.from_kv_cache_config(
            kv_cache_config,
            max_object_bytes=max_object_bytes,
            call_timeout_ms=call_timeout_ms,
            put_timeout_ms=put_timeout_ms,
            get_timeout_ms=get_timeout_ms,
        )
        return cls._from_config(config, gpu_tensors=gpu_tensors)

    @property
    def instance_id(self) -> str:
        return self._config.instance_id

    @property
    def instance_group(self) -> str:
        return self._config.instance_group

    def is_transfer_quiescent(self, error: BaseException) -> bool:
        """True when this failed call is known not to leave pending DMA."""
        from .lookup import KvMetaLookupError

        if isinstance(error, KvMetaLookupError) and error.reason == "discovery":
            return True  # Refresh failed before submitting save/load to the SDK.
        return (
            isinstance(error, self._client_error_type)
            and getattr(error, "operation", None) in ("save", "load")
            and getattr(error, "data_transfer_may_be_inflight", True) is False
        )

    def _refresh_metadata_addresses(self):
        # Serialize discovery publication with Close, not with payload IO. The
        # SDK atomically swaps metadata channels and retains the same consumer.
        with self._metadata_lock:
            if self._closed:
                raise RuntimeError("KVCM client is closed")
            if not self._config.vipserver_domain:
                return
            from .lookup import (
                KvMetaLookupError,
                _cached_vipserver_resolver,
                _resolve_vipserver_addresses,
            )

            try:
                addresses = _resolve_vipserver_addresses(
                    self._config.vipserver_domain, _cached_vipserver_resolver
                )
                if set(addresses) != set(self._metadata_addresses):
                    self._client.update_addresses(addresses)
                    self._metadata_addresses = tuple(addresses)
            except Exception:
                # No mutation or payload IO was submitted, and the SDK retains
                # its old channel snapshot on failure. Never replay a mutation.
                error = KvMetaLookupError(
                    "KVMeta service discovery unavailable", reason="discovery"
                )
                error.data_transfer_may_be_inflight = False
                raise error from None

    def save(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: str | None = None,
    ) -> None:
        """Save contiguous CPU/CUDA tensors under exact object keys."""

        self._refresh_metadata_addresses()
        self._client.save(keys, tensors, trace_id=trace_id)

    def save_one(self, key: str, tensor: Any, *, trace_id: str | None = None) -> None:
        """Save one exact-size tensor without single-item sequence boilerplate."""

        self.save((key,), (tensor,), trace_id=trace_id)

    def load(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: str | None = None,
    ) -> None:
        """Load objects into caller-allocated contiguous CPU/CUDA tensors.

        This API does not pin the remote allocation across a concurrent
        Remove or reclaim. Callers must coordinate object lifetime. On any
        exception, every destination may already have been written and must
        be discarded rather than consumed by inference.
        """

        self._refresh_metadata_addresses()
        self._client.load(keys, tensors, trace_id=trace_id)

    def load_one(
        self, key: str, tensor: _TensorT, *, trace_id: str | None = None
    ) -> _TensorT:
        """Fill and return one exact-size tensor; discard it on any exception."""

        self.load((key,), (tensor,), trace_id=trace_id)
        return tensor

    def buffer_safe_after_failure(self, error: Exception) -> bool:
        """Require the wheel's explicit completion contract, as in prefill."""
        from .lookup import KvMetaLookupError

        if isinstance(error, KvMetaLookupError) and error.reason == "discovery":
            return True  # The native transfer has not been submitted.
        if not isinstance(error, self._client_error_type):
            return False
        from kv_cache_manager.client.pybind import kvcm_py_client

        codes = kvcm_py_client.ClientErrorCode
        if (
            getattr(kvcm_py_client, "KV_META_OBJECT_TRANSFER_STATE_VERSION", 0) != 1
            or getattr(error, "data_transfer_may_be_inflight", None) is not False
            or not isinstance(getattr(error, "code", None), codes)
            or not hasattr(codes, "ER_SDK_ERROR_BUFFER_SAFE")
            or not hasattr(codes, "ER_SDK_TIMEOUT_BUFFER_SAFE")
        ):
            return False
        if error.code == getattr(codes, "ER_SERVICE_WRITE_IN_PROGRESS", None):
            # A known conflict rejects this save before the SDK starts I/O.
            return (
                getattr(error, "operation", None) == "save"
                and getattr(error, "unknown_outcome", None) is False
            )
        # Keep legacy numeric values, generic SDK timeouts and unknown codes
        # conservative. The three validation codes are pre-I/O rejections.
        return error.code in (
            codes.ER_SDK_ERROR_BUFFER_SAFE,
            codes.ER_SDK_TIMEOUT_BUFFER_SAFE,
            getattr(codes, "ER_INVALID_PARAMS", None),
            getattr(codes, "ER_INVALID_LOCAL_BUFFERS", None),
            getattr(codes, "ER_INVALID_SDKWRAPPER_CONFIG", None),
        )

    def object_size(self, key: str, *, trace_id=None, timeout_ms=None):
        """Return exact size or None from KVMeta Get; never expose a storage URI."""
        from .lookup import KvMetaLookup

        with self._lookup_lock:
            if self._closed:
                raise RuntimeError("KVCM client is closed")
            if self._lookup is None:
                self._lookup = KvMetaLookup(self._config)
            lookup = self._lookup
        return lookup.size(key, trace_id=trace_id, timeout_ms=timeout_ms)

    def save_object(self, key: str, value: Any, *, devices=None, trace_id=None):
        """Save one complete tensor tree; synchronize GPU producer streams first."""
        from .tensor_object import pack_object, plan_object

        plan = plan_object(
            value, devices=devices, max_bytes=self._config.max_object_bytes
        )
        self.save_one(key, pack_object(plan), trace_id=trace_id)

    def load_object(
        self,
        key: str,
        *,
        trace_id=None,
        cuda_device=None,
        restore_devices=True,
        timeout_ms=None,
    ):
        """Restore a complete object; return None on a metadata cache miss."""
        import torch

        from .tensor_object import unpack_object

        size = self.object_size(key, trace_id=trace_id, timeout_ms=timeout_ms)
        if size is None:
            return None
        buffer = torch.empty(size, dtype=torch.uint8)
        self.load_one(key, buffer, trace_id=trace_id)
        return unpack_object(
            buffer,
            cuda_device=cuda_device,
            restore_devices=restore_devices,
            max_bytes=self._config.max_object_bytes,
        )

    def try_load(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: str | None = None,
    ) -> bool:
        """Load objects and return ``False`` for a normal cache miss.

        Only KVCM's exact ``NOT_FOUND`` result is converted to ``False``.
        Timeouts, size mismatches, service failures, and malformed client
        errors remain exceptions.  A ``False`` result does not make the
        destination contents valid: overwrite or discard every tensor before
        use.
        """

        try:
            self.load(keys, tensors, trace_id=trace_id)
        except self._client_error_type as error:
            code = getattr(error, "code", None)
            if (
                type(code) is type(self._not_found_code)
                and code == self._not_found_code
            ):
                return False
            raise
        return True

    def try_load_one(
        self, key: str, tensor: Any, *, trace_id: str | None = None
    ) -> bool:
        """Load one object and return ``False`` for a normal cache miss."""

        return self.try_load((key,), (tensor,), trace_id=trace_id)

    def remove(self, keys: Sequence[str], *, trace_id: str | None = None) -> None:
        """Remove exact object keys."""

        self._refresh_metadata_addresses()
        self._client.remove(keys, trace_id=trace_id)

    def remove_one(self, key: str, *, trace_id: str | None = None) -> None:
        """Remove one exact-size tensor object."""

        self.remove((key,), trace_id=trace_id)

    def close(self) -> None:
        """Release local KVCM resources; safe to call more than once."""

        with self._metadata_lock:
            try:
                self._client.close()
            finally:
                self._close_lookup()

    def _close_lookup(self):
        with self._lookup_lock:
            self._closed = True
            lookup, self._lookup = self._lookup, None
        if lookup is not None:
            lookup.close()

    def __enter__(self) -> RtpKvMetaObjectClient:  # noqa: PYI034 - Python 3.10.
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Cleanup must never turn a body failure into apparent success, even if
        # a future optional client accidentally returns a truthy value here.
        with self._metadata_lock:
            try:
                self._client.__exit__(exc_type, exc_value, traceback)
            finally:
                self._close_lookup()
