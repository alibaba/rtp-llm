"""RTP-facing client for KVCM's exact-size embedding object service."""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, TypeVar

from . import lookup
from ._config import (
    DEFAULT_MAX_OBJECT_BYTES,
    RtpKvMetaObjectClientConfig,
    _resolve_vipserver_addresses,
)
from .lookup import KvMetaLookupError, KvMetaLookupTimeout

_MISSING_WHEEL_MESSAGE = (
    "KVCM EMB storage requires the kvcm_py_client wheel with KVMeta object support"
)
_INCOMPATIBLE_WHEEL_MESSAGE = (
    "KVCM EMB storage requires KVMeta object API version 2 or 3"
)
_SUPPORTED_KVCM_OBJECT_API_VERSIONS = (2, 3)
_ENDPOINT_REFRESH_SECONDS = 1.0
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
        or KV_META_OBJECT_API_VERSION not in _SUPPORTED_KVCM_OBJECT_API_VERSIONS
    ):
        raise RuntimeError(_INCOMPATIBLE_WHEEL_MESSAGE)
    try:
        from kv_cache_manager.client import KvMetaObjectClientError
        from kv_cache_manager.client.pybind import kvcm_py_client
    except ImportError as error:
        raise RuntimeError(_INCOMPATIBLE_WHEEL_MESSAGE) from error
    metadata_version = getattr(kvcm_py_client, "KV_META_OBJECT_METADATA_VERSION", None)
    if (
        type(metadata_version) is not int
        or metadata_version != 1
        or not callable(getattr(KvMetaObjectClient, "get_object_meta", None))
        or not callable(getattr(KvMetaObjectClient, "load_from_meta", None))
    ):
        raise RuntimeError(
            "KVCM EMB storage requires native object metadata API version 1"
        )
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
        self._metadata_lock = threading.Lock()
        self._metadata_addresses = frozenset(config.addresses)
        self._closed = False
        self._metadata_stop = threading.Event()
        if config.vipserver_domain:
            # Keep the SDK's last usable endpoints available during discovery.
            threading.Thread(
                target=self._watch_metadata_addresses,
                args=(
                    weakref.ref(self),
                    self._metadata_stop,
                    _ENDPOINT_REFRESH_SECONDS,
                ),
                name="emb-metadata-address-refresh",
                daemon=True,
            ).start()

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
        if isinstance(error, KvMetaLookupError) and error.reason == "discovery":
            return True  # Refresh failed before submitting save/load to the SDK.
        return (
            isinstance(error, self._client_error_type)
            and getattr(error, "operation", None) in ("save", "load")
            and getattr(error, "data_transfer_may_be_inflight", True) is False
        )

    def _discover_addresses(self):
        try:
            return _resolve_vipserver_addresses(
                self._config.vipserver_domain, lookup._cached_vipserver_resolver
            )
        except Exception:
            raise KvMetaLookupError("KVMeta service discovery unavailable") from None

    @staticmethod
    def _watch_metadata_addresses(client_ref, stop, interval):
        last_error = None
        while not stop.wait(interval):
            client = client_ref()
            if client is None:
                return
            try:
                client._refresh_metadata_addresses()
                last_error = None
            except KvMetaLookupError as error:
                if str(error) != last_error:
                    last_error = str(error)
                    logging.error("EMB KVCM endpoint refresh failed: %s", error)
            finally:
                # Do not hold the client alive while the thread is sleeping.
                del client

    def _refresh_metadata_addresses(self):
        addresses = self._discover_addresses()
        snapshot = frozenset(addresses)
        with self._metadata_lock:
            if self._closed or snapshot == self._metadata_addresses:
                return
            try:
                self._client.update_addresses(addresses)
            except Exception:
                raise KvMetaLookupError("KVMeta endpoint update failed") from None
            self._metadata_addresses = snapshot

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("KVCM client is closed")

    def save(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: str | None = None,
    ) -> None:
        """Save contiguous CPU/CUDA tensors under exact object keys."""

        self._ensure_open()
        self._client.save(keys, tensors, trace_id=trace_id)

    def save_one(self, key: str, tensor: Any, *, trace_id: str | None = None) -> None:
        """Save one exact-size tensor without single-item sequence boilerplate."""

        self.save((key,), (tensor,), trace_id=trace_id)

    def native_object_api(self):
        self._ensure_open()
        export = getattr(type(self._client), "native_object_api", None)
        return self._client.native_object_api() if callable(export) else None

    def native_save_error(self, code, may_be_inflight):
        return self.native_transfer_error("save", code, may_be_inflight)

    def native_transfer_error(self, operation, code, may_be_inflight):
        from kv_cache_manager.client.pybind.kvcm_py_client import ClientErrorCode

        try:
            native_code = ClientErrorCode(code)
        except ValueError:
            native_code = code
        return self._client_error_type(
            operation,
            native_code,
            unknown_outcome=operation == "save"
            and (code == 2 or type(native_code) is int),
            data_transfer_may_be_inflight=may_be_inflight,
        )

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

        self._ensure_open()
        self._client.load(keys, tensors, trace_id=trace_id)

    def load_one(
        self, key: str, tensor: _TensorT, *, trace_id: str | None = None
    ) -> _TensorT:
        """Fill and return one exact-size tensor; discard it on any exception."""

        self.load((key,), (tensor,), trace_id=trace_id)
        return tensor

    def load_from_meta(self, meta, buffer: _TensorT, *, trace_id=None) -> _TensorT:
        """Read directly from the native metadata for this request."""
        self._client.load_from_meta(meta, buffer, trace_id=trace_id)
        return buffer

    def buffer_safe_after_failure(self, error: Exception) -> bool:
        """Require the wheel's explicit completion contract, as in prefill."""
        if isinstance(error, KvMetaLookupError) and error.reason == "discovery":
            return True  # The native transfer has not been submitted.
        if not isinstance(error, self._client_error_type):
            return False
        from kv_cache_manager.client.pybind import kvcm_py_client

        codes = kvcm_py_client.ClientErrorCode
        # The versioned flag describes local DMA completion, independently of
        # whether a metadata mutation's outcome is known. Legacy/untyped errors
        # cannot provide this proof and remain quarantined.
        return (
            getattr(kvcm_py_client, "KV_META_OBJECT_TRANSFER_STATE_VERSION", 0) == 1
            and getattr(error, "data_transfer_may_be_inflight", None) is False
            and isinstance(getattr(error, "code", None), codes)
            and hasattr(codes, "ER_SDK_ERROR_BUFFER_SAFE")
            and hasattr(codes, "ER_SDK_TIMEOUT_BUFFER_SAFE")
        )

    @property
    def supports_native_object_load(self):
        return getattr(self._client, "supports_native_object_load", False) is True

    def get_object_meta(
        self, key: str, *, trace_id=None, timeout_ms=None, native_load=False
    ):
        """Fetch metadata once, optionally with its matching native load capability.

        A miss returns None, or (None, None) in native load mode.
        """
        self._ensure_open()
        if timeout_ms is None:
            timeout_ms = self._config.call_timeout_ms
        if timeout_ms <= 0:
            raise KvMetaLookupTimeout("KVMeta lookup deadline exceeded")
        try:
            if native_load:
                return self._client.get_object_meta_with_native_load(
                    key, timeout_ms, trace_id=trace_id
                )
            return self._client.get_object_meta(key, timeout_ms, trace_id=trace_id)
        except self._client_error_type as error:
            if getattr(error.code, "name", None) == "ER_SDK_TIMEOUT":
                raise KvMetaLookupTimeout(
                    "KVMeta Get returned ER_SDK_TIMEOUT"
                ) from error
            raise

    def object_size(self, key: str, *, trace_id=None, timeout_ms=None):
        """Return exact size or None without exposing a storage location."""
        meta = self.get_object_meta(key, trace_id=trace_id, timeout_ms=timeout_ms)
        return meta.size_bytes if meta is not None else None

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

        meta = self.get_object_meta(key, trace_id=trace_id, timeout_ms=timeout_ms)
        if meta is None:
            return None
        buffer = torch.empty(meta.size_bytes, dtype=torch.uint8)
        self.load_from_meta(meta, buffer, trace_id=trace_id)
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

        self._ensure_open()
        self._client.remove(keys, trace_id=trace_id)

    def remove_one(self, key: str, *, trace_id: str | None = None) -> None:
        """Remove one exact-size tensor object."""

        self.remove((key,), trace_id=trace_id)

    def close(self) -> None:
        """Release local KVCM resources; safe to call more than once."""

        self._metadata_stop.set()
        with self._metadata_lock:
            self._closed = True
            self._client.close()

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
        self._metadata_stop.set()
        with self._metadata_lock:
            self._closed = True
            self._client.__exit__(exc_type, exc_value, traceback)
