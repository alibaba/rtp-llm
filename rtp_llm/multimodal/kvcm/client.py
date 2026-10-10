"""RTP-facing client for KVCM's exact-size embedding object service."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, TypeVar

from ._config import RtpKvMetaObjectClientConfig

_MISSING_WHEEL_MESSAGE = (
    "KVCM EMB storage requires the kvcm_py_client wheel with KVMeta object support"
)
_TensorT = TypeVar("_TensorT")


def _load_kvcm_client_types() -> tuple[type[Any], type[Any]]:
    """Import the optional KVCM package only when a client is constructed."""

    try:
        from kv_cache_manager.client import KvMetaObjectClient, KvMetaObjectClientConfig
    except ImportError as error:
        raise RuntimeError(_MISSING_WHEEL_MESSAGE) from error
    return KvMetaObjectClient, KvMetaObjectClientConfig


class RtpKvMetaObjectClient:
    """Small RTP facade over KVCM's generic exact-object Python client.

    Construct without arguments when the process already has RTP's established
    ``RECO_*`` variables, or use :meth:`from_kv_cache_config` when RTP has
    already parsed those variables and CLI overrides. Construction validates
    the complete config and registers the derived ``kve_`` instance. One
    instance is thread-safe and should be reused for the worker lifetime.
    """

    def __init__(self) -> None:
        self._initialize(RtpKvMetaObjectClientConfig.from_env())

    @classmethod
    def _from_config(cls, config: RtpKvMetaObjectClientConfig) -> RtpKvMetaObjectClient:
        instance = object.__new__(cls)
        instance._initialize(config)
        return instance

    def _initialize(self, config: RtpKvMetaObjectClientConfig) -> None:
        if not isinstance(config, RtpKvMetaObjectClientConfig):
            raise TypeError("config must be a RtpKvMetaObjectClientConfig")
        client_type, config_type = _load_kvcm_client_types()
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
        client = client_type(generic_config)

        self._config = config
        self._client = client
        self._lookup = None
        self._lookup_lock = threading.Lock()
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        max_object_bytes: int = 1024 * 1024 * 1024,
    ) -> RtpKvMetaObjectClient:
        """Create from ``environ``, or the current process's ``RECO_*`` values."""

        config = RtpKvMetaObjectClientConfig.from_env(
            environ=environ,
            max_object_bytes=max_object_bytes,
        )
        return cls._from_config(config)

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: Any,
        *,
        max_object_bytes: int = 1024 * 1024 * 1024,
    ) -> RtpKvMetaObjectClient:
        """Create and register from RTP's parsed ``KVCacheConfig`` object."""

        config = RtpKvMetaObjectClientConfig.from_kv_cache_config(
            kv_cache_config,
            max_object_bytes=max_object_bytes,
        )
        return cls._from_config(config)

    @property
    def instance_id(self) -> str:
        return self._config.instance_id

    @property
    def instance_group(self) -> str:
        return self._config.instance_group

    def save(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: str | None = None,
    ) -> None:
        """Save contiguous CPU/CUDA tensors under exact object keys."""

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
        """Load objects into caller-allocated contiguous CPU/CUDA tensors."""

        self._client.load(keys, tensors, trace_id=trace_id)

    def load_one(
        self, key: str, tensor: _TensorT, *, trace_id: str | None = None
    ) -> _TensorT:
        """Fill and return one caller-allocated tensor of the exact size."""

        self.load((key,), (tensor,), trace_id=trace_id)
        return tensor

    def object_size(self, key: str, *, trace_id=None, timeout_ms=None):
        """Return exact object size or None, using the existing KVMeta Get RPC."""
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

    def remove(self, keys: Sequence[str], *, trace_id: str | None = None) -> None:
        """Remove exact object keys."""

        self._client.remove(keys, trace_id=trace_id)

    def remove_one(self, key: str, *, trace_id: str | None = None) -> None:
        """Remove one exact-size tensor object."""

        self.remove((key,), trace_id=trace_id)

    def close(self) -> None:
        """Release local KVCM resources; safe to call more than once."""

        try:
            self._client.close()
        finally:
            with self._lookup_lock:
                self._closed = True
                lookup = self._lookup
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
        try:
            self._client.__exit__(exc_type, exc_value, traceback)
        finally:
            with self._lookup_lock:
                self._closed = True
                lookup = self._lookup
            if lookup is not None:
                lookup.close()
