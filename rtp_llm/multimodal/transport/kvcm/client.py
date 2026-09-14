"""RTP-specific facade over KVCM's generic exact-object Python client."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Type

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import MMKvcmConfig

_MISSING_WHEEL_MESSAGE = (
    "KVCM EMB storage requires the kvcm_py_client wheel with KVMeta object support"
)


def _load_kvcm_client_types() -> Tuple[Type[Any], Type[Any]]:
    """Load the optional KVCM package only after RTP selects KVMeta transport."""

    try:
        from kv_cache_manager.client import (
            KvMetaObjectClient,
            KvMetaObjectClientConfig,
        )
    except ImportError as error:
        raise RuntimeError(_MISSING_WHEEL_MESSAGE) from error
    return KvMetaObjectClient, KvMetaObjectClientConfig


class RtpKvMetaObjectClient:
    """Small RTP-facing adapter that owns one generic KVCM object client.

    ``kvcm_config`` is the canonical RTP startup config already derived from
    ``RECO_CLIENT_CONFIG``. Constructing this adapter constructs the generic
    client, whose native ``Create`` path validates the transfer config and
    registers the prefixed KVMeta instance.
    """

    def __init__(self, kvcm_config: "MMKvcmConfig") -> None:
        client_type, config_type = _load_kvcm_client_types()

        # Snapshot every RTP setting before constructing the generic config.
        # The startup config normally has plain attributes, but reading each
        # field once also makes the registered identity deterministic if a
        # caller mutates a config object concurrently.
        addresses = tuple(kvcm_config.addresses)
        instance_id = kvcm_config.instance_id
        instance_group = kvcm_config.instance_group
        user_data = kvcm_config.user_data
        transfer_client_config = kvcm_config.transfer_client_config
        call_timeout_ms = kvcm_config.call_timeout_ms
        write_timeout_seconds = kvcm_config.write_timeout_seconds
        max_object_bytes = kvcm_config.max_object_bytes
        try:
            generic_config = config_type(
                addresses=addresses,
                instance_id=instance_id,
                instance_group=instance_group,
                user_data=user_data,
                transfer_client_config=transfer_client_config,
                call_timeout_ms=call_timeout_ms,
                write_timeout_seconds=write_timeout_seconds,
                max_object_bytes=max_object_bytes,
            )
            client = client_type(generic_config)
        except ImportError as error:
            # The package can be importable while its native extension or one
            # of that extension's runtime libraries is missing.
            raise RuntimeError(_MISSING_WHEEL_MESSAGE) from error

        self._client = client
        self._instance_id = instance_id
        self._instance_group = instance_group

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def instance_group(self) -> str:
        return self._instance_group

    def save(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        self._client.save(keys, tensors, trace_id=trace_id)

    def load(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        self._client.load(keys, tensors, trace_id=trace_id)

    def remove(self, keys: Sequence[str], *, trace_id: Optional[str] = None) -> None:
        self._client.remove(keys, trace_id=trace_id)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RtpKvMetaObjectClient":
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self._client.__exit__(exc_type, exc_value, traceback)
