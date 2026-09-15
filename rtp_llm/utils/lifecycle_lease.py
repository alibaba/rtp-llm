"""Instance-wide serialization for lifecycle operations."""

import json
import logging
import uuid
from typing import Any, Callable, Dict, Optional


class LifecycleLease:
    KEY = "rtp_llm_instance_lifecycle_lease"

    def __init__(
        self,
        store: Optional[Any],
        store_factory: Optional[Callable[[], Optional[Any]]],
        required: bool,
    ):
        self._store = store
        self._store_factory = store_factory
        self._required = required or store is not None
        self._holder = uuid.uuid4().hex

    def _get_store(self) -> Optional[Any]:
        if self._store is None and self._store_factory is not None:
            try:
                self._store = self._store_factory()
            except Exception as e:
                logging.error("failed to establish lifecycle TCPStore: %s", e)
        return self._store

    @property
    def required(self) -> bool:
        return self._required

    @required.setter
    def required(self, value: bool) -> None:
        self._required = bool(value)

    def record(self, operation: str) -> str:
        return json.dumps(
            {"holder": self._holder, "operation": operation},
            sort_keys=True,
            separators=(",", ":"),
        )

    def acquire(self, operation: str) -> tuple[Optional[str], Dict[str, Any]]:
        store = self._get_store()
        if store is None:
            if not self._required:
                return None, {}
            return None, {
                "error": "instance-wide lifecycle coordination is unavailable",
                "grpc_status": "FAILED_PRECONDITION",
            }
        record = self.record(operation)
        try:
            current = store.compare_set(self.KEY, "", record)
            if isinstance(current, bytes):
                current = current.decode("utf-8")
        except Exception as e:
            return None, {
                "error": f"instance-wide lifecycle coordination failed: {e}",
                "grpc_status": "FAILED_PRECONDITION",
            }
        if current != record:
            return None, {
                "error": "another lifecycle operation holds the instance lease",
                "grpc_status": "FAILED_PRECONDITION",
            }
        return record, {}

    def release(self, record: Optional[str]) -> None:
        if record is None or self._store is None:
            return
        try:
            self._store.compare_set(self.KEY, record, "")
        except Exception as e:
            logging.error("failed to release instance lifecycle lease: %s", e)
