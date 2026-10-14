from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Protocol

from rtp_llm.kv_cache_subscriber.config import SubscriberConfig
from rtp_llm.kv_cache_subscriber.models import CacheDiff

if TYPE_CHECKING:
    from rtp_llm.kv_cache_subscriber.manager import KvcmHttpManagerClient


class KvcmReporter(Protocol):
    async def start(self, block_size: int) -> None: ...

    async def register_node(self) -> None: ...

    async def report_host_down(self) -> None: ...

    async def report_heartbeat(self) -> None: ...

    async def report_diff(self, diff: CacheDiff, block_size: int) -> None: ...

    async def close(self) -> None: ...


def _engine_config() -> dict[str, Any]:
    try:
        value = json.loads(os.environ.get("DS_LLM_ENGINE_CONFIG", "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: object, default: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


class HttpKvcmReporter:
    """Translate RTP full-snapshot diffs into KVCM ReportEvent requests."""

    def __init__(
        self,
        config: SubscriberConfig,
        manager: KvcmHttpManagerClient | None = None,
    ) -> None:
        self._config = config
        if manager is None:
            from rtp_llm.kv_cache_subscriber.manager import KvcmHttpManagerClient

            manager = KvcmHttpManagerClient(
                config.kvcm_url, config.kvcm_request_timeout_s
            )
        self._manager = manager
        self._engine_config = _engine_config()
        self._started = False

    def _trace_id(self, operation: str) -> str:
        return f"rtp_subscriber_{operation}_{time.monotonic_ns()}"

    def _spec_name(self, block_size: int) -> str:
        return f"rtp_llm_{block_size}"

    def _event_request(self, events: list[dict[str, object]]) -> dict[str, object]:
        return {
            "trace_id": self._trace_id("report_event"),
            "instance_id": self._config.instance_id,
            "host_ip_port": self._config.host_ip_port,
            "storage_type": self._config.storage_type,
            "events": events,
        }

    def _register_instance_request(self, block_size: int) -> dict[str, object]:
        config = self._engine_config
        dtype = config.get("dtype")
        model_name = config.get("model_name")
        return {
            "trace_id": self._trace_id("register_instance"),
            "instance_group": self._config.instance_group,
            "instance_id": self._config.instance_id,
            "block_size": block_size,
            "location_spec_infos": [
                {"name": self._spec_name(block_size), "size": block_size}
            ],
            "model_deployment": {
                "model_name": (
                    model_name
                    if isinstance(model_name, str) and model_name
                    else self._config.model_name
                ),
                "dtype": (
                    dtype
                    if isinstance(dtype, str) and dtype
                    else self._config.model_dtype
                ),
                "use_mla": bool(config.get("use_mla", self._config.use_mla)),
                "tp_size": _positive_int(
                    config.get("tensor_parallel_size"), self._config.tp_size
                ),
                "dp_size": _positive_int(
                    config.get("data_parallel_size"), self._config.dp_size
                ),
                "lora_name": "",
                "pp_size": _positive_int(
                    config.get("pipeline_parallel_size"), self._config.pp_size
                ),
                "extra": "",
                "user_data": "",
            },
            "location_spec_groups": [
                {"name": "default", "spec_names": [self._spec_name(block_size)]}
            ],
        }

    async def start(self, block_size: int) -> None:
        await self._manager.start()
        await self._manager.register_instance(
            self._register_instance_request(block_size)
        )
        self._started = True

    async def _report(self, events: list[dict[str, object]]) -> None:
        if not self._started:
            raise RuntimeError("KVCM reporter has not been started")
        await self._manager.report_event(self._event_request(events))

    async def register_node(self) -> None:
        await self._report(
            [
                {
                    "event_type": "EVENT_NODE_REGISTER",
                    "node_register": {"mediums": [self._config.medium]},
                }
            ]
        )

    async def report_host_down(self) -> None:
        await self._report(
            [{"event_type": "EVENT_HOST_DOWN", "host_down": {}}]
        )

    async def report_heartbeat(self) -> None:
        await self._report(
            [
                {
                    "event_type": "EVENT_HEARTBEAT",
                    "heartbeat": {"system_status": {}},
                }
            ]
        )

    def _block_add(self, key: int, block_size: int) -> dict[str, object]:
        return {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": str(key),
                "medium": self._config.medium,
                "specs": [
                    {
                        "name": self._spec_name(block_size),
                        "uri": (
                            f"rtp-llm://{self._config.host_ip_port}/"
                            f"{self._config.medium}"
                        ),
                    }
                ],
            },
        }

    def _block_delete(self, key: int) -> dict[str, object]:
        return {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {
                "block_key": str(key),
                "medium": self._config.medium,
            },
        }

    async def report_diff(self, diff: CacheDiff, block_size: int) -> None:
        events = [self._block_add(key, block_size) for key in diff.added]
        events.extend(self._block_delete(key) for key in diff.removed)
        batch_size = self._config.kvcm_report_batch_size
        for offset in range(0, len(events), batch_size):
            await self._report(events[offset : offset + batch_size])

    async def close(self) -> None:
        await self._manager.close()
        self._started = False
