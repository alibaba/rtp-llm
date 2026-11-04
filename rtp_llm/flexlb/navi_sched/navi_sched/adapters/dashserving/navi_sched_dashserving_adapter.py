from __future__ import annotations

import atexit
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..vllm.NaviSchedAdapterConfig import NaviSchedAdapterConfig
from ..vllm.NaviSchedClient import NaviSchedClient
from ..vllm.VLLMAdapterUtils import split_address
from ..vllm.VLLMBlockHash import stable_request_id
from ...util import DEFAULT_CLIENT_RUNTIME_NAME


class DashServingRequestError(ValueError):
    pass


class DashServingRoutingError(RuntimeError):
    pass


@dataclass(frozen=True)
class DashServingRoute:
    request_id: str
    scheduler_request_id: int
    target_address: str
    target_host: str
    target_port: int
    schedule_latency_ms: float


def _read_positive_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    try:
        value = int(environ.get(name, str(default)).strip())
    except (AttributeError, ValueError) as exc:
        raise DashServingRoutingError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise DashServingRoutingError(f"{name} must be a positive integer")
    return value


def _config_from_env(
    environ: Mapping[str, str] | None = None,
) -> NaviSchedAdapterConfig:
    values = environ if environ is not None else os.environ
    addresses = tuple(
        address.strip()
        for address in values.get("DS_NAVI_SCHED_MASTER_ADDRESSES", "").split(",")
        if address.strip()
    )
    config = NaviSchedAdapterConfig(
        master_biz_name=values.get("DS_NAVI_SCHED_MASTER_BIZ_NAME", "").strip(),
        target_biz_name=values.get("DS_NAVI_SCHED_TARGET_BIZ_NAME", "").strip(),
        client_biz_name=values.get(
            "DS_NAVI_SCHED_CLIENT_BIZ_NAME", "dashserving_router"
        ).strip(),
        master_addresses=addresses,
        master_vip_domain=values.get("DS_NAVI_SCHED_MASTER_VIP_DOMAIN", "").strip(),
        master_vip_jmenv=values.get(
            "DS_NAVI_SCHED_MASTER_VIP_JMENV", "jmenv.tbsite.net"
        ).strip(),
        timeout_ms=_read_positive_int(values, "DS_NAVI_SCHED_TIMEOUT_MS", 1000),
        block_size=_read_positive_int(values, "DS_NAVI_SCHED_BLOCK_SIZE", 16),
    )
    try:
        config.validate_client()
    except ValueError as exc:
        raise DashServingRoutingError(str(exc)) from exc
    return config


class NaviSchedDashServingRouter:
    def __init__(
        self,
        config: NaviSchedAdapterConfig | None = None,
        client_factory=None,
    ) -> None:
        self._config = config or _config_from_env()
        self._client_factory = client_factory or NaviSchedClient
        self._pid = os.getpid()
        self._client = None
        self._lock = threading.Lock()

    def _make_client(self):
        try:
            return self._client_factory(
                self._config,
                name=DEFAULT_CLIENT_RUNTIME_NAME,
            )
        except Exception as exc:
            raise DashServingRoutingError(
                "failed to initialize the NaviSched Client"
            ) from exc

    def _get_client(self):
        current_pid = os.getpid()
        if current_pid != self._pid:
            self._pid = current_pid
            self._client = None
            self._lock = threading.Lock()
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                self._client = self._make_client()
            return self._client

    @staticmethod
    def _option_value(source: Any, name: str) -> Any:
        if isinstance(source, Mapping):
            return source.get(name)
        return getattr(source, name, None)

    @classmethod
    def _validate_request_options(
        cls,
        prompt: Mapping[str, Any],
        params: Any,
        extra_params: Mapping[str, Any] | None,
    ) -> None:
        sources = [prompt, params, extra_params]
        params_extra = cls._option_value(params, "extra")
        if isinstance(params_extra, Mapping):
            sources.append(params_extra)

        multimodal = (
            "multi_modal_data",
            "modalities",
            "mm_processor_kwargs",
            "mm_extra_kwargs",
        )
        if any(
            cls._option_value(source, name) is not None
            for source in sources
            if source is not None
            for name in multimodal
        ):
            raise DashServingRequestError(
                "NaviSched routing does not support multimodal block hashes"
            )

        hash_options = (
            "cache_salt",
            "lora_request",
            "lora_name",
            "lora_id",
        )
        if any(
            cls._option_value(source, name) is not None
            for source in sources
            if source is not None
            for name in hash_options
        ):
            raise DashServingRequestError(
                "NaviSched routing does not support cache_salt or LoRA block hashes"
            )

        candidate_count = cls._option_value(params, "n")
        if candidate_count is not None:
            try:
                candidate_count = int(candidate_count)
            except (TypeError, ValueError) as exc:
                raise DashServingRequestError(
                    "NaviSched routing requires exactly one generated candidate"
                ) from exc
            if candidate_count != 1:
                raise DashServingRequestError(
                    "NaviSched routing requires exactly one generated candidate"
                )

    @staticmethod
    def _prompt_token_ids(
        prompt: Any,
    ) -> list[int]:
        if not isinstance(prompt, Mapping):
            raise DashServingRequestError("prompt must contain prompt_token_ids")
        token_ids = prompt.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise DashServingRequestError("prompt_token_ids must be non-empty")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise DashServingRequestError("prompt_token_ids must contain integers")
        return token_ids

    def route_sync(
        self,
        *,
        request_id: str,
        prompt: Any,
        params: Any = None,
        extra_params: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> DashServingRoute:
        if not isinstance(request_id, str) or not request_id:
            raise DashServingRequestError("request_id is required")
        if not isinstance(prompt, Mapping):
            raise DashServingRequestError("prompt must contain prompt_token_ids")
        self._validate_request_options(prompt, params, extra_params)
        token_ids = self._prompt_token_ids(prompt)
        scheduler_request_id = stable_request_id(request_id)
        started_at = time.monotonic()
        try:
            decision = self._get_client().route(
                token_ids,
                request_id=scheduler_request_id,
            )
        except DashServingRoutingError:
            raise
        except Exception as exc:
            raise DashServingRoutingError("NaviSched scheduling failed") from exc
        target = getattr(decision, "target", None)
        target_address = getattr(target, "address", "")
        if not target_address:
            raise DashServingRoutingError("NaviSched returned no Prefill target")
        try:
            target_host, target_port = split_address(target_address)
        except (TypeError, ValueError) as exc:
            raise DashServingRoutingError(
                "NaviSched returned an invalid Prefill target"
            ) from exc
        return DashServingRoute(
            request_id=request_id,
            scheduler_request_id=scheduler_request_id,
            target_address=target_address,
            target_host=target_host,
            target_port=target_port,
            schedule_latency_ms=(time.monotonic() - started_at) * 1000.0,
        )

    def close(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()


_singleton_lock = threading.Lock()
_singleton_router: NaviSchedDashServingRouter | None = None


def _reset_after_fork() -> None:
    global _singleton_lock, _singleton_router
    _singleton_lock = threading.Lock()
    _singleton_router = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _get_router() -> NaviSchedDashServingRouter:
    global _singleton_router
    if _singleton_router is not None:
        return _singleton_router
    with _singleton_lock:
        if _singleton_router is None:
            _singleton_router = NaviSchedDashServingRouter()
        return _singleton_router


def route_sync(**kwargs: Any) -> DashServingRoute:
    return _get_router().route_sync(**kwargs)


def shutdown() -> None:
    global _singleton_router
    with _singleton_lock:
        router, _singleton_router = _singleton_router, None
    if router is not None:
        router.close()


atexit.register(shutdown)
