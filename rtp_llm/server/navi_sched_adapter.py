import asyncio
import time
import importlib
import os
from typing import Mapping, Optional

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException

MASTER_TYPE_ENV = "RTP_LLM_MASTER_TYPE"
FLEXLB_MASTER_TYPE = "flexlb"
NAVI_SCHED_MASTER_TYPE = "navi_sched"
NAVI_SCHED_CONFIG_JSON_ENV = "RTP_LLM_NAVI_SCHED_CONFIG_JSON"


def get_master_type(environ: Optional[Mapping[str, str]] = None) -> str:
    values = environ if environ is not None else os.environ
    master_type = values.get(MASTER_TYPE_ENV, "").strip().lower()
    if not master_type:
        return FLEXLB_MASTER_TYPE
    if master_type not in (FLEXLB_MASTER_TYPE, NAVI_SCHED_MASTER_TYPE):
        raise ValueError(
            f"{MASTER_TYPE_ENV} must be {FLEXLB_MASTER_TYPE} or "
            f"{NAVI_SCHED_MASTER_TYPE}, got {master_type}"
        )
    return master_type


def _load_adapter():
    return importlib.import_module("navi_sched.adapters.rtp_llm")


def create_navi_sched_client(host_service):
    if not navi_sched_enabled():
        return None
    return _load_adapter().create_navi_sched_client(host_service)


def get_navi_sched_backend_roles(default_roles, navi_sched_client):
    config = getattr(navi_sched_client, "config", None)
    get_target_role = getattr(config, "get_target_role", None)
    if get_target_role is None:
        return default_roles
    return [get_target_role()]


def apply_navi_sched_request_overrides(input):
    if not os.environ.get(NAVI_SCHED_CONFIG_JSON_ENV, "").strip():
        return
    _load_adapter().apply_frontend_request_overrides(input)


def create_master_client(flexlb_client, host_service):
    if not navi_sched_enabled():
        return flexlb_client

    return NaviSchedMasterClient(flexlb_client, _load_adapter())


class FlexlbScheduleError(FtRuntimeException):
    pass


class NaviSchedMasterClient:
    requires_flexlb_route = True
    _now = staticmethod(time.monotonic)

    def __init__(self, flexlb_client, adapter):
        self._flexlb = flexlb_client
        self._adapter = adapter
        client_class = importlib.import_module(
            "navi_sched.adapters.rtp_llm.NaviSchedClient").NaviSchedClient
        self._client = client_class(enable_agent_forward=True, deferred_master=True)

    def __getattr__(self, name):
        return getattr(self._flexlb, name)

    async def get_backend_role_addrs(self, **kwargs):
        if kwargs.get("vit_only"):
            return await self._flexlb.get_backend_role_addrs(**kwargs)
        config = kwargs["input"].generate_config
        timeout_ms = config.ttft_timeout_ms or config.timeout_ms
        if timeout_ms is None or timeout_ms <= 0:
            timeout_ms = self._flexlb.master_config.master_default_timeout_ms
        timeout_s = timeout_ms / 1000.0
        if kwargs.get("timeout_s") is not None:
            timeout_s = min(timeout_s, kwargs["timeout_s"])
        deadline = self._now() + timeout_s
        scheduled = None

        async def schedule_rpc(target, request, _timeout_s, metadata):
            nonlocal scheduled
            ready = asyncio.get_running_loop().create_future()

            def received(response):
                if not ready.done():
                    ready.set_result(response)

            async def run():
                try:
                    ip, port = self._adapter.split_address(target)
                    flexlb_info = self._adapter.FlexlbInfo(
                        ip=ip, port=port, request=request, deadline=deadline,
                        metadata=metadata, _on_response=received)
                    return await self._client.get_backend_role_addrs(
                        block_cache_keys=kwargs["block_cache_keys"],
                        cache_key_block_size=kwargs["cache_key_block_size"],
                        input=kwargs["input"], request_id=kwargs["request_id"],
                        input_pb=kwargs.get("input_pb"), flexlb_info=flexlb_info)
                except BaseException as error:
                    if not ready.done():
                        ready.set_exception(error)
                    raise

            scheduled = asyncio.create_task(run())
            try:
                return await asyncio.shield(ready)
            except BaseException:
                scheduled.cancel()
                await asyncio.gather(scheduled, return_exceptions=True)
                raise

        first_request = dict(kwargs)
        first_request.update(
            skip_prefill=True, schedule_rpc=schedule_rpc, timeout_s=timeout_s)
        try:
            try:
                parsed = await self._flexlb.get_backend_role_addrs(**first_request)
            except Exception as error:
                if isinstance(error, FtRuntimeException):
                    raise FlexlbScheduleError(
                        error.exception_type, error.message,
                        error.admission_reject_reason) from error
                raise FlexlbScheduleError(
                    ExceptionType.ADMISSION_UNAVAILABLE,
                    f"FlexLB scheduling failed: {error}") from error
            if parsed.connection_failed:
                raise FlexlbScheduleError(
                    ExceptionType.ADMISSION_UNAVAILABLE,
                    "FlexLB is unavailable; Prefill was not delegated")
            if scheduled is None:
                return parsed
            return (await scheduled).with_flexlb_result(parsed)
        finally:
            if scheduled is not None and not scheduled.done():
                scheduled.cancel()
                await asyncio.gather(scheduled, return_exceptions=True)

    async def close(self):
        await asyncio.gather(self._flexlb.close(), self._client.close())



def navi_sched_enabled():
    master_type = os.environ.get(MASTER_TYPE_ENV, "").strip()
    if master_type:
        return get_master_type() == NAVI_SCHED_MASTER_TYPE
    return bool(os.environ.get(NAVI_SCHED_CONFIG_JSON_ENV, "").strip())
