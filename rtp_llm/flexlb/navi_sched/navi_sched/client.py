from typing import Optional, Union

from .alog import configureNaviSchedAlog
from .sched import NaviSchedDecision, NaviSchedRequest, decisionFromNative, makeRequest
from .util import (
    DEFAULT_CLIENT_RUNTIME_NAME,
    RoleFacade,
    makeRuntimeConfig,
    makeServiceConfig,
    naviSchedModule,
)

_navi_sched = naviSchedModule()


class NaviSchedClient(RoleFacade):
    def __init__(
        self,
        biz_name: str,
        master_biz_name: str = "",
        runtime_config=None,
        enable_agent_forward: bool = False,
        expect_agent_response_stream: bool = True,
        name: str = DEFAULT_CLIENT_RUNTIME_NAME,
        enable_sample_collection: bool = True,
        **options,
    ):
        configureNaviSchedAlog()
        self._master_biz_name = str(master_biz_name)
        navi_sched = naviSchedModule()
        service_config = makeServiceConfig(navi_sched, name, options)
        config = makeRuntimeConfig(
            navi_sched,
            navi_sched.NaviSchedClientConfig,
            biz_name,
            runtime_config,
        )
        config.service = service_config
        config.master_biz_name = str(master_biz_name)
        config.enable_sample_collection = bool(enable_sample_collection)
        config.enable_agent_forward = bool(enable_agent_forward)
        config.expect_agent_response_stream = bool(expect_agent_response_stream)
        service = navi_sched.NaviSchedClient(config)
        super().__init__(navi_sched, service)

    def schedule(
        self,
        request: Union[NaviSchedRequest, int],
        target_biz_name: str,
        master_biz_name: str = "",
        block_hashes=None,
        timeout_ms: int = 1000,
        request_id: Optional[int] = None,
        block_size: int = 256,
        collect_replay: bool = False,
        forward_payload: bytes = b"",
    ) -> NaviSchedDecision:
        effective_master_biz_name = (
            str(master_biz_name) or self._master_biz_name
        )
        if not target_biz_name:
            raise ValueError("target_biz_name is required")
        if not effective_master_biz_name:
            raise ValueError("master_biz_name is required")
        if not isinstance(request, NaviSchedRequest):
            request = makeRequest(
                request,
                () if block_hashes is None else block_hashes,
                request_id=request_id,
                block_size=block_size,
                forward_payload=forward_payload,
            )
        elif not 0 < int(request.request_id) < (1 << 64):
            raise ValueError("request.request_id must be a nonzero uint64")
        if int(request.prompt_token_count) < 0:
            raise ValueError("request.prompt_token_count must be nonnegative")
        if int(request.block_size) <= 0:
            raise ValueError("request.block_size must be positive")
        if len(request.block_hashes) > (
            int(request.prompt_token_count) // int(request.block_size)
        ):
            raise ValueError("request.block_hashes exceed complete prompt blocks")
        native_request = _navi_sched.NaviSchedScheduleRequest()
        native_request.request_id = int(request.request_id)
        native_request.biz_name = str(target_biz_name)
        native_request.prompt_token_count = int(request.prompt_token_count)
        native_request.block_size = int(request.block_size)
        native_request.block_hashes = request.block_hashes
        native_request.timeout_ms = int(timeout_ms)
        native_request.forward_payload = request.forward_payload
        result = self._service.schedule(
            native_request,
            effective_master_biz_name,
            bool(collect_replay),
        )
        return decisionFromNative(result)

    def take_next_forward_response(self, timeout_ms: int = 0):
        return self._service.take_next_forward_response(int(timeout_ms))

    def abort_forward_request(self, request_id: int) -> bool:
        return bool(self._service.abort_forward_request(int(request_id)))

    def allocate_request_id(self) -> int:
        return int(self._service.allocate_request_id())

    def has_master_biz(self, master_biz_name: str = "") -> bool:
        effective_master_biz_name = (
            str(master_biz_name) or self._master_biz_name
        )
        return bool(self._service.has_master_biz(effective_master_biz_name))

    def report_local_fallback(self) -> None:
        self._service.report_local_fallback()

    @property
    def request_received_count(self) -> int:
        return int(self._service.get_request_received_count())

__all__ = [
    "NaviSchedClient",
    "NaviSchedDecision",
    "NaviSchedRequest",
]
