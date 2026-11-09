from navi_sched.util import DEFAULT_CLIENT_RUNTIME_NAME

from navi_sched import NaviSchedClient as CoreNaviSchedClient

from .NaviSchedAdapterConfig import NaviSchedAdapterConfig
from .NaviSchedEndpointRouter import NaviSchedEndpointRouter
from .VLLMAdapterUtils import make_kmon_tags, split_address


class NaviSchedClient:
    def __init__(
        self,
        config: NaviSchedAdapterConfig,
        native_client=None,
        kmon_tags=None,
        **role_options,
    ):
        config.validate_client()
        self._config = config
        self._client = native_client
        if self._client is None:
            enable_agent_forward = bool(role_options.pop("enable_agent_forward", False))
            name = role_options.pop("name", DEFAULT_CLIENT_RUNTIME_NAME)
            local_nodes = []
            for address in config.master_addresses:
                ip, grpc_port = split_address(address)
                local_nodes.append(
                    {
                        "biz_name": config.master_biz_name,
                        "ip": ip,
                        "grpc_port": grpc_port,
                    }
                )
            runtime_config = {"local_nodes": local_nodes}
            if config.master_vip_domain:
                runtime_config["vip_subscription"] = {
                    "jmenv_domain": config.master_vip_jmenv,
                    "services": [config.master_vip_domain],
                    "grpc_port_diff": config.master_vip_grpc_port_diff,
                }
            if config.master_spectrum_service_id:
                runtime_config["spectrum_subscription"] = {
                    "service_id": config.master_spectrum_service_id,
                    "grpc_port_diff": 1,
                }
            self._client = CoreNaviSchedClient(
                config.client_biz_name,
                config.master_biz_name,
                runtime_config=runtime_config,
                enable_agent_forward=enable_agent_forward,
                expect_agent_response_stream=True,
                name=name,
                enable_kmon=True,
                kmon_tags=(
                    make_kmon_tags("client") if kmon_tags is None else kmon_tags
                ),
                wait_no_query=False,
                **role_options,
            )
            self._client.init()
        self._router = NaviSchedEndpointRouter(
            self._client,
            config.target_biz_name,
            timeout_ms=config.timeout_ms,
            block_size=config.block_size,
            route_worker_count=config.route_worker_count,
            replay_sample_interval=config.replay_sample_interval,
        )

    def route(self, prompt_token_ids, **request_options):
        return self._router.route(prompt_token_ids, **request_options)

    async def route_async(self, prompt_token_ids, **request_options):
        return await self._router.route_async(
            prompt_token_ids,
            **request_options,
        )

    def route_generate_request(self, request, **request_options):
        return self._router.route_generate_request(
            request,
            **request_options,
        )

    async def route_generate_request_async(self, request, **request_options):
        return await self._router.route_generate_request_async(
            request,
            **request_options,
        )

    def take_next_forward_response(self, timeout_ms=0):
        if self._client is None:
            return None
        return self._client.take_next_forward_response(timeout_ms)

    def abort_forward_request(self, request_id: int) -> bool:
        return bool(
            self._client is not None
            and self._client.abort_forward_request(int(request_id))
        )

    def allocate_request_id(self) -> int:
        if self._client is None:
            raise RuntimeError("NaviSched client is closed")
        return int(self._client.allocate_request_id())

    def has_master_biz(self) -> bool:
        return bool(
            self._client is not None
            and self._client.has_master_biz(self._config.master_biz_name)
        )

    def report_local_fallback(self) -> None:
        if self._client is not None:
            self._client.report_local_fallback()

    def close(self) -> None:
        self._router.close()
        if self._client is not None:
            self._client.stop()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
