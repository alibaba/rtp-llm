from urllib.parse import urlsplit

import aiohttp
from vllm_router.routers.routing_logic import RoutingInterface

from .NaviSchedAdapterConfig import NaviSchedAdapterConfig
from .NaviSchedClient import NaviSchedClient
from .VLLMBlockHash import stable_request_id


class NaviSchedProductionStackRouter(RoutingInterface):
    requires_request_json = True

    def __init__(self, config=None, client_factory=None):
        if hasattr(self, "_initialized"):
            return
        self._config = config or NaviSchedAdapterConfig.from_env()
        client_class = client_factory or NaviSchedClient
        self._client = client_class(self._config, enable_agent_forward=True)
        self._initialized = True

    async def route_request(
        self,
        endpoints,
        engine_stats,
        request_stats,
        request,
        request_json,
    ):
        del engine_stats, request_stats
        if not endpoints:
            raise ValueError("NaviSched requires at least one vLLM endpoint")
        self._validate_hash_inputs(request_json)
        token_ids = await self._get_token_ids(endpoints[0].url, request, request_json)
        engine_request_id = self._get_engine_request_id(request, request_json)
        decision = await self._client.route_async(
            token_ids,
            request_id=stable_request_id(engine_request_id),
            forward_payload=engine_request_id.encode("utf-8"),
        )
        if decision.target is None:
            raise RuntimeError("NaviSched returned no target")
        return self._find_endpoint(endpoints, decision.target.address)

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    async def _get_token_ids(self, endpoint, request, request_json):
        supplied = self._get_supplied_token_ids(request_json)
        if supplied is not None:
            return supplied
        payload = self._make_tokenize_payload(request_json)
        session = request.app.state.aiohttp_client_wrapper()
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_ms / 1000.0)
        async with session.post(
            endpoint.rstrip("/") + "/tokenize",
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            result = await response.json()
        token_ids = result.get("tokens")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("vLLM tokenize endpoint returned no tokens")
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _get_supplied_token_ids(request_json):
        prompt = request_json.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            return None
        if all(isinstance(token_id, int) for token_id in prompt):
            return [int(token_id) for token_id in prompt]
        return None

    @staticmethod
    def _validate_hash_inputs(request_json):
        if request_json.get("cache_salt") is not None:
            raise ValueError("NaviSched does not support cache_salt block hashes")
        for name in ("multi_modal_data", "mm_processor_kwargs"):
            if request_json.get(name) is not None:
                raise ValueError("NaviSched does not support multimodal block hashes")

    @staticmethod
    def _make_tokenize_payload(request_json):
        payload = {"model": request_json.get("model")}
        if "messages" in request_json:
            payload["messages"] = request_json["messages"]
            names = (
                "add_generation_prompt",
                "continue_final_message",
                "add_special_tokens",
                "chat_template",
                "chat_template_kwargs",
                "media_io_kwargs",
                "mm_processor_kwargs",
                "tools",
            )
        elif isinstance(request_json.get("prompt"), str):
            payload["prompt"] = request_json["prompt"]
            names = ("add_special_tokens",)
        else:
            raise ValueError("NaviSched requires one text or token-id prompt")
        for name in names:
            if name in request_json:
                payload[name] = request_json[name]
        return payload

    @staticmethod
    def _get_request_id(request, request_json):
        return str(
            request.headers.get("X-Request-Id")
            or getattr(request.state, "request_id", "")
            or request_json.get("request_id")
            or ""
        )

    @classmethod
    def _get_engine_request_id(cls, request, request_json):
        request_id = cls._get_request_id(request, request_json)
        path = str(getattr(getattr(request, "url", None), "path", ""))
        prefixes = {
            "/v1/completions": "cmpl-",
            "/v1/chat/completions": "chatcmpl-",
        }
        engine_request_id = prefixes.get(path, "") + request_id
        if path == "/v1/completions":
            engine_request_id += "-0"
        return engine_request_id

    @classmethod
    def _get_sched_request_id(cls, request, request_json):
        engine_request_id = cls._get_engine_request_id(request, request_json)
        return stable_request_id(engine_request_id)

    @classmethod
    def _find_endpoint(cls, endpoints, target_address):
        if not target_address:
            raise RuntimeError("NaviSched returned no target address")
        target = cls._normalize_address(target_address)
        for endpoint in endpoints:
            if cls._normalize_address(endpoint.url) == target:
                return endpoint.url
        raise RuntimeError(
            f"NaviSched target {target_address} is not an eligible vLLM endpoint"
        )

    @staticmethod
    def _normalize_address(address):
        value = str(address).strip()
        parsed = urlsplit(value if "://" in value else "//" + value)
        if parsed.hostname is None or parsed.port is None:
            raise ValueError(f"invalid vLLM endpoint address: {address}")
        return parsed.hostname, parsed.port
