import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from rtp_llm.config.exceptions import ExceptionType
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.master_client import FlexlbResponse

from ...util import DEFAULT_CLIENT_RUNTIME_NAME
from .NaviSchedAdapterConfig import NaviSchedAdapterConfig
from .RtpLLMAdapterUtils import (
    get_schedule_timeout_ms,
    make_kmon_options,
    make_runtime_options,
    split_address,
)
from .RtpLLMUnifiedConfig import (
    NAVI_SCHED_CONFIG_JSON_ENV,
    load_navi_sched_config,
)
from .ServiceDiscovery import (
    RtpLLMServiceRoute,
    ServiceEndpoint,
    make_discovery_runtime,
    master_grpc_port_diff,
)


class NaviSchedClient:
    uses_prompt_token_ids = False

    def __init__(
        self,
        config=None,
        host_service=None,
        native_client=None,
        navi_sched_module=None,
        enable_agent_forward=False,
    ):
        self.config = config or NaviSchedAdapterConfig.from_env()
        self.config.validate_client()
        self._host_service = host_service
        self._module = navi_sched_module
        self._enable_agent_forward = enable_agent_forward
        self._client = native_client
        self._schedule_sequence = 0
        if self._client is None:
            self._init_client()

    def _make_native_client(self, module, master_biz_name, runtime_config):
        name = DEFAULT_CLIENT_RUNTIME_NAME
        client = module.NaviSchedClient(
            "rtp_llm_navi_sched_client",
            master_biz_name,
            runtime_config=runtime_config,
            enable_agent_forward=self._enable_agent_forward,
            expect_agent_response_stream=False,
            name=name,
            wait_no_query=False,
            **make_kmon_options("client", os.environ),
            **make_runtime_options(name, self.config.log_path),
        )
        return client.init()

    def _init_client(self) -> None:
        master_vip = getattr(self._host_service, "master_vip", None)
        module = self._module or __import__("navi_sched")
        if self.config.master_address:
            master_domain = self.config.master_address
            master_biz_name = self.config.master_biz_name
            if not master_biz_name:
                raise ValueError(
                    "RTP_LLM_NAVI_SCHED_MASTER_BIZ_NAME is required with "
                    "RTP_LLM_NAVI_SCHED_MASTER_ADDRESS"
                )
            local_nodes = []
            for master_address in master_domain.split(","):
                master_ip, master_port = split_address(master_address)
                local_nodes.append({
                    "biz_name": master_biz_name,
                    "ip": master_ip,
                    "grpc_port": master_port,
                })
            runtime_config = {"local_nodes": local_nodes}
        else:
            route = RtpLLMServiceRoute.from_env(os.environ, required=False)
            if route is not None and route.master_endpoint is not None:
                endpoint = route.master_endpoint
                master_domain = endpoint.address
                master_biz_name = self.config.master_biz_name
                port_diff = master_grpc_port_diff(endpoint)
                runtime_config = make_discovery_runtime(
                    endpoint,
                    master_biz_name,
                    route.use_local,
                    port_diff,
                    os.environ,
                )
            else:
                master_domain = str(getattr(master_vip, "domain", "") or "")
                use_local = bool(getattr(master_vip, "use_local", False))
                master_biz_name = self.config.master_biz_name
                if not master_domain or not master_biz_name:
                    raise ValueError(
                        "NaviSched Master service discovery is not configured"
                    )
                if use_local:
                    local_nodes = []
                    for master_address in master_domain.split(","):
                        master_ip, master_port = split_address(master_address)
                        local_nodes.append({
                            "biz_name": master_biz_name,
                            "ip": master_ip,
                            "grpc_port": master_port,
                        })
                    runtime_config = {"local_nodes": local_nodes}
                else:
                    endpoint = ServiceEndpoint(master_domain, "grpc")
                    runtime_config = make_discovery_runtime(
                        endpoint, master_biz_name, False, 0, os.environ
                    )
        self._client = self._make_native_client(
            module, master_biz_name, runtime_config
        )
        logging.info(
            "navi_sched_effective_client_config=%s",
            json.dumps(
                {
                    "adapter": asdict(self.config),
                    "enable_agent_forward": self._enable_agent_forward,
                    "master_biz_name": master_biz_name,
                    "master_domain": master_domain,
                    "runtime_config": runtime_config,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    async def close(self) -> None:
        clients = [self._client]
        self._client = None
        for client in clients:
            if client is not None:
                await asyncio.to_thread(client.stop)

    def get_latest_queue_length(self) -> int:
        return 0

    def is_available(self) -> bool:
        return bool(self.config.master_address) or bool(
            getattr(getattr(self._host_service, "master_vip", None), "domain", "")
        )

    @staticmethod
    def _apply_batch_decision(input, decision) -> None:
        expected_batch_size = int(decision.expected_batch_size)
        try:
            batch_id = int(decision.batch_id or 0)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid NaviSched batch_id: {decision.batch_id}"
            ) from error
        if batch_id <= 0 or expected_batch_size <= 0:
            input.group_id = -1
            input.group_size = 1
            input.navi_sched_assigned_batch = False
            return
        input.group_id = batch_id
        input.group_size = expected_batch_size
        input.navi_sched_assigned_batch = True
        input.navi_sched_predicted_queue_latency_ms = float(
            decision.predicted_queue_latency_ms
        )
        input.navi_sched_predicted_cache_hit_tokens = int(
            decision.predicted_cache_hit_tokens
        )
        input.navi_sched_predicted_execution_latency_ms = float(
            decision.predicted_execution_latency_ms
        )
        input.navi_sched_schedule_time_us = int(
            getattr(decision, "schedule_time_us", 0)
        )

    def _make_role_addrs(self, decision):
        role_by_target = {
            1: RoleType.PREFILL,
            2: RoleType.DECODE,
        }
        target = decision.target
        if target is None or not target.address:
            return []
        role = role_by_target.get(
            int(getattr(target, "role", 0)),
            self.config.get_target_role(),
        )
        server_ip, server_port = split_address(target.address)
        return [RoleAddr(
            role=role,
            ip=server_ip,
            http_port=max(0, server_port - 1),
            grpc_port=server_port,
        )]

    def _adapt_decision(self, input, decision) -> FlexlbResponse:
        if decision.target is None:
            error_type = (
                ExceptionType.TRAFFIC_LIMIT_ERROR
                if decision.reason == "rate_limited"
                else ExceptionType.MASTER_NO_AVAILABLE_WORKER
            )
            return FlexlbResponse.error_response(
                int(error_type),
                decision.reason or "navi sched returned no target",
            )
        selected_addrs = self._make_role_addrs(decision)
        if not selected_addrs:
            error_type = (
                ExceptionType.TRAFFIC_LIMIT_ERROR
                if decision.reason == "rate_limited"
                else ExceptionType.MASTER_NO_AVAILABLE_WORKER
            )
            return FlexlbResponse.error_response(
                int(error_type),
                decision.reason or "navi sched returned no target",
            )
        selected_role = selected_addrs[0].role
        role_addrs = selected_addrs + [
            address
            for address in input.generate_config.role_addrs
            if address.role != selected_role
        ]
        self._apply_batch_decision(input, decision)
        return FlexlbResponse.ok(
            role_addrs, enqueued_by_master=self._enable_agent_forward
        )

    def _should_collect_replay(self) -> bool:
        self._schedule_sequence += 1
        interval = self.config.replay_sample_interval
        return interval > 0 and (self._schedule_sequence - 1) % interval == 0

    def _write_replay(self, decision, prompt_token_count, block_hashes) -> None:
        replay = decision.replay
        if replay is None:
            return
        output_dir = Path(self.config.replay_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timing = (
            asdict(decision.request_timing)
            if decision.request_timing is not None
            else None
        )
        payload = {
            "capture_time_ns": int(time.time() * 1000000000),
            "process_id": os.getpid(),
            "request_id": decision.request_id,
            "prompt_token_count": prompt_token_count,
            "block_hashes": block_hashes,
            "decision": {
                "schedule_id": decision.schedule_id,
                "schedule_loss": decision.schedule_loss,
                "schedule_latency_us": decision.schedule_latency_us,
                "batch_id": decision.batch_id,
                "expected_batch_size": decision.expected_batch_size,
                "predicted_cache_hit_tokens": (
                    decision.predicted_cache_hit_tokens
                ),
                "predicted_queue_latency_ms": decision.predicted_queue_latency_ms,
                "predicted_execution_latency_ms": (
                    decision.predicted_execution_latency_ms
                ),
                "target": (
                    decision.target.address if decision.target is not None else ""
                ),
                "request_timing": timing,
            },
            "replay": json.loads(replay.to_json()),
        }
        filename = (
            f"schedule_{decision.schedule_id}_request_{decision.request_id}.json"
        )
        output_path = output_dir / filename
        temporary_path = output_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(temporary_path, output_path)

    async def get_backend_role_addrs(
        self,
        block_cache_keys,
        cache_key_block_size,
        input,
        request_id,
        input_pb=None,
    ) -> FlexlbResponse:
        token_ids = input.token_ids
        prompt_token_count = (
            int(token_ids.numel())
            if hasattr(token_ids, "numel")
            else len(token_ids)
        )
        schedule_options = {
            "target_biz_name": self.config.biz_name,
            "timeout_ms": get_schedule_timeout_ms(input.generate_config),
            "request_id": int(request_id),
            "block_size": int(cache_key_block_size),
        }
        native_block_hashes = [
            int(value) & ((1 << 64) - 1) for value in block_cache_keys
        ]
        if self._enable_agent_forward and input_pb is None:
            raise ValueError("NaviSched Agent forwarding requires input_pb")
        forward_payload = (
            input_pb.SerializeToString()
            if self._enable_agent_forward
            else b""
        )
        collect_replay = self._should_collect_replay()
        timing_probe = (
            os.environ.get("RTP_LLM_NAVI_SCHED_ROUTE_TIMING_PROBE") == "1"
        )
        schedule_call_ns = time.time_ns()
        worker_start_ns = schedule_call_ns
        client = self._client
        if timing_probe:
            def schedule_with_timestamp():
                worker_start = time.time_ns()
                worker_decision = client.schedule(
                    prompt_token_count,
                    block_hashes=native_block_hashes,
                    collect_replay=collect_replay,
                    forward_payload=forward_payload,
                    **schedule_options,
                )
                return worker_start, time.time_ns(), worker_decision

            worker_start_ns, worker_return_ns, decision = await asyncio.to_thread(
                schedule_with_timestamp
            )
        else:
            decision = await asyncio.to_thread(
                client.schedule,
                prompt_token_count,
                block_hashes=native_block_hashes,
                collect_replay=collect_replay,
                forward_payload=forward_payload,
                **schedule_options,
            )
            worker_return_ns = time.time_ns()
        schedule_return_ns = time.time_ns()
        if timing_probe:
            timing = decision.request_timing
            logging.info(
                "NAVI_SCHED_ROUTE_TIMING request_id=%s schedule_call_ns=%s "
                "worker_start_ns=%s worker_return_ns=%s schedule_return_ns=%s "
                "master_schedule_time_us=%s batch_id=%s expected_batch_size=%s "
                "predicted_queue_ms=%.3f predicted_execution_ms=%.3f "
                "predicted_cache_hit_tokens=%s "
                "executor_wait_ms=%.3f event_loop_resume_ms=%.3f "
                "schedule_cost_ms=%.3f "
                "batch_wait_ms=%s queue_wait_ms=%s execution_latency_ms=%s "
                "client_total_latency_ms=%s",
                request_id,
                schedule_call_ns,
                worker_start_ns,
                worker_return_ns,
                schedule_return_ns,
                decision.schedule_time_us,
                decision.batch_id,
                decision.expected_batch_size,
                decision.predicted_queue_latency_ms,
                decision.predicted_execution_latency_ms,
                decision.predicted_cache_hit_tokens,
                (worker_start_ns - schedule_call_ns) / 1_000_000.0,
                (schedule_return_ns - worker_return_ns) / 1_000_000.0,
                (schedule_return_ns - schedule_call_ns) / 1_000_000.0,
                getattr(timing, "batch_wait_ms", None),
                getattr(timing, "queue_wait_ms", None),
                getattr(timing, "execution_latency_ms", None),
                getattr(timing, "client_total_latency_ms", None),
            )
        input.navi_sched_block_hashes = native_block_hashes
        if collect_replay:
            self._write_replay(
                decision, prompt_token_count, native_block_hashes
            )
        return self._adapt_decision(input, decision)


@lru_cache(maxsize=8)
def _configured_force_max_new_tokens(raw_config: str) -> int:
    unified_config = load_navi_sched_config({
        NAVI_SCHED_CONFIG_JSON_ENV: raw_config,
    })
    value = int(
        unified_config.get("client", {}).get("force_max_new_tokens", 0) or 0
    )
    if value < 0:
        raise ValueError("client.force_max_new_tokens must be nonnegative")
    if value > 0:
        logging.info(
            "NaviSched test request override enabled: force_max_new_tokens=%s",
            value,
        )
    return value


def apply_frontend_request_overrides(input, config=None) -> None:
    max_new_tokens = (
        config.force_max_new_tokens
        if config is not None
        else _configured_force_max_new_tokens(
            os.environ.get(NAVI_SCHED_CONFIG_JSON_ENV, "").strip()
        )
    )
    if max_new_tokens <= 0:
        return
    input.generate_config.max_new_tokens = max_new_tokens


def create_navi_sched_client(
    host_service, enable_agent_forward=True, config=None
):
    config = config or NaviSchedAdapterConfig.from_env()
    if not config.enabled:
        return None
    role_type = os.environ.get("ROLE_TYPE", "").strip().upper()
    if role_type and role_type != "FRONTEND":
        return None
    return NaviSchedClient(
        config,
        host_service=host_service,
        enable_agent_forward=enable_agent_forward,
    )
