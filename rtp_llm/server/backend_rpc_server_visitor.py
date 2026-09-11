import asyncio
import logging
import os
import time
from dataclasses import replace
from typing import TYPE_CHECKING, AsyncGenerator, Callable, List, Optional, Set

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient, trans_input
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.ops import SpeculativeExecutionConfig, VitSeparation, get_block_cache_keys
from rtp_llm.server.cache_key_routing import route_cache_keys_for_page_rr
from rtp_llm.server.host_service import HostService, HostServiceArgs
from rtp_llm.server.master_client import FlexlbResponse, MasterClient
from rtp_llm.server.misc import format_exception
from rtp_llm.server.recent_cache_key_window import RecentCacheKeyWindow
from rtp_llm.server.request_headers import (
    extract_correlation_request_id,
    extract_trace_id,
)
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutputs,
    RequestInfo,
)
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool
from rtp_llm.utils.time_util import Timer

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs

route_logger = logging.getLogger("route_logger")


def get_role_names(role_addrs: List[RoleAddr]) -> Set[str]:
    """Return the set of human-readable role names from a list of RoleAddr."""
    return {role_addr.role.name for role_addr in role_addrs}


PD_ROUTE_RETRY_ON_UNAVAILABLE_ENV = "RTP_LLM_PD_ROUTE_RETRY_ON_UNAVAILABLE"
DEFAULT_PD_ROUTE_RETRY_ON_UNAVAILABLE = 3
_TERMINAL_ROUTE_EXCEPTION_TYPES = frozenset(
    {
        ExceptionType.PRIORITY_PREEMPTED,
        ExceptionType.PRIORITY_ADMISSION_REJECTED,
        ExceptionType.RESOURCE_EXHAUSTED,
        ExceptionType.ADMISSION_UNAVAILABLE,
        # A scheduling deadline is already a completed admission outcome.
        # Retrying it with a new request id silently starts a second admission
        # attempt with a fresh identity and can hide the original timeout.
        ExceptionType.BATCH_SLO_EXPIRED,
    }
)


class BackendRPCServerVisitor:
    def __init__(
        self,
        max_seq_len: int,  # max_seq_len_ from ModelConfig
        seq_size_per_block: int,  # seq_size_per_block_ from ModelConfig
        pd_sep_config,  # PDSepConfig from ops
        addresses: list[str],  # RPC addresses for data parallel communication
        sp_config: Optional[SpeculativeExecutionConfig] = None,
        grpc_config=None,  # Optional GrpcConfig
        vit_separation: Optional[VitSeparation] = None,  # Optional VitSeparation
        server_config=None,
        master_config=None,
        parallelism_config=None,
        prefill_cp_config=None,
        source_role: str = "frontend",
        dsv4_image_token_id: Optional[int] = None,
    ) -> None:
        """Initialize BackendRPCServerVisitor.

        Args:
            max_seq_len: Maximum sequence length from ModelConfig
            seq_size_per_block: Sequence size per block from ModelConfig
            pd_sep_config: PDSepConfig from ops
            addresses: List of RPC addresses for data parallel communication
            sp_config: Optional SpeculativeExecutionConfig
            grpc_config: Optional GrpcConfig for client configuration
            vit_separation: Optional VitSeparation for multimodal models
            server_config: Optional ServerConfig for master configuration
            master_config: Optional MasterConfig for master client configuration
            parallelism_config: Optional ParallelismConfig for page-RR route cache keys
            prefill_cp_config: Optional PrefillCPConfig for page-RR route cache keys
            source_role: Caller role used for request-info correlation fields.
        """
        self.max_seq_len = max_seq_len
        self.seq_size_per_block = seq_size_per_block
        self.pd_sep_config = pd_sep_config
        self.sp_config = sp_config
        self.source_role = source_role
        self.dsv4_image_token_id = (
            dsv4_image_token_id
            if vit_separation == VitSeparation.VIT_SEPARATION_REMOTE
            else None
        )
        self.source_ip = str(getattr(server_config, "ip", "") or "")
        assert self.max_seq_len > 0

        # Get max_rpc_timeout_ms and decode_entrance from pd_sep_config
        max_rpc_timeout_ms = pd_sep_config.max_rpc_timeout_ms
        decode_entrance = pd_sep_config.decode_entrance

        # Get client_config from grpc_config if provided, otherwise use empty dict
        if grpc_config is not None:
            client_config = grpc_config.get_client_config()
        else:
            client_config = {}

        self.model_rpc_client = ModelRpcClient(
            addresses=addresses,
            client_config=client_config,
            max_rpc_timeout_ms=max_rpc_timeout_ms,
            decode_entrance=decode_entrance,
        )
        self._vit_channel_pool = GrpcHostChannelPool(
            options=list(client_config.items())
        )

        host_args = HostServiceArgs.create_from_env()
        self.backend_role_list = self.get_backend_role_list(
            self.pd_sep_config, host_args, vit_separation
        )
        self.host_service = HostService(host_args)
        self.master_config = master_config
        self._page_rr_route_cache_keys = False
        self._page_rr_cp_size = 1
        self._prefill_cp_active = False
        if parallelism_config is not None:
            cp_config = prefill_cp_config or getattr(
                parallelism_config, "prefill_cp_config", None
            )
            self._prefill_cp_active = bool(
                cp_config and (cp_config.is_enabled() or cp_config.is_prefill_enabled())
            )
            tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)
            kv_cache_sharded = bool(getattr(cp_config, "kv_cache_sharded", False))
            if kv_cache_sharded and tp_size > 1:
                self._page_rr_route_cache_keys = True
                self._page_rr_cp_size = tp_size
        self.master_client = MasterClient(
            host_service=self.host_service,
            server_config=server_config,
            master_config=master_config,
        )
        self.recent_cache_key_window = RecentCacheKeyWindow()
        self.pd_route_retry_on_unavailable = self._pd_route_retry_on_unavailable()
        self.request_id_factory: Optional[Callable[[], int]] = None

    async def close(self):
        await self.model_rpc_client.close()
        await self.master_client.close()
        await self._vit_channel_pool.close()

    def set_request_id_factory(self, factory: Callable[[], int]) -> None:
        self.request_id_factory = factory

    @staticmethod
    def _pd_route_retry_on_unavailable() -> int:
        raw = os.environ.get(PD_ROUTE_RETRY_ON_UNAVAILABLE_ENV, "")
        if not raw:
            return DEFAULT_PD_ROUTE_RETRY_ON_UNAVAILABLE
        try:
            return max(0, int(raw))
        except ValueError:
            route_logger.warning(
                "Invalid %s=%r, falling back to default retry count %s",
                PD_ROUTE_RETRY_ON_UNAVAILABLE_ENV,
                raw,
                DEFAULT_PD_ROUTE_RETRY_ON_UNAVAILABLE,
            )
            return DEFAULT_PD_ROUTE_RETRY_ON_UNAVAILABLE

    @staticmethod
    def _is_retryable_route_rpc_error(e: BaseException) -> bool:
        # Use isinstance instead of getattr duck-typing — only FtRuntimeException
        # carries exception_type; gRPC RpcError and other exceptions do not.
        if isinstance(e, FtRuntimeException):
            try:
                exception_type = int(e.exception_type)
                # These are completed admission decisions, not transient route
                # transport failures.  A new request id would change request
                # identity and hide the typed 429 result selected by Master.
                if any(
                    exception_type == int(terminal_type)
                    for terminal_type in _TERMINAL_ROUTE_EXCEPTION_TYPES
                ):
                    return False
                return exception_type >= 8000
            except (TypeError, ValueError):
                pass
        text = str(e)
        return (
            "StatusCode.UNAVAILABLE" in text
            or "grpc_status:14" in text
            or "recvmsg:Connection timed out" in text
            or "Socket closed" in text
        )

    @staticmethod
    def get_backend_role_list(
        pd_sep_config,
        host_args: HostServiceArgs,
        vit_separation: Optional[VitSeparation] = None,
    ) -> List[RoleType]:
        logging.info(f"pd_sep_config: {pd_sep_config.to_string()}")
        role_list: List[RoleType] = []

        if (
            vit_separation == VitSeparation.VIT_SEPARATION_REMOTE
            and host_args.vit_domain
        ):
            role_list.append(RoleType.VIT)
            logging.info("Added VIT role")

        config_role_type = pd_sep_config.role_type

        if config_role_type == RoleType.PREFILL and not pd_sep_config.decode_entrance:
            role_list.append(RoleType.DECODE)
            logging.info("Added DECODE role for PREFILL type")
        elif config_role_type == RoleType.DECODE and pd_sep_config.decode_entrance:
            role_list.append(RoleType.PREFILL)
            logging.info("Added PREFILL role for DECODE type")
        elif config_role_type == RoleType.FRONTEND:
            logging.info(
                f"Checking FRONTEND roles: decode_domain={host_args.decode_domain}, prefill_domain={host_args.prefill_domain}, pdfusion_domain={host_args.pdfusion_domain}"
            )
            if host_args.decode_domain:
                role_list.append(RoleType.DECODE)
                logging.info("Added DECODE role for FRONTEND type")
            if host_args.prefill_domain:
                role_list.append(RoleType.PREFILL)
                logging.info("Added PREFILL role for FRONTEND type")
            if host_args.pdfusion_domain:
                role_list.append(RoleType.PDFUSION)
                logging.info("Added PDFUSION role for FRONTEND type")

        logging.info(f"configured backend role list: {role_list}")
        return role_list

    async def get_master_route_addrs(
        self, input: GenerateInput
    ) -> Optional[FlexlbResponse]:
        """
        Resolve role addrs from FlexLB master (and slave on connection failure).
        Returns None on success; on failure returns FlexlbResponse for routing decisions.
        request_id is frontend-generated and is not overwritten.
        """
        token_ids = (
            input.token_ids.tolist()[0]
            if len(input.token_ids.shape) == 2
            else input.token_ids.tolist()
        )
        selected_vit = None
        routing_input = input
        schedule_kwargs = {}
        try:
            image_token_id = getattr(self, "dsv4_image_token_id", None)
            if image_token_id is not None and input.mm_inputs:
                timeout_ms = input.generate_config.ttft_timeout_ms
                if timeout_ms is None or timeout_ms <= 0:
                    timeout_ms = input.generate_config.timeout_ms
                if timeout_ms is None or timeout_ms <= 0:
                    timeout_ms = (
                        self.master_client.master_config.master_default_timeout_ms
                    )
                deadline = time.monotonic() + timeout_ms / 1000.0

                def remaining_timeout():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise FtRuntimeException(
                            ExceptionType.DEADLINE_EXCEEDED,
                            "ViT and P/D routing deadline exceeded",
                        )
                    return remaining

                image_locs = [
                    i
                    for i, token_id in enumerate(token_ids)
                    if token_id == image_token_id
                ]
                if len(image_locs) != len(input.mm_inputs):
                    raise FtRuntimeException(
                        ExceptionType.INVALID_PARAMS,
                        "DeepSeek image placeholders and inputs must match",
                    )
                for i, (loc, mm_input) in enumerate(zip(image_locs, input.mm_inputs)):
                    phase = loc if i == 0 else 1 + loc - (image_locs[i - 1] + 1)
                    mm_input.config = replace(
                        mm_input.config, image_block_start_mod4=phase % 4
                    )

                vit_route = await self.master_client.get_backend_role_addrs(
                    block_cache_keys=[],
                    cache_key_block_size=self._cache_key_block_size(),
                    input=input,
                    request_id=input.request_id,
                    vit_only=True,
                    timeout_s=remaining_timeout(),
                )
                if not vit_route.is_ok:
                    return vit_route
                selected_vit = vit_route.role_addrs[0]
                raw_input = replace(
                    input,
                    generate_config=input.generate_config.model_copy(
                        update={"role_addrs": [selected_vit]}
                    ),
                )
                input_pb = trans_input(raw_input)
                image_token_ids = await self._get_vit_token_ids(
                    selected_vit, input_pb, remaining_timeout()
                )
                expanded = []
                previous_end = 0
                for loc, image_ids in zip(image_locs, image_token_ids):
                    expanded.extend(token_ids[previous_end:loc])
                    expanded.extend(image_ids)
                    previous_end = loc + 1
                expanded.extend(token_ids[previous_end:])
                if len(expanded) >= self.max_seq_len:
                    raise FtRuntimeException(
                        ExceptionType.LONG_PROMPT_ERROR,
                        "DeepSeek image routing length exceeds max_seq_len",
                    )
                # Only the routing estimate uses these IDs. Enqueue receives
                # raw tokens/URLs and the selected ViT; P hashes actual features.
                routing_input = replace(
                    raw_input, token_ids=torch.tensor(expanded, dtype=torch.int32)
                )
                token_ids = expanded
                route_logger.info(
                    "ViT routing metadata request_id=%s vit=%s raw_tokens=%s routing_tokens=%s",
                    input.request_id,
                    selected_vit.ip,
                    raw_input.prompt_length,
                    len(expanded),
                )
                schedule_kwargs["timeout_s"] = remaining_timeout()
            else:
                input_pb = trans_input(input)

            # Hash physical blocks before page-RR samples canonical route keys.
            full_block_cache_keys = get_block_cache_keys(
                token_ids, self.seq_size_per_block
            )
            block_cache_keys = self._route_cache_keys(full_block_cache_keys)
            self._report_recent_cache_key_metrics(block_cache_keys)
            route_result = await self.master_client.get_backend_role_addrs(
                block_cache_keys=block_cache_keys,
                cache_key_block_size=self._cache_key_block_size(),
                input=routing_input,
                request_id=input.request_id,
                input_pb=input_pb,
                **schedule_kwargs,
            )
        except BaseException as e:
            exception_json = format_exception(e)
            kmonitor.report(
                AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
                1,
                {"error_code": exception_json.get("error_code_str", "")},
            )
            raise

        if route_result.is_ok:
            input.generate_config.role_addrs = (
                [selected_vit]
                + [
                    addr
                    for addr in route_result.role_addrs
                    if addr.role != RoleType.VIT
                ]
                if selected_vit is not None
                else route_result.role_addrs
            )
            input.enqueued_by_master = route_result.enqueued_by_master
            if selected_vit is not None:
                route_logger.info(
                    "ViT routing P/D scheduled request_id=%s vit=%s enqueued_by_master=%s",
                    input.request_id,
                    selected_vit.ip,
                    input.enqueued_by_master,
                )
            route_logger.debug(
                "master route success, request_id=%s, addrs=%s",
                input.request_id,
                route_result.role_addrs,
            )
            kmonitor.report(AccMetrics.MASTER_ROUTE_QPS_METRIC, 1)
            return None

        route_logger.error(
            "master route failed, request_id=%s, connection_failed=%s, error_code=%s, error_message=%s",
            input.request_id,
            route_result.connection_failed,
            route_result.error_code,
            route_result.error_message or "",
        )
        if route_result.connection_failed:
            route_error = FtRuntimeException(
                ExceptionType.GET_CONNECTION_FAILED,
                "FlexLB master and slave connections failed",
            )
            error_code = format_exception(route_error)["error_code_str"]
        elif route_result.error_code is not None:
            try:
                route_error = FtRuntimeException(
                    ExceptionType(route_result.error_code),
                    route_result.error_message or "master route failed",
                )
                error_code = format_exception(route_error)["error_code_str"]
            except ValueError:
                error_code = str(route_result.error_code)
        else:
            route_error = FtRuntimeException(
                ExceptionType.ROUTE_ERROR,
                route_result.error_message or "master route failed",
            )
            error_code = format_exception(route_error)["error_code_str"]
        kmonitor.report(
            AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
            1,
            {"error_code": error_code},
        )
        return route_result

    async def _get_vit_token_ids(self, vit: RoleAddr, input_pb, timeout_s: float):
        request = MultimodalInputsPB(metadata_only=True)
        request.multimodal_inputs.extend(input_pb.multimodal_inputs)
        channel = await self._vit_channel_pool.get(f"{vit.ip}:{vit.grpc_port}")
        response = await MultimodalRpcServiceStub(channel).RemoteMultimodalEmbedding(
            request, timeout=timeout_s
        )
        outputs = response.multimodal_outputs
        if len(outputs) != len(request.multimodal_inputs) or any(
            not output.token_ids for output in outputs
        ):
            raise FtRuntimeException(
                ExceptionType.ROUTE_ERROR,
                "ViT metadata must contain non-empty token IDs for every image",
            )
        return [list(output.token_ids) for output in outputs]

    def _report_recent_cache_key_metrics(self, block_cache_keys: List[int]) -> None:
        try:
            snapshot = self.recent_cache_key_window.record(block_cache_keys)
            kmonitor.report(
                AccMetrics.RECENT_CACHE_KEY_REQUEST_COUNT_METRIC,
                1,
            )
            if snapshot.request_occurrences <= 0:
                kmonitor.report(
                    AccMetrics.RECENT_CACHE_KEY_EMPTY_REQUEST_COUNT_METRIC,
                    1,
                )
            kmonitor.report(
                AccMetrics.RECENT_CACHE_KEY_HIT_COUNT_METRIC,
                snapshot.request_hit_occurrences,
            )
            kmonitor.report(
                AccMetrics.RECENT_CACHE_KEY_TOTAL_COUNT_METRIC,
                snapshot.request_occurrences,
            )
            kmonitor.report(
                GaugeMetrics.RECENT_CACHE_KEY_HIT_RATIO_METRIC,
                snapshot.request_hit_ratio,
            )
        except Exception:
            route_logger.exception("failed to report recent cache key metrics")

    def _route_cache_keys(self, block_cache_keys: List[int]) -> List[int]:
        return route_cache_keys_for_page_rr(
            block_cache_keys, self._page_rr_route_cache_keys, self._page_rr_cp_size
        )

    def _cache_key_block_size(self) -> int:
        if self._page_rr_route_cache_keys and self._page_rr_cp_size > 1:
            return self.seq_size_per_block * self._page_rr_cp_size
        return self.seq_size_per_block

    async def get_domain_route_addrs(self, input: GenerateInput):
        specified_roles = {addr.role for addr in input.generate_config.role_addrs}
        missing_roles = [
            role for role in self.backend_role_list if role not in specified_roles
        ]
        role_addrs: List[RoleAddr] = self.host_service.get_backend_role_addrs(
            missing_roles
        )
        if role_addrs:
            input.generate_config.role_addrs.extend(role_addrs)
            route_logger.warning(
                "fallback to host service, request_id=%s, addrs=%s",
                input.request_id,
                role_addrs,
            )
            kmonitor.report(
                AccMetrics.DOMAIN_ROUTE_QPS_METRIC,
                1,
            )
        else:
            route_logger.error(
                "host service failed, request_id=%s, missing_roles=%s",
                input.request_id,
                missing_roles,
            )

    async def route_ips(self, input: GenerateInput):
        # proactive rejection: check cached queue length before making request to master
        if self.master_config:
            threshold = self.master_config.master_queue_reject_threshold
            queue_length = self.host_service.get_queue_length()
            if queue_length > threshold:
                route_logger.warning(
                    f"FlexLb cached queue length {queue_length} exceeds threshold "
                    f"{threshold}, "
                    f"proactively rejecting request <{input.request_id}>"
                )
                kmonitor.report(AccMetrics.MASTER_QUEUE_REJECT_QPS_METRIC, 1)
                raise FtRuntimeException(
                    exception_type=ExceptionType.TRAFFIC_LIMIT_ERROR,
                    message=f"Flexlb queue length {queue_length} exceeds threshold {threshold}",
                )
        with Timer() as route_timer:
            role_addrs_specified = bool(input.generate_config.role_addrs)
            master_addr = self.host_service.get_master_addr()
            route_logger.debug("routing to master: %s", master_addr)

            input_token_batched = False
            if len(input.token_ids.shape) == 2 and input.token_ids.size(0) != 1:
                input_token_batched = True

            master_route_result: Optional[FlexlbResponse] = None
            if not role_addrs_specified and master_addr and not input_token_batched:
                with Timer() as master_route_timer:
                    master_route_result = await self.get_master_route_addrs(input)
                kmonitor.report(
                    GaugeMetrics.MASTER_ROUTE_RT_METRIC, master_route_timer.cost_ms()
                )
            elif not role_addrs_specified:
                route_logger.warning(
                    "master address: %s or input token batched: %s is not valid, fallback to domain routing",
                    master_addr,
                    input_token_batched,
                )
            specified_roles = {addr.role for addr in input.generate_config.role_addrs}
            need_domain_routing = not set(self.backend_role_list).issubset(
                specified_roles
            )
            allow_domain_fallback = master_route_result is None or (
                master_route_result.connection_failed
            )
            if (
                not input.generate_config.role_addrs or need_domain_routing
            ) and allow_domain_fallback:
                with Timer() as domain_route_timer:
                    await self.get_domain_route_addrs(input)
                kmonitor.report(
                    GaugeMetrics.DOMAIN_ROUTE_RT_METRIC, domain_route_timer.cost_ms()
                )
            route_logger.debug("routing to master done")

        kmonitor.report(GaugeMetrics.ROUTE_RT_METRIC, route_timer.cost_ms())
        if not input.generate_config.role_addrs:
            route_error = FtRuntimeException(
                ExceptionType.ROUTE_ERROR,
                "request_id=%s no backend role addresses found after routing"
                % input.request_id,
            )
            if (
                master_route_result is not None
                and not master_route_result.is_ok
                and master_route_result.error_code is not None
            ):
                route_error.rtp_error_code = master_route_result.error_code
            raise route_error

    def check_sp_supported(self, input: GenerateInput):
        if not self.sp_config or not self.sp_config.model_type:
            return
        if input.generate_config.force_disable_sp_run:
            return

        # speculative decoding does not support batched input
        if len(input.token_ids.shape) == 2 and input.token_ids.size(0) != 1:
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "speculative decoding does not support batched input",
            )
        # speculative decoding does not support num_return_sequences > 1 or num_beams > 1
        if (
            input.generate_config.num_return_sequences > 1
            or input.generate_config.num_beams > 1
        ):
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "speculative decoding does not support num_return_sequences > 1 or num_beams > 1",
            )
        # speculative decoding does not support return_all_probs
        if input.generate_config.return_all_probs:
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "speculative decoding does not support return_all_probs",
            )

    def check_prefill_cp_supported(self, input: GenerateInput) -> None:
        if not self._prefill_cp_active:
            return

        unsupported = []
        if input.generate_config.calculate_loss:
            unsupported.append("calculate_loss")
        if input.generate_config.return_all_hidden_states:
            unsupported.append("return_all_hidden_states")
        if unsupported:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS,
                "prefill context parallelism does not support request option(s): "
                + ", ".join(unsupported),
            )

    def fill_request_info(self, input: GenerateInput) -> None:
        if getattr(input, "request_info", None) is None:
            input.request_info = RequestInfo()

        request_info = input.request_info
        if not request_info.source_role:
            request_info.source_role = self.source_role

        source_role = (request_info.source_role or self.source_role).lower()
        if source_role == "dash":
            if not request_info.dash_ip:
                request_info.dash_ip = self.source_ip
        elif not request_info.frontend_ip:
            request_info.frontend_ip = self.source_ip

        trace_id = str(
            getattr(input.generate_config, "trace_id", "")
            or extract_trace_id(getattr(input, "headers", None))
            or ""
        )
        if not request_info.trace_id:
            request_info.trace_id = trace_id
        if not getattr(input.generate_config, "trace_id", "") and request_info.trace_id:
            input.generate_config.trace_id = request_info.trace_id

        if not request_info.request_id:
            request_info.request_id = (
                extract_correlation_request_id(getattr(input, "headers", None))
                or request_info.trace_id
                or str(input.request_id)
            )

    @torch.inference_mode()
    async def enqueue(
        self, input: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        def set_aux_info(e: BaseException) -> None:
            if getattr(e, "aux_info", None):
                return
            aux_info = {
                "input_len": input.prompt_length,
                "output_len": 0,
                "step_output_len": 0,
                "reuse_len": 0,
            }
            role_addrs = input.generate_config.role_addrs or []
            if role_addrs:
                aux_info["role_addrs"] = [
                    role_addr.model_dump(mode="json") for role_addr in role_addrs
                ]
                roles = {
                    str(getattr(role_addr.role, "name", role_addr.role))
                    for role_addr in role_addrs
                }
                aux_info["pd_sep"] = {"PREFILL", "DECODE"}.issubset(roles)
            e.aux_info = aux_info

        try:
            self.fill_request_info(input)
            input.generate_config.validate()
            if input.prompt_length <= 0:
                raise FtRuntimeException(
                    ExceptionType.LONG_PROMPT_ERROR,
                    f"model tokens can not be empty, request length is {input.prompt_length}",
                )

            self.check_sp_supported(input)
            self.check_prefill_cp_supported(input)

            max_new_tokens = min(
                self.max_seq_len - input.prompt_length,
                input.generate_config.max_new_tokens,
            )
            if max_new_tokens <= 0:
                raise FtRuntimeException(
                    ExceptionType.LONG_PROMPT_ERROR,
                    f"model max tokens is {self.max_seq_len}, "
                    f"request length is {input.prompt_length}, max_new_tokens is {max_new_tokens}",
                )

        except BaseException as e:
            set_aux_info(e)
            raise

        async def route_and_enqueue(attempt_input: GenerateInput):
            if self.host_service.service_available:
                await self.route_ips(attempt_input)
            return self.model_rpc_client.enqueue(attempt_input)

        async def stream_with_aux_info():
            attempt = 0
            attempt_input = input
            is_streaming = bool(getattr(input.generate_config, "is_streaming", False))
            first_exc: Optional[BaseException] = None
            while True:
                yielded_output = False
                try:
                    stream = await route_and_enqueue(attempt_input)
                    if is_streaming:
                        async for output in stream:
                            yielded_output = True
                            yield output
                    else:
                        buffered_outputs = []
                        async for output in stream:
                            buffered_outputs.append(output)
                        yielded_output = True
                        for output in buffered_outputs:
                            yield output
                    return
                except BaseException as e:
                    set_aux_info(e)
                    if first_exc is None:
                        first_exc = e
                    if (
                        yielded_output
                        or attempt >= self.pd_route_retry_on_unavailable
                        or not self._is_retryable_route_rpc_error(e)
                    ):
                        # After retries, re-raise the ORIGINAL exception to
                        # preserve its error category (e.g. CAPACITY->429 from
                        # the first route failure).  A later attempt may have
                        # hit a different error (e.g. a model-RPC
                        # INTERNAL->500) whose category does not reflect the
                        # root cause; the caller should see the original
                        # exception so that servicer/frontend maps it to the
                        # correct HTTP status code.
                        # A later terminal admission decision is authoritative;
                        # do not hide it behind an earlier retryable transport
                        # or legacy capacity failure.
                        is_terminal_route_decision = (
                            isinstance(e, FtRuntimeException)
                            and e.exception_type in _TERMINAL_ROUTE_EXCEPTION_TYPES
                        )
                        if (
                            first_exc is not None
                            and first_exc is not e
                            and not is_terminal_route_decision
                        ):
                            raise first_exc
                        raise
                    request_id_factory = getattr(self, "request_id_factory", None)
                    if request_id_factory is None:
                        raise
                    attempt += 1
                    attempt_input = replace(
                        input,
                        request_id=request_id_factory(),
                        generate_config=input.generate_config.model_copy(
                            update={"role_addrs": []}
                        ),
                        enqueued_by_master=False,
                    )
                    route_logger.warning(
                        "retrying PD route after retryable RPC error, "
                        "request_id=%s, attempt=%s/%s, error=%s",
                        attempt_input.request_id,
                        attempt,
                        self.pd_route_retry_on_unavailable,
                        e,
                    )
                    await asyncio.sleep(min(0.2, 0.05 * attempt))

        return stream_with_aux_info()

    def is_backend_service_ready(self, refresh: bool = False) -> bool:
        roles: List[RoleAddr] = self.host_service.get_backend_role_addrs(
            self.backend_role_list, refresh
        )
        if not roles:
            return False
        for role in self.backend_role_list:
            if role not in [r.role for r in roles]:
                logging.warning(f"role {role} not in available roles {roles}")
                return False
        return True


def get_dsv4_image_token_id(model_config) -> Optional[int]:
    if getattr(model_config, "model_type", None) == "deepseek_v4":
        mm_params = model_config.mm_related_params
        if mm_params.config.get("vision_n_layers", 0) > 0:
            return mm_params.special_token_ids["image_token_id"]
    return None


def create_backend_rpc_server_visitor(
    py_env_configs: "PyEnvConfigs",
    model_config,
    source_role: str = "frontend",
) -> "BackendRPCServerVisitor":
    """Build a `BackendRPCServerVisitor` from `PyEnvConfigs` + a lightweight `ModelConfig`.

    Used by `DashScApp` without tokenizer/pipeline initialization. `FrontendWorker`
    constructs its visitor through `Pipeline`; both paths share the image-token
    configuration helper. Produce `model_config` via `ModelFactory.create_model_config`.
    """
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.distribute.distributed_server import (
        get_dp_addrs_from_world_info,
        get_world_info,
    )

    engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)
    world_info = get_world_info(
        server_config=py_env_configs.server_config,
        distribute_config=py_env_configs.distribute_config,
        parallelism_config=py_env_configs.parallelism_config,
    )
    addresses = get_dp_addrs_from_world_info(
        world_info=world_info,
        parallelism_config=engine_config.parallelism_config,
    )
    vit_separation = None
    if py_env_configs.vit_config:
        vit_separation = py_env_configs.vit_config.vit_separation

    return BackendRPCServerVisitor(
        max_seq_len=model_config.max_seq_len,
        seq_size_per_block=model_config.attn_config.tokens_per_block,
        pd_sep_config=engine_config.pd_sep_config,
        addresses=addresses,
        sp_config=py_env_configs.sp_config,
        grpc_config=py_env_configs.grpc_config,
        vit_separation=vit_separation,
        server_config=py_env_configs.server_config,
        master_config=py_env_configs.master_config,
        parallelism_config=engine_config.parallelism_config,
        prefill_cp_config=py_env_configs.prefill_cp_config,
        source_role=source_role,
        dsv4_image_token_id=get_dsv4_image_token_id(model_config),
    )
