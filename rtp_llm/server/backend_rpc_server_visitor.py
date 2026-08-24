import asyncio
import logging
import os
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, List, Optional, Set

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient, trans_input
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
from rtp_llm.utils.time_util import Timer

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs

route_logger = logging.getLogger("route_logger")
_STRIP_FRONTEND_STOP_TOKEN_IDS = "RTP_LLM_STRIP_FRONTEND_STOP_TOKEN_IDS"
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}


def _strip_frontend_stop_token_ids_enabled() -> bool:
    value = os.environ.get(_STRIP_FRONTEND_STOP_TOKEN_IDS)
    return value is not None and value.strip().lower() in _TRUE_VALUES


def _iter_ints(value: Any):
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
        return
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("-").isdigit():
            yield int(text)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_ints(item)


def _normalize_stop_ids(eos_token_id=None, stop_word_ids_list=None) -> List[List[int]]:
    stop_ids = [[token_id] for token_id in _iter_ints(eos_token_id)]
    for ids in stop_word_ids_list or []:
        seq = [int(token_id) for token_id in _iter_ints(ids)]
        if seq:
            stop_ids.append(seq)
    seen = set()
    result = []
    for seq in stop_ids:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            result.append(seq)
    return result


def _first_stop_index(token_ids: List[int], stop_ids: List[List[int]]) -> int:
    first = len(token_ids)
    for seq in stop_ids:
        for i in range(len(token_ids) - len(seq) + 1):
            if token_ids[i : i + len(seq)] == seq:
                first = min(first, i)
                break
    return first


def _pending_stop_prefix_len(token_ids: List[int], stop_ids: List[List[int]]) -> int:
    max_len = max((len(seq) for seq in stop_ids), default=1) - 1
    for suffix_len in range(min(max_len, len(token_ids)), 0, -1):
        suffix = token_ids[-suffix_len:]
        if any(
            len(seq) > suffix_len and seq[:suffix_len] == suffix for seq in stop_ids
        ):
            return suffix_len
    return 0


def _to_output_ids_tensor(token_ids: List[int], like: torch.Tensor) -> torch.Tensor:
    shape = list(like.shape) or [0]
    shape[-1] = len(token_ids)
    if token_ids:
        return torch.tensor(token_ids, dtype=like.dtype, device=like.device).reshape(
            shape
        )
    return torch.empty(shape, dtype=like.dtype, device=like.device)


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
        self.frontend_stop_word_ids_list: List[List[int]] = []
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

        host_args = HostServiceArgs.create_from_env()
        self.backend_role_list = self.get_backend_role_list(
            self.pd_sep_config, host_args, vit_separation
        )
        self.host_service = HostService(host_args)
        self.master_config = master_config
        self._page_rr_route_cache_keys = False
        self._page_rr_cp_size = 1
        if parallelism_config is not None:
            cp_config = prefill_cp_config or getattr(
                parallelism_config, "prefill_cp_config", None
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
        # Keep hash generation at the physical KV block granularity. Page-RR
        # routing samples canonical keys from this full logical-block key list;
        # it must not recompute request hashes with the virtual block size.
        full_block_cache_keys = get_block_cache_keys(token_ids, self.seq_size_per_block)
        block_cache_keys = self._route_cache_keys(full_block_cache_keys)
        self._report_recent_cache_key_metrics(block_cache_keys)
        input_pb = trans_input(input)

        try:
            route_result = await self.master_client.get_backend_role_addrs(
                block_cache_keys=block_cache_keys,
                cache_key_block_size=self._cache_key_block_size(),
                input=input,
                request_id=input.request_id,
                input_pb=input_pb,
            )
        except BaseException as e:
            exception_json = format_exception(e)
            metric_tags = dict(getattr(input, "frontend_metric_tags", {}) or {})
            metric_tags["error_code"] = exception_json.get("error_code_str", "")
            kmonitor.report(
                AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
                1,
                metric_tags,
            )
            raise

        if route_result.is_ok:
            input.generate_config.role_addrs = route_result.role_addrs
            input.enqueued_by_master = route_result.enqueued_by_master
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
        metric_tags = dict(getattr(input, "frontend_metric_tags", {}) or {})
        metric_tags["error_code"] = error_code
        kmonitor.report(
            AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
            1,
            metric_tags,
        )
        return route_result

    def _report_recent_cache_key_metrics(
        self,
        block_cache_keys: List[int],
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        try:
            snapshot = self.recent_cache_key_window.record(block_cache_keys)
            kmonitor.report(
                AccMetrics.RECENT_CACHE_KEY_REQUEST_COUNT_METRIC,
                1,
                tags or {},
            )
            if snapshot.request_occurrences <= 0:
                kmonitor.report(
                    AccMetrics.RECENT_CACHE_KEY_EMPTY_REQUEST_COUNT_METRIC,
                    1,
                    tags or {},
                )
            kmonitor.report(
                AccMetrics.RECENT_CACHE_KEY_HIT_COUNT_METRIC,
                snapshot.request_hit_occurrences,
                tags or {},
            )
            kmonitor.report(
                AccMetrics.RECENT_CACHE_KEY_TOTAL_COUNT_METRIC,
                snapshot.request_occurrences,
                tags or {},
            )
            kmonitor.report(
                GaugeMetrics.RECENT_CACHE_KEY_HIT_RATIO_METRIC,
                snapshot.request_hit_ratio,
                tags or {},
            )
        except Exception:
            route_logger.exception("failed to report recent cache key metrics")

    def _report_frontend_cache_key_metrics(self, input: GenerateInput) -> None:
        token_ids = input.token_ids.tolist()
        prompts = token_ids if len(input.token_ids.shape) == 2 else [token_ids]
        tags = dict(getattr(input, "frontend_metric_tags", {}) or {})
        for prompt_token_ids in prompts:
            full_cache_keys = get_block_cache_keys(
                prompt_token_ids,
                self.seq_size_per_block,
            )
            self._report_recent_cache_key_metrics(
                self._route_cache_keys(full_cache_keys),
                tags,
            )

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
        # speculative decoding does not support multiple returns or any fixed/
        # variable beam-search schedule.
        if (
            input.generate_config.num_return_sequences > 1
            or input.generate_config.has_num_beams()
        ):
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "speculative decoding does not support num_return_sequences > 1 or beam search",
            )
        # speculative decoding does not support return_all_probs
        if input.generate_config.return_all_probs:
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "speculative decoding does not support return_all_probs",
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

    def set_frontend_stop_word_ids(
        self, eos_token_id=None, stop_word_ids_list=None
    ) -> None:
        self.frontend_stop_word_ids_list = _normalize_stop_ids(
            eos_token_id=eos_token_id,
            stop_word_ids_list=stop_word_ids_list,
        )

    @staticmethod
    def strip_frontend_stop_word_ids(
        outputs: GenerateOutputs,
        stop_word_ids_list: List[List[int]],
        pending: dict[int, List[int]],
    ) -> GenerateOutputs:
        for index, output in enumerate(outputs.generate_outputs):
            if output.output_ids is None:
                continue
            token_ids = pending.pop(index, [])
            token_ids.extend(
                output.output_ids.detach().cpu().reshape(-1).int().tolist()
            )
            stop_index = _first_stop_index(token_ids, stop_word_ids_list)
            if output.finished or stop_index < len(token_ids):
                token_ids = token_ids[:stop_index]
            else:
                pending_len = _pending_stop_prefix_len(token_ids, stop_word_ids_list)
                if pending_len:
                    pending[index] = token_ids[-pending_len:]
                    token_ids = token_ids[:-pending_len]
            output.output_ids = _to_output_ids_tensor(token_ids, output.output_ids)
        return outputs

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

        # This is deliberately outside route_ips(): route retries must not
        # duplicate samples, and domain/explicit-address routes need the same
        # theoretical cache-hit metric as FlexLB master routing.
        self._report_frontend_cache_key_metrics(input)

        async def route_and_enqueue(attempt_input: GenerateInput):
            if self.host_service.service_available:
                await self.route_ips(attempt_input)
            return self.model_rpc_client.enqueue(attempt_input)

        async def stream_with_aux_info():
            attempt = 0
            strip_stop_ids: List[List[int]] = []
            if _strip_frontend_stop_token_ids_enabled():
                request_stop_ids = getattr(
                    input.generate_config, "stop_words_list", None
                )
                strip_stop_ids = _normalize_stop_ids(
                    stop_word_ids_list=(
                        self.frontend_stop_word_ids_list + list(request_stop_ids or [])
                    )
                )
            attempt_input = input
            is_streaming = bool(getattr(input.generate_config, "is_streaming", False))

            def observe_backend_output(output: Any, output_attempt: int) -> None:
                observer = getattr(input, "frontend_metric_observer", None)
                if observer is None:
                    return
                try:
                    observer(output, output_attempt)
                except Exception:
                    # Metrics are best-effort and must never change inference
                    # response or retry behavior.
                    logging.exception("failed to observe raw backend frontend metrics")

            first_exc: Optional[BaseException] = None
            while True:
                pending_stop_prefix: dict[int, List[int]] = {}
                yielded_output = False
                try:
                    stream = await route_and_enqueue(attempt_input)
                    if is_streaming:
                        async for output in stream:
                            observe_backend_output(output, attempt)
                            if bool(getattr(output, "frontend_metric_only", False)):
                                continue
                            yielded_output = True
                            if strip_stop_ids:
                                output = self.strip_frontend_stop_word_ids(
                                    output, strip_stop_ids, pending_stop_prefix
                                )
                            yield output
                    else:
                        buffered_outputs = []
                        async for output in stream:
                            observe_backend_output(output, attempt)
                            if bool(getattr(output, "frontend_metric_only", False)):
                                continue
                            if strip_stop_ids:
                                output = self.strip_frontend_stop_word_ids(
                                    output, strip_stop_ids, pending_stop_prefix
                                )
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


def create_backend_rpc_server_visitor(
    py_env_configs: "PyEnvConfigs",
    model_config,
    source_role: str = "frontend",
) -> "BackendRPCServerVisitor":
    """Build a `BackendRPCServerVisitor` from `PyEnvConfigs` + a lightweight `ModelConfig`.

    Used by both `FrontendWorker` (historically inline) and `DashScApp` (independent
    process) so they open equivalent channels to the backend without dragging in the
    tokenizer/pipeline machinery. `model_config` only needs `max_seq_len` and
    `attn_config.tokens_per_block`; produce it via `ModelFactory.create_model_config`.
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
    )
