import logging
import random
import time
from typing import AsyncGenerator, List, Optional

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
from rtp_llm.ops import SpeculativeExecutionConfig, VitSeparation
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.ops import get_block_cache_keys
from rtp_llm.server.host_service import HostService, HostServiceArgs
from rtp_llm.server.master_client import MasterClient
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs
from rtp_llm.utils.time_util import Timer

route_logger = logging.getLogger("route_logger")


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
"""
        self.max_seq_len = max_seq_len
        self.seq_size_per_block = seq_size_per_block
        self.pd_sep_config = pd_sep_config
        self.sp_config = sp_config
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
        self.master_client = MasterClient(
            host_service=self.host_service,
            server_config=server_config,
            master_config=master_config
        )

    @staticmethod
    def get_backend_role_list(
        pd_sep_config, host_args: HostServiceArgs, vit_separation: Optional[VitSeparation] = None
    ) -> List[RoleType]:
        logging.info(f"pd_sep_config: {pd_sep_config.to_string()}")
        role_list: List[RoleType] = []

        if vit_separation == VitSeparation.VIT_SEPARATION_REMOTE and host_args.vit_domain:
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

    async def get_master_route_addrs(self, input: GenerateInput):
        token_ids = []
        if len(input.token_ids.shape) == 2:
            token_ids: List[int] = input.token_ids.tolist()[0]  # type: ignore
        else:
            token_ids: List[int] = input.token_ids.tolist()  # type: ignore
        block_cache_keys = get_block_cache_keys(
            token_ids, self.seq_size_per_block
        )
        start = time.time()
        request_id = random.randint(1, 2**63 - 1)

        try:
            role_addrs, inter_request_id = (
                await self.master_client.get_backend_role_addrs(
                    block_cache_keys=block_cache_keys,
                    input=input,
                    request_id=request_id
                )
            )
        except BaseException as e:
            exception_json = format_exception(e)
            error_code_str = exception_json.get("error_code_str", "")
            kmonitor.report(
                AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
                1,
                {"error_code": error_code_str},
            )
            logging.error(f"master route failed, request <{request_id}> {exception_json}, start={start}, rt={time.time() - start}")
            raise e

        if not role_addrs:
            route_logger.error(
                f"master route failed, request <{input.request_id}> no role addresses returned"
            )
        else:
            input.generate_config.role_addrs = role_addrs
            input.generate_config.inter_request_id = inter_request_id
            if inter_request_id != -1:
                input.request_id = inter_request_id
            route_logger.debug(
                f"master route success, request <{input.request_id}> route to address: {role_addrs}, inter_request_id: {inter_request_id}"
            )
            kmonitor.report(AccMetrics.MASTER_ROUTE_QPS_METRIC, 1)

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
                f"fallback to host service, request <{input.request_id}> route to address: {role_addrs}"
            )
            kmonitor.report(
                AccMetrics.DOMAIN_ROUTE_QPS_METRIC,
                1,
            )
        else:
            route_logger.error(f"host service failed, request <{input.request_id}>, missing roles: {missing_roles}")

    async def route_ips(self, input: GenerateInput):
        # proactive rejection: check cached queue length before making request to master
        queue_length = self.host_service.get_queue_length()
        if queue_length > StaticConfig.master_config.master_queue_reject_threshold:
            route_logger.warning(
                f"FlexLb cached queue length {queue_length} exceeds threshold "
                f"{StaticConfig.master_config.master_queue_reject_threshold}, "
                f"proactively rejecting request <{input.request_id}>"
            )
            raise FtRuntimeException(
                exception_type=ExceptionType.TRAFFIC_LIMIT_ERROR,
                message=f"Flexlb queue length {queue_length} exceeds threshold {StaticConfig.master_config.master_queue_reject_threshold}",
            )
        with Timer() as route_timer:
            # Check if role_addrs is already specified in the request
            role_addrs_specified = bool(input.generate_config.role_addrs)

            # master don't support schedule batched input yet
            input_token_batched = False
            if len(input.token_ids.shape) == 2 and input.token_ids.size(0) != 1:
                input_token_batched = True

            # Only get route from master if role_addrs is not specified
            if not role_addrs_specified and not input_token_batched:
                with Timer() as master_route_timer:
                    await self.get_master_route_addrs(input)
                kmonitor.report(
                    GaugeMetrics.MASTER_ROUTE_RT_METRIC, master_route_timer.cost_ms()
                )
            elif not role_addrs_specified:
                route_logger.warning(
                    f"input token batched: {input_token_batched} is not valid, fallback to domain routing"
                )
            specified_roles = {addr.role for addr in input.generate_config.role_addrs}
            # 预先计算是否需要调用
            need_domain_routing = not set(self.backend_role_list).issubset(
                specified_roles
            )
            if not input.generate_config.role_addrs or need_domain_routing:
                with Timer() as domain_route_timer:
                    await self.get_domain_route_addrs(input)
                kmonitor.report(
                    GaugeMetrics.DOMAIN_ROUTE_RT_METRIC, domain_route_timer.cost_ms()
                )
            route_logger.debug(f"routing to master done")

        kmonitor.report(GaugeMetrics.ROUTE_RT_METRIC, route_timer.cost_ms())
        if not input.generate_config.role_addrs:
            raise FtRuntimeException(
                ExceptionType.ROUTE_ERROR,
                f"request <{input.request_id}> no backend role addresses found after routing",
            )

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

    @torch.inference_mode()
    async def enqueue(
        self, input: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
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

        if self.host_service.service_available:
            await self.route_ips(input)

        return self.model_rpc_client.enqueue(input)

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
