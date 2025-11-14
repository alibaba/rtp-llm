from typing import AsyncGenerator, Dict, Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.ops import TaskType
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import EngineScheduleInfo, KVCacheInfo, WorkerStatusInfo
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class RPCEngine(BaseEngine):
    def __init__(
        self, model: BaseModel, gang_info, propose_model: Optional[ProposeModel] = None
    ) -> None:
        self.model = model
        self.propose_model = propose_model
        self.gang_info = gang_info
        self.tokenizer = model.tokenizer
        # BaseModel no longer has config attribute, use model_config instead
        self.config = model.model_config
        self.token_processor = TokenProcessor(
            self.tokenizer, self.model.model_config.special_tokens
        )
        if self.model.is_multimodal():
            self.mm_engine = MMProcessEngine(self.model, self.model.vit_config)
        else:
            self.mm_engine = None
        self.rtp_llm_op_ = RtpLLMOp(
            model, self.mm_engine, propose_model, self.token_processor
        )
        self.model_rpc_client = ModelRpcClient(
            self.model.engine_config.parallelism_config.ffn_disaggregate_config,
            self.model.engine_config.pd_sep_config.max_rpc_timeout_ms,
            self.model.engine_config.pd_sep_config.decode_entrance,
            gang_info=gang_info,
        )

    @override
    def _start(self) -> None:
        self.rtp_llm_op_.start()

        # Start HTTP server for language model tasks
        if (
            self.config.task_type == TaskType.LANGUAGE_MODEL
            and self.gang_info is not None
        ):
            self.rtp_llm_op_.ft_op.start_http_server(
                self.model.model_weights_loader,
                self.model.model_config.lora_infos,
                self.gang_info,
                self.tokenizer,
                None,  # chat_renderer is not needed for HTTP server startup
            )

    @override
    def _stop(self) -> None:
        self.rtp_llm_op_.stop()

    @override
    def decode(self, input: GenerateInput) -> AsyncGenerator[GenerateOutputs, None]:
        return self.model_rpc_client.enqueue(input)

    @override
    def get_worker_status_info(self, latest_finished_version: int) -> WorkerStatusInfo:
        return self.rtp_llm_op_.get_worker_status_info(latest_finished_version)

    @override
    def get_cache_status_info(self, latest_cache_version: int) -> KVCacheInfo:
        return self.rtp_llm_op_.get_cache_status_info(latest_cache_version)

    @override
    def get_engine_schedule_info(
        self, latest_finised_version: int
    ) -> EngineScheduleInfo:
        return self.rtp_llm_op_.get_engine_schedule_info(latest_finised_version)

    @override
    def update_scheduler_info(self, scheduler_info: str) -> None:
        self.rtp_llm_op_.update_scheduler_info(scheduler_info)

    @override
    def update_eplb_config(self, req: Dict[str, str]) -> bool:
        return self.rtp_llm_op_.update_eplb_config(req)

    @override
    def _pause(self) -> None:
        self.started = False
        return self.rtp_llm_op_.pause()

    @override
    def _restart(self) -> None:
        return self.rtp_llm_op_.restart()
