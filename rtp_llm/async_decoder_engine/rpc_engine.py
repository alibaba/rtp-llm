from typing import AsyncGenerator, Dict, Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.task_type import TaskType
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel, GenerateInput, GenerateOutputs
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class RPCEngine(BaseEngine):
    def __init__(
        self,
        model: BaseModel,
        propose_model: Optional[ProposeModel] = None,
        gang_info=None,
    ) -> None:
        super().__init__(model)
        self.propose_model = propose_model
        self.tokenizer = model.tokenizer

        self.gang_info = gang_info
        self.token_processor = TokenProcessor(
            self.tokenizer, self.model.config.special_tokens
        )
        if self.model.is_multimodal():
            self.mm_engine = MMProcessEngine(self.model)
        else:
            self.mm_engine = None
        self.rtp_llm_op_ = RtpLLMOp(
            model, self.mm_engine, propose_model, self.token_processor
        )
        self.model_rpc_client = ModelRpcClient(self.config)

    @override
    def _start(self) -> None:
        self.rtp_llm_op_.start()

        # Start HTTP server for language model tasks
        if (
            self.model.task_type == TaskType.LANGUAGE_MODEL
            and self.gang_info is not None
        ):
            self.rtp_llm_op_.ft_op.start_http_server(
                self.model.model_weights_loader,
                self.model.config.lora_infos,
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
    def _pause(self) -> None:
        self.started = False
        return self.rtp_llm_op_.pause()

    @override
    def _restart(self) -> None:
        return self.rtp_llm_op_.restart()
