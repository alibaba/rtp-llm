from typing import Dict, Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import TaskType
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class LanguageCppEngine(BaseEngine):
    def __init__(
        self,
        model: BaseModel,
        engine_config: EngineConfig,
        world_info=None,
        propose_model: Optional[ProposeModel] = None,
    ) -> None:
        """Initialize RPCEngine with model and engine configuration.

        Args:
            model: BaseModel instance
            engine_config: EngineConfig instance containing engine and parallelism configs
            world_info: Optional WorldInfo instance from DistributedServer (used for HTTP server)
            propose_model: Optional propose model for speculative decoding
        """
        self.model = model
        self.propose_model = propose_model
        self.tokenizer = model.tokenizer
        self.world_info = world_info
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
            engine_config, model, self.mm_engine, propose_model, self.token_processor
        )

    @override
    def _start(self) -> None:
        self.rtp_llm_op_.start()

        # Start HTTP server for language model tasks
        if (
            self.config.task_type == TaskType.LANGUAGE_MODEL
            and self.world_info is not None
        ):
            self.rtp_llm_op_.ft_op.start_http_server(
                self.model.model_weights_loader,
                self.model.model_config.lora_infos,
                self.world_info,
                self.tokenizer,
                None,  # chat_renderer is not needed for HTTP server startup
            )

    @override
    def _stop(self) -> None:
        self.rtp_llm_op_.stop()
