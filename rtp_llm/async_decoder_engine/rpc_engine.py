import logging
import time
from typing import Dict, Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.ops import TaskType, VitSeparation
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.utils.time_util import timer_wrapper


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

        self.mm_process_engine = None
        if (
            self.model.is_multimodal()
            and self.model.vit_config.vit_separation
            == VitSeparation.VIT_SEPARATION_LOCAL
            and engine_config.parallelism_config.tp_rank == 0
        ):
            self.mm_process_engine = (
                MultimodalMixinFactory.create_multimodal_process_engine(
                    model_config=self.model.model_config,
                    engine_config=engine_config,
                    vit_config=self.model.vit_config,
                    device=f"cuda:{engine_config.parallelism_config.local_rank}",
                )
            )
        self.rtp_llm_op_ = RtpLLMOp(
            engine_config,
            model,
            propose_model,
            self.token_processor,
            self.mm_process_engine,
        )

    @timer_wrapper(description="start async engine")
    @override
    def _start(self) -> None:
        start_time = time.time()
        self.rtp_llm_op_.start()
        consume_s = time.time() - start_time
        logging.info(f"start rtp_llm_op_ took {consume_s:.2f}s")
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
