import logging
import os
from typing import Dict, List, Optional

from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp
from rtp_llm.config.engine_config import EngineConfig

class RtpLLMOp:
    def __init__(
        self,
        engine_config: EngineConfig,
        model: BaseModel,
        propose_model: Optional[ProposeModel] = None,
        token_processor: Optional[TokenProcessor] = None,
    ):
        self.engine_config = engine_config
        self.model = model
        self.propose_model = propose_model
        self.ft_op = CppRtpLLMOp()
        self.token_processor = token_processor

    def start(self):
        self.weight = self.model.weight
        logging.info("engine_config: %s", self.engine_config.to_string())
        self.ft_op.init(  # type: ignore
            self.model,
            self.engine_config,
            self.model.vit_config,
            self.propose_model,
            self.token_processor,
        )

    def stop(self):
        self.ft_op.stop()  # type: ignore