import logging
from typing import Dict, List, Optional

from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp
from rtp_llm.ops import get_block_cache_keys as cpp_get_block_cache_keys
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class RtpLLMOp:
    def __init__(
        self,
        model: BaseModel,
        mm_engine: Optional[MMProcessEngine] = None,
        propose_model: Optional[ProposeModel] = None,
        token_processor: Optional[TokenProcessor] = None,
    ):
        super().__init__()
        self.model = model
        self.mm_engine = mm_engine
        self.propose_model = propose_model
        self.ft_op = CppRtpLLMOp()
        self.token_processor = token_processor

    def start(self):
        self.weight = self.model.weight
        self.ft_op.init(  # type: ignore
            self.model, self.mm_engine, self.propose_model, self.token_processor
        )

    def stop(self):
        self.ft_op.stop()  # type: ignore

    def pause(self):
        return self.ft_op.pause()

    def restart(self):
        return self.ft_op.restart()
