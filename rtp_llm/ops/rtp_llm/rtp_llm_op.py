import logging
from typing import Dict, Optional, Tuple
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine
from rtp_llm.utils.token_processor import TokenProcessor
from rtp_llm.ops import LoadBalanceInfo, EngineScheduleInfo, EplbConfig, EplbMode


class RtpLLMOp():
    def __init__(
            self,
            model: BaseModel,
            mm_engine: Optional[MMProcessEngine] = None,
            propose_model: Optional[ProposeModel] = None,
            token_processor: Optional[TokenProcessor] = None
        ):
        super().__init__()
        self.model = model
        self.mm_engine = mm_engine
        self.propose_model = propose_model
        self.ft_op = CppRtpLLMOp()
        self.token_processor = token_processor

    def start(self):
        self.weight = self.model.weight
        self.ft_op.init( # type: ignore
            self.model,
            self.mm_engine,
            self.propose_model,
            self.token_processor)

    def stop(self):
        self.ft_op.stop() # type: ignore

    def ready(self):
        return self.ft_op.ready()

    def get_load_balance_info(self) -> LoadBalanceInfo:
        return self.ft_op.get_load_balance_info() # type: ignore

    def get_engine_schedule_info(self) -> EngineScheduleInfo:
        return self.ft_op.get_engine_schedule_info() # type: ignore

    def update_scheduler_info(self, scheduler_info: str):
        self.ft_op.update_scheduler_info(scheduler_info) # type: ignore

    def update_eplb_config(self, req: Dict[str, str]) -> bool:
        try:
            config = EplbConfig()
            config.mode = EplbMode.__members__[req.get("mode", "NONE")]
            config.update_time = int(req.get("update_time", 1000))
            return self.ft_op.update_eplb_config(config)
        except Exception as e:
            logging.error(f"update eplb config error: {e}")
            return False