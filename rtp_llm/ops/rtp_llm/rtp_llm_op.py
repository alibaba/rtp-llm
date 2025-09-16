import logging
import os
from typing import Dict, List, Optional

from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import EngineScheduleInfo, EplbConfig, EplbMode, KVCacheInfo
from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp
from rtp_llm.ops import WorkerStatusInfo
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

    def get_worker_status_info(self, latest_finished_version: int) -> WorkerStatusInfo:
        return self.ft_op.get_worker_status_info(latest_finished_version)

    def get_cache_status_info(self, latest_cache_version: int) -> KVCacheInfo:
        return self.ft_op.get_cache_status_info(latest_cache_version)

    def get_engine_schedule_info(
        self, latest_finised_version: int
    ) -> EngineScheduleInfo:
        return self.ft_op.get_engine_schedule_info(latest_finised_version)  # type: ignore

    def update_scheduler_info(self, scheduler_info: str):
        self.ft_op.update_scheduler_info(scheduler_info)  # type: ignore

    def update_eplb_config(self, req: Dict[str, str]) -> bool:
        try:
            config = EplbConfig()
            config.mode = EplbMode.__members__[req.get("mode", "NONE")]
            config.update_time = int(req.get("update_time", 1000))
            return self.ft_op.update_eplb_config(config)
        except Exception as e:
            logging.error(f"update eplb config error: {e}")
            return False

    def pause(self):
        return self.ft_op.pause()

    def restart(self):
        return self.ft_op.restart()
