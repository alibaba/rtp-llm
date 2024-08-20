from typing import Optional, Tuple
from maga_transformer.models.base_model import BaseModel
from maga_transformer.models.propose_model.propose_model import ProposeModel
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.ops import RtpLLMOp as CppRtpLLMOp
from maga_transformer.utils.mm_process_engine import MMProcessEngine


class RtpLLMOp(FTOPBase):
    def __init__(
            self,
            model: BaseModel,
            mm_engine: Optional[MMProcessEngine] = None,
            propose_model: Optional[ProposeModel] = None,
        ):
        super().__init__()
        self.model = model
        self.mm_engine = mm_engine
        self.propose_model = propose_model
        self.ft_op = CppRtpLLMOp()


    def start(self):
        self.weight = self.model.weight
        self.ft_op.init( # type: ignore
            self.model,
            self.mm_engine,
            self.propose_model)

    def stop(self):
        self.ft_op.stop() # type: ignore

    def get_kv_cache_info(self) -> Tuple[int, int]:
        available_kv_cache, total_kv_cache = self.ft_op.get_kv_cache_info() # type: ignore
        return available_kv_cache, total_kv_cache
