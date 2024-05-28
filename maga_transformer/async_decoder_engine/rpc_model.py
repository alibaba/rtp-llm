import torch
import asyncio
import logging
from maga_transformer.cpp.model_rpc.model_rpc_client import ModelRpcClient
from typing import Optional, List, Dict, Union, Tuple
from maga_transformer.models.base_model import BaseModel, GenerateInput
from maga_transformer.utils.time_util import Timer
from maga_transformer.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.task_type import TaskType
from PIL import Image

class RpcModel:
    def __init__(self, model: BaseModel, sp_model: Optional[BaseModel] = None) -> None:
        self.model = model
        self.sp_model = sp_model
        self.tokenizer = model.tokenizer
        self.config = model.config
        self.vit_expand_token_id_lock = asyncio.Lock()
        self.rtp_llm_op_ = RtpLLMOp(model.config, False)
        self.rtp_llm_op_.set_weight(model.weight)
        self.model_rpc_client = ModelRpcClient(self.model.weight.lora_resource)

    def is_multimodal(self) -> bool:
        return self.model.is_multimodal()

    async def expand_token_id(self, token_ids: List[int], images: List[Image.Image]) -> Tuple[List[int], Union[torch.Tensor, List[torch.Tensor]]]:
        assert self.is_multimodal()
        raise Exception("not support yet")

    @property
    def task_type(self) -> TaskType:
        return self.model.task_type

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config

    def stop(self):
        self.rtp_llm_op_.stop()

    def update(self, lora_infos: Dict[str, str]):
        with Timer() as timer:
            self.rtp_llm_op_.update_lora(lora_infos)
        logging.info(f'update lora weights time: {timer.cost_ms() / 1000 :.2f} s')

    @torch.no_grad()
    def enqueue(self, input: GenerateInput):
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            raise Exception('bug, not supposed to be here')
        return self.model_rpc_client.enqueue(input)
