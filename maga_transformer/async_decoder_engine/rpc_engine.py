from typing import AsyncGenerator, Optional, Dict
import asyncio
import logging
from typing_extensions import override
from maga_transformer.utils.time_util import Timer
from maga_transformer.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from maga_transformer.models.base_model import BaseModel, GenerateInput, GenerateOutputs
from maga_transformer.async_decoder_engine.base_engine import BaseEngine, KVCacheInfo
from maga_transformer.cpp.model_rpc.model_rpc_client import ModelRpcClient

class RPCEngine(BaseEngine):
    def __init__(self,
                 model: BaseModel,
                 sp_model: Optional[BaseModel] = None) -> None:
        self.model = model
        self.sp_model = sp_model
        self.tokenizer = model.tokenizer
        self.config = model.config
        self.vit_expand_token_id_lock = asyncio.Lock()
        self.rtp_llm_op_ = RtpLLMOp(model.config, False)
        self.rtp_llm_op_.set_linear_bias_slopes(model.linear_bias_slopes)
        self.model_rpc_client = ModelRpcClient(self.model.weight.lora_resource)
        self.model.weight.lora_resource.ft_op = [self.rtp_llm_op_]

    @override
    def start(self) -> None:
        # op engine init is in set_weight
        assert self.model.weight
        self.rtp_llm_op_.set_weight(self.model.weight)

    @override
    def stop(self) -> None:
        self.rtp_llm_op_.stop()

    @override
    def decode(self,
               input: GenerateInput) -> AsyncGenerator[GenerateOutputs, None]:
        return self.model_rpc_client.enqueue(input)

    @override
    def update_lora(self, lora_infos: Dict[str, str]) -> None:
        with Timer() as timer:
            self.model.weight.lora_resource.update(lora_infos)
        logging.info(f'update lora weights time: {timer.cost_ms() / 1000 :.2f} s')

    @override
    def get_kv_cache_info(self) -> KVCacheInfo:
        available_kv_cache, total_kv_cache = self.rtp_llm_op_.get_kv_cache_info()
        return KVCacheInfo(available_kv_cache=available_kv_cache, total_kv_cache=total_kv_cache)

