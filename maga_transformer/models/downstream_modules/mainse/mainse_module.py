import torch
import orjson
import numpy as np
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters


from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomHandler, CustomRenderer
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EngineInputs, EngineOutputs

class MainseHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        self.is_causal = config.is_causal

    def tensor_info(self) -> List[str]:
        return ["w_out.weight", "w_out.bias"]

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        return hidden_states

class MainseRenderer(CustomRenderer):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        self.config_ = config
        self.tokenizer_ = tokenizer

    async def render_request(self, request_json: Dict[str, Any]):
        return orjson.loads(request_json['request'])

    async def create_input(self, request: Dict[str, Any]) -> EngineInputs:
        user_bert_ids = request["user_bert_ids"]
        user_bert_mask_len = request["user_bert_mask_len"]
        item_bert_ids = request["item_bert_ids"]
        item_bert_mask_len = request["item_bert_mask_len"]
        batch_size = len(item_bert_ids)
        user_length = user_bert_mask_len[0]
        combo_tokens_np = np.zeros((batch_size * user_bert_mask_len[0] + sum(item_bert_mask_len)), dtype=np.int32)
        token_type_ids_np = np.zeros((batch_size * user_bert_mask_len[0] + sum(item_bert_mask_len)), dtype=np.int32)
        bias = 0
        for i, item_length in enumerate(item_bert_mask_len):
            combo_tokens_np[bias: bias + user_length] = user_bert_ids
            combo_tokens_np[bias + user_length: bias + user_length + item_length]= item_bert_ids[i][:item_length]
            token_type_ids_np[bias + user_length: bias + user_length + item_length] = 1
            bias += user_length + item_length
        seq_length_np = np.array(item_bert_mask_len, dtype=np.int32) + user_length

        if len(combo_tokens_np) > self.config_.max_context_batch_size * self.config_.max_seq_len:
            raise Exception(f"max context batch size: {self.config_.max_context_batch_size}, max_seq_len: {self.config_.max_seq_len}  < input_length: {len(combo_tokens_np)}")

        input = EngineInputs(token_ids = torch.from_numpy(combo_tokens_np),
                             token_type_ids=torch.from_numpy(token_type_ids_np),
                             input_lengths=torch.from_numpy(seq_length_np))
        return input

    async def render_response(self, request: Any, outputs: EngineOutputs) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "score": outputs.outputs,
                "__@_pos_@__": request['__@_pos_@__']
            }
        }


class MainseModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = MainseRenderer(config, tokenizer)
        self.handler = MainseHandler(config)
