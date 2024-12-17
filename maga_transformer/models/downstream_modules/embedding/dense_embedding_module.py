import os
import sys
import json
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union
from sentence_transformers.util import import_from_string
from sentence_transformers.models import Transformer, Normalize

from transformers import PreTrainedTokenizerBase
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomHandler
from maga_transformer.utils.tensor_utils import get_last_token_from_combo_tokens, get_first_token_from_combo_tokens
from maga_transformer.models.downstream_modules.embedding.misc import combo_to_batch, EmbeddingRendererBase
from maga_transformer.models.downstream_modules.embedding.api_datatype import EmbeddingResponseType, EmbeddingResponseFormat, OpenAIEmbeddingRequest, SimilarityRequest

class DenseEmbeddingModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = DenseEmbeddingRenderer(config, tokenizer)
        if os.path.exists(os.path.join(self.config_.ckpt_path, 'modules.json')):
            self.handler = SentenceTransformerHandler(config)
        else:
            self.handler = NormalHandler(config)


class DenseEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, ** kwargs)
        self.embedding_type = EmbeddingResponseType.DENSE

    def render_request(self, request_json: Dict[str, Any]) -> Union[SimilarityRequest, OpenAIEmbeddingRequest]:
        if 'left' in request_json:
            return SimilarityRequest(**request_json)
        else:
            return OpenAIEmbeddingRequest(**request_json)

    def similar_func(self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat) -> float:
        return float(torch.tensor(left.embedding) @ torch.tensor(right.embedding).T)

    def embedding_func(self, request: Any, res: torch.Tensor, input_length: int, input_tokens: torch.Tensor) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res.tolist()

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if 'data' in log_response:
            del log_response['data']
        return log_response

class NormalHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        self.is_causal = config.is_causal

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        if self.is_causal:
            ts = get_last_token_from_combo_tokens(hidden_states, input_lengths)
        else:
            ts = get_first_token_from_combo_tokens(hidden_states, input_lengths)
        ts = torch.nn.functional.normalize(ts, dim=1)
        return ts

class SentenceTransformerHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        sys.path.append(config.ckpt_path)
        dtype = to_torch_dtype(config.data_type)
        modules_config_path = os.path.join(config.ckpt_path, 'modules.json')
        assert os.path.exists(modules_config_path), "not found modules.json from sentence_transformer"
        with open(modules_config_path) as fIn:
            modules_config = json.load(fIn)
        modules: OrderedDict[str, nn.Module] = OrderedDict()
        for module_config in modules_config:
            module_class = import_from_string(module_config["type"])
            # For Transformer, don't load the full directory, rely on `transformers` instead
            # But, do load the config file first.
            if module_class == Transformer and module_config["path"] == "":
                pass
            else:
                # Normalize does not require any files to be loaded
                if module_class == Normalize:
                    module_path = None
                else:
                    module_path = os.path.join(config.ckpt_path, module_config["path"])
                module = module_class.load(module_path)
                modules[module_config["name"]] = module
        self.model = nn.Sequential(modules).cuda().to(dtype)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> List[Any]:
        batch_input_ids, batch_hidden_states, batch_attention_mask = combo_to_batch(hidden_states, input_ids, input_lengths)
        return self.forward_internal(batch_input_ids, batch_hidden_states, batch_attention_mask)
    
    def forward_internal(self, batch_input_ids: torch.Tensor, batch_hidden_states: torch.Tensor, batch_attention_mask: torch.Tensor):
        input =  {
            "token_embeddings": batch_hidden_states,
            "attention_mask": batch_attention_mask
        }
        return self.model(input)['sentence_embedding']