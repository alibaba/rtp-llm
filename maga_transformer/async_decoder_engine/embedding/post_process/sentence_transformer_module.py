import os
import json
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import List, Dict
from maga_transformer.utils.util import to_cuda
from maga_transformer.async_decoder_engine.embedding.embedding_batch_query import EmbeddingBatchQuery, EmbeddingOutput

from sentence_transformers.util import import_from_string
from sentence_transformers.models import Transformer, Pooling, Normalize

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.post_process.post_process_module import PostProcessModule

class SentenceTransformerModule(PostProcessModule):
    def __init__(self, config: GptInitModelParameters):
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
        self.model = nn.Sequential(modules)
        
    def _reorder_input(self, all_hidden_states: torch.Tensor, batch_query: EmbeddingBatchQuery, attention_mask: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        inputs: List[Dict[str, torch.Tensor]] = []
        start_idx = 0
        for idx, query in enumerate(batch_query.context_streams):
            inputs.append({
                "token_embeddings": all_hidden_states[start_idx: start_idx + query.input.input_length].unsqueeze_(0),
                "attention_mask": attention_mask[idx][:query.input.input_length]
            })
            start_idx += query.input.input_length
        return inputs

    
    def process(self, batch_query: EmbeddingBatchQuery, all_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> List[EmbeddingOutput]:
        batch_inputs = self._reorder_input(all_hidden_states, batch_query, attention_mask)
        outputs: List[EmbeddingOutput] = []
        # 这里合并成batch算可能性能会更好，但是由于这部分占总时间1/1000，所以先不纠结
        for input in batch_inputs:
            outputs.append(EmbeddingOutput(sentence_embedding=self.model(input)['sentence_embedding'].squeeze_(0)))
        return outputs