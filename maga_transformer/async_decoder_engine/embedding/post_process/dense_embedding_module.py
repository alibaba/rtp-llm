import os
import json
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import List, Dict, Union

from sentence_transformers.util import import_from_string
from sentence_transformers.models import Transformer, Normalize

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class DenseEmbeddingModule(object):
    def __call__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_length: List[int], do_normalize: bool) -> torch.Tensor:
        raise NotImplementedError()
    
def init_dense_embedding_module(config: GptInitModelParameters, dtype: Union[str, torch.dtype]) -> DenseEmbeddingModule:
    if os.path.exists(os.path.join(config.ckpt_path, 'modules.json')):
        dense_embedding_module = SentenceTransformerModule(config, dtype)
    else:
        dense_embedding_module =  NormalModule(config.is_causal)
    return dense_embedding_module

class NormalModule(DenseEmbeddingModule):
    def __init__(self, is_casual: bool):
        self.is_casual = is_casual

    def __call__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_lengths: List[int], do_normalize: bool) -> torch.Tensor:
        batch_size = len(input_lengths)
        if self.is_casual:
            ts =  torch.stack([hidden_states[idx][pos - 1] for idx, pos in enumerate(input_lengths)])
        else:
            ts = torch.stack([hidden_states[idx][0] for idx, pos in enumerate(input_lengths)])

        if do_normalize:
            ts = torch.nn.functional.normalize(ts, dim=1)
        return ts

class SentenceTransformerModule(DenseEmbeddingModule):
    def __init__(self, config: GptInitModelParameters, dtype: Union[str, torch.dtype]):
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

    def __call__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_length: List[int], do_normalize: bool) -> torch.Tensor:
        input =  {
            "token_embeddings": hidden_states,
            "attention_mask": attention_mask
        }
        return self.model(input)['sentence_embedding']