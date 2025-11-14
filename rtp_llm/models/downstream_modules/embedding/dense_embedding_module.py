import copy
import json
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Set, Union

import torch
import torch.nn as nn
from sentence_transformers.models import Normalize, Transformer
from sentence_transformers.util import import_from_string

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    SimilarityRequest,
)
from rtp_llm.models.downstream_modules.embedding.misc import (
    EmbeddingRendererBase,
    combo_to_batch_data,
)
from rtp_llm.utils.tensor_utils import (
    get_first_token_from_combo_tokens,
    get_last_token_from_combo_tokens,
)
from rtp_llm.utils.util import to_torch_dtype


class DenseEmbeddingModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = DenseEmbeddingRenderer(config, tokenizer)
        if os.path.exists(os.path.join(self.config_.ckpt_path, "modules.json")):
            self.handler = SentenceTransformerHandler(config)
        else:
            self.handler = NormalHandler(config)


class DenseEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embedding_type = EmbeddingResponseType.DENSE

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, OpenAIEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return OpenAIEmbeddingRequest(**request_json)

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ) -> float:
        return float(torch.tensor(left.embedding) @ torch.tensor(right.embedding).T)

    def embedding_func(
        self,
        request: Any,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res.tolist()

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if "data" in log_response:
            del log_response["data"]
        return log_response


class NormalHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.is_causal = config.attn_config.is_causal

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_causal:
            ts = get_last_token_from_combo_tokens(hidden_states, input_lengths)
        else:
            ts = get_first_token_from_combo_tokens(hidden_states, input_lengths)
        ts = torch.nn.functional.normalize(ts, dim=1)
        return ts


class SentenceTransformerHandler(CustomHandler):
    arg_name_mapping = {
        "token_embeddings": "hidden_states",
    }

    @classmethod
    def rename_args(cls, lst: List[str]) -> List[str]:
        return [cls.arg_name_mapping.get(x, x) for x in lst]

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        sys.path.append(config.ckpt_path)
        dtype = to_torch_dtype(config.data_type)

        modules_config_path = os.path.join(config.ckpt_path, "modules.json")
        assert os.path.exists(
            modules_config_path
        ), "not found modules.json from sentence_transformer"
        with open(modules_config_path) as fIn:
            modules_config = json.load(fIn)

        modules: OrderedDict[str, nn.Module] = OrderedDict()
        fallback_args = [
            "input_lengths",
            "input_ids",
            "attention_mask",
            "token_embeddings",
        ]
        extend_forward_args_list: Set[str] = set()
        for module_idx, module_config in enumerate(modules_config):
            # import module
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

                # collect module args
                module_args = (
                    module.forward_args()
                    if hasattr(module, "forward_args")
                    else fallback_args
                )
                if isinstance(module_args, list):
                    for arg_idx, module_arg in enumerate(module_args):
                        if isinstance(module_arg, str):
                            extend_forward_args_list.add(module_arg)
                        else:
                            logging.warning(
                                f'unexpected {arg_idx}th module_arg of module {module_idx}: "{module_arg}", ignored'
                            )
                else:
                    logging.warning(
                        f'unexpected module_args of module {module_idx}: "{module_args}", ignored'
                    )

        self.model = nn.Sequential(modules).cuda().to(dtype)
        self.raw_extend_forward_args_list = [*extend_forward_args_list]
        self.extend_forward_args_list = self.rename_args(
            self.raw_extend_forward_args_list
        )
        # require input_lengths for combo conversion
        if "input_lengths" not in self.extend_forward_args_list:
            self.extend_forward_args_list.append("input_lengths")

        logging.info(
            f"original extend forward args: {self.raw_extend_forward_args_list}"
        )
        logging.info(f"extend forward args: {self.extend_forward_args_list}")

    def extend_forward_args(self) -> List[str]:
        return self.extend_forward_args_list

    def extend_forward(
        self,
        **combo_data: Any,
    ) -> List[Any]:
        input_lengths: torch.Tensor = combo_data["input_lengths"]
        batch_data = combo_to_batch_data(input_lengths, combo_data)
        return self.extend_forward_internal(batch_data)

    def extend_forward_internal(self, data: Dict[str, torch.Tensor]):
        data = {
            k: data[self.arg_name_mapping.get(k, k)]
            for k in self.raw_extend_forward_args_list
        }

        return self.model(data)["sentence_embedding"]
