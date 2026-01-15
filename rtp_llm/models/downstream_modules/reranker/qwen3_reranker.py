from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.qwen3_reranker_renderer import Qwen3RerankerRenderer
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.tensor_utils import get_last_token_from_combo_tokens
from rtp_llm.utils.util import to_torch_dtype


class Qwen3RerankerModule(CustomModule):

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = Qwen3RerankerRenderer(self.config_, self.tokenizer_)
        token_false_id = self.tokenizer_.convert_tokens_to_ids("no")
        token_true_id = self.tokenizer_.convert_tokens_to_ids("yes")
        self.handler = Qwen3RerankerHandler(self.config_, token_false_id, token_true_id)


class Qwen3RerankerHandler(CustomHandler):

    def __init__(self, config: ModelConfig, token_false_id: int, token_true_id: int):
        super().__init__(config)
        self.token_false_id = token_false_id
        self.token_true_id = token_true_id
        self.tie_word_embeddings = config.tie_word_embeddings
        self.lm_head_weight_name = (
            "model.embed_tokens.weight"
            if self.tie_word_embeddings
            else "lm_head.weight"
        )

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = [self.lm_head_weight_name]
        weights = []
        for k in w_list:
            weights.append(
                CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            )
        return weights

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        linear_weight = tensor_map[self.lm_head_weight_name]
        self.linear = torch.nn.Linear(linear_weight.shape[1], linear_weight.shape[0])
        self.linear.weight.data = linear_weight
        self.linear = self.linear.to(data_type).eval().to(self.device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        last_token = get_last_token_from_combo_tokens(hidden_states, input_lengths)
        last_token_logits = self.linear(last_token)
        true_vector = last_token_logits[:, self.token_true_id]
        false_vector = last_token_logits[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp()
        return scores
