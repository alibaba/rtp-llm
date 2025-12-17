import threading
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.reranker.api_datatype import (
    VoyageRerankerRequest,
)
from rtp_llm.models.downstream_modules.reranker.reranker_module import RerankerRenderer
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.tensor_utils import get_last_token_from_combo_tokens
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import to_torch_dtype


class Qwen3RerankerModule(CustomModule):

    def __init__(
        self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase
    ):
        super().__init__(config, tokenizer)
        self.renderer = Qwen3RerankerRenderer(self.config_, self.tokenizer_)
        token_false_id = self.tokenizer_.convert_tokens_to_ids("no")
        token_true_id = self.tokenizer_.convert_tokens_to_ids("yes")
        self.handler = Qwen3RerankerHandler(self.config_, token_false_id, token_true_id)


class Qwen3RerankerRenderer(RerankerRenderer):
    def __init__(
        self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase
    ):
        super().__init__(config, tokenizer)
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens_ = self.tokenizer_.encode(
            prefix, add_special_tokens=False, return_tensors="np"
        )[0]
        self.suffix_tokens_ = self.tokenizer_.encode(
            suffix, add_special_tokens=False, return_tensors="np"
        )[0]
        self.lock = threading.Lock()

    def format_instruction(
        self, instruction: Optional[str], query: str, doc: str
    ) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def create_input(self, formated_request: VoyageRerankerRequest):
        with self.lock:
            begin_time = current_time_ms()
            input: List[str] = [
                self.format_instruction(
                    formated_request.instruction, formated_request.query, doc
                )
                for doc in formated_request.documents
            ]
            inputs = self.tokenizer_(
                input,
                padding=False,
                truncation="longest_first",
                return_tensors="np",
                return_attention_mask=False,
                max_length=self.config_.max_seq_len
                - len(self.prefix_tokens_)
                - len(self.suffix_tokens_),
            )
            inputs = [
                np.concatenate([self.prefix_tokens_, input, self.suffix_tokens_])
                for input in inputs["input_ids"]
            ]
            input_lengths_t = torch.tensor([len(input) for input in inputs]).to(
                torch.int32
            )
            input_t = torch.from_numpy(np.concatenate(inputs)).to(torch.int32)
            kmonitor.report(
                GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )
            kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(input_t))

            return EngineInputs(
                token_ids=input_t,
                token_type_ids=torch.zeros_like(input_t),
                input_lengths=input_lengths_t,
                multimodal_inputs=[],
            )


class Qwen3RerankerHandler(CustomHandler):

    def __init__(
        self, config: GptInitModelParameters, token_false_id: int, token_true_id: int
    ):
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
