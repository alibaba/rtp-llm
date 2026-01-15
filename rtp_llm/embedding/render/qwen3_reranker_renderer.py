import threading
from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.reranker.api_datatype import VoyageRerankerRequest
from rtp_llm.embedding.render.reranker_renderer import RerankerRenderer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.utils.time_util import current_time_ms


class Qwen3RerankerRenderer(RerankerRenderer):
    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase):
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
