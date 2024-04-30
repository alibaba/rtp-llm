import torch
from typing import Any, List, Union, Dict, Tuple
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.metrics import kmonitor, GaugeMetrics

from transformers import PreTrainedTokenizerBase
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingInput

class CommonInputGenerator(object):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: GptInitModelParameters):
        self.tokenizer_ = tokenizer
        self.config_ = config

    @torch.inference_mode()
    async def generate( # type: ignore
        self,
        prompt: Union[List[str], str, List[Tuple[str, str]]]
    ) -> Tuple[List[EmbeddingInput], int]:
        if isinstance(prompt, str):
            prompt = [prompt]
        begin_time = current_time_ms()
        # align images and prompts
        # do batch encode and split into embedding input per batch
        assert self.tokenizer_ is not None, "tokenizer should not be None"
        # truncate with tokenizer max_seq_len
        encoded = self.tokenizer_(prompt, max_length=self.config_.max_seq_len, return_attention_mask=False, padding=False, return_length=True, truncation='longest_first')

        input_lengths: List[int] = encoded['length']
        token_ids: List[List[int]] = encoded['input_ids']
        token_type_ids: List[List[int]] = encoded.get("token_type_ids", [[0] * input_length for input_length in input_lengths])
        total_length = sum(input_lengths)
        # double check input length < self.model.config.max_seq_len
        for length in input_lengths:
            if length > self.config_.max_seq_len:
                raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, f"one of prompt length: {length} > max_length: {self.model.config.max_seq_len}")

        kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, total_length)

        inputs = [EmbeddingInput(token_ids=token_id,
                                 token_type_ids=token_type_id,
                                 input_length=input_length) \
            for token_id, token_type_id, input_length in zip(token_ids, token_type_ids, input_lengths)]
        return inputs, sum([len(x.token_ids) for x in inputs])