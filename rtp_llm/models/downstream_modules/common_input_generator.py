import logging
import threading
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
)
from rtp_llm.models.downstream_modules.openai_render import (
    OpenAIRenderBasicInfo,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms


class CommonInputGenerator(object):
    def __init__(self, tokenizer: BaseTokenizer, config: ModelConfig):
        self.tokenizer_ = tokenizer
        self.config_ = config
        self.openai_render_info: OpenAIRenderBasicInfo = OpenAIRenderBasicInfo(
            tokenizer, config
        )
        self.lock = threading.Lock()

    @torch.inference_mode()
    def generate(  # type: ignore
        self,
        prompt: Union[
            str,
            List[str],
            List[Tuple[str, str]],
            List[Tuple[str, List[MultimodalInput]]],
            List[ChatMessage],
            List[List[ChatMessage]],
        ],
        truncate: bool = True,
        tokenizer_config: Dict[str, Any] = {},
    ) -> EngineInputs:
        with self.lock:
            begin_time = current_time_ms()
            # align images and prompts
            # do batch encode and split into embedding input per batch
            assert self.tokenizer_ is not None, "tokenizer should not be None"
            # truncate with tokenizer max_seq_len
            truncate_length = self.config_.max_seq_len
            if self.config_.position_ids_style == 1:
                truncate_length = self.config_.max_seq_len - (
                    self.config_.special_tokens.pad_token_id + 1
                )
            multimodal_inputs = []
            # (text, images) pairs must be matched before the plain-string
            # branch: their tuples also satisfy its isinstance check.
            if (
                isinstance(prompt, list)
                and isinstance(prompt[0], Tuple)
                and len(prompt[0]) == 2
                and isinstance(prompt[0][1], list)
            ):
                encoded = self.tokenizer_(
                    [text for text, _ in prompt],
                    max_length=truncate_length,
                    return_attention_mask=False,
                    padding=False,
                    return_length=True,
                    truncation=truncate,
                    return_tensors="np",
                    **tokenizer_config,
                )
                combo_tokens = torch.from_numpy(
                    np.concatenate(encoded["input_ids"])
                ).to(torch.int32)
                input_lengths = torch.from_numpy(encoded["length"]).to(torch.int32)
                # The C++ processor pairs images with prompts by order, so the
                # per-prompt lists are flattened in prompt order.
                multimodal_inputs = [
                    mm for _, images in prompt for mm in images
                ]
            elif (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and isinstance(prompt[0], str))
                or (isinstance(prompt, list) and isinstance(prompt[0], Tuple))
            ):
                prompt = [prompt] if isinstance(prompt, str) else prompt
                encoded = self.tokenizer_(
                    prompt,
                    max_length=truncate_length,
                    return_attention_mask=False,
                    padding=False,
                    return_length=True,
                    truncation=truncate,
                    return_tensors="np",
                    **tokenizer_config,
                )
                combo_tokens = torch.from_numpy(
                    np.concatenate(encoded["input_ids"])
                ).to(torch.int32)
                input_lengths = torch.from_numpy(encoded["length"]).to(torch.int32)
            elif isinstance(prompt, list) and isinstance(prompt[0], ChatMessage):
                chat_request = ChatCompletionRequest(messages=prompt)
                rendered_input = self.openai_render_info.chat_renderer.render_chat(
                    chat_request
                )
                encoded = rendered_input.input_ids
                combo_tokens = torch.from_numpy(np.array(encoded)).to(torch.int32)
                input_lengths = torch.from_numpy(np.array([len(encoded)])).to(
                    torch.int32
                )
                multimodal_inputs = rendered_input.multimodal_inputs
            else:
                logging.error(f"Unsupported prompt type: {type(prompt)}")
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    f"Unsupported prompt type: {type(prompt)}",
                )

            if "token_type_ids" in encoded:
                combo_token_types = torch.from_numpy(
                    np.concatenate(encoded["token_type_ids"])
                ).to(torch.int32)
            else:
                combo_token_types = torch.zeros_like(combo_tokens, dtype=torch.int32)

            for length in input_lengths:
                if length > self.config_.max_seq_len:
                    raise FtRuntimeException(
                        ExceptionType.LONG_PROMPT_ERROR,
                        f"one of prompt length: {length} > max_length: {self.config_.max_seq_len}",
                    )

            kmonitor.report(
                GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )
            kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(combo_tokens))
            return EngineInputs(
                token_ids=combo_tokens,
                token_type_ids=combo_token_types,
                input_lengths=input_lengths,
                multimodal_inputs=multimodal_inputs,
            )
