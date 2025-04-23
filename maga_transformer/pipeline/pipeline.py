import os
import logging
import torch
import asyncio
import threading
import platform
import queue
import json
from typing import Any, List, Union, Iterator, Tuple, Callable, Optional, Dict, Generator, AsyncGenerator
from concurrent.futures import Future
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from maga_transformer.utils.util import AtomicCounter
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.metrics import kmonitor, GaugeMetrics

from maga_transformer.models.base_model import BaseModel, GenerateOutput, GenerateOutputs, GenerateResponse, GenerateInput
from maga_transformer.utils.multimodal_util import MultimodalInput
from maga_transformer.model_factory import ModelFactory, AsyncModel, ModelConfig
from maga_transformer.async_decoder_engine.backend_rpc_server_visitor import BackendRPCServerVisitor
from maga_transformer.pipeline.pipeline_custom_func import PipelineCustomFunc, get_piple_custom_func
from maga_transformer.utils.word_util import remove_padding_eos, get_stop_word_slices, \
            truncate_response_with_stop_words, truncate_token_with_stop_word_id, match_stop_words
from maga_transformer.utils.tokenizer_utils import DecodingState
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.utils.mm_process_engine import MMProcessEngine
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

request_counter = AtomicCounter()

class Pipeline(object):
    def __init__(self, model_cls: Union["BaseModel", BaseModel],
                model_config: GptInitModelParameters, tokenizer: Optional[PreTrainedTokenizerBase]):
        self.model_cls = model_cls
        self.model_config = model_config
        self.tokenizer = tokenizer
        self._special_tokens: int = self.model_config.special_tokens
        self._mm_token: str = self.model_config.mm_related_params.special_tokens.get('default_mm_token', '')
        self.piple_funcs: PipelineCustomFunc = get_piple_custom_func(self.model_cls)
        self.backend_rpc_server_visitor = BackendRPCServerVisitor(model_config)

    def stop(self):
        if isinstance(self.model_cls, AsyncModel):
            logging.info("async model stop")
            self.model_cls.stop()

    def encode(self, prompt: str):
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt)

    def decode(self, token_id: int):
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id])

    @staticmethod
    def create_generate_config(generate_config: Union[GenerateConfig, Dict[str, Any]], vocab_size: int,
                               special_tokens: Any, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> GenerateConfig:
        if isinstance(generate_config, dict):
            config = GenerateConfig.create_generate_config(generate_config, **kwargs)
        else:
            # 认为是从frontend_worker传递进来的，不需要再处理一遍
            config = generate_config
        config.add_special_tokens(special_tokens)
        config.convert_select_tokens(vocab_size, tokenizer)
        config.add_thinking_params(tokenizer)
        config.add_stop_ids_from_str(tokenizer)
        config.validate()
        return config

    def __call__(self, prompt: str, urls: Optional[List[str]] = None, **kwargs: Any) -> Iterator[GenerateResponse]:
        # if not multimodal model, just pass [[]] * len(prompt)
        return self.pipeline(prompt, urls = urls, **kwargs)

    def pipeline(self,
                 prompt: str,
                 request_id: int = None,
                 urls: Optional[List[str]] = None,
                 **kwargs: Any) -> Iterator[GenerateResponse]:

        q = queue.Queue()

        async def generator():
            res = None
            try:
                res = self.pipeline_async(prompt, request_id, urls, **kwargs)
                async for x in res:
                    q.put(x)
                q.put(None)
            except Exception as e:
                q.put(e)
            finally:
                # if pipline break, should call aclose() to remove async_generator task from loop
                if res is not None:
                    res.aclose()

        def start_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generator())

        backgroud_thread = threading.Thread(target=start_loop)
        backgroud_thread.start()
        try:
            while True:
                try:
                    r = q.get(timeout=0.01)
                    if r is None:
                        break
                    if isinstance(r, Exception):
                        raise r
                    yield r
                except queue.Empty:
                    continue
        finally:
            backgroud_thread.join()

    @torch.inference_mode()
    def pipeline_async( # type: ignore
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()

        if request_id == None:
            request_id = request_counter.increment()

        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.create_generate_config(generate_config_json, self.model_config.vocab_size,
                                                      self.model_config.special_tokens, self.tokenizer, **kwargs)
        # for delete stop word from output
        prompt = self.piple_funcs.modify_prompt_func(prompt, generate_config=generate_config.model_dump(), **kwargs)
        mm_inputs = []
        if self.model_config.is_multimodal:
            prompt, mm_inputs = self.piple_funcs.multimodal_modify_prompt_func(prompt, urls, self._mm_token,
                    generate_config=generate_config.model_dump(), **kwargs)

        token_ids = self.piple_funcs.process_encode_func(prompt,
                                             generate_config=generate_config.model_dump(),
                                             tokenizer=self.tokenizer,
                                             add_special_tokens=self.model_config.add_special_tokens,
                                             special_tokens=self._special_tokens,
                                             **kwargs)

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(generate_config.sp_advice_prompt) 

        kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.num_beams)
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))
        return self.generate_stream(request_id, token_ids, mm_inputs, generate_config, **kwargs)

    def process_stop_id(self,
                        generate_config: GenerateConfig,
                        generate_output: GenerateOutput,
                        tokens,
                        stop_word_ids: List[List[int]],
                        stop_word_id_slices: List[List[int]]):
        if not generate_config.print_stop_words:
            if not generate_output.finished:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_id_slices)
            else:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_ids)
        return tokens

    def process_stop_str(self,
                    generate_config: GenerateConfig,
                    generate_output: GenerateOutput,
                    text: str, all_text: str,
                    stop_word_str_list: List[str],
                    stop_word_str_slices: List[str],
                    token_buffer: str,
                    **kwargs: Any):
        generate_output.finished = self.piple_funcs.stop_generate_func(all_text, **kwargs) or generate_output.finished
        if stop_word_str_list and not generate_output.finished and match_stop_words(all_text, stop_word_str_list):
            generate_output.finished = True

        if not generate_config.print_stop_words:
            if not generate_config.return_incremental:
                if not generate_output.finished:
                    text = truncate_response_with_stop_words(text, stop_word_str_slices, generate_config.is_streaming)
                else:
                    text = truncate_response_with_stop_words(text, stop_word_str_list, generate_config.is_streaming)
            else:
                if not generate_output.finished:
                    text = token_buffer + text
                    trunc_text = truncate_response_with_stop_words(text, stop_word_str_slices, generate_config.is_streaming)
                    token_buffer = text[len(trunc_text):]
                    text = trunc_text
                else:
                    text = truncate_response_with_stop_words(token_buffer + text, stop_word_str_list, generate_config.is_streaming)
        return text, token_buffer

    def decode_tokens(self,
                      generate_config: GenerateConfig,
                      generate_outputs: GenerateOutputs,
                      stop_word_str_list: List[str],
                      stop_word_str_slices: List[str],
                      stop_word_ids: List[int],
                      stop_word_id_slices: List[int],
                      decoding_states: List[DecodingState],
                      token_buffers: List[str],
                      ouput_tokens_list: List[torch.Tensor],
                      **kwargs: Any) -> Tuple[List[str], List[int], List[DecodingState], List[str], List[torch.Tensor]]:
        texts = []
        all_texts = []
        output_lens = []
        if len(decoding_states) == 0:
            if generate_config.num_beams == 1 and generate_config.is_streaming:
                decoding_states = [DecodingState() for _ in range(len(generate_outputs.generate_outputs))]
            else:
                # num_beams不等于1的情况下，不能进行增量decode，因为过去的token id会变化
                decoding_states = [None] * len(generate_outputs.generate_outputs)

        if len(token_buffers) == 0:
            token_buffers = [""] * len(generate_outputs.generate_outputs)

        if len(ouput_tokens_list) == 0:
            ouput_tokens_list = [torch.empty(0, dtype=torch.int32) for _ in range(len(generate_outputs.generate_outputs))]

        # TODO(xinfei.sxf) remove i
        i = 0
        for generate_output in generate_outputs.generate_outputs:
            # all model incremental return output_ids
            if generate_config.num_beams == 1:
                ouput_tokens_list[i] = torch.cat((ouput_tokens_list[i], generate_output.output_ids), dim=1)
                generate_output.output_ids = ouput_tokens_list[i]
            tokens = generate_output.output_ids
            tokens = remove_padding_eos(tokens, self._special_tokens.eos_token_id)
            output_lens.append(tokens.nelement())

            tokens = self.process_stop_id(generate_config, generate_output, tokens.tolist(), stop_word_ids, stop_word_id_slices)

            text, all_text = self.piple_funcs.process_decode_func(tokens,
                                          generate_config=generate_config.model_dump(),
                                          tokenizer=self.tokenizer,
                                          decoding_state=decoding_states[i],
                                          return_incremental=generate_config.return_incremental,
                                          **kwargs)

            text, token_buffers[i] = self.process_stop_str(generate_config, generate_output, text, all_text, stop_word_str_list,
                    stop_word_str_slices, token_buffers[i], **kwargs)

            text = self.piple_funcs.modify_response_func(
                    text, hidden_states=generate_output.hidden_states,
                    generate_config=generate_config.model_dump(),
                    **kwargs)

            texts.append(text)
            all_texts.append(all_text)
            i += 1
        return texts, output_lens, decoding_states, token_buffers, ouput_tokens_list

    @torch.inference_mode()
    async def generate_stream(self, request_id: int, token_ids: List[int], mm_inputs: List[MultimodalInput],
                            generate_config: GenerateConfig, **kwargs: Any) -> AsyncGenerator[GenerateResponse, None]:
        token_type_ids = []

        token_ids = torch.tensor(token_ids, dtype=torch.int)

        input = GenerateInput(request_id=request_id,
                              token_ids=token_ids,
                              mm_inputs=mm_inputs,
                              generate_config=generate_config,
                              tokenizer=self.tokenizer,
                              token_type_ids=token_type_ids)
        stop_word_strs = generate_config.stop_words_str
        stop_word_str_slices = get_stop_word_slices(stop_word_strs)
        stop_word_ids = generate_config.stop_words_list
        stop_word_id_slices = get_stop_word_slices(stop_word_ids)

        stream: AsyncGenerator[GenerateOutputs, None] = self.backend_rpc_server_visitor.enqueue(input)

        decoding_states: List[DecodingState] = []
        ouput_tokens_list: List[torch.Tensor] = []
        token_buffers: List[str] = []
        generate_outputs_cache = GenerateOutputs()

        # TODO(xinfei.sxf) add batch and stop test
        async for generate_outputs in stream:
            if not generate_outputs_cache.generate_outputs:
                generate_outputs_cache.generate_outputs = generate_outputs.generate_outputs
            else:
                generate_outputs_cache.generate_outputs = [out if out.finished else generate_outputs.generate_outputs[i]
                                                           for i, out in enumerate(generate_outputs_cache.generate_outputs)]
            assert len(generate_outputs_cache.generate_outputs) == len(generate_outputs.generate_outputs)
            begin_time = current_time_ms()
            generate_texts, output_lens, decoding_states, token_buffers, ouput_tokens_list = self.decode_tokens(
                generate_config, generate_outputs_cache, stop_word_strs, stop_word_str_slices,
                stop_word_ids, stop_word_id_slices, decoding_states, token_buffers, ouput_tokens_list, **kwargs)

            kmonitor.report(GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time)

            yield GenerateResponse(generate_outputs=generate_outputs_cache, generate_texts=generate_texts)
            if all(output.finished for output in generate_outputs_cache.generate_outputs):
                kmonitor.report(GaugeMetrics.FT_ITERATE_COUNT_METRIC, generate_outputs_cache.generate_outputs[0].aux_info.iter_count)
                for l in output_lens:
                    kmonitor.report(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, l)
                break
