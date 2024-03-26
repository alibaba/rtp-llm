import logging
import torch
import asyncio
import threading
import queue
import json
from typing import Any, List, Union, Iterator, Tuple, Callable, Optional, Dict, Generator, AsyncGenerator
from PIL import Image
from concurrent.futures import Future
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.metrics import kmonitor, GaugeMetrics

from maga_transformer.models.base_model import BaseModel, GenerateOutput, GenerateResponse
from maga_transformer.model_factory import ModelFactory, AsyncModel, ModelConfig
from maga_transformer.pipeline.pipeline_custom_func import PipelineCustomFunc, get_piple_custom_func
from maga_transformer.async_decoder_engine.generate_stream import GenerateInput
from maga_transformer.utils.word_util import remove_padding_eos, get_stop_word_slice_list, truncate_response_with_stop_words
from maga_transformer.utils.tokenizer_utils import DecodingState
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.utils.multimodal_download import DownloadEngine

class Pipeline(object):
    def __init__(self, model: Union[AsyncModel, BaseModel], tokenizer: Optional[PreTrainedTokenizerBase]):
        self.model = model
        self.tokenizer = tokenizer
        self._special_tokens: int = self.model.config.special_tokens
        self._img_token: str = self.model.config.vit_related_params.vit_special_tokens.get('default_image_token', '')
        self.piple_funcs: PipelineCustomFunc = get_piple_custom_func(self.model)
        self.download_engine: DownloadEngine = DownloadEngine()

    def stop(self):
        if isinstance(self.model, AsyncModel):
            self.model.stop()

    def encode(self, prompt: str):
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt)

    def decode(self, token_id: int):
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id])

    # just for perf test
    def enable_perf_test_schedule_strategy(self):
        if isinstance(self.model, AsyncModel):
            self.model.enable_perf_test_schedule_strategy()

    @staticmethod
    def create_generate_config(generate_config: Union[GenerateConfig, Dict[str, Any]], vocab_size: int,
                               special_tokens: Any, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> GenerateConfig:
        if isinstance(generate_config, dict):
            config = GenerateConfig.create_generate_config(generate_config, **kwargs)
        else:
            # 认为是从inference_worker传递进来的，不需要再处理一遍
            config = generate_config
        config.add_special_tokens(special_tokens)
        config.convert_select_tokens(vocab_size, tokenizer)
        return config

    def _get_stop_word_strs(self, tokenizer: PreTrainedTokenizerBase, generate_config: GenerateConfig) -> List[str]:
        return generate_config.stop_words_str + [self.piple_funcs.process_decode_func(ids, tokenizer=self.tokenizer) for ids in generate_config.stop_words_list]

    def __call__(self, prompt: List[str], images: Optional[List[List[str]]] = None, **kwargs: Any) -> Iterator[GenerateResponse]:
        # if not multimodal model, just pass images = [[]] * len(prompt)
        return self.pipeline(prompt, images, **kwargs)

    def pipeline(self,
                 prompt: str,
                 images: Optional[List[str]] = None,
                 **kwargs: Any) -> Iterator[GenerateResponse]:
        q = queue.Queue()

        async def generator():
            res = None
            try:
                res = self.pipeline_async(prompt, images, **kwargs)
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
        images: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()
        # align images and prompts
        if images is None or len(images) == 0:
            images = []
        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.create_generate_config(generate_config_json, self.model.config.vocab_size,
                                                      self.model.config.special_tokens, self.tokenizer, **kwargs)
        # for delete stop word from output
        prompt = self.piple_funcs.modify_prompt_func(prompt, generate_config=generate_config.model_dump(), images=images, **kwargs)

        if self.model.is_multimodal():
            prompt, images = self.piple_funcs.multimodal_modify_prompt_func(prompt, images=images, img_token=self._img_token,
                                                                            generate_config=generate_config.model_dump(), **kwargs)
            if len(images) > 0:
                images = self.download_engine.submit(images)

        token_ids = self.piple_funcs.process_encode_func(prompt,
                                             generate_config=generate_config.model_dump(),
                                             tokenizer=self.tokenizer,
                                             special_tokens=self._special_tokens,
                                             **kwargs)

        kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.num_beams)
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))

        return self.generate_stream(token_ids, images, generate_config, **kwargs)

    def decode_tokens(self,
                      generate_output: GenerateOutput,
                      generate_config: GenerateConfig,
                      stop_word_str_list: List[str],
                      stop_word_str_slice_list: List[str],
                      decoding_state: Optional[DecodingState],
                      token_buffer: List[str],
                      **kwargs: Any) -> Tuple[List[Any], List[List[int]], List[str]]:
        tokens = generate_output.output_ids.cpu()
        tokens = remove_padding_eos(tokens, self._special_tokens.eos_token_id)
        output_lens = [t.nelement() for t in tokens]
        texts = [self.piple_funcs.process_decode_func(tokens.tolist(),
                                          generate_config=generate_config.model_dump(),
                                          tokenizer=self.tokenizer,
                                          decoding_state=decoding_state,
                                          return_incremental=generate_config.return_incremental,
                                          **kwargs) \
                 for tokens in tokens]
        # custom stop logic, update origin finsihed tensor
        if len(token_buffer) == 0:
            token_buffer = [""] * len(texts)

        generate_output.finished = self.piple_funcs.stop_generate_func(texts[0], **kwargs) or generate_output.finished
        if not generate_config.print_stop_words:
            if not generate_config.return_incremental:
                if not generate_output.finished:
                    texts = [truncate_response_with_stop_words(text, stop_word_str_slice_list) for text in texts]
                else:
                    texts = [truncate_response_with_stop_words(text, stop_word_str_list) for text in texts]
            else:
                if not generate_output.finished:
                    new_texts = []
                    for i in range(len(texts)):
                        text = token_buffer[i] + texts[i]
                        trunc_text = truncate_response_with_stop_words(text, stop_word_str_slice_list)
                        token_buffer[i] = text[len(trunc_text):]
                        new_texts.append(trunc_text)
                    texts = new_texts
                else:
                    texts = [truncate_response_with_stop_words(token_buffer[i] + texts[i], stop_word_str_list) for i in range(len(texts))]
        return texts, output_lens, token_buffer

    @torch.inference_mode()
    async def generate_stream(self, token_ids: List[int], images: List[Future[Image.Image]], generate_config: GenerateConfig, **kwargs: Any) -> AsyncGenerator[GenerateResponse, None]:
        # TODO(xinfei.sxf) stop words etc 直接带入raw query中去

        if self.model.is_multimodal() and len(images) > 0:
            images = await self.download_engine.get(images)
            token_ids, images = await self.model.expand_token_id(token_ids, images)

        token_ids = torch.tensor(token_ids, dtype=torch.int, pin_memory=True)

        input = GenerateInput(token_ids=token_ids,
                              images=images,
                              generate_config=generate_config,
                              tokenizer=self.tokenizer)
        stream = self.model.enqueue(input)

        stop_word_strs = self._get_stop_word_strs(self.tokenizer, input.generate_config)
        stop_word_str_slices = get_stop_word_slice_list(stop_word_strs)
        num_beams = input.generate_config.num_beams
        decoding_state = DecodingState() if num_beams == 1 else None

        token_buffer: List[str] = []

        async for generate_output in stream:
            begin_time = current_time_ms()
            generate_texts, output_lens, token_buffer = self.decode_tokens(
                generate_output,
                input.generate_config,
                stop_word_strs, stop_word_str_slices, decoding_state, token_buffer, **kwargs)

            hidden_states = generate_output.hidden_states
            if num_beams == 1:
                generate_texts[0] = self.piple_funcs.modify_response_func(
                    generate_texts[0], hidden_states=hidden_states,
                    generate_config=input.generate_config.model_dump(),
                    **kwargs)

            if generate_output.finished:
                kmonitor.report(GaugeMetrics.FT_ITERATE_COUNT_METRIC, generate_output.aux_info.iter_count)
                for l in output_lens:
                    kmonitor.report(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, l)
            kmonitor.report(GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time)

            yield GenerateResponse(generate_output=generate_output, generate_texts=generate_texts)
