import logging
import torch
import asyncio
import threading
import queue
import json
from typing import Any, List, Union, Iterator, Tuple, Callable, Optional, Dict, Generator, AsyncGenerator
from maga_transformer.utils.time_util import current_time_ms
from torch.nn.utils.rnn import pad_sequence
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.config.generate_config import GenerateConfig, RequestFormat
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.utils.ft_plugin import plguin_loader, \
    ModifyResponseCallable, DecodeCallable, EncodeCallable, \
    StopGenerateCallable, ModifyPromptCallable,  MultiModalModifyPromptCallable
from maga_transformer.models.base_model import BaseModel, TokenizerBase, GenerateOutput, GenerateResponse
from maga_transformer.model_factory import ModelFactory, AsyncModel
from maga_transformer.pipeline.chatapi_format import encode_chatapi
from maga_transformer.async_decoder_engine.generate_stream import GenerateInput
from maga_transformer.utils.word_util import get_stop_word_slice_list
from maga_transformer.utils.tokenizer_utils import DecodingState, IncrementDecodingUtils
from transformers import PreTrainedTokenizerBase

class DefaultFunc(object):
    @staticmethod
    def modify_prompt_func(prompt: str, **kwargs: Any) -> str:
        return prompt

    @staticmethod
    def multimodal_modify_prompt_func(prompt: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        return prompt, kwargs['image']

    @staticmethod
    def modify_response_func(response: str, **kwargs: Any) -> str:
        return response

    @staticmethod
    def stop_generatre_func(response: str, **kwargs: Any) -> bool:
        return False

    @staticmethod
    def process_encode_func(prompt: str, generate_config: Dict[str, Any], special_tokens: Any, tokenizer: TokenizerBase, **kwargs: Any) -> List[int]:
        if len(prompt) == 0:
            raise FtRuntimeException(ExceptionType.EMPTY_PROMPT_ERROR, "prompt should have at least one token!")
        if generate_config['request_format'] == RequestFormat.CHAT_API:
            return encode_chatapi(prompt, special_tokens, tokenizer)
        if type(prompt) is not str:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "expect string prompt, actual: " + str(prompt))
        return tokenizer.encode(prompt)

    @staticmethod
    def tokenids_decode_func(tokens: List[int], tokenizer: Union[TokenizerBase, PreTrainedTokenizerBase],
                             decoding_state: Optional[DecodingState] = None, return_incremental: bool = False, **kwargs: Any) -> str:
        if decoding_state is None:
            return tokenizer.decode(tokens)

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            new_text = IncrementDecodingUtils.detokenize_incrementally(tokenizer, tokens, decoding_state)
            decoding_state.all_text += new_text
        else:
            all_text = tokenizer.decode(tokens)
            new_text = all_text[len(decoding_state.all_text): ]
            decoding_state.all_text = all_text

        return new_text if return_incremental == True else decoding_state.all_text

class Pipeline(object):
    modify_prompt_func: ModifyPromptCallable
    multimodal_modify_prompt_func: MultiModalModifyPromptCallable
    modify_response_func: ModifyResponseCallable
    process_encode_func: EncodeCallable
    process_decode_func: DecodeCallable
    stop_generate_func:  StopGenerateCallable

    def __init__(self, model: Union[AsyncModel, BaseModel], tokenizer: Optional[TokenizerBase]):
        self.model = model
        self.tokenizer = tokenizer
        self._special_tokens: int = self.model.config.special_tokens
        self._img_token: str = self.model.config.vit_related_params.vit_special_tokens.get('default_image_token', '')
        self.has_init_decode_stop_words: bool = False
        self._init_pipeline_func()

    def stop(self):
        if isinstance(self.model, AsyncModel):
            self.model.stop()

    # just for perf test
    def reset_perf_test_schedule_strategy(self):
        if isinstance(self.model, AsyncModel):
            self.model.reset_perf_test_schedule_strategy()

    def _get_func(self, func_name: str, default: Callable[..., Any]) -> Callable[..., Any]:
        ft_plugin = plguin_loader.get_plugin()
        if getattr(ft_plugin, func_name, None) is not None:
            logging.info(f"using {func_name} implement in ft_plugin")
            return getattr(ft_plugin, func_name)
        if getattr(self.model, func_name, None) is not None:
            logging.info(f"using {func_name} implement in model")
            return getattr(self.model, func_name)
        if isinstance(self.model, AsyncModel) and getattr(self.model.model, func_name, None) is not None:
            logging.info(f"using {func_name} implement in model")
            return getattr(self.model.model, func_name)
        logging.info(f"using {func_name} default implement")
        return default

    def _init_pipeline_func(self):
        self.modify_prompt_func = self._get_func("modify_prompt_plugin", DefaultFunc.modify_prompt_func)
        self.multimodal_modify_prompt_func = self._get_func("multimodal_modify_prompt_plugin", DefaultFunc.multimodal_modify_prompt_func)
        self.process_encode_func = self._get_func("process_encode_plugin", DefaultFunc.process_encode_func)
        self.process_decode_func = self._get_func("process_decode_plugin", DefaultFunc.tokenids_decode_func)
        self.modify_response_func  = self._get_func("modify_response_plugin", DefaultFunc.modify_response_func)
        self.stop_generate_func= self._get_func("stop_generate_plugin", DefaultFunc.stop_generatre_func)

    def __call__(self, prompt: List[str], images: Optional[List[List[str]]] = None, **kwargs: Any) -> Iterator[GenerateResponse]:
        # if not multimodal model, just pass images = [[]] * len(prompt)
        return self.pipeline(prompt, images, **kwargs)

    def remove_padding_eos(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        # token_ids shape: [beam_width, max_length]
        out_token_ids = [tokens.cpu().numpy() for tokens in token_ids]
        out_token_ids = [tokens[tokens != self._special_tokens.eos_token_id].tolist() for tokens in out_token_ids]
        return [torch.IntTensor(x) for x in out_token_ids]

    def _slice_response_with_stop_words(self, response: str, stop_word_strs: List[str]):
        for stop_word in stop_word_strs:
            if stop_word and response.endswith(stop_word):
                response = response[:(-len(stop_word))]
        return response

    def decode_tokens(self,
                      generate_output: GenerateOutput,
                      generate_config: GenerateConfig,
                      stop_word_str_list: List[str],
                      stop_word_str_slice_list: List[str],
                      decoding_state: Optional[DecodingState],
                      **kwargs: Any) -> Tuple[List[Any], List[List[int]]]:
        tokens = generate_output.output_ids.cpu()
        tokens = self.remove_padding_eos(tokens)
        output_lens = [t.nelement() for t in tokens]
        texts = [self.process_decode_func(tokens.tolist(),
                                          generate_config=generate_config.model_dump(),
                                          tokenizer=self.tokenizer,
                                          decoding_state=decoding_state,
                                          return_incremental=generate_config.return_incremental,
                                          **kwargs) \
                 for tokens in tokens]
        # custom stop logic, update origin finsihed tensor
        generate_output.finished = self.stop_generate_func(texts[0], **kwargs) or generate_output.finished
        if not generate_config.print_stop_words:
            if not generate_output.finished and not generate_config.return_incremental:
                texts = [self._slice_response_with_stop_words(text, stop_word_str_slice_list) for text in texts]
            else:
                texts = [self._slice_response_with_stop_words(text, stop_word_str_list) for text in texts]
        return texts, output_lens

    # static for ut
    @staticmethod
    def get_stop_words_list(special_tokens,
                            tokenizer: TokenizerBase,
                            stop_words_list: List[List[int]],
                            stop_words_str: List[str]):
        stop_words_list = stop_words_list + special_tokens.stop_words_list
        for stop_words in stop_words_str + special_tokens.stop_words_str:
            stop_words_list.append(tokenizer.encode(stop_words)) # type: ignore
        return stop_words_list

    # static for ut
    @staticmethod
    def create_generate_config(generate_config: Dict[str, Any], special_tokens: Any, **kwargs: Any) -> GenerateConfig:
        generate_config.update(kwargs)
        try:
            config = GenerateConfig(**generate_config)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_GENERATE_CONFIG_FORMAT, f"generate_config validate failed: {str(e)}")
        # 如果同时在外面和里面都有设置采样参数，选择使用外面的
        # 这里假设外部传进来的stop_word_list和stop_word_str都不包含batch维度
        config.stop_words_list += special_tokens.stop_words_list
        config.stop_words_str += special_tokens.stop_words_str
        config.check_data_type()
        return config

    def _get_stop_word_strs(self, tokenizer: TokenizerBase, generate_config: GenerateConfig) -> List[str]:
        return generate_config.stop_words_str + [self.process_decode_func(ids, tokenizer=self.tokenizer) for ids in generate_config.stop_words_list]

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
        generate_config = self.create_generate_config(generate_config_json, self.model.config.special_tokens, **kwargs)
        # for delete stop word from output
        prompt = self.modify_prompt_func(prompt, generate_config=generate_config.model_dump(), images=images, **kwargs)

        if self.model.is_multimodal:
            prompt, images = self.multimodal_modify_prompt_func(prompt, images=images, img_token=self._img_token, generate_config=generate_config.model_dump(), **kwargs)

        token_ids = self.process_encode_func(prompt,
                                             generate_config=generate_config.model_dump(),
                                             tokenizer=self.tokenizer,
                                             special_tokens=self._special_tokens,
                                             **kwargs)

        kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.num_beams)
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))

        token_ids = torch.tensor(token_ids, dtype=torch.int, pin_memory=True)
        input = GenerateInput(token_ids=token_ids,
                              images=images,
                              generate_config=generate_config,
                              tokenizer=self.tokenizer)
        return self.generate_stream(input, **kwargs)

    @torch.inference_mode()
    async def generate_stream(self, input: GenerateInput, **kwargs: Any) -> AsyncGenerator[GenerateResponse, None]:
        # TODO(xinfei.sxf) stop words etc 直接带入raw query中去
        stop_word_strs = self._get_stop_word_strs(self.tokenizer, input.generate_config)
        stop_word_str_slices = get_stop_word_slice_list(stop_word_strs)

        stream = self.model.generate_stream(input)

        decoding_state = DecodingState() if input.generate_config.num_beams == 1 else None
        async for generate_output in stream:
            begin_time = current_time_ms()
            generate_texts, output_lens = self.decode_tokens(
                generate_output,
                input.generate_config,
                stop_word_strs, stop_word_str_slices, decoding_state, **kwargs)
            num_beams = input.generate_config.num_beams
            hidden_states = generate_output.hidden_states
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.view(num_beams, -1)
            if num_beams == 1:
                generate_texts[0] = self.modify_response_func(
                    generate_texts[0], hidden_states=hidden_states,
                    generate_config=input.generate_config.model_dump(),
                    **kwargs)
            if generate_output.finished:
                kmonitor.report(GaugeMetrics.FT_ITERATE_COUNT_METRIC, generate_output.aux_info.iter_count)
                for l in output_lens:
                    kmonitor.report(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, l)
            kmonitor.report(GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
            yield GenerateResponse(generate_output=generate_output, generate_texts=generate_texts)
