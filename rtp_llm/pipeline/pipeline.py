import asyncio
import logging
import queue
import threading
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple, Union

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.word_util import (
    get_stop_word_slices,
    match_stop_words,
    remove_padding_eos,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)

request_counter = AtomicCounter()


class Pipeline(object):
    def __init__(
        self,
        model_config: GptInitModelParameters,
        tokenizer: Optional[BaseTokenizer],
        separated_frontend: bool = False,
    ):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self._special_tokens: int = self.model_config.special_tokens
        self._mm_token: str = self.model_config.mm_related_params.special_tokens.get(
            "default_mm_token", ""
        )
        self.backend_rpc_server_visitor = BackendRPCServerVisitor(
            model_config, separated_frontend
        )

    def encode(self, prompt: str):
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt)

    def decode(self, token_id: int):
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id])

    @staticmethod
    def create_generate_config(
        generate_config: Union[GenerateConfig, Dict[str, Any]],
        vocab_size: int,
        special_tokens: Any,
        tokenizer: BaseTokenizer,
        **kwargs: Any
    ) -> GenerateConfig:
        if isinstance(generate_config, dict):
            config = GenerateConfig.create_generate_config(generate_config, **kwargs)
        else:
            # 认为是从frontend_worker传递进来的，不需要再处理一遍
            config = generate_config
        config.add_special_tokens(special_tokens)
        config.convert_select_tokens(vocab_size, tokenizer)
        config.add_thinking_params(tokenizer)
        config.add_stop_ids_from_str(tokenizer)
        return config

    def __call__(
        self, prompt: str, urls: Optional[List[str]] = None, **kwargs: Any
    ) -> Iterator[GenerateResponse]:
        # if not multimodal model, just pass [[]] * len(prompt)
        return self.pipeline(prompt, urls=urls, **kwargs)

    def pipeline(
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Iterator[GenerateResponse]:

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
    def pipeline_async(  # type: ignore
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
        generate_config = self.create_generate_config(
            generate_config_json,
            self.tokenizer.vocab_size,
            self.model_config.special_tokens,
            self.tokenizer,
            **kwargs
        )
        mm_inputs = [MultimodalInput(url) for url in urls] if urls is not None else []

        if len(prompt) == 0:
            raise FtRuntimeException(
                ExceptionType.EMPTY_PROMPT_ERROR,
                "prompt should have at least one token!",
            )
        if type(prompt) is not str:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "expect string prompt, actual: " + str(prompt),
            )
        token_ids = self.tokenizer.encode(prompt)

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        kmonitor.report(
            GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
        )
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.max_num_beams())
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))
        return self.generate_stream(
            request_id, token_ids, mm_inputs, generate_config, **kwargs
        )

    def process_stop_id(
        self,
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        tokens,
        stop_word_ids: List[List[int]],
        stop_word_id_slices: List[List[int]],
    ):
        if not generate_config.print_stop_words:
            if not generate_output.finished:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_id_slices)
            else:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_ids)
        return tokens

    def process_stop_str(
        self,
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        text: str,
        all_text: str,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        token_buffer: str,
        **kwargs: Any
    ):
        if generate_config.return_incremental:
            text = token_buffer + text

        if stop_word_str_list:
            stop_idx, stop_len = match_stop_words(text, stop_word_str_list)
            if stop_idx != -1:
                if not generate_config.print_stop_words:
                    text = text[:stop_idx]
                else:
                    text = text[:stop_idx + stop_len]
                token_buffer = ""
                generate_output.finished = True

        if generate_output.finished:
            return text, token_buffer

        if generate_config.return_incremental or not generate_config.print_stop_words:
            trunc_text = truncate_response_with_stop_words(text, stop_word_str_slices, generate_config.is_streaming, True)
            if generate_config.return_incremental:
                token_buffer = text[len(trunc_text) :]
            text = trunc_text

        return text, token_buffer

    def decode_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        decoding_states: List[DecodingState],
        token_buffers: List[str],
        ouput_tokens_list: List[torch.Tensor],
        **kwargs: Any
    ) -> Tuple[
        List[str], List[int], List[DecodingState], List[str], List[torch.Tensor]
    ]:
        texts = []
        all_texts = []
        output_lens = []
        if len(decoding_states) == 0:
            if not generate_config.has_num_beams() and generate_config.is_streaming:
                decoding_states = [
                    DecodingState()
                    for _ in range(len(generate_outputs.generate_outputs))
                ]
            else:
                # num_beams不等于1的情况下，不能进行增量decode，因为过去的token id会变化
                decoding_states = [None] * len(generate_outputs.generate_outputs)

        if len(token_buffers) == 0:
            token_buffers = [""] * len(generate_outputs.generate_outputs)

        if len(ouput_tokens_list) == 0:
            ouput_tokens_list = [
                torch.empty(0, dtype=torch.int32)
                for _ in range(len(generate_outputs.generate_outputs))
            ]

        def tokenids_decode_func(
            tokens: List[int],
            tokenizer: BaseTokenizer,
            decoding_state: Optional[DecodingState] = None,
            return_incremental: bool = False,
            **kwargs: Any
        ) -> Tuple[str, str]:
            if decoding_state is None:
                all_text = tokenizer.decode(tokens, **kwargs)
                # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
                while (len(all_text) > 0) and ("\uFFFD" == all_text[-1]):
                    all_text = all_text[:-1]
                return all_text, all_text

            new_text = IncrementDecodingUtils.detokenize_incrementally(
                tokenizer, tokens, decoding_state
            )
            decoding_state.all_text += new_text
            return (
                new_text if return_incremental == True else decoding_state.all_text
            ), decoding_state.all_text

        # TODO(xinfei.sxf) remove i
        i = 0
        for generate_output in generate_outputs.generate_outputs:
            # all model incremental return output_ids
            if not generate_config.has_num_beams():
                ouput_tokens_list[i] = torch.cat(
                    (ouput_tokens_list[i], generate_output.output_ids), dim=1
                )
                generate_output.output_ids = ouput_tokens_list[i]
            tokens = generate_output.output_ids
            if not generate_config.ignore_eos:
                tokens = remove_padding_eos(tokens, self._special_tokens.eos_token_id)
            else:
                tokens = tokens.reshape(-1)
            output_lens.append(tokens.nelement())
            tokens = self.process_stop_id(
                generate_config,
                generate_output,
                tokens.tolist(),
                stop_word_ids,
                stop_word_id_slices,
            )

            text, all_text = tokenids_decode_func(
                tokens,
                generate_config=generate_config.model_dump(),
                tokenizer=self.tokenizer,
                decoding_state=decoding_states[i],
                return_incremental=generate_config.return_incremental,
                skip_special_tokens=generate_config.skip_special_tokens,
                **kwargs
            )

            text, token_buffers[i] = self.process_stop_str(
                generate_config,
                generate_output,
                text,
                all_text,
                stop_word_str_list,
                stop_word_str_slices,
                token_buffers[i],
                **kwargs
            )

            if generate_config.out_prefix:
                text = generate_config.out_prefix + text

            texts.append(text)
            all_texts.append(all_text)
            i += 1
        return texts, output_lens, decoding_states, token_buffers, ouput_tokens_list

    @torch.inference_mode()
    async def generate_stream(
        self,
        request_id: int,
        token_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        **kwargs: Any
    ) -> AsyncGenerator[GenerateResponse, None]:
        token_type_ids = []

        token_ids = torch.tensor(token_ids, dtype=torch.int)

        input = GenerateInput(
            request_id=request_id,
            token_ids=token_ids,
            mm_inputs=mm_inputs,
            generate_config=generate_config,
            tokenizer=self.tokenizer,
            token_type_ids=token_type_ids,
        )

        stop_word_strs = generate_config.stop_words_str
        stop_word_str_slices = get_stop_word_slices(stop_word_strs)
        stop_word_ids = generate_config.stop_words_list
        stop_word_id_slices = get_stop_word_slices(stop_word_ids)

        stream: AsyncGenerator[GenerateOutputs, None] = (
            await self.backend_rpc_server_visitor.enqueue(input)
        )

        decoding_states: List[DecodingState] = []
        ouput_tokens_list: List[torch.Tensor] = []
        token_buffers: List[str] = []
        generate_outputs_cache = GenerateOutputs()

        # TODO(xinfei.sxf) add batch and stop test
        async for generate_outputs in stream:
            if not generate_outputs_cache.generate_outputs:
                generate_outputs_cache.generate_outputs = (
                    generate_outputs.generate_outputs
                )
            else:
                generate_outputs_cache.generate_outputs = [
                    out if out.finished else generate_outputs.generate_outputs[i]
                    for i, out in enumerate(generate_outputs_cache.generate_outputs)
                ]
            assert len(generate_outputs_cache.generate_outputs) == len(
                generate_outputs.generate_outputs
            )
            begin_time = current_time_ms()
            (
                generate_texts,
                output_lens,
                decoding_states,
                token_buffers,
                ouput_tokens_list,
            ) = self.decode_tokens(
                generate_config,
                generate_outputs_cache,
                stop_word_strs,
                stop_word_str_slices,
                stop_word_ids,
                stop_word_id_slices,
                decoding_states,
                token_buffers,
                ouput_tokens_list,
                **kwargs
            )

            kmonitor.report(
                GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )

            yield GenerateResponse(
                generate_outputs=generate_outputs_cache, generate_texts=generate_texts
            )
            if all(
                output.finished for output in generate_outputs_cache.generate_outputs
            ):
                kmonitor.report(
                    GaugeMetrics.FT_ITERATE_COUNT_METRIC,
                    generate_outputs_cache.generate_outputs[0].aux_info.iter_count,
                )
                for l in output_lens:
                    kmonitor.report(GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, l)
                break
