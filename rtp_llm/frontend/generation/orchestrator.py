import asyncio
import queue
import threading
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.generation.config_factory import create_generate_config
from rtp_llm.frontend.generation.decoder import GenerationDecoder
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.ops import SpecialTokens, SpeculativeExecutionConfig, VitSeparation
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.word_util import get_stop_word_slices

request_counter = AtomicCounter()


class GenerationOrchestrator:
    """Orchestrates end-to-end generation and decoding."""

    def __init__(
        self,
        special_tokens: SpecialTokens,
        pd_sep_config,
        max_seq_len: int,
        seq_size_per_block: int,
        tokenizer: BaseTokenizer,
        sp_config: Optional[SpeculativeExecutionConfig] = None,
        mm_related_params: Optional[Any] = None,
        vit_separation: Optional[VitSeparation] = None,
        backend_rpc_server_visitor: BackendRPCServerVisitor = None,
    ):
        self.pd_sep_config = pd_sep_config
        self.tokenizer = tokenizer
        self._special_tokens: SpecialTokens = special_tokens
        self._mm_token: str = ""
        if mm_related_params:
            self._mm_token = mm_related_params.special_tokens.get(
                "default_mm_token", ""
            )

        self.backend_rpc_server_visitor = backend_rpc_server_visitor
        self.decoder = GenerationDecoder(tokenizer, special_tokens)

    @staticmethod
    def stop_word_slices(stop_words):
        return GenerationDecoder.stop_word_slices(stop_words)

    @staticmethod
    def process_stop_id(*args, **kwargs):
        return GenerationDecoder.process_stop_id(*args, **kwargs)

    @staticmethod
    def process_stop_str(*args, **kwargs):
        return GenerationDecoder.process_stop_str(*args, **kwargs)

    @staticmethod
    def create_generate_config(*args, **kwargs):
        return create_generate_config(*args, **kwargs)

    def decode_non_incremental_tokens(self, *args, **kwargs):
        return self.decoder.decode_non_incremental_tokens(*args, **kwargs)

    def decode_incremental_tokens(self, *args, **kwargs):
        return self.decoder.decode_incremental_tokens(*args, **kwargs)

    def encode(self, prompt: str):
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt)

    def decode(self, token_id: int):
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id])

    def __call__(
        self, prompt: str, urls: Optional[List[str]] = None, **kwargs: Any
    ) -> Iterator[GenerateResponse]:
        return self.pipeline(prompt, urls=urls, **kwargs)

    def pipeline(
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()

        if request_id is None:
            request_id = request_counter.increment()

        generate_config_json = kwargs.pop("generate_config", {})
        generate_env_config = kwargs.pop("generate_env_config", None)
        generate_config = create_generate_config(
            generate_config_json,
            len(self.tokenizer),
            self._special_tokens,
            self.tokenizer,
            generate_env_config,
            **kwargs,
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

    @torch.inference_mode()
    async def generate_stream(
        self,
        request_id: int,
        token_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        **kwargs: Any,
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

        decoding_states: List[Any] = []
        ouput_tokens_list: List[torch.Tensor] = []
        token_buffers: List[str] = []
        generate_outputs_cache = GenerateOutputs()

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
            is_incremental = (
                not generate_config.has_num_beams() and generate_config.is_streaming
            )
            if is_incremental:
                (
                    generate_texts,
                    output_lens,
                    decoding_states,
                    token_buffers,
                    ouput_tokens_list,
                ) = self.decoder.decode_incremental_tokens(
                    generate_config,
                    generate_outputs_cache,
                    stop_word_strs,
                    stop_word_str_slices,
                    stop_word_ids,
                    stop_word_id_slices,
                    decoding_states,
                    token_buffers,
                    ouput_tokens_list,
                    **kwargs,
                )
            else:
                (
                    generate_texts,
                    output_lens,
                    ouput_tokens_list,
                ) = self.decoder.decode_non_incremental_tokens(
                    generate_config,
                    generate_outputs_cache,
                    stop_word_strs,
                    stop_word_str_slices,
                    stop_word_ids,
                    stop_word_id_slices,
                    ouput_tokens_list,
                    **kwargs,
                )

            kmonitor.report(
                GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )

            yield GenerateResponse(
                generate_outputs=generate_outputs_cache, generate_texts=generate_texts
            )
            if (
                all(
                    output.finished
                    for output in generate_outputs_cache.generate_outputs
                )
                and generate_config.aux_info
            ):
                kmonitor.report(
                    GaugeMetrics.FT_ITERATE_COUNT_METRIC,
                    generate_outputs_cache.generate_outputs[0].aux_info.iter_count,
                )
                if len(output_lens) > 0:
                    kmonitor.report(
                        GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC,
                        sum(output_lens) / len(output_lens),
                    )
                break
