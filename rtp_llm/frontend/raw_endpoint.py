import asyncio
import logging
import queue
import threading
import traceback
from dataclasses import asdict
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union

import torch
from pydantic import BaseModel

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.structure.request_extractor import Request, RequestExtractor
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.word_util import (
    batch_remove_padding_eos,
    get_stop_word_slices,
    match_stop_words,
    remove_padding_eos_with_numpy,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)

request_counter = AtomicCounter()

from rtp_llm.frontend.helpers.parallel_processor import ParallelProcessor

# Import helper classes for modular design
from rtp_llm.frontend.helpers.pipeline_response import (
    BatchPipelineResponse,
    MultiSequencesPipelineResponse,
    PipelineResponse,
    TokenizerEncodeResponse,
)
from rtp_llm.frontend.helpers.response_collector import ResponseCollector
from rtp_llm.frontend.helpers.response_formatter import ResponseFormatter
from rtp_llm.frontend.helpers.token_decoder import TokenDecoder


class RawEndpoint:
    """
    Raw API endpoint that integrates FrontendWorker and Pipeline functionality
    Unified naming convention, symmetric with OpenaiEndpoint
    """

    def __init__(
        self,
        model_config,
        tokenizer,
        backend_rpc_server_visitor,
        separated_frontend: bool,
    ) -> None:
        logging.info("starting raw endpoint (formerly frontend worker)")

        # Use passed-in components to avoid duplicate creation
        if tokenizer is None:
            raise AttributeError("tokenizer is none!")
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor

        # Save separated_frontend parameter
        self.separated_frontend = separated_frontend

        # Set multimodal token (integrated from original Pipeline)
        self._mm_token: str = self.model_config.mm_related_params.special_tokens.get(
            "default_mm_token", ""
        )

        # Initialize helper classes - modular design
        self.token_decoder = TokenDecoder(self.tokenizer, self.model_config)
        self.response_formatter = ResponseFormatter()
        self.parallel_processor = ParallelProcessor()

        logging.info("raw endpoint start done.")

    def tokenizer_offset_mapping(self, prompt: str) -> Any:
        """Get tokenizer's offset mapping"""
        return self.tokenizer(
            prompt, return_offsets_mapping=True, return_attention_mask=False
        )

    def tokenizer_encode(self, prompt: str) -> Tuple[List[int], List[str]]:
        """
        Encode and return token ids and tokens
        Maintain interface compatibility with original FrontendWorker
        """
        token_ids = self.encode(prompt)
        token_ids = [int(id) for id in token_ids]
        tokens = [self.decode(id) for id in token_ids]
        return token_ids, tokens

    def completion(self, **kwargs: Any) -> CompleteResponseAsyncGenerator:
        """
        Main inference interface, corresponding to OpenaiEndpoint.chat_completion()
        Integrated from original FrontendWorker.inference() logic
        """
        default_generate_config = GenerateConfig()
        request_extractor = RequestExtractor(default_generate_config)
        request, kwargs = request_extractor.extract_request(kwargs)

        if request.is_streaming is False and request.incremental:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "request is non_stream but use incremental decoder",
            )

        response_generator = self._generate(request, **kwargs)

        complete_response_collect_func = partial(
            ResponseCollector.collect_complete_response,
            incremental=request.incremental,
            batch_infer=request.batch_infer,
            num_return_sequences=request.num_return_sequences,
        )
        return CompleteResponseAsyncGenerator(
            response_generator, complete_response_collect_func
        )

    def extract_generation_config(
        self, generate_config: Union[GenerateConfig, Dict[str, Any]], **kwargs: Any
    ) -> GenerateConfig:
        """
        Extract generation configuration, unified naming corresponding to OpenaiEndpoint._extract_generation_config()
        Integrated from original Pipeline.create_generate_config() logic
        """
        return self.create_generate_config(
            generate_config,
            len(self.tokenizer),
            self.model_config.special_tokens,
            self.tokenizer,
            **kwargs,
        )

    def render_input(
        self, prompt: str, urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Render input, corresponding to OpenaiEndpoint.render_chat()
        """
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

        token_ids = self.encode(prompt)
        mm_inputs = [MultimodalInput(url) for url in urls] if urls is not None else []

        return {
            "prompt": prompt,
            "token_ids": token_ids,
            "mm_inputs": mm_inputs,
            "urls": urls or [],
        }

    def _generate(self, request: Request, **kwargs: Any):
        """
        Internal generation method, integrated from original FrontendWorker._inference() logic
        """
        if (
            len(request.input_texts) > 1
            or request.batch_infer
            or request.num_return_sequences > 0
        ):
            generators: List[AsyncGenerator[Dict[str, Any], None]] = []
            for i, (text, urls, generate_config) in enumerate(
                zip(request.input_texts, request.input_urls, request.generate_configs)
            ):
                generators.append(
                    self._yield_generate(
                        request.request_id + i * 10000,
                        text,
                        urls,
                        generate_config=generate_config,
                        **kwargs,
                    )
                )
            return self.parallel_processor.parallel_batch_async_generators(
                request.incremental,
                generators,
                request.batch_infer,
            )
        else:
            return self._yield_generate(
                request.request_id,
                request.input_texts[0],
                request.input_urls[0],
                generate_config=request.generate_configs[0],
                **kwargs,
            )

    async def _yield_generate(
        self,
        request_id: int,
        text: str,
        urls: List[str],
        generate_config: GenerateConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generation wrapper, integrated from original FrontendWorker._yield_generate() and Pipeline.pipeline_async() logic
        """
        stream = self.inference_stream_async(
            prompt=text,
            request_id=request_id,
            urls=urls,
            generate_config=generate_config,
            **kwargs,
        )
        async for generate_response in stream:
            yield self.response_formatter.format_response_new(
                generate_response, generate_config
            )

    def inference_stream(
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        """
        Synchronous inference streaming method, integrated from original Pipeline.pipeline() logic
        """
        q = queue.Queue()

        async def generator():
            res = None
            try:
                res = self.inference_stream_async(prompt, request_id, urls, **kwargs)
                async for x in res:
                    q.put(x)
                q.put(None)
            except Exception as e:
                q.put(e)
            finally:
                if res is not None:
                    await res.aclose()

        def start_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generator())

        background_thread = threading.Thread(target=start_loop)
        background_thread.start()
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
            background_thread.join()

    @torch.inference_mode()
    async def inference_stream_async(
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        """
        Asynchronous inference streaming method, integrated from original Pipeline.pipeline_async() logic
        """
        begin_time = current_time_ms()

        if request_id is None:
            request_id = request_counter.increment()

        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.extract_generation_config(generate_config_json, **kwargs)
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
        token_ids = self.encode(prompt)

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.encode(
                generate_config.sp_advice_prompt
            )

        kmonitor.report(
            GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
        )
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.max_num_beams())
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))

        async for response in self.generate_stream(
            request_id, token_ids, mm_inputs, generate_config, **kwargs
        ):
            yield response

    @torch.inference_mode()
    async def generate_stream(
        self,
        request_id: int,
        token_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        """
        Core streaming generation method, integrated from original Pipeline.generate_stream() logic
        """
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
        output_tokens_list: List[torch.Tensor] = []
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
                    output_tokens_list,
                ) = self.token_decoder.decode_incremental_tokens(
                    generate_config,
                    generate_outputs_cache,
                    stop_word_strs,
                    stop_word_str_slices,
                    stop_word_ids,
                    stop_word_id_slices,
                    decoding_states,
                    token_buffers,
                    output_tokens_list,
                    **kwargs,
                )
            else:
                (
                    generate_texts,
                    output_lens,
                    output_tokens_list,
                ) = self.token_decoder.decode_non_incremental_tokens(
                    generate_config,
                    generate_outputs_cache,
                    stop_word_strs,
                    stop_word_str_slices,
                    stop_word_ids,
                    stop_word_id_slices,
                    output_tokens_list,
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

    def is_streaming(self, req: Dict[str, Any]) -> bool:
        """Check if request is streaming"""
        return RequestExtractor.is_streaming(req) or req.get("stream", False)
