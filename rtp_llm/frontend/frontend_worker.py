import asyncio
import json
import logging
import os
import pathlib
import queue
import sys
import threading
import traceback
from functools import partial
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch

from rtp_llm.config.py_config_modules import StaticConfig

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))

from dataclasses import asdict

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
from rtp_llm.utils.response_utils import append_string_field, update_field_with_latest
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


class GenerationResponse(BaseModel):
    response: str = ""
    finished: bool = True
    aux_info: Dict[str, Any] = {}
    hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    loss: Optional[Union[float, List[float]]] = None
    logits: Optional[Union[List[float], List[List[float]]]] = None
    output_ids: Optional[List[List[int]]] = None
    input_ids: Optional[List[List[int]]] = None


class MultiSequenceGenerationResponse(BaseModel):
    response: List[str]
    finished: bool
    aux_info: List[Dict[str, Any]] = {}


class BatchGenerationResponse(BaseModel):
    response_batch: List[Union[GenerationResponse, MultiSequenceGenerationResponse]]


class TokenizationResponse(BaseModel):
    token_ids: List[int] = []
    offset_mapping: Optional[List[Any]] = None
    tokens: List[str] = []
    error: str = ""


class FrontendWorker:
    def __init__(self, model_config, tokenizer, backend_rpc_server_visitor) -> None:
        logging.info("starting frontend worker")
        self.model_config = model_config
        self.tokenizer = tokenizer
        self._special_tokens: int = self.model_config.special_tokens
        self._mm_token: str = self.model_config.mm_related_params.special_tokens.get(
            "default_mm_token", ""
        )
        self.backend_rpc_server_visitor = backend_rpc_server_visitor
        logging.info("frontend worker start done.")

    def handle_request(self, **kwargs: Any) -> CompleteResponseAsyncGenerator:
        default_generate_config = GenerateConfig()
        request_extractor = RequestExtractor(default_generate_config)
        request, kwargs = request_extractor.extract_request(kwargs)

        if request.is_streaming is False and request.incremental:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "request is non_stream but use incremental decoder",
            )

        response_generator = self._create_generation_streams(request, **kwargs)

        complete_response_collect_func = partial(
            FrontendWorker.aggregate_incremental_responses,
            incremental=request.incremental,
            batch_infer=request.batch_infer,
            num_return_sequences=request.num_return_sequences,
        )
        return CompleteResponseAsyncGenerator(
            response_generator, complete_response_collect_func
        )

    def _create_generation_streams(self, request: Request, **kwargs: Any):
        if (
            len(request.input_texts) > 1
            or request.batch_infer
            or request.num_return_sequences > 0
        ):
            num_return_sequences = request.generate_configs[0].num_return_sequences
            generators: List[AsyncGenerator[Dict[str, Any], None]] = []
            # TODO temp fix sp with batch infer, will change request_id to str later
            for i, (text, urls, generate_config) in enumerate(
                zip(request.input_texts, request.input_urls, request.generate_configs)
            ):
                generators.append(
                    self._stream_formatted_results(
                        request.request_id + i * 10000,
                        text,
                        urls,
                        generate_config=generate_config,
                        **kwargs,
                    )
                )
            return self._merge_parallel_streams(
                request.incremental,
                generators,
                request.batch_infer,
            )
        else:
            return self._stream_formatted_results(
                request.request_id,
                request.input_texts[0],
                request.input_urls[0],
                generate_config=request.generate_configs[0],
                **kwargs,
            )

    def _format_generation_response(
        self, gen_responses: GenerateResponse, generate_config: GenerateConfig
    ) -> Dict[str, Any]:
        """Format generation response to standard output format"""
        generate_texts = gen_responses.generate_texts

        # Multi-sequence response
        if generate_config.num_return_sequences > 0:
            aux_info = (
                [
                    asdict(seq.aux_info)
                    for seq in gen_responses.generate_outputs.generate_outputs
                ]
                if generate_config.aux_info
                else []
            )

            return MultiSequenceGenerationResponse(
                response=generate_texts,
                finished=all(
                    seq.finished
                    for seq in gen_responses.generate_outputs.generate_outputs
                ),
                aux_info=aux_info,
            )

        # Single sequence response
        first_output = gen_responses.generate_outputs.generate_outputs[0]
        aux_info = first_output.aux_info if generate_config.aux_info else None

        # Handle special case for beam search
        if aux_info and generate_config.has_num_beams():
            aux_info.beam_responses = generate_texts

        return GenerationResponse(
            response=generate_texts[0],
            finished=first_output.finished,
            aux_info=asdict(aux_info) if aux_info else {},
            hidden_states=self._to_list_if_needed(
                first_output.hidden_states, generate_config.return_hidden_states
            ),
            loss=self._to_list_if_needed(
                first_output.loss, generate_config.calculate_loss
            ),
            logits=self._to_list_if_needed(
                first_output.logits, generate_config.return_logits
            ),
            output_ids=self._to_list_if_needed(
                first_output.output_ids, generate_config.return_output_ids
            ),
            input_ids=self._to_list_if_needed(
                first_output.input_ids, generate_config.return_input_ids
            ),
        )

    @staticmethod
    def _to_list_if_needed(tensor, should_convert: bool):
        """Helper method: conditionally convert tensor to list"""
        return tensor.tolist() if should_convert and tensor is not None else None

    async def _stream_formatted_results(
        self,
        request_id: int,
        text: str,
        urls: List[str],
        generate_config: GenerateConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        stream = self.generate_async(
            prompt=text,
            request_id=request_id,
            urls=urls,
            generate_config=generate_config,
            **kwargs,
        )
        async for generate_response in stream:
            yield self._format_generation_response(generate_response, generate_config)

    async def _merge_parallel_streams(
        self,
        incremental: bool,
        generators: List[AsyncGenerator[Dict[str, Any], None]],
        batch_infer: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Merge multiple generator output streams in parallel"""
        iterators = [gen.__aiter__() for gen in generators]
        done_idxs: Set[int] = set()
        batch_state: List[Any] = [None] * len(iterators)

        while len(done_idxs) < len(iterators):
            # Collect tasks from unfinished iterators
            pending_tasks = [
                (idx, itr.__anext__())
                for idx, itr in enumerate(iterators)
                if idx not in done_idxs
            ]

            if not pending_tasks:
                break

            # Execute all tasks in parallel
            results = await asyncio.gather(
                *(task for _, task in pending_tasks), return_exceptions=True
            )

            # Process each result
            for (idx, _), result in zip(pending_tasks, results):
                if isinstance(result, StopAsyncIteration):
                    self._handle_stream_completion(
                        batch_state, idx, incremental, done_idxs
                    )
                elif isinstance(result, Exception):
                    self._handle_stream_error(result)
                else:
                    batch_state[idx] = result

            # Yield current batch state
            yield (
                BatchGenerationResponse(response_batch=batch_state)
                if batch_infer
                else batch_state[0]
            )

    @staticmethod
    def _handle_stream_completion(
        batch_state: List[Any], idx: int, incremental: bool, done_idxs: Set[int]
    ):
        """Handle stream completion"""
        done_idxs.add(idx)
        if batch_state[idx] is None or incremental:
            batch_state[idx] = GenerationResponse()

    @staticmethod
    def _handle_stream_error(error: Exception):
        """Handle stream error"""
        error_msg = f'ErrorMsg: {str(error)} \n Traceback: {"".join(traceback.format_tb(error.__traceback__))}'
        logging.warning(error_msg)
        raise error

    @staticmethod
    async def aggregate_incremental_responses(
        all_responses: List[
            Union[
                GenerationResponse,
                MultiSequenceGenerationResponse,
                BatchGenerationResponse,
            ]
        ],
        incremental: bool,
        batch_infer: bool,
        num_return_sequences: int,
    ) -> Union[
        GenerationResponse, MultiSequenceGenerationResponse, BatchGenerationResponse
    ]:
        """Aggregate incremental responses into complete response"""
        if not incremental:
            return await CompleteResponseAsyncGenerator.get_last_value(all_responses)

        if batch_infer:
            return await FrontendWorker._aggregate_batch_responses(
                all_responses, num_return_sequences
            )

        if num_return_sequences > 0:
            return await FrontendWorker._aggregate_multi_sequence_responses(
                all_responses
            )

        return await FrontendWorker._aggregate_single_responses(all_responses)

    @staticmethod
    async def _aggregate_batch_responses(all_responses, num_return_sequences: int):
        """Aggregate batch responses"""
        batch_response_streams = None

        async for response in all_responses:
            if batch_response_streams is None:
                batch_response_streams = [[item] for item in response.response_batch]
            else:
                for idx, single_response in enumerate(response.response_batch):
                    batch_response_streams[idx].append(single_response)

        complete_batch_response = []
        async for stream in CompleteResponseAsyncGenerator.generate_from_list(
            batch_response_streams
        ):
            single_response_gen = CompleteResponseAsyncGenerator.generate_from_list(
                stream
            )
            complete_response = await FrontendWorker.aggregate_incremental_responses(
                single_response_gen, True, False, num_return_sequences
            )
            complete_batch_response.append(complete_response)

        return BatchGenerationResponse(response_batch=complete_batch_response)

    @staticmethod
    async def _aggregate_multi_sequence_responses(all_responses):
        """Aggregate multi-sequence responses"""
        responses = None
        aux_infos = None
        finished = False

        async for response in all_responses:
            if responses is None:
                responses = list(response.response)
                aux_infos = list(response.aux_info)
                finished = response.finished

            for idx, seq_response in enumerate(response.response):
                responses[idx] = append_string_field(responses[idx], seq_response)
                if response.aux_info and response.aux_info[idx]:
                    aux_infos[idx] = response.aux_info[idx]

            if response.finished:
                finished = True

        return MultiSequenceGenerationResponse(
            response=responses,
            aux_info=aux_infos,
            finished=finished,
        )

    @staticmethod
    async def _aggregate_single_responses(all_responses):
        """Aggregate single responses"""
        result = GenerationResponse(response="", finished=False)

        async for response in all_responses:
            result.response = append_string_field(result.response, response.response)
            result.finished = result.finished or response.finished
            result.aux_info = update_field_with_latest(
                result.aux_info, response.aux_info
            )
            result.output_ids = update_field_with_latest(
                result.output_ids, response.output_ids
            )
            result.input_ids = update_field_with_latest(
                result.input_ids, response.input_ids
            )
            result.logits = update_field_with_latest(result.logits, response.logits)

        return result

    @staticmethod
    def build_generation_config(
        generate_config: Union[GenerateConfig, Dict[str, Any]],
        vocab_size: int,
        special_tokens: Any,
        tokenizer: BaseTokenizer,
        **kwargs: Any,
    ) -> GenerateConfig:
        if isinstance(generate_config, dict):
            config = GenerateConfig.create_generate_config(generate_config, **kwargs)
        else:
            # Assume it's passed from frontend_worker, no need to process again
            config = generate_config
        config.add_special_tokens(special_tokens)
        config.convert_select_tokens(vocab_size, tokenizer)
        config.add_thinking_params(tokenizer)
        config.add_stop_ids_from_str(tokenizer)
        return config

    def generate_sync(
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[GenerateResponse]:
        """Synchronous generation interface, bridging async generator via thread and queue"""
        result_queue = queue.Queue()

        async def async_generator_wrapper():
            """Async generator wrapper that puts results into queue"""
            generator = None
            try:
                generator = self.generate_async(prompt, request_id, urls, **kwargs)
                async for result in generator:
                    result_queue.put(result)
                result_queue.put(None)  # Completion marker
            except Exception as e:
                result_queue.put(e)
            finally:
                if generator is not None:
                    await generator.aclose()

        def run_async_in_thread():
            """Run event loop in new thread"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(async_generator_wrapper())

        # Start background thread
        background_thread = threading.Thread(target=run_async_in_thread, daemon=False)
        background_thread.start()

        try:
            while True:
                try:
                    result = result_queue.get(timeout=0.01)
                    if result is None:  # Completion marker
                        break
                    if isinstance(result, Exception):
                        raise result
                    yield result
                except queue.Empty:
                    continue
        finally:
            background_thread.join()

    @torch.inference_mode()
    def generate_async(  # type: ignore
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()

        if request_id == None:
            request_id = request_counter.increment()

        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.build_generation_config(
            generate_config_json,
            len(self.tokenizer),
            self.model_config.special_tokens,
            self.tokenizer,
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
        return self.stream_from_backend(
            request_id, token_ids, mm_inputs, generate_config, **kwargs
        )

    def _truncate_by_stop_token_ids(
        self,
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        tokens: List[int],
        stop_word_ids: List[List[int]],
        stop_word_id_slices: List[List[int]],
    ) -> List[int]:
        """Truncate tokens by stop word IDs"""
        if generate_config.print_stop_words:
            return tokens

        stop_patterns = (
            stop_word_ids if generate_output.finished else stop_word_id_slices
        )
        return truncate_token_with_stop_word_id(tokens, stop_patterns)

    def _truncate_by_stop_strings(
        self,
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        text: str,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        token_buffer: str,
    ) -> Tuple[str, str]:
        """Truncate text by stop word strings"""
        if generate_config.return_incremental:
            text = append_string_field(token_buffer, text)

        # Check for complete stop word match
        if stop_word_str_list:
            stop_idx, stop_len = match_stop_words(text, stop_word_str_list)
            if stop_idx != -1:
                text = (
                    text[: stop_idx + stop_len]
                    if generate_config.print_stop_words
                    else text[:stop_idx]
                )
                generate_output.finished = True
                return text, ""

        if generate_output.finished:
            return text, token_buffer

        # Truncate partially matched stop words
        if generate_config.return_incremental or not generate_config.print_stop_words:
            truncated = truncate_response_with_stop_words(
                text, stop_word_str_slices, generate_config.is_streaming, True
            )
            new_buffer = (
                text[len(truncated) :] if generate_config.return_incremental else ""
            )
            return truncated, new_buffer

        return text, token_buffer

    async def decode_complete_tokens(
        self,
        output_generator: AsyncGenerator[GenerateOutputs, None],
        generate_config: GenerateConfig,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        """Decode complete tokens (for beam search or non-streaming mode)"""
        output_tokens_list: List[torch.Tensor] = []
        generate_outputs_cache = GenerateOutputs()

        async for generate_outputs in output_generator:
            # Update cache (keep finished outputs)
            if not generate_outputs_cache.generate_outputs:
                generate_outputs_cache.generate_outputs = (
                    generate_outputs.generate_outputs
                )
            else:
                generate_outputs_cache.generate_outputs = [
                    (
                        cached_out
                        if cached_out.finished
                        else generate_outputs.generate_outputs[i]
                    )
                    for i, cached_out in enumerate(
                        generate_outputs_cache.generate_outputs
                    )
                ]

            # Decode tokens
            begin_time = current_time_ms()

            # Prepare token lists
            tokens_for_decode = self._prepare_tokens_for_complete_decode(
                generate_config, generate_outputs_cache, output_tokens_list
            )

            # Truncate stop word tokens and decode
            generate_texts, output_lens = self._decode_and_truncate_tokens(
                generate_config,
                generate_outputs_cache,
                tokens_for_decode,
                stop_word_str_list,
                stop_word_str_slices,
                stop_word_ids,
                stop_word_id_slices,
                **kwargs,
            )

            kmonitor.report(
                GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )

            yield GenerateResponse(
                generate_outputs=generate_outputs_cache, generate_texts=generate_texts
            )

            # Check if all finished and report metrics
            if all(out.finished for out in generate_outputs_cache.generate_outputs):
                if generate_config.aux_info:
                    kmonitor.report(
                        GaugeMetrics.FT_ITERATE_COUNT_METRIC,
                        generate_outputs_cache.generate_outputs[0].aux_info.iter_count,
                    )
                if output_lens:
                    kmonitor.report(
                        GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC,
                        sum(output_lens) / len(output_lens),
                    )
                break

    def _prepare_tokens_for_complete_decode(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        output_tokens_list: List[torch.Tensor],
    ) -> List[List[int]]:
        """Prepare tokens for complete decoding"""
        if generate_config.has_num_beams():
            return self._prepare_beam_tokens(generate_config, generate_outputs)

        return self._prepare_regular_tokens(
            generate_config, generate_outputs, output_tokens_list
        )

    def _prepare_beam_tokens(
        self, generate_config: GenerateConfig, generate_outputs: GenerateOutputs
    ):
        """Prepare tokens for beam search"""
        all_output_ids = torch.cat(
            [go.output_ids for go in generate_outputs.generate_outputs], dim=0
        )
        all_output_ids_np = all_output_ids.cpu().numpy()

        if generate_config.ignore_eos:
            return all_output_ids_np.tolist()

        processed_tokens = batch_remove_padding_eos(
            all_output_ids_np, self._special_tokens.eos_token_id
        )
        return [tokens.tolist() for tokens in processed_tokens]

    def _prepare_regular_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        output_tokens_list: List[torch.Tensor],
    ):
        """Prepare regular tokens"""
        if not output_tokens_list:
            output_tokens_list[:] = [
                torch.empty(0, dtype=torch.int32)
                for _ in range(len(generate_outputs.generate_outputs))
            ]

        tokens_lists = []
        for i, generate_output in enumerate(generate_outputs.generate_outputs):
            # Accumulate tokens
            if len(output_tokens_list[i]) == 0:
                output_tokens_list[i] = generate_output.output_ids
            else:
                output_tokens_list[i] = torch.cat(
                    (output_tokens_list[i], generate_output.output_ids), dim=1
                )
                generate_output.output_ids = output_tokens_list[i]

            # Handle EOS
            tokens = generate_output.output_ids.cpu().numpy().flatten()
            if not generate_config.ignore_eos:
                tokens = remove_padding_eos_with_numpy(
                    tokens, self._special_tokens.eos_token_id
                )
            else:
                tokens = tokens.reshape(-1)

            tokens_lists.append(tokens)

        return tokens_lists

    def _decode_and_truncate_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        tokens_lists: List[List[int]],
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        **kwargs: Any,
    ) -> Tuple[List[str], List[int]]:
        """Decode tokens and truncate stop words"""
        output_lens = []
        token_lists_to_decode = []

        # Truncate stop word tokens
        for i, (tokens_list, generate_output) in enumerate(
            zip(tokens_lists, generate_outputs.generate_outputs)
        ):
            output_lens.append(len(tokens_list))
            processed_tokens = self._truncate_by_stop_token_ids(
                generate_config,
                generate_output,
                tokens_list,
                stop_word_ids,
                stop_word_id_slices,
            )
            token_lists_to_decode.append(processed_tokens)

        # Batch decode
        decoded_batch = self.tokenizer.batch_decode(
            token_lists_to_decode,
            skip_special_tokens=generate_config.skip_special_tokens,
            **kwargs,
        )
        decoded_texts = [text.rstrip("\uFFFD") for text in decoded_batch]

        # Truncate stop word strings and add prefix
        final_texts = []
        for i, (text, generate_output) in enumerate(
            zip(decoded_texts, generate_outputs.generate_outputs)
        ):
            processed_text, _ = self._truncate_by_stop_strings(
                generate_config,
                generate_output,
                text,
                stop_word_str_list,
                stop_word_str_slices,
                "",
            )

            if generate_config.out_prefix:
                processed_text = generate_config.out_prefix + processed_text

            final_texts.append(processed_text)

        return final_texts, output_lens

    async def decode_streaming_tokens(
        self,
        output_generator: AsyncGenerator[GenerateOutputs, None],
        generate_config: GenerateConfig,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        """Streaming incremental token decoding"""
        decoding_states: List[DecodingState] = []
        output_tokens_list: List[torch.Tensor] = []
        token_buffers: List[str] = []
        generate_outputs_cache = GenerateOutputs()

        async for generate_outputs in output_generator:
            # Update cache (keep finished outputs)
            if not generate_outputs_cache.generate_outputs:
                generate_outputs_cache.generate_outputs = (
                    generate_outputs.generate_outputs
                )
            else:
                generate_outputs_cache.generate_outputs = [
                    (
                        cached_out
                        if cached_out.finished
                        else generate_outputs.generate_outputs[i]
                    )
                    for i, cached_out in enumerate(
                        generate_outputs_cache.generate_outputs
                    )
                ]

            # Decode tokens
            begin_time = current_time_ms()

            num_outputs = len(generate_outputs_cache.generate_outputs)

            # Initialize states
            token_buffers = token_buffers or [""] * num_outputs
            decoding_states = decoding_states or [
                DecodingState() for _ in range(num_outputs)
            ]
            output_tokens_list = output_tokens_list or [
                torch.empty(0, dtype=torch.int32) for _ in range(num_outputs)
            ]

            # Incrementally decode each output
            newly_decoded_texts = []
            all_texts = []
            output_lens = []

            for i, generate_output in enumerate(
                generate_outputs_cache.generate_outputs
            ):
                # Accumulate tokens
                output_tokens_list[i] = torch.cat(
                    (output_tokens_list[i], generate_output.output_ids), dim=1
                )
                tokens_np = output_tokens_list[i].cpu().numpy().flatten()

                # Handle EOS
                if not generate_config.ignore_eos:
                    tokens_list = remove_padding_eos_with_numpy(
                        tokens_np, self._special_tokens.eos_token_id
                    ).tolist()
                else:
                    tokens_list = tokens_np.tolist()

                output_lens.append(len(tokens_list))

                # Truncate stop word tokens
                processed_tokens = self._truncate_by_stop_token_ids(
                    generate_config,
                    generate_output,
                    tokens_list,
                    stop_word_ids,
                    stop_word_id_slices,
                )

                # Incremental decoding
                new_text = IncrementDecodingUtils.detokenize_incrementally(
                    self.tokenizer, processed_tokens, decoding_states[i]
                )
                decoding_states[i].all_text += new_text

                text_to_return = (
                    new_text
                    if generate_config.return_incremental
                    else decoding_states[i].all_text
                )
                newly_decoded_texts.append(text_to_return)
                all_texts.append(decoding_states[i].all_text)

            # Truncate stop word strings and add prefix
            generate_texts = []
            for i in range(num_outputs):
                text, token_buffers[i] = self._truncate_by_stop_strings(
                    generate_config,
                    generate_outputs_cache.generate_outputs[i],
                    newly_decoded_texts[i],
                    stop_word_str_list,
                    stop_word_str_slices,
                    token_buffers[i],
                )

                if generate_config.out_prefix:
                    text = generate_config.out_prefix + text

                generate_texts.append(text)

            kmonitor.report(
                GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )

            yield GenerateResponse(
                generate_outputs=generate_outputs_cache, generate_texts=generate_texts
            )

            # Check if all finished and report metrics
            if all(out.finished for out in generate_outputs_cache.generate_outputs):
                if generate_config.aux_info:
                    kmonitor.report(
                        GaugeMetrics.FT_ITERATE_COUNT_METRIC,
                        generate_outputs_cache.generate_outputs[0].aux_info.iter_count,
                    )
                if output_lens:
                    kmonitor.report(
                        GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC,
                        sum(output_lens) / len(output_lens),
                    )
                break

    @torch.inference_mode()
    async def stream_from_backend(
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

        # Prepare stop word matching patterns
        stop_word_strs = generate_config.stop_words_str
        stop_word_str_slices = get_stop_word_slices(stop_word_strs)
        stop_word_ids = generate_config.stop_words_list
        stop_word_id_slices = get_stop_word_slices(stop_word_ids)

        output_generator: AsyncGenerator[GenerateOutputs, None] = (
            await self.backend_rpc_server_visitor.enqueue(input)
        )

        is_incremental = (
            not generate_config.has_num_beams() and generate_config.is_streaming
        )

        if is_incremental:
            # Streaming decoding path
            async for response in self.decode_streaming_tokens(
                output_generator,
                generate_config,
                stop_word_strs,
                stop_word_str_slices,
                stop_word_ids,
                stop_word_id_slices,
                **kwargs,
            ):
                yield response
        else:
            # Complete decoding path
            async for response in self.decode_complete_tokens(
                output_generator,
                generate_config,
                stop_word_strs,
                stop_word_str_slices,
                stop_word_ids,
                stop_word_id_slices,
                **kwargs,
            ):
                yield response
