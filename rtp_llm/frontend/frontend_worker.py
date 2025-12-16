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
from rtp_llm.frontend.base_endpoint import BaseEndpoint
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.structure.request_extractor import (
    Request,
    RequestExtractor,
    request_id_field_name,
)
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter, check_with_info
from rtp_llm.utils.word_util import (
    batch_remove_padding_eos,
    get_stop_word_slices,
    match_stop_words,
    remove_padding_eos_with_numpy,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)

request_counter = AtomicCounter()


class PipelineResponse(BaseModel):
    response: str = ""
    finished: bool = True
    aux_info: Dict[str, Any] = {}
    hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    loss: Optional[Union[float, List[float]]] = None
    logits: Optional[Union[List[float], List[List[float]]]] = None
    output_ids: Optional[List[List[int]]] = None
    input_ids: Optional[List[List[int]]] = None


class MultiSequencesPipelineResponse(BaseModel):
    response: List[str]
    finished: bool
    aux_info: List[Dict[str, Any]] = {}


class BatchPipelineResponse(BaseModel):
    response_batch: List[Union[PipelineResponse, MultiSequencesPipelineResponse]]


class TokenizerEncodeResponse(BaseModel):
    token_ids: List[int] = []
    offset_mapping: Optional[List[Any]] = None
    tokens: List[str] = []
    error: str = ""


class FrontendWorker(BaseEndpoint):
    def __init__(
        self,
        model_config,
        tokenizer,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
        rank_id,
        server_id,
    ) -> None:
        logging.info("starting frontend worker")
        super().__init__(
            model_config, tokenizer, backend_rpc_server_visitor, rank_id, server_id
        )
        self._special_tokens: int = self.model_config.special_tokens
        self._mm_token: str = self.model_config.mm_related_params.special_tokens.get(
            "default_mm_token", ""
        )
        logging.info("frontend worker start done.")

    def _check_request(self, req: Dict[str, Any], req_id: int) -> None:
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            if "master_info" in req:
                request_id = req["master_info"].get("request_id")
                check_with_info(
                    request_id != None and isinstance(request_id, int),
                    "request_id in master_info is None or not int",
                )
                req[request_id_field_name] = request_id
            else:
                req[request_id_field_name] = req_id
            return req
        except Exception as e:
            return self._handle_exception(req, e)

    def inference(self, **kwargs: Any) -> CompleteResponseAsyncGenerator:
        default_generate_config = GenerateConfig()
        request_extractor = RequestExtractor(default_generate_config)
        request, kwargs = request_extractor.extract_request(kwargs)

        if request.is_streaming is False and request.incremental:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "request is non_stream but use incremental decoder",
            )

        response_generator = self._inference(request, **kwargs)

        complete_response_collect_func = partial(
            FrontendWorker.collect_complete_response,
            incremental=request.incremental,
            batch_infer=request.batch_infer,
            num_return_sequences=request.num_return_sequences,
        )
        return CompleteResponseAsyncGenerator(
            response_generator, complete_response_collect_func
        )

    def _inference(self, request: Request, **kwargs: Any):
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
                    self._yield_generate(
                        request.request_id + i * 10000,
                        text,
                        urls,
                        generate_config=generate_config,
                        **kwargs,
                    )
                )
            return self._parallel_batch_async_generators(
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

    def _format_response(
        self, gen_responses: GenerateResponse, generate_config: GenerateConfig
    ) -> Dict[str, Any]:
        generate_texts = gen_responses.generate_texts
        finished = gen_responses.generate_outputs.generate_outputs[0].finished
        if generate_config.aux_info:
            aux_info = gen_responses.generate_outputs.generate_outputs[0].aux_info
            if generate_config.has_num_beams():
                aux_info.beam_responses = generate_texts
        hidden_states = gen_responses.generate_outputs.generate_outputs[0].hidden_states
        output_ids = gen_responses.generate_outputs.generate_outputs[0].output_ids
        input_ids = gen_responses.generate_outputs.generate_outputs[0].input_ids
        loss = gen_responses.generate_outputs.generate_outputs[0].loss
        logits = gen_responses.generate_outputs.generate_outputs[0].logits

        response = PipelineResponse(
            response=generate_texts[0],
            finished=finished,
            aux_info=asdict(aux_info) if generate_config.aux_info else {},
            hidden_states=(
                hidden_states.tolist()
                if generate_config.return_hidden_states and hidden_states is not None
                else None
            ),
            loss=(
                loss.tolist()
                if generate_config.calculate_loss and loss is not None
                else None
            ),
            logits=(
                logits.tolist()
                if generate_config.return_logits and logits is not None
                else None
            ),
            output_ids=(
                output_ids.tolist()
                if generate_config.return_output_ids and output_ids is not None
                else None
            ),
            input_ids=(
                input_ids.tolist()
                if generate_config.return_input_ids and input_ids is not None
                else None
            ),
        )

        return response

    def _format_response_new(
        self, gen_responses: GenerateResponse, generate_config: GenerateConfig
    ) -> Dict[str, Any]:
        generate_texts = gen_responses.generate_texts
        if generate_config.num_return_sequences > 0:
            aux_info = []
            if generate_config.aux_info:
                aux_info = [
                    asdict(seq.aux_info)
                    for seq in gen_responses.generate_outputs.generate_outputs
                ]
            sequences_pipeline_response = MultiSequencesPipelineResponse(
                response=generate_texts,
                finished=all(
                    [
                        seq.finished
                        for seq in gen_responses.generate_outputs.generate_outputs
                    ]
                ),
                aux_info=aux_info,
            )
            return sequences_pipeline_response
        else:
            return self._format_response(gen_responses, generate_config)

    async def _yield_generate(
        self,
        request_id: int,
        text: str,
        urls: List[str],
        generate_config: GenerateConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        stream = self.pipeline_async(
            prompt=text,
            request_id=request_id,
            urls=urls,
            generate_config=generate_config,
            **kwargs,
        )
        async for generate_response in stream:
            yield self._format_response_new(generate_response, generate_config)

    def is_streaming(self, req: Dict[str, Any]):
        return RequestExtractor.is_streaming(req) or req.get("stream", False)

    async def _parallel_batch_async_generators(
        self,
        incremental: bool,
        generators: List[AsyncGenerator[Dict[str, Any], None]],
        batch_infer: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        iterators = [gen.__aiter__() for gen in generators]
        done_idxs: Set[int] = set()
        batch_state: List[Any] = [None] * len(iterators)

        while True:
            # 创建并行任务
            tasks = []
            for idx, itr in enumerate(iterators):
                if idx not in done_idxs:  # 仅为未完成的迭代器创建任务
                    tasks.append((idx, itr.__anext__()))

            # 使用 asyncio.gather() 获取结果
            if tasks:
                results = await asyncio.gather(
                    *(task[1] for task in tasks), return_exceptions=True
                )
                for idx, result in zip((task[0] for task in tasks), results):
                    if isinstance(result, Exception):
                        # 处理异常情况，如 StopAsyncIteration
                        if isinstance(result, StopAsyncIteration):
                            done_idxs.add(idx)
                            if batch_state[idx] is None:
                                batch_state[idx] = PipelineResponse()
                            if incremental:
                                batch_state[idx] = PipelineResponse()
                        else:
                            error_msg = f'ErrorMsg: {str(result)} \n Traceback: {"".join(traceback.format_tb(result.__traceback__))}'
                            logging.warning(error_msg)
                            raise result
                    else:
                        batch_state[idx] = result

            # 检查是否所有迭代器都完成
            if len(done_idxs) == len(iterators):
                break

            # 处理 batch 数据
            batch = batch_state
            if batch_infer:
                yield BatchPipelineResponse(response_batch=batch)
            else:
                yield batch[0]

    # TODO(xinfei.sxf) 这个函数将逻辑又写了一遍
    @staticmethod
    async def collect_complete_response(
        all_responses: List[
            Union[
                PipelineResponse, MultiSequencesPipelineResponse, BatchPipelineResponse
            ]
        ],
        incremental: bool,
        batch_infer: bool,
        num_return_sequences: int,
    ) -> Union[PipelineResponse, MultiSequencesPipelineResponse, BatchPipelineResponse]:

        if not incremental:
            return await CompleteResponseAsyncGenerator.get_last_value(all_responses)

        if batch_infer:
            batch_response_incremental_stream = None
            async for response in all_responses:
                if not batch_response_incremental_stream:
                    batch_response_incremental_stream = [
                        [_] for _ in response.response_batch
                    ]
                else:
                    for batch_idx, single_response in enumerate(
                        response.response_batch
                    ):
                        batch_response_incremental_stream[batch_idx].append(
                            single_response
                        )
            complete_batch_response = []
            async for (
                single_response_incremental_stream
            ) in CompleteResponseAsyncGenerator.generate_from_list(
                batch_response_incremental_stream
            ):
                single_yield_response = (
                    CompleteResponseAsyncGenerator.generate_from_list(
                        single_response_incremental_stream
                    )
                )
                single_complete_response = (
                    await FrontendWorker.collect_complete_response(
                        single_yield_response, incremental, False, num_return_sequences
                    )
                )
                complete_batch_response.append(single_complete_response)
            return BatchPipelineResponse(response_batch=complete_batch_response)

        if num_return_sequences > 0:
            complete_multi_seq_response = None
            complete_multi_seq_finished = None
            complete_multi_seq_aux_info = None
            async for response in all_responses:
                if not complete_multi_seq_response:
                    complete_multi_seq_response = [_ for _ in response.response]
                    complete_multi_seq_aux_info = [_ for _ in response.aux_info]
                    complete_multi_seq_finished = response.finished
                for seq_idx, seq_reponse in enumerate(response.response):
                    complete_multi_seq_response[seq_idx] = (
                        complete_multi_seq_response[seq_idx] + seq_reponse
                    )
                    if response.aux_info and response.aux_info[seq_idx]:
                        complete_multi_seq_aux_info[seq_idx] = response.aux_info[
                            seq_idx
                        ]
                    if response.finished:
                        complete_multi_seq_finished = True
            return MultiSequencesPipelineResponse(
                response=complete_multi_seq_response,
                aux_info=complete_multi_seq_aux_info,
                finished=complete_multi_seq_finished,
            )
        complete_response = ""
        finished = False
        aux_info = None
        output_ids = None
        input_ids = None
        logits = None
        async for response in all_responses:
            complete_response = complete_response + response.response
            if response.finished:
                finished = response.finished
            if response.aux_info:
                aux_info = response.aux_info
            if response.output_ids:
                output_ids = response.output_ids
            if response.input_ids:
                input_ids = response.input_ids
            if response.logits:
                logits = response.logits
        return PipelineResponse(
            response=complete_response,
            finished=finished,
            aux_info=aux_info,
            output_ids=output_ids,
            input_ids=input_ids,
            logits=logits,
        )

    @staticmethod
    def create_generate_config(
        generate_config: Union[GenerateConfig, Dict[str, Any]],
        vocab_size: int,
        special_tokens: Any,
        tokenizer: BaseTokenizer,
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()

        if request_id == None:
            request_id = request_counter.increment()

        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.create_generate_config(
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
        **kwargs: Any,
    ):
        if generate_config.return_incremental:
            text = token_buffer + text

        if stop_word_str_list:
            stop_idx, stop_len = match_stop_words(text, stop_word_str_list)
            if stop_idx != -1:
                if not generate_config.print_stop_words:
                    text = text[:stop_idx]
                else:
                    text = text[: stop_idx + stop_len]
                token_buffer = ""
                generate_output.finished = True

        if generate_output.finished:
            return text, token_buffer

        if generate_config.return_incremental or not generate_config.print_stop_words:
            trunc_text = truncate_response_with_stop_words(
                text, stop_word_str_slices, generate_config.is_streaming, True
            )
            if generate_config.return_incremental:
                token_buffer = text[len(trunc_text) :]
            text = trunc_text

        return text, token_buffer

    def decode_non_incremental_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        ouput_tokens_list: List[torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[List[str], List[int]]:
        tokens_lists_for_decode_input = []
        output_lens = []
        token_lists_to_decode = []
        if generate_config.has_num_beams():
            all_output_ids = torch.cat(
                [go.output_ids for go in generate_outputs.generate_outputs], dim=0
            )
            all_output_ids_np = all_output_ids.cpu().numpy()
            if not generate_config.ignore_eos:
                processed_tokens_np_list = batch_remove_padding_eos(
                    all_output_ids_np, self._special_tokens.eos_token_id
                )
                tokens_lists_for_decode_input = [
                    tokens.tolist() for tokens in processed_tokens_np_list
                ]
            else:
                tokens_lists_for_decode_input = all_output_ids_np.tolist()
        else:
            if len(ouput_tokens_list) == 0:
                ouput_tokens_list = [
                    torch.empty(0, dtype=torch.int32)
                    for _ in range(len(generate_outputs.generate_outputs))
                ]
            for i, generate_output in enumerate(generate_outputs.generate_outputs):
                if len(ouput_tokens_list[i]) == 0:
                    ouput_tokens_list[i] = generate_output.output_ids
                else:
                    ouput_tokens_list[i] = torch.cat(
                        (ouput_tokens_list[i], generate_output.output_ids), dim=1
                    )
                    generate_output.output_ids = ouput_tokens_list[i]
                tokens = generate_output.output_ids.cpu().numpy().flatten()
                if not generate_config.ignore_eos:
                    tokens = remove_padding_eos_with_numpy(
                        tokens, self._special_tokens.eos_token_id
                    )
                else:
                    tokens = tokens.reshape(-1)
                tokens_lists_for_decode_input.append(tokens)
        for i, generate_output in enumerate(generate_outputs.generate_outputs):
            tokens_list = tokens_lists_for_decode_input[i]
            output_lens.append(len(tokens_list))
            processed_tokens = self.process_stop_id(
                generate_config,
                generate_output,
                tokens_list,
                stop_word_ids,
                stop_word_id_slices,
            )
            token_lists_to_decode.append(processed_tokens)

        decoded_batch = self.tokenizer.batch_decode(
            token_lists_to_decode,
            skip_special_tokens=generate_config.skip_special_tokens,
            **kwargs,
        )
        newly_decoded_texts = [text.rstrip("\uFFFD") for text in decoded_batch]
        all_texts = newly_decoded_texts

        final_texts = []
        for i in range(len(all_texts)):
            processed_text, _ = self.process_stop_str(
                generate_config,
                generate_outputs.generate_outputs[i],
                newly_decoded_texts[i],
                all_texts[i],
                stop_word_str_list,
                stop_word_str_slices,
                "",
                **kwargs,
            )

            if generate_config.out_prefix:
                processed_text = generate_config.out_prefix + processed_text

            final_texts.append(processed_text)

        return (final_texts, output_lens, ouput_tokens_list)

    def decode_incremental_tokens(
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
        **kwargs: Any,
    ) -> Tuple[List[str], List[int]]:
        """处理增量解码的逻辑。"""
        num_outputs = len(generate_outputs.generate_outputs)
        if len(token_buffers) == 0:
            token_buffers = [""] * num_outputs

        if len(decoding_states) == 0:
            decoding_states = [DecodingState() for _ in range(num_outputs)]

        if len(ouput_tokens_list) == 0:
            ouput_tokens_list = [
                torch.empty(0, dtype=torch.int32) for _ in range(num_outputs)
            ]

        newly_decoded_texts = []
        all_texts = []
        output_lens = []
        ignore_eos = generate_config.ignore_eos
        for i, generate_output in enumerate(generate_outputs.generate_outputs):
            ouput_tokens_list[i] = torch.cat(
                (ouput_tokens_list[i], generate_output.output_ids), dim=1
            )
            full_tokens_tensor = ouput_tokens_list[i]
            tokens_np = full_tokens_tensor.cpu().numpy().flatten()
            if not ignore_eos:
                tokens_list = remove_padding_eos_with_numpy(
                    tokens_np, self._special_tokens.eos_token_id
                ).tolist()
            else:
                tokens_list = tokens_np.tolist()

            output_lens.append(len(tokens_list))

            processed_tokens = self.process_stop_id(
                generate_config,
                generate_output,
                tokens_list,
                stop_word_ids,
                stop_word_id_slices,
            )
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

        final_texts = []
        for i in range(len(all_texts)):
            processed_text, token_buffers[i] = self.process_stop_str(
                generate_config,
                generate_outputs.generate_outputs[i],
                newly_decoded_texts[i],
                all_texts[i],
                stop_word_str_list,
                stop_word_str_slices,
                token_buffers[i],
                **kwargs,
            )

            if generate_config.out_prefix:
                processed_text = generate_config.out_prefix + processed_text

            final_texts.append(processed_text)

        return (
            final_texts,
            output_lens,
            decoding_states,
            token_buffers,
            ouput_tokens_list,
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
                ) = self.decode_incremental_tokens(
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
                ) = self.decode_non_incremental_tokens(
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
