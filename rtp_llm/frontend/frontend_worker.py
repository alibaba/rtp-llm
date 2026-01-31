import asyncio
import json
import logging
import os
import pathlib
import sys
import traceback
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))

from dataclasses import asdict

from pydantic import BaseModel

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.model_config import (
    update_stop_words_from_env,
    update_tokenizer_special_tokens,
)
from rtp_llm.distribute.distributed_server import WorldInfo, get_world_info
from rtp_llm.distribute.worker_info import ParallelInfo, g_parallel_info, g_worker_info
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.ops import ParallelismConfig, SpecialTokens, VitSeparation
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.structure.request_extractor import Request, RequestExtractor
from rtp_llm.utils.base_model_datatypes import GenerateResponse
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


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


def get_dp_addrs_from_world_info(
    world_info: WorldInfo, parallelism_config: ParallelismConfig
) -> list[str]:
    """Get data parallel addresses from world_info.

    Args:
        world_info: WorldInfo containing all worker members
        parallelism_config: ParallelismConfig containing parallelism configuration
        address: Optional address to use when dp_size == 1 (defaults to localhost:rpc_server_port)

    Returns:
        List of RPC addresses for data parallel communication
    """
    addresses = []

    ffn_disaggregate_config = parallelism_config.ffn_disaggregate_config
    # If FFN disaggregate is enabled, limit addresses to serving ranks
    if ffn_disaggregate_config.enable_ffn_disaggregate:
        serving_ranks = (
            ffn_disaggregate_config.attention_tp_size
            * ffn_disaggregate_config.attention_dp_size
        )
        members = world_info.members[:serving_ranks]
        logging.info(
            f"FFN disaggregate enabled, limiting addresses to {serving_ranks} serving ranks: {members}"
        )
    else:
        # Get all addresses from world_info members with tp_rank == 0
        members = [
            member
            for member in world_info.members
            if (member.world_rank % parallelism_config.tp_size) == 0
        ]

    addresses = [f"{member.ip}:{member.rpc_server_port}" for member in members]
    logging.info(
        f"[world_rank: {parallelism_config.world_rank}] "
        f"using addresses from world_info: {addresses}"
    )

    return addresses


class FrontendWorker:
    def __init__(self, py_env_configs, model_config, special_tokens) -> None:
        logging.info("starting frontend worker")

        self.tokenizer = TokenizerFactory.create(
            model_config.ckpt_path, model_config.tokenizer_path, model_config.model_type
        )

        # Create engine_config with world_info
        engine_config = EngineConfig.create(py_env_configs)

        # Get world_info from distribute_config
        world_info = get_world_info(
            server_config=py_env_configs.server_config,
            distribute_config=py_env_configs.distribute_config,
        )

        # Get addresses from distribute_info
        addresses = get_dp_addrs_from_world_info(
            world_info=world_info,
            parallelism_config=engine_config.parallelism_config,
        )

        vit_separation = None
        if py_env_configs.vit_config:
            vit_separation = py_env_configs.vit_config.vit_separation

        self.pipeline = Pipeline(
            special_tokens=special_tokens,
            pd_sep_config=engine_config.pd_sep_config,
            addresses=addresses,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.attn_config.tokens_per_block,
            tokenizer=self.tokenizer,
            sp_config=py_env_configs.sp_config,
            mm_related_params=None,  # Frontend doesn't need mm_related_params
            grpc_config=py_env_configs.grpc_config,
            vit_separation=vit_separation,
        )
        self.backend_rpc_server_visitor = self.pipeline.backend_rpc_server_visitor
        self.generate_env_config = py_env_configs.generate_env_config

        logging.info("frontend worker start done.")

    def tokenizer_offset_mapping(self, prompt: str) -> Any:
        return self.pipeline.tokenizer(
            prompt, return_offsets_mapping=True, return_attention_mask=False
        )

    def tokenizer_encode(self, prompt: str) -> Tuple[List[int], List[str]]:
        token_ids = self.pipeline.encode(prompt)
        token_ids = [int(id) for id in token_ids]
        tokens = [self.pipeline.decode(id) for id in token_ids]
        return token_ids, tokens

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
                        request.request_id,
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
        stream = self.pipeline.pipeline_async(
            prompt=text,
            request_id=request_id,
            urls=urls,
            generate_config=generate_config,
            generate_env_config=self.generate_env_config,
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
