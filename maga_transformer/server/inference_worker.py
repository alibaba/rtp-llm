import os
import copy
import sys
import json
import logging
import pathlib
import torch
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, Union, NamedTuple, AsyncGenerator, Set

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))

from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.utils.util import copy_gemm_config
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.models.base_model import GenerateResponse, GenerateConfig
from maga_transformer.model_factory import ModelFactory, AsyncModel
from maga_transformer.structure.request_extractor import RequestExtractor, Request

from pydantic import BaseModel

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
    tokens: List[str] = []
    error: str = ""

class InferenceWorker():
    def __init__(self) -> None:
        copy_gemm_config()
        logging.info("starting InferenceWorker")
        if not torch.cuda.is_available():
            raise Exception("GPU not found")

        self.model: AsyncModel = ModelFactory.create_from_env()
        self.pipeline = Pipeline(self.model, self.model.tokenizer)
        logging.info("Load model done.")

    def tokenizer_encode(self, prompt: str) -> Tuple[List[int], List[str]]:
        token_ids = self.pipeline.encode(prompt)
        token_ids = [int(id) for id in token_ids]
        tokens = [self.pipeline.decode(id) for id in token_ids]
        return token_ids, tokens

    def inference(self, **kwargs: Any) -> CompleteResponseAsyncGenerator:
        request_extractor = RequestExtractor(self.model.default_generate_config)
        request, kwargs = request_extractor.extract_request(kwargs)

        if request.is_streaming is False and request.incremental:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "request is non_stream but use incremental decoder")

        response_generator = self._inference(request, **kwargs)

        complete_response_collect_func = partial(InferenceWorker.collect_complete_response,
                                                 incremental=request.incremental,
                                                 batch_infer=request.batch_infer,
                                                 num_return_sequences=request.num_return_sequences)
        return CompleteResponseAsyncGenerator(response_generator, complete_response_collect_func)


    def _inference(self, request: Request, **kwargs: Any):
        if len(request.input_texts) > 1 or request.batch_infer or request.num_return_sequences > 0:
            num_return_sequences = request.generate_configs[0].num_return_sequences
            generators = [self._yield_generate(text, images, generate_config=generate_config, **kwargs)
                          for text, images, generate_config in zip(request.input_texts, request.input_images, request.generate_configs)]
            return self._batch_async_generators(request.incremental, num_return_sequences, generators, request.batch_infer)
        else:
            return self._yield_generate(request.input_texts[0], request.input_images[0], generate_config=request.generate_configs[0], **kwargs)

    def stop(self) -> None:
        if isinstance(self.model, AsyncModel):
            logging.info("stoping InferenceWorker")
            self.model.stop()

    def _format_response(self, gen_responses: GenerateResponse, generate_config: GenerateConfig) -> Dict[str, Any]:
        generate_texts = gen_responses.generate_texts
        finished = gen_responses.generate_output.finished
        beam_width = gen_responses.generate_output.output_ids.shape[0]
        aux_info = gen_responses.generate_output.aux_info
        hidden_states = gen_responses.generate_output.hidden_states
        output_ids = gen_responses.generate_output.output_ids
        input_ids = gen_responses.generate_output.input_ids
        loss = gen_responses.generate_output.loss
        logits = gen_responses.generate_output.logits

        if beam_width > 1:
            aux_info.beam_responses = generate_texts
        response = PipelineResponse(
            response=generate_texts[0],
            finished=finished,
            aux_info=aux_info.model_dump(mode='json'),
            hidden_states=hidden_states.tolist() if generate_config.return_hidden_states and hidden_states is not None else None,
            loss=loss.tolist() if generate_config.calculate_loss and loss is not None else None,
            logits=logits.tolist() if generate_config.return_logits and logits is not None else None,
            output_ids=output_ids.tolist() if generate_config.return_output_ids and output_ids is not None else None,
            input_ids=input_ids.tolist() if generate_config.return_input_ids and input_ids is not None else None
        )

        return response

    async def _yield_generate(self, text: str, images: List[str], generate_config: GenerateConfig, **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        stream = self.pipeline.pipeline_async(prompt=text, images=images, generate_config=generate_config, **kwargs)
        async for generate_response in stream:
            yield self._format_response(generate_response, generate_config)

    def is_streaming(self, req: Dict[str, Any]):
        return RequestExtractor.is_streaming(req) or req.get('stream', False)

    def update(self, version_info: VersionInfo):
        lora_infos: Dict[str, Any] = dict()
        if version_info.peft_info != None:
            lora_infos = version_info.peft_info.get("lora_info", {})
        return self.model.update(lora_infos)


    @staticmethod
    async def _batch_async_generators(incremental: bool, num_return_sequences: int,
                                      generators: List[AsyncGenerator[Dict[str, Any], None]],
                                      batch_infer: bool) -> AsyncGenerator[Dict[str, Any], None]:
        iterators = [gen.__aiter__() for gen in generators]
        done_idxs: Set[int] = set()
        batch_state: List[Any] = [None] * len(iterators)
        while True:
            for idx, itr in enumerate(iterators):
                try:
                    batch_state[idx] = await itr.__anext__()
                except StopAsyncIteration:
                    done_idxs.add(idx)
                if idx in done_idxs:
                    if batch_state[idx] is None:
                        batch_state[idx] = PipelineResponse()
                    if incremental:
                        batch_state[idx] = PipelineResponse()
            if len(done_idxs) == len(iterators):
                break
            batch = batch_state
            if num_return_sequences > 0:
                batch_size = int(len(batch_state) / num_return_sequences)
                new_batch: List[Any] = []
                for batch_idx in range(batch_size):
                    seqs = batch_state[batch_idx * num_return_sequences:(batch_idx + 1) * num_return_sequences]
                    sequences_pipeline_response = MultiSequencesPipelineResponse(
                        response=[seq.response for seq in seqs],
                        finished=all([seq.finished for seq in seqs]),
                        aux_info=[seq.aux_info for seq in seqs]
                    )
                    new_batch.append(sequences_pipeline_response)
                batch = new_batch
            if batch_infer:
                yield BatchPipelineResponse(response_batch=batch)
            else:
                yield batch[0]

    @staticmethod
    async def collect_complete_response(
        all_responses: List[Union[PipelineResponse, MultiSequencesPipelineResponse, BatchPipelineResponse]],
        incremental: bool,
        batch_infer: bool,
        num_return_sequences: int
    ) -> Union[PipelineResponse, MultiSequencesPipelineResponse, BatchPipelineResponse]:

        if not incremental:
            return await CompleteResponseAsyncGenerator.get_last_value(all_responses)

        if batch_infer:
            batch_response_incremental_stream = None
            async for response in all_responses:
                if not batch_response_incremental_stream:
                    batch_response_incremental_stream = [[_] for _ in response.response_batch]
                else:
                    for batch_idx, single_response in enumerate(response.response_batch):
                        batch_response_incremental_stream[batch_idx].append(single_response)
            complete_batch_response = []
            async for single_response_incremental_stream in CompleteResponseAsyncGenerator.generate_from_list(batch_response_incremental_stream):
                single_yield_response = CompleteResponseAsyncGenerator.generate_from_list(single_response_incremental_stream)
                single_complete_response = await InferenceWorker.collect_complete_response(single_yield_response, incremental, False, num_return_sequences)
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
                    complete_multi_seq_response[seq_idx] = complete_multi_seq_response[seq_idx] + seq_reponse
                    if response.aux_info and response.aux_info[seq_idx]:
                        complete_multi_seq_aux_info[seq_idx] = response.aux_info[seq_idx]
                    if response.finished:
                        complete_multi_seq_finished = True
            return MultiSequencesPipelineResponse(response=complete_multi_seq_response,
                                                  aux_info=complete_multi_seq_aux_info,
                                                  finished=complete_multi_seq_finished)
        complete_response = ""
        finished = False
        aux_info = None
        output_ids = None
        input_ids = None
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
        return PipelineResponse(
            response=complete_response,
            finished=finished,
            aux_info=aux_info,
            output_ids=output_ids,
            input_ids=input_ids
        )
