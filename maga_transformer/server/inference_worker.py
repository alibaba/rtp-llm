import os
import copy
import sys
import json
import logging
import pathlib
import torch
from typing import Any, Dict, List, AsyncGenerator, Set

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))

from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.utils.util import copy_gemm_config
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.models.base_model import GenerateResponse, GenerateConfig
from maga_transformer.model_factory import ModelFactory, AsyncModel
from maga_transformer.structure.request_extractor import RequestExtractor

class InferenceWorker():
    def __init__(self) -> None:
        copy_gemm_config()
        logging.info("starting InferenceWorker")
        if not torch.cuda.is_available():
            raise Exception("GPU not found")

        self.model = ModelFactory.create_from_env()
        self.pipeline = Pipeline(self.model, self.model.tokenizer)
        logging.info("Load model done.")

    def inference(self, **kwargs: Any):
        request_extractor = RequestExtractor(self.model.default_generate_config)
        request, kwargs = request_extractor.extract_request(kwargs)
        
        if len(request.input_texts) > 1 or request.batch_infer:
            generators = [self._yield_generate(text, images, generate_config=generate_config, **kwargs)
                          for text, images, generate_config in zip(request.input_texts, request.input_images, request.generate_configs)]
            incremental = request.generate_configs[0].return_incremental
            num_return_sequences = request.generate_configs[0].num_return_sequences
            return self._batch_async_generators(incremental, num_return_sequences, generators, request.batch_infer)
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

        response: Dict[str, Any] = {
            "response": generate_texts[0],
            "finished": finished,
            "aux_info": aux_info.model_dump(mode='json'),
        }
        if generate_config.return_hidden_states:
            response["hidden_states"] = hidden_states.tolist()
        if generate_config.calculate_loss and loss is not None:
            response['loss'] = loss.tolist()
        if generate_config.return_logits:
            response['logits'] = logits.tolist()
        if generate_config.return_output_ids:
            response['output_ids'] = output_ids.tolist()
        if generate_config.return_input_ids:
            response['input_ids'] = input_ids.tolist()
        return response

    async def _yield_generate(self, text: str, images: List[str], generate_config: GenerateConfig, **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        stream = self.pipeline.pipeline_async(prompt=text, images=images, generate_config=generate_config, **kwargs)
        async for generate_response in stream:
            yield self._format_response(generate_response, generate_config)

    def is_streaming(self, req: Dict[str, Any]):
        normal_stream = req.get(
            'yield_generator',
            req.get('generation_config',
                    req.get('generate_config', {})
                    ).get('yield_generator', False))
        openai_stream = req.get('stream', False)
        return normal_stream or openai_stream

    def update(self, version_info: VersionInfo):
        lora_infos = dict()
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
                        batch_state[idx] = {"response": '', 'finished':True, 'aux_info':{}}
                    if incremental:
                        batch_state[idx]['response'] = ''
            if len(done_idxs) == len(iterators):
                break
            batch_size = int(len(batch_state) / num_return_sequences)
            batch = batch_state
            if num_return_sequences > 1:
                new_batch: List[Any] = []
                for batch_idx in range(batch_size):
                    seqs = batch_state[batch_idx * num_return_sequences:(batch_idx + 1) * num_return_sequences]
                    new_batch.append({"response": [seq['response'] for seq in seqs],
                                      'finished': [seq['finished'] for seq in seqs],
                                      'aux_info':[seq['aux_info'] for seq in seqs]})
                batch = new_batch
            if batch_infer:
                yield {'response_batch':batch}
            else:
                yield batch[0]
