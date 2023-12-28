import gc
import torch
import logging
import traceback
from typing import Optional, Iterator, List, Any, Generator, AsyncGenerator, Dict
from maga_transformer.utils.util import get_mem_info
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.models.base_model import BaseModel
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.decoder_engine import DecoderEngine
from maga_transformer.async_decoder_engine.engine_creator import create_engine
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.async_decoder_engine.ptuning import get_ptuning_params

class AsyncModel:
    def __init__(self, model: BaseModel, sp_model: Optional[BaseModel] = None) -> None:
        self.model = model
        self.sp_model = sp_model
        self.config = model.config
        assert self.config.max_seq_len > 0
        self.tokenizer = model.tokenizer
        ptuning_args = get_ptuning_params(self.model, self.tokenizer)
        logging.info(f'first mem info: used:{get_mem_info().used} free: {get_mem_info().free}')
        if self.sp_model is not None:
            assert ptuning_args is None, "speculative don't support ptuning yet"
            self.decoder_engine_ = create_engine(self.model, self.config, None, self.sp_model, self.sp_model.config)
        else:            
            self.decoder_engine_ = create_engine(model, self.config, ptuning_args)

    @property
    def is_multimodal(self) -> bool:
        return self.model.is_multimodal

    @property
    def default_generate_config(self) -> Dict[str, Any]:
        return self.model.default_generate_config

    def stop(self):
        self.decoder_engine_.stop()

    @torch.no_grad()
    async def generate_stream(self, # type: ignore
                        input_token_ids: torch.Tensor,
                        input_lengths: Optional[torch.Tensor],
                        images: List[List[str]],
                        generate_config: GenerateConfig) -> AsyncGenerator[GenerateOutput, None]:
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            return
        max_new_tokens = min(self.config.max_seq_len - input_token_ids.shape[1], generate_config.max_new_tokens)
        if input_token_ids.shape[1] <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, f"model tokens can not be empty, request length is {input_token_ids.shape[1]}")
        if max_new_tokens <= 0:
           raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, f"model max tokens is {self.config.max_seq_len}, request length is {input_token_ids.shape[1]}, max_new_tokens is {max_new_tokens}")
        inputs_np = input_token_ids.cpu().numpy()
        if input_lengths is None:
            input_lengths = torch.IntTensor([len(v[v != self.config.special_tokens.eos_token_id]) for v in inputs_np])
        try:
            batch_stream = self.decoder_engine_.decoder(input_token_ids, input_lengths, images, generate_config)
            async for hidden_states, token_streams, finished, aux_info, loss, logits in batch_stream:
                yield GenerateOutput(hidden_states, token_streams, finished, aux_info, loss, logits)
        except Exception as e:
            logging.error(f'generate error: {e}, Traceback: {traceback.format_exc()}')
            raise e
