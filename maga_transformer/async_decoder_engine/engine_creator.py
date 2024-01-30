import logging
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union, Any, Dict
from maga_transformer.utils.util import get_mem_info
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.scheduler import Scheduler
from maga_transformer.async_decoder_engine.cache_manager import CacheConfigGenerator
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.base_model import BaseModel
from maga_transformer.ops.gpt_ops.gpt_op import GptOp
from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.async_decoder_engine.ptuning import PrefixParams
from maga_transformer.async_decoder_engine.normal_model_executor import NormalModelExecutor, ModelOps, ModelType, ExecutorBase
from maga_transformer.async_decoder_engine.speculative.sp_model_executor import SpModelExecutor
from maga_transformer.async_decoder_engine.medusa.medusa_model_executor import MedusaModelExecutor
from maga_transformer.async_decoder_engine.medusa.utils import generate_medusa_buffers
from maga_transformer.async_decoder_engine.decoder_engine import DecoderEngine

class ExecutorType(Enum):
    Normal = "normal"
    Speculative = "speculative"
    Medusa = "medusa"

def check_exeutor_type(model: BaseModel, config: GptInitModelParameters, speculative_model: Any = None, speculative_config: Optional[GptInitModelParameters] = None):
    if speculative_model is not None:
        assert speculative_config is not None and speculative_model is not None, "speculative_config should not be None"
        return ExecutorType.Speculative
    if model.medusa_head is not None:
        return ExecutorType.Medusa
    return ExecutorType.Normal

def create_engine(model: BaseModel, config: GptInitModelParameters, ptuning_args: Optional[PrefixParams],
                  speculative_model: Any = None, speculative_config: Optional[GptInitModelParameters] = None) -> DecoderEngine:
    executor_type = check_exeutor_type(model, config, speculative_model, speculative_config)
    logging.info(f"executor_type: {executor_type}")
    if executor_type == ExecutorType.Normal:
        return _create_normal_engine(model, config, ptuning_args)
    elif executor_type == ExecutorType.Speculative:
        return _create_sp_engine(model, config, speculative_model, speculative_config)
    elif executor_type == ExecutorType.Medusa:
        return _create_medusa_engine(model, config)
    else:
        raise Exception(f"unsupported executor type: {executor_type}")

def _create_normal_engine(model: BaseModel, config: GptInitModelParameters, ptuning_args: Optional[PrefixParams]) -> DecoderEngine:
    model_ops = _create_ops(ModelType.Normal, model, config)    
    logging.info(f'ft op mem info: used: {get_mem_info().used} free: {get_mem_info().free}')
    nccl_op = NcclOp()
    cache_config = CacheConfigGenerator.create_config(config)
    scheduler = Scheduler(config, cache_config, ptuning_args, 1, nccl_op)
    executor = NormalModelExecutor(model_ops, scheduler)
    model.weight.lora_resource.ft_op = [model_ops.gpt_op]
    return DecoderEngine(executor, scheduler, config)

def _create_sp_engine(model: BaseModel, config: GptInitModelParameters, sp_model: BaseModel, sp_config: GptInitModelParameters) -> DecoderEngine:
    model_ops = _create_ops(ModelType.Normal, model, config)    
    sp_model_ops = _create_ops(ModelType.Speculative, sp_model, sp_config)
    assert model.prefix_encoder is None and sp_model.prefix_encoder is None, "speculative not support prefix yet"
    nccl_op = NcclOp()
    cache_config, sp_cache_config = CacheConfigGenerator.create_sp_config(config, sp_config)
    scheduler = Scheduler(config, cache_config, None, sp_config.gen_num_per_circle, nccl_op)
    validate_executor = NormalModelExecutor(model_ops, scheduler)
    
    sp_scheduler = Scheduler(sp_config, sp_cache_config, None, sp_config.gen_num_per_circle, nccl_op)
    sp_excutor = NormalModelExecutor(sp_model_ops, sp_scheduler)
    
    executor = SpModelExecutor(validate_executor, sp_excutor, sp_config.gen_num_per_circle)    
    return DecoderEngine(executor, scheduler, config)

def _create_medusa_engine(model: BaseModel, config: GptInitModelParameters, **kwargs: Any) -> DecoderEngine:
    assert model.medusa_head is not None
    assert config.medusa_config is not None
    medusa_buffer = generate_medusa_buffers(config.medusa_config.medusa_choices, config.medusa_config.top_k)
    model_ops = _create_ops(ModelType.Medusa, model, config)
    # in medusa, len(token) always equals to len(kvcache), so we need plus 1 to fix
    gen_num_per_circle = len(medusa_buffer.medusa_position_ids) + 1
    logging.info(f'ft op mem info: used: {get_mem_info().used} free: {get_mem_info().free}')
    nccl_op = NcclOp()
    cache_config = CacheConfigGenerator.create_config(config)
    scheduler = Scheduler(config, cache_config, None, gen_num_per_circle, nccl_op)
    executor = MedusaModelExecutor(model_ops, scheduler, medusa_buffer)
    return DecoderEngine(executor, scheduler, config)

def _create_ops(type: ModelType, model: BaseModel, config: GptInitModelParameters) -> ModelOps:
    gpt_op = GptOp.from_config(config)
    gpt_op.set_weight(model.weight)
    generate_config = GenerateConfig(
        using_hf_sampling=False
    )
    sampler = model.create_sampler(generate_config)
    return ModelOps(type, model, config, gpt_op, generate_config, sampler)
