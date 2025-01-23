from dataclasses import dataclass
import json
import logging
import os
from typing import List, Optional
import transformers
from transformers import AutoConfig, PretrainedConfig
from maga_transformer.models.base_model import BaseModel, ModelConfig
from maga_transformer.utils.weight_type import WEIGHT_TYPE, get_weight_type_from_env
from maga_transformer.tools.api.hf_model_helper import get_hf_model_info, HfStyleModelInfo
from maga_transformer.model_factory import ModelFactory
from maga_transformer.model_factory_register import ModelDict

from maga_transformer.utils.fuser import fetch_remote_file_to_local, umount_file


GENERAL_HF_MODEL = "HuggingFacePipeline"

def _parse_hf_model_type(model_link):
    hf_model_info = get_hf_model_info(model_link)
    ft_model_type = hf_model_info.ft_model_type
    return {
        "hf_model_type": GENERAL_HF_MODEL,
        "ft_model_type": ft_model_type
    }

def parse_ft_model_type(model_path):
    # load config.json
    config = _get_raw_config(model_path)
    ft_model_type = ModelDict.get_ft_model_type_by_config(config)
    return {
        "ft_model_type": ft_model_type
    }


def _analyze_model_type(model_path):
    if HfStyleModelInfo.is_from_hf(model_path):
        res = _parse_hf_model_type(model_path)
    else:
        model_path = fetch_remote_file_to_local(model_path)
        res =  parse_ft_model_type(model_path)

    return {k: v for k, v in res.items() if v is not None}

def _get_raw_config(model_path):
    if HfStyleModelInfo.is_from_hf(model_path):
        hf_model_info = get_hf_model_info(model_path)
        return hf_model_info.model_config
    else:
        model_path = fetch_remote_file_to_local(model_path)
        try:
            config = ModelFactory.get_config_json(model_path)
        except Exception as e:
            logging.error(f"parse tf model_type failed: {str(e)}")
            config = {}
        return config


@dataclass
class QuantConfig:
    weight_bits: int = 16
    quant_algo: Optional[str] = None

@dataclass
class ModelBasicInfo:
    ft_model_type: Optional[str] = None
    hidden_size : Optional[int] = None
    architectures : Optional[List[str]] = None
    llm_architectures: Optional[List[str]] = None
    is_quant_weight: bool = False
    quant_config:Optional[QuantConfig] = None
    model_size: Optional[int] = None
    param_count: Optional[int] = None


def _load_as_hf_style(model_path, ft_model_type, env_params) -> ModelBasicInfo:
    total_size = None
    param_count = None
    config_file = None
    # if HfStyleModelInfo.is_from_hf(model_path):
    hf_model_info = get_hf_model_info(model_path)
    config_file = hf_model_info.model_config_file
    config_dict = hf_model_info.model_config
    total_size = hf_model_info.total_size
    param_count = hf_model_info.param_count
    # else:
    #     config_file = os.path.join(model_path, "config.json")
    #     if os.path.exists(config_file):
    #         with open(config_file, "r") as f:
    #             config_dict = json.load(f)
    #     else:
    #         config_dict = {}

    methods = [
        lambda: AutoConfig.from_pretrained(
            os.path.dirname(config_file), trust_remote_code=True
        ),
        lambda: PretrainedConfig.from_dict(config_dict),
    ]
    last_exception = None
    config = None
    for method in methods:
        try:
            config = method()
            break
        except Exception as e:
            last_exception = e  # Store the last exception
            logging.error(f"Failed to parse model config: {str(e)}")
            continue  # Try the next method
    if config is None:
        # If we reach here, all methods have failed
        logging.error(f"Failed to parse model config: {str(last_exception)}")
        return ModelBasicInfo()

    # config = PretrainedConfig.from_dict(config_dict)
    logging.info(f"config:{config}")
    hidden_size = config.hidden_size if hasattr(config, "hidden_size") else None
    # num_hidden_layers = config.num_hidden_layers
    # num_attention_heads = config.num_attention_heads
    # vocab_size = config.vocab_size
    quant_config = None
    is_quant_weight = False
    if hasattr(config, "quantization_config"):
        is_quant_weight = True
        quant_method = config.quantization_config.get('quant_method', '')
        weight_bits = 16
        if quant_method.lower() == "fp8":
            weight_bits = 8
        elif quant_method.lower() == "gptq":
            weight_bits = config.quantization_config.get('bits', 4)
        elif quant_method.lower() == "awq":
            weight_bits = config.quantization_config.get('bits', 4)

        quant_config = QuantConfig(
            weight_bits = weight_bits,
            quant_algo = quant_method,
        )

    return ModelBasicInfo(
        ft_model_type=ft_model_type if ft_model_type else None,
        hidden_size=hidden_size,
        architectures=config.architectures,
        llm_architectures=None,
        is_quant_weight = is_quant_weight,
        quant_config=quant_config,
        model_size=total_size,
        param_count=param_count
    )


def _load_as_ft_style(model_path, is_from_hf, ft_model_type, env_params) -> ModelBasicInfo:
    model_cls = ModelFactory.get_model_cls(ft_model_type)

    if is_from_hf:
        hf_model_info = get_hf_model_info(model_path)
        if hf_model_info and hf_model_info.model_config_file is not None:
            model_path = os.path.dirname(hf_model_info.model_config_file)
    model_config = ModelConfig(
        model_type=ft_model_type,
        ckpt_path=model_path,
        weight_type=get_weight_type_from_env(env_params),
        ptuning_path=None,
        max_seq_len=int(env_params.get("MAX_SEQ_LEN", "0")),
        tokenizer_path=None
    )
    config: GptInitModelParameters = model_cls.create_config(model_config)
    is_quant_weight = config.quant_algo.isQuant()
    quant_config = None
    if is_quant_weight:
        quant_method = None
        if config.quant_algo.isGptq():
            quant_method = "gptq"
        elif config.quant_algo.isAwq():
            quant_method = "awq"
        elif config.quant_algo.isSmoothQuant():
            quant_method = "smooth_quant"
        elif config.quant_algo.isOmniQuant():
            quant_method = "omni_quant"
        elif config.quant_algo.isFp8():
            quant_method = "fp8"
        elif config.quant_algo.isPerTensorQuant():
            quant_method = "pertensor_quant"
        elif config.quant_algo.isWeightOnlyPerCol():
            quant_method = "weight_only_per_col"

        quant_config = QuantConfig(
            weight_bits = config.quant_algo.getWeightBits(),
            quant_algo = quant_method,
        )

    logging.info(f"config:{config}, {config.size_per_head} * {config.head_num}")
    param_count = None
    try:
        param_count = model_cls.eval_model_param_count(config)
    except Exception as e:
        param_count = BaseModel.eval_model_param_count(config)
        logging.error(f"eval model param count failed: {str(e)}")
    total_size = None
    try:
        total_size = model_cls.eval_model_size(config)
    except Exception as e:
        total_size = BaseModel.eval_model_size(config)
        logging.error(f"eval model param count failed: {str(e)}")

    raw_config_dict = _get_raw_config(model_path)

    return ModelBasicInfo(
        ft_model_type=ft_model_type,
        param_count=param_count,
        model_size=total_size,
        hidden_size=config.gpt_init_params.hidden_size,
        architectures=raw_config_dict.get('architectures',None),
        llm_architectures=None,
        is_quant_weight=is_quant_weight,
        quant_config=quant_config
    )

def parse_model_basic_info(model_path, env_params=dict({}), ft_model_type=None):
    is_from_hf = HfStyleModelInfo.is_from_hf(model_path)
    model_path = model_path if is_from_hf else fetch_remote_file_to_local(model_path)
    ft_model_type = _analyze_model_type(model_path).get("ft_model_type", None) if not ft_model_type else ft_model_type
    try:
        # 尝试使用 ft_model_type 加载模型
        res = _load_as_ft_style(model_path, is_from_hf, ft_model_type, env_params)
    except Exception as e:
        # 如果发生异常，使用 hf_model_type 加载模型
        logging.warning(f"try load model with ft_model_type failed: {str(e)}")
        res = _load_as_hf_style(model_path, ft_model_type, env_params)
    if not is_from_hf and model_path:
        umount_file(model_path)
    return res
