import json
import logging
from typing import Any, Dict, Union

from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.py_config_modules import QuantizationConfig
from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import ProfilingDebugLoggingConfig
from rtp_llm.tools.api.hf_model_helper import HfStyleModelInfo, get_hf_model_info
from rtp_llm.tools.api.utils import handler_error
from rtp_llm.utils.fuser import fetch_remote_file_to_local, umount_file


def _get_quantization_from_env_params(env_params: Dict[str, str]) -> str:
    """Get quantization setting from environment parameters.
    
    Compatibility function for tools that need to extract quantization from env_params.
    
    Args:
        env_params: Dictionary of environment parameters
        
    Returns:
        Quantization string or empty string
    """
    QUANTIZATION_KEY = "QUANTIZATION"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    INT8_MODE = "INT8_MODE"
    
    quantization = env_params.get(QUANTIZATION_KEY, "")
    
    # Compatibility logic: if int8_mode == 1 or weight_type is INT8, set quantization to INT8
    if not quantization:
        int8_mode = env_params.get(INT8_MODE, "0")
        weight_type = env_params.get(WEIGHT_TYPE, "").upper()
        if int(int8_mode) == 1 or weight_type == "INT8":
            quantization = "INT8"
    
    return quantization or ""


def eval_model_size(env_params, model_type, model_path, ptuning_path):
    model_cls = ModelFactory.get_model_cls(model_type)
    quantization = _get_quantization_from_env_params(env_params)
    logging.info(f"env_params: {env_params}, quantization: {quantization}")
    # Use _create_config to get base config from C++
    config: ModelConfig = model_cls._create_config(model_path)
    # Apply settings from env_params
    model_args = ModelArgs()
    model_args.model_type = model_type
    model_args.ckpt_path = model_path
    model_args.tokenizer_path = model_path
    model_args.ptuning_path = ptuning_path
    model_args.max_seq_len = int(env_params.get("MAX_SEQ_LEN", "0"))
    
    quantization_config = QuantizationConfig()
    quantization_config.quantization = quantization

    kv_cache_config = KVCacheConfig()
    kv_cache_config.int8_kv_cache = int(env_params.get("INT8_KV_CACHE", "0")) == 1
    kv_cache_config.fp8_kv_cache = int(env_params.get("FP8_KV_CACHE", "0")) == 1

    build_model_config(config,
                       model_args=model_args,
                       kv_cache_config=kv_cache_config,
                       profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
                       quantization_config=quantization_config)

    return config.eval_model_size(), config.model_param_count()


def calc_hf_model_size(req: Dict[str, Any]):
    model_path = req.get("model_path")
    # maybe ft_model_type
    model_type = req.get("ft_model_type", None)
    env_params: Any | dict[Any, Any] = req.get("env_params", {})

    hf_model_info: HfStyleModelInfo = get_hf_model_info(model_path)
    logging.info(f"hf_model_info: {hf_model_info}, {env_params}, {model_type}")
    if model_type:
        try:
            return eval_model_size(
                env_params, model_type, hf_model_info.hf_local_dir, None
            )
        except Exception as e:
            logging.exception(f"eval ft model size failed: e: {e}")

    param_count = hf_model_info.param_count
    total_size = hf_model_info.total_size
    quantization = _get_quantization_from_env_params(env_params)

    logging.info(f"req: {req}, quantization: {quantization}")
    if param_count:
        quant_config = init_quant_config(quantization) if quantization else None
        logging.info(f"req: {req}, quant_config: {quant_config}")
        if quant_config:
            return param_count * quant_config.bits / 8, param_count
        else:
            return param_count * 2, param_count
    return param_count, total_size


def cacl_ft_model_size(req: Dict[str, Any]) -> int:
    env_params = req.get("env_params", {})
    model_type = req.get("ft_model_type")
    model_path = fetch_remote_file_to_local(req.get("model_path"))
    ptuning_path = (
        fetch_remote_file_to_local(req.get("ptuning_path"))
        if req.get("ptuning_path", None)
        else None
    )

    if not model_type or not model_path:
        return handler_error(Exception.ERROR_INPUT_FORMAT_ERROR, "bad_input")

    res = eval_model_size(env_params, model_type, model_path, ptuning_path)
    if model_path:
        umount_file(model_path)
    if ptuning_path:
        umount_file(ptuning_path)
    return res


from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/calc_model_size")
@router.get("/calc_model_size")
def calc_mdoel_size(req: Union[str, Dict[Any, Any]]):
    if isinstance(req, str):
        req = json.loads(req)

    if HfStyleModelInfo.is_from_hf(req.get("model_path")):
        model_size, param_count = calc_hf_model_size(req)
    else:
        model_size, param_count = cacl_ft_model_size(req)
    response = {}
    if model_size:
        response["model_size"] = model_size
    if param_count:
        response["param_count"] = param_count

    return JSONResponse(content=response)
