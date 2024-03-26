import json
import logging
import logging.config
import torch
from typing import Any, Dict, Union

from maga_transformer.model_factory import ModelFactory
from maga_transformer.models.base_model import ModelConfig
from maga_transformer.tools.api.utils import handler_error
from maga_transformer.utils.util import WEIGHT_TYPE, get_weight_type_from_env
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.tools.api.hf_model_helper import HF_MODEL_INFO_HELPER, HfModelInfo, HfModelInfoHelper

def eval_model_size(env_params, model_type, model_path, ptuning_path):
    model_cls = ModelFactory.get_model_cls(model_type)
    model_config = ModelConfig(
        model_type=model_type,
        ckpt_path=model_path,
        weight_type=get_weight_type_from_env(env_params),
        ptuning_path=ptuning_path,
        max_seq_len=0,
        tokenizer_path=None
    )
    config: GptInitModelParameters = model_cls.create_config(model_config)
    return model_cls.eval_model_size(config)

def calc_hf_model_size(req: Dict[str, Any]):
    model_path = req.get("model_path")
    # maybe ft_model_type
    model_type = req.get("ft_model_type", None)
    env_params: Any | dict[Any, Any] = req.get("env_params", {})

    hf_model_info:HfModelInfo = HfModelInfoHelper.get_instance().get_hf_model_info(model_path)
    if model_type:
        try:
            return eval_model_size(env_params, model_type, hf_model_info.hf_local_dir, None)
        except Exception as e:
            logging.exception(f"eval ft model size failed: e: {e}")

    param_count =  hf_model_info.param_count
    if param_count :
        weight_type = get_weight_type_from_env(env_params)
        if weight_type == WEIGHT_TYPE.INT8:
            return param_count
        else:
            return param_count * 2
    return None

def cacl_ft_model_size(req: Dict[str, Any]) -> int:
    env_params = req.get("env_params", {})
    model_type = req.get("ft_model_type")
    model_path = fetch_remote_file_to_local(req.get("model_path"))
    ptuning_path = fetch_remote_file_to_local(req.get("ptuning_path")) if req.get("ptuning_path", None) else None

    if not model_type or not model_path:
        return handler_error(Exception.ERROR_INPUT_FORMAT_ERROR, "bad_input")

    model_size = eval_model_size(env_params, model_type, model_path, ptuning_path)
    return model_size


from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/calc_model_size")
@router.get("/calc_model_size")
def calc_mdoel_size(req: Union[str,Dict[Any, Any]]):
    if isinstance(req, str):
        req = json.loads(req)

    if HfModelInfoHelper.is_from_hf(req.get("model_path")):
        model_size = calc_hf_model_size(req)
    else:
        model_size = cacl_ft_model_size(req)
    response = {"model_size":model_size} if model_size else {}

    return JSONResponse(content = response)
