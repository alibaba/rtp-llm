import json
import logging
import logging.config
from typing import Any, Dict, Union

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import ModelConfig
from rtp_llm.tools.api.hf_model_helper import HfStyleModelInfo, get_hf_model_info
from rtp_llm.tools.api.utils import handler_error
from rtp_llm.utils.fuser import fetch_remote_file_to_local, umount_file


def eval_model_size(env_params, model_type, model_path, ptuning_path):
    model_cls = ModelFactory.get_model_cls(model_type)
    quantization = ModelConfig.get_quantization_from_params(env_params)
    logging.info(f"env_params: {env_params}, quantization: {quantization}")
    model_config = ModelConfig(
        model_type=model_type,
        ckpt_path=model_path,
        act_type=env_params.get(ModelConfig.ACT_TYPE),
        ptuning_path=ptuning_path,
        max_seq_len=int(env_params.get("MAX_SEQ_LEN", "0")),
        tokenizer_path=None,
        quantization=quantization,
    )
    config: GptInitModelParameters = model_cls.create_config(model_config)
    return model_cls.eval_model_size(config), model_cls.eval_model_param_count(config)


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
    quantization = ModelConfig.get_quantization_from_params(env_params)

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
