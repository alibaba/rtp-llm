
import json
import logging.config
from typing import Any, Dict, Union
from maga_transformer.tools.api.model_basic_info_analyzer import ModelBasicInfo, _analyze_model_type, parse_model_basic_info

from maga_transformer.tools.api.utils import handler_error





from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/analyze_model_type")
@router.get("/analyze_model_type")
def analyze_model_type(req: Union[str,Dict[Any, Any]]):
    if isinstance(req, str):
        req = json.loads(req)
    model_path = req.get("model_path")

    if not model_path:
        return handler_error(Exception.ERROR_INPUT_FORMAT_ERROR, "bad_input")

    model_type = _analyze_model_type(model_path)
    response = {"model_type":model_type}
    return JSONResponse(content = response)

@router.post("/analyze_model_basic_info")
@router.get("/analyze_model_basic_info")
def analyze_model_basic_info(req: Union[str,Dict[Any, Any]]):
    if isinstance(req, str):
        req = json.loads(req)
    model_path = req.get("model_path")

    if not model_path:
        return handler_error(Exception.ERROR_INPUT_FORMAT_ERROR, "bad_input")
    from dataclasses import asdict
    ft_model_type = req.get("ft_model_type", None)
    env_params: Any | dict[Any, Any] = req.get("env_params", {})
    model_basic_info: ModelBasicInfo = parse_model_basic_info(model_path, env_params, ft_model_type)
    response: dict[str, ModelBasicInfo] = asdict(model_basic_info)
    return JSONResponse(content = response)
