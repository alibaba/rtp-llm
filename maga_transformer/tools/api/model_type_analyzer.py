
import json
import logging
import logging.config
from typing import Any, Dict, Union

from maga_transformer.model_factory import ModelFactory
from maga_transformer.tools.api.hf_model_helper import HF_MODEL_INFO_HELPER, HfModelInfoHelper
from maga_transformer.model_factory_register import ModelDict
from maga_transformer.tools.api.utils import handler_error

from maga_transformer.utils.fuser import fetch_remote_file_to_local


GENERAL_HF_MODEL = "HuggingFacePipeline"

def parse_hf_model_type(model_link):
    hf_model_info = HF_MODEL_INFO_HELPER.get_hf_model_info(model_link)
    ft_model_type = hf_model_info.ft_model_type
    return {
        "hf_model_type": GENERAL_HF_MODEL,
        "ft_model_type": ft_model_type
    }

def parse_ft_model_type(model_path):
    # load config.json
    try:
        config = ModelFactory.get_config_json(model_path)
    except Exception as e:
        logging.error(f"parse tf model_type failed: {str(e)}")
        return {
        }
    ft_model_type = ModelDict.get_ft_model_type_by_config(config)
    return {
        "ft_model_type": ft_model_type
    }



def _analyze_model_type(model_path):
    if HfModelInfoHelper.is_from_hf(model_path):
        res = parse_hf_model_type(model_path)
    else:
        model_path = fetch_remote_file_to_local(model_path)
        res =  parse_ft_model_type(model_path)

    return {k: v for k, v in res.items() if v is not None}


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
