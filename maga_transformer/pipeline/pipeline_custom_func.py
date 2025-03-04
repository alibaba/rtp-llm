import logging
from typing import Any, List, Union, Iterator, Tuple, Callable, Optional, Dict

from maga_transformer.models.base_model import BaseModel
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.utils.ft_plugin import plguin_loader, \
    ModifyResponseCallable, DecodeCallable, EncodeCallable, \
    StopGenerateCallable, ModifyPromptCallable,  MultiModalModifyPromptCallable
from maga_transformer.pipeline.default_plugin import DefaultPlugin

class PipelineCustomFunc:
    modify_prompt_func: ModifyPromptCallable
    multimodal_modify_prompt_func: MultiModalModifyPromptCallable
    modify_response_func: ModifyResponseCallable
    process_encode_func: EncodeCallable
    process_decode_func: DecodeCallable
    stop_generate_func:  StopGenerateCallable
    
def get_custom_func(model_cls: Union[BaseModel, "BaseModel"],
                    func_name: str, default_func: Callable[..., Any]) -> Callable[..., Any]:    
    ft_plugin = plguin_loader.get_plugin()
    
    if getattr(ft_plugin, func_name, None) is not None:
        logging.info(f"using {func_name} implement in ft_plugin")
        return getattr(ft_plugin, func_name)

    if getattr(model_cls, func_name, None) is not None:
        logging.info(f"using {func_name} implement in model")
        return getattr(model_cls, func_name)
    
    logging.info(f"using {func_name} default implement")
    return default_func

def get_piple_custom_func(model_cls: Union[BaseModel, "BaseModel"]):
    funcs = PipelineCustomFunc()
    funcs.modify_prompt_func = get_custom_func(model_cls, "modify_prompt_plugin", DefaultPlugin.modify_prompt_func)
    funcs.multimodal_modify_prompt_func = get_custom_func(model_cls, "multimodal_modify_prompt_plugin", DefaultPlugin.multimodal_modify_prompt_func)
    funcs.process_encode_func = get_custom_func(model_cls, "process_encode_plugin", DefaultPlugin.process_encode_func)
    funcs.process_decode_func = get_custom_func(model_cls, "process_decode_plugin", DefaultPlugin.tokenids_decode_func)
    funcs.modify_response_func = get_custom_func(model_cls, "modify_response_plugin", DefaultPlugin.modify_response_func)
    funcs.stop_generate_func = get_custom_func(model_cls, "stop_generate_plugin", DefaultPlugin.stop_generate_func)
    return funcs