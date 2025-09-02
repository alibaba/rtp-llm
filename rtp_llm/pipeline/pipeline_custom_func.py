import logging
from typing import Any, Callable, Union

from rtp_llm.models.base_model import BaseModel
from rtp_llm.pipeline.default_plugin import DefaultPlugin
from rtp_llm.utils.ft_plugin import (
    DecodeCallable,
    EncodeCallable,
    ModifyPromptCallable,
    ModifyResponseCallable,
    MultiModalModifyPromptCallable,
    StopGenerateCallable,
    plguin_loader,
)


class PipelineCustomFunc:
    modify_prompt_func: ModifyPromptCallable
    multimodal_modify_prompt_func: MultiModalModifyPromptCallable
    modify_response_func: ModifyResponseCallable
    process_encode_func: EncodeCallable
    process_decode_func: DecodeCallable
    stop_generate_func: StopGenerateCallable


def get_custom_func(
    model_cls: Union[BaseModel, "BaseModel"],
    func_name: str,
    default_func: Callable[..., Any],
) -> Callable[..., Any]:
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
    funcs.modify_prompt_func = get_custom_func(
        model_cls, "modify_prompt_plugin", DefaultPlugin.modify_prompt_func
    )
    funcs.multimodal_modify_prompt_func = get_custom_func(
        model_cls,
        "multimodal_modify_prompt_plugin",
        DefaultPlugin.multimodal_modify_prompt_func,
    )
    funcs.process_encode_func = get_custom_func(
        model_cls, "process_encode_plugin", DefaultPlugin.process_encode_func
    )
    funcs.process_decode_func = get_custom_func(
        model_cls, "process_decode_plugin", DefaultPlugin.tokenids_decode_func
    )
    funcs.modify_response_func = get_custom_func(
        model_cls, "modify_response_plugin", DefaultPlugin.modify_response_func
    )
    funcs.stop_generate_func = get_custom_func(
        model_cls, "stop_generate_plugin", DefaultPlugin.stop_generate_func
    )
    return funcs
