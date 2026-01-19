import logging
import pathlib
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Union

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    RendererInfo,
    RoleEnum,
    UsageInfo,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
    StreamResponseObject,
)
from rtp_llm.tokenizer_factory.tokenizers import QWenTokenizer, QWenV2Tokenizer
from rtp_llm.utils.base_model_datatypes import GenerateOutputs
from rtp_llm.utils.word_util import (
    get_stop_word_slices,
    truncate_response_with_stop_words,
)

current_file_path = pathlib.Path(__file__).parent.absolute()
import sys

sys.path.insert(0, str(current_file_path))
from qwen_agent.llm import get_chat_model

QwenTokenizerTypes = Union[QWenTokenizer, QWenV2Tokenizer]


@dataclass
class ProcessedOutput:
    output_str: str
    output_token_length: int
    finish_reason: Optional[FinisheReason]


class QwenAgentRenderer(CustomChatRenderer):
    def __init__(
        self,
        tokenizer: QwenTokenizerTypes,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        self.add_extra_stop_words(["✿RESULT✿", "✿RETURN✿"])

        self.template_chat_renderer: Optional[BasicRenderer] = None
        try:
            if tokenizer.chat_template != None:
                logging.info(
                    f"qwen model has chat_template [{tokenizer.chat_template}], "
                    "which will be used for non-function call dialogue."
                )
                self.template_chat_renderer = BasicRenderer(
                    tokenizer,
                    renderer_params,
                    generate_env_config,
                    render_config,
                    ckpt_path,
                    misc_config,
                    vit_config,
                )
        except AttributeError:
            pass

        llm_cfg = {
            "model": "qwen"
        }  # model 设置以qwen开头，且没设置server，会直接路由初始化 qwen_dashcope 这个类，我们要用里面的处理逻辑

        # Get misc_config if available
        if misc_config is not None:
            # misc_config might be PyMiscellaneousConfig, extract the actual misc_config
            if hasattr(misc_config, "misc_config"):
                llm_cfg["misc_config"] = misc_config.misc_config
            else:
                llm_cfg["misc_config"] = misc_config

        self.qwen_llm = get_chat_model(llm_cfg)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        # 将request转换为qwen的idx输入
        if (self.template_chat_renderer != None) and (
            (request.functions == None) or (len(request.functions) == 0)
        ):
            return self.template_chat_renderer.render_chat(request)

        messages = []
        functions = []
        extra_generate_cfg = (
            request.extra_configs.dict() if request.extra_configs else {}
        )

        if request.messages:
            messages = [msg.model_dump() for msg in request.messages]
        if request.functions:
            for func in request.functions:
                func = func.model_dump()
                # 删除不必要的提前赋值 name_for_model name_for_human description_for_model
                for k in ["name_for_model", "name_for_human", "description_for_model"]:
                    if k in func and func[k] == None:
                        del func[k]
                functions.append(func)

        completion_prompt = self.qwen_llm.generate_completion_prompt(
            messages, functions, extra_generate_cfg
        )
        input_ids = self.tokenizer.encode(completion_prompt)
        return RenderedInputs(input_ids=input_ids)

    def _parse_function_response(self, response: str) -> Optional[DeltaMessage]:
        # 处理function call 的流式信息
        func_name, func_args = "", ""
        i = response.rfind("✿FUNCTION✿:")
        j = response.rfind("\n✿ARGS✿:")
        k = response.rfind("\n✿RESULT✿")
        if 0 <= i < j:  # i index ✿FUNCTION✿
            if k < j:  # j index \n✿ARGS✿ k index \n✿RESULT✿
                # because the output text may have discarded the stop word.
                response = response.rstrip() + "\n✿RESULT✿"  # Add it back.
            k = response.rfind("\n✿RESULT✿")
            func_name = response[i + len("✿FUNCTION✿:") : j].strip()
            func_args = response[j + len("\n✿ARGS✿:") : k].strip()
        logging.info(
            f"parsed function from response: [{response}]: {func_name}, {func_args}"
        )
        if func_name:
            return DeltaMessage(
                content=response[:i],
                function_call=FunctionCall(name=func_name, arguments=func_args),
            )
        return None

    def _process_output_ids_tensor(
        self,
        input_length,
        output_ids_tensor: torch.Tensor,
        max_new_tokens: int,
        finished: bool = False,
        input_with_functions: bool = False,
    ) -> ProcessedOutput:
        output_ids_tensor = output_ids_tensor.cpu().reshape([-1])
        # TODO(wangyin): This slicing shouldn't be done here.
        # model should return output length, ids should be sliced with output length.
        output_ids = output_ids_tensor[output_ids_tensor != self.eos_token_id].tolist()
        finish_reason = (
            self._check_finish_reason(output_ids, input_length, max_new_tokens)
            if finished
            else None
        )

        output_length = len(output_ids)
        output_ids = self._remove_stop_word_ids(output_ids, output_ids)
        output_str = self.tokenizer.decode(output_ids)

        # following qwen agent function_calling.py process
        if input_with_functions:
            if output_str.startswith(": "):
                output_str = output_str[2:]
            elif output_str.startswith(":"):
                output_str = output_str[1:]

        output_str = output_str.strip("\uFFFD")
        for stop_word in self.stop_words_str_list:
            output_str = output_str.replace(stop_word, "")
        return ProcessedOutput(output_str, output_length, finish_reason)

    async def render_response_stream(
        self,
        output_generator: AsyncGenerator[GenerateOutputs, None],
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        index = 0
        output_string = ""
        output_length = 0
        responded_string = ""
        responded_length = 0
        output_token_length = 0
        finish_reason: Optional[FinisheReason] = None
        generating_function_call = False
        stop_word_slice_list = get_stop_word_slices(generate_config.stop_words_str)
        output_tokens_list = torch.empty(0, dtype=torch.int32)
        input_with_functions = not (
            request.functions == None or len(request.functions) == 0
        )

        async for output in output_generator:
            if output_token_length == 0:
                yield StreamResponseObject(
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(role=RoleEnum.assistant),
                        )
                    ]
                )
            output = output.generate_outputs[0]
            # all mode incremental return output_ids
            input_token_length = output.aux_info.input_len
            output_tokens_list = torch.cat(
                (output_tokens_list, output.output_ids), dim=1
            )
            output.output_ids = output_tokens_list

            processed_output = self._process_output_ids_tensor(
                input_token_length,
                output.output_ids,
                generate_config.max_new_tokens,
                output.finished,
                input_with_functions,
            )
            output_string = processed_output.output_str.strip()
            # print(f"==============> {output_string}")
            output_length = len(processed_output.output_str)
            finish_reason = processed_output.finish_reason
            output_token_length = processed_output.output_token_length

            if input_with_functions and output_string.endswith("✿FUNCTION✿:"):
                generating_function_call = True
                continue

            if generating_function_call:
                continue

            if output_length > responded_length + len("✿FUNCTION✿:"):
                delta_string = output_string[
                    responded_length : output_length - len("✿FUNCTION✿:")
                ]
                trunc_string = truncate_response_with_stop_words(
                    delta_string,
                    stop_word_slice_list,
                    generate_config.is_streaming,
                    True,
                )
                if trunc_string != delta_string:
                    continue
                responded_string = output_string[: output_length - len("✿FUNCTION✿:")]
                responded_length = len(responded_string)

                yield StreamResponseObject(
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(
                                content=delta_string,
                            ),
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length,
                    ),
                )

        if generating_function_call:
            function_message = self._parse_function_response(
                output_string[responded_length:]
            )
            if function_message == None:
                logging.warn(
                    f"output [{output_string}] failed to parse function from [{responded_length}]. "
                    "regarded as normal output."
                )
            else:
                finish_reason = FinisheReason.function_call
                responded_string = output_string
                responded_length = output_length
                yield StreamResponseObject(
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=function_message,
                            # finish_reason=finish_reason,
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length,
                    ),
                )

        if finish_reason == None:
            logging.debug(
                f"output [{responded_string}] found no stop reason! use stop as default."
            )
            finish_reason = FinisheReason.stop

        if responded_length < output_length:
            index += 1
            yield StreamResponseObject(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            content=truncate_response_with_stop_words(
                                output_string[responded_length:],
                                generate_config.stop_words_str,
                                generate_config.is_streaming,
                            ),
                        ),
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=input_token_length,
                    total_tokens=input_token_length + output_token_length,
                    completion_tokens=output_token_length,
                ),
            )

        yield StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=index + 1,
                    delta=DeltaMessage(
                        content="",
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length,
            ),
            aux_info=output.aux_info if request.aux_info else None,
        )

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        if self.template_chat_renderer:
            renderer_info.template = self.template_chat_renderer.chat_template
        return renderer_info


register_renderer("qwen_agent", QwenAgentRenderer)
