from typing import Optional, List, Union
from maga_transformer.openai.renderers.qwen_agent_renderer import QwenAgentRenderer
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers import Qwen2Tokenizer
from maga_transformer.openai.api_datatype import (
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
)
from maga_transformer.openai.renderers.custom_renderer import (
    StreamResponseObject,
    RenderedInputs,
    StreamStatus,
)
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.openai.renderers.qwen_agent.utils.tool_function_converter import (
    convert_tool_to_function_request,
    convert_function_to_tool_response,
)

QwenTokenizerTypes = Union[QWenTokenizer, Qwen2Tokenizer]

class QwenAgentToolRenderer(QwenAgentRenderer):

    # override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        # 转换request从tool协议到function协议
        function_request_dict = convert_tool_to_function_request(
            request.model_dump(exclude_none=True)
        )

        function_request = ChatCompletionRequest(**function_request_dict)

        return super().render_chat(function_request)

    # override
    def _parse_function_response(self, response: str) -> Optional[DeltaMessage]:
        delta_message = super()._parse_function_response(response)
        if delta_message and delta_message.function_call:
            # 转换function_call成tool_call
            tool_delta_dict = convert_function_to_tool_response(
                **delta_message.model_dump(exclude_none=True)
            )

            return DeltaMessage(**tool_delta_dict)

        return delta_message

    # override
    async def _generate_final(
        self, buffer_list: List[StreamStatus], request: ChatCompletionRequest
    ):
        stream_response_object: StreamResponseObject = await super()._generate_final(
            buffer_list, request
        )
        if (
            stream_response_object.choices[0].finish_reason
            == FinisheReason.function_call
        ):
            stream_response_object.choices[0].finish_reason = FinisheReason.tool_calls
        return stream_response_object


register_renderer("qwen_agent_tool", QwenAgentToolRenderer)
