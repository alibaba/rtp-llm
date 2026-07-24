import logging
from typing import Any, Dict, List, Optional

from typing_extensions import override

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
)


class KimiK3Renderer(CustomChatRenderer):
    """Render Kimi K3's Python-defined XTML chat encoding.

    K3 deliberately has no Jinja ``chat_template``.  Its remote tokenizer
    renders a sequence of trusted structural segments and untrusted text
    segments, then encodes the two with different special-token policies.
    Calling ``encode`` on the final debug string would lose that distinction,
    so this renderer consumes the tokenizer's tokenized result directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_extra_stop_words(["<|end_of_msg|>"])

    @staticmethod
    def _request_dict(request: ChatCompletionRequest) -> Dict[str, Any]:
        return request.model_dump(exclude_none=True, mode="json")

    @staticmethod
    def _ensure_text_only(messages: List[Dict[str, Any]]) -> None:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            unsupported = [
                part.get("type")
                for part in content
                if isinstance(part, dict) and part.get("type") != "text"
            ]
            if unsupported:
                raise ValueError(
                    "Kimi K3 RTP-LLM bring-up is text-only; unsupported content "
                    f"types: {unsupported}"
                )

    @staticmethod
    def _tools(request_dict: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        tools = request_dict.get("tools")
        if tools:
            return tools
        functions = request_dict.get("functions")
        if functions:
            return [
                {"type": "function", "function": function}
                for function in functions
            ]
        return None

    @staticmethod
    def _template_kwargs(
        request: ChatCompletionRequest, request_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if request.chat_template_kwargs:
            kwargs.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
        ):
            kwargs.update(request.extra_configs.chat_template_kwargs)

        # RTP's public spelling is enable_thinking; K3's tokenizer spelling is
        # simply thinking.  An explicit request flag wins over template kwargs.
        if "enable_thinking" in kwargs and "thinking" not in kwargs:
            kwargs["thinking"] = bool(kwargs.pop("enable_thinking"))
        else:
            kwargs.pop("enable_thinking", None)
        kwargs.setdefault("thinking", not request.disable_thinking())
        if request.enable_thinking is not None:
            kwargs["thinking"] = request.enable_thinking

        if request.reasoning_effort is not None:
            kwargs["thinking_effort"] = request.reasoning_effort
        if request_dict.get("tool_choice") is not None:
            kwargs["tool_choice"] = request_dict["tool_choice"]
        if request_dict.get("response_format") is not None:
            kwargs["response_format"] = request_dict["response_format"]
        return kwargs

    @staticmethod
    def _as_token_ids(value: Any) -> List[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(
                    "Kimi K3 renderer expected one conversation, got a token batch"
                )
            value = value[0]
        if not isinstance(value, list) or not all(
            isinstance(token_id, int) for token_id in value
        ):
            raise TypeError(
                "Kimi K3 tokenizer.apply_chat_template must return List[int] "
                f"for a single request, got {type(value).__name__}"
            )
        return value

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request_dict = self._request_dict(request)
        messages = request_dict["messages"]
        self._ensure_text_only(messages)
        tools = self._tools(request_dict)
        template_kwargs = self._template_kwargs(request, request_dict)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not isinstance(rendered_prompt, str):
            raise TypeError(
                "Kimi K3 tokenizer.apply_chat_template must return str when "
                f"tokenize=False, got {type(rendered_prompt).__name__}"
            )
        token_ids = self._as_token_ids(input_ids)
        logging.debug("Kimi K3 rendered %d XTML prompt tokens", len(token_ids))
        return RenderedInputs(input_ids=token_ids, rendered_prompt=rendered_prompt)

    @override
    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        del generate_config
        tool_choice = request.tool_choice
        if tool_choice is None or tool_choice in ("auto", "none", "required"):
            return
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS,
            "Kimi K3 currently supports tool_choice='auto', 'none', or "
            "'required'; named tool_choice is not implemented",
        )


register_renderer("kimi_k3", KimiK3Renderer)
