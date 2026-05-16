import importlib.util
import json
import logging
import os
import sys
from typing import Any, Optional, Tuple

from typing_extensions import override

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import ChatCompletionRequest, DeltaMessage
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import OutputDelta, RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    rtp_tools_to_sglang_tools,
    streaming_parse_result_to_tool_calls,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.utils.base_model_datatypes import GenerateOutput


def _dsv4_renderer_debug_enabled() -> bool:
    return os.environ.get("RTP_LLM_DSV4_RENDERER_DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _preview_text(text: str, limit: int = 512) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}...[{len(text)} chars]...{text[-half:]}"


def _split_reasoning_before_dsml(text: str) -> Optional[Tuple[str, str]]:
    marker = "<｜DSML｜tool_calls>"
    idx = text.find(marker)
    if idx == -1:
        return None

    reasoning_text = (
        text[:idx].replace("<think>", "").replace("</think>", "").strip()
    )
    return reasoning_text, text[idx:]


def _longest_suffix_prefix(text: str, tokens: list[str]) -> int:
    max_len = 0
    for token in tokens:
        for length in range(1, min(len(text), len(token) - 1) + 1):
            if token.startswith(text[-length:]):
                max_len = max(max_len, length)
    return max_len


class DeepseekV4Renderer(ReasoningToolBaseRenderer):
    """DeepSeek V4.0 Renderer

    This renderer uses a dedicated Python encoding script instead of Jinja templates.
    The encoding script is loaded from the checkpoint's "encoding" folder.

    Key features:
    1. Loads encoding_dsv4.py from checkpoint/encoding folder
    2. Uses encode_messages function for rendering
    3. Supports thinking mode and tool calls
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        # Load the encoding module before calling super().__init__()
        self.encoding_module = self._load_encoding_module(renderer_params.ckpt_path)
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )

    def _load_encoding_module(self, ckpt_path: str):
        """
        Load the encoding_dsv4.py module from the checkpoint's encoding folder.

        Args:
            ckpt_path: Path to the checkpoint directory

        Returns:
            The loaded encoding module

        Raises:
            FileNotFoundError: If the encoding script is not found
            ImportError: If the encoding script cannot be loaded
        """
        encoding_folder = os.path.join(ckpt_path, "encoding")
        encoding_script_path = os.path.join(encoding_folder, "encoding_dsv4.py")

        if not os.path.exists(encoding_script_path):
            raise FileNotFoundError(
                f"DeepSeek V4.0 encoding script not found at {encoding_script_path}. "
                f"Please ensure the checkpoint includes the 'encoding' folder with encoding_dsv4.py"
            )

        try:
            spec = importlib.util.spec_from_file_location(
                "encoding_dsv4", encoding_script_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {encoding_script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["encoding_dsv4"] = module
            spec.loader.exec_module(module)

            logging.info(
                f"Successfully loaded DeepSeek V4.0 encoding module from {encoding_script_path}"
            )
            return module
        except Exception as e:
            raise ImportError(
                f"Failed to load DeepSeek V4.0 encoding module from {encoding_script_path}: {str(e)}"
            )

    @override
    def _setup_chat_template(self, template_file_name: str = "chat_template.jinja"):
        """
        DeepSeek V4.0 doesn't use Jinja templates.
        The chat_template attribute is set to None to indicate custom rendering.
        """
        self.chat_template = None

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        """
        Check if thinking mode is enabled.

        Supports both parent class logic and per-request enable_thinking parameter.

        Args:
            request: Chat completion request

        Returns:
            True if thinking mode is enabled, False otherwise
        """
        # Check parent class logic first
        thinking_enabled = super().in_think_mode(request)

        # Check if enable_thinking is explicitly set in request kwargs
        if request.chat_template_kwargs and request.chat_template_kwargs.get(
            "enable_thinking"
        ):
            thinking_enabled = True
        if (
            request.extra_configs
            and request.extra_configs.chat_template_kwargs
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
            and request.extra_configs.chat_template_kwargs.get("enable_thinking")
        ):
            thinking_enabled = True

        return thinking_enabled

    def _normalize_reasoning_effort(self, effort: Any) -> Optional[str]:
        if not isinstance(effort, str) or effort == "none":
            return None
        if effort in ("max", "xhigh"):
            return "max"
        return "high"

    def _normalize_tool_arguments(self, arguments: Any) -> str:
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        """
        Build prompt string using the DeepSeek V4.0 encoding script.

        Args:
            request: Chat completion request

        Returns:
            str: Rendered prompt string
        """
        # Convert request messages to the format expected by encoding_dsv4
        messages = []
        for msg in request.messages:
            message_dict = {"role": msg.role.value, "content": msg.content}

            # Add tool_calls if present (on assistant messages)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": self._normalize_tool_arguments(
                                tc.function.arguments
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Add reasoning_content if present
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                message_dict["reasoning_content"] = msg.reasoning_content

            messages.append(message_dict)

        # Add tools from request level to the first system message
        # According to encoding_dsv4 format, tools must be attached to a system message
        if request.tools:
            tools_data = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    },
                }
                for tool in request.tools
            ]

            # Find the first system message and add tools to it
            has_system = False
            for msg in messages:
                if msg["role"] == "system":
                    msg["tools"] = tools_data
                    has_system = True
                    break

            # If no system message exists, create one with tools
            if not has_system:
                messages.insert(
                    0, {"role": "system", "content": "", "tools": tools_data}
                )

        # Determine thinking mode
        thinking_mode = "thinking" if self.in_think_mode(request) else "chat"

        # Configure encoding
        # drop_thinking=True: Remove reasoning_content from historical assistant messages
        # add_default_bos_token=True: Always add BOS token since we encode full messages
        encode_config = {
            "thinking_mode": thinking_mode,
            "drop_thinking": True,
            "add_default_bos_token": True,
            "reasoning_effort": self._normalize_reasoning_effort(
                request.reasoning_effort
            ),
        }

        # Override with custom configs if provided
        # Note: context parameter is not used since RTP-LLM always provides full message history
        if request.chat_template_kwargs:
            encode_config.update(request.chat_template_kwargs)

        if (
            request.extra_configs
            and request.extra_configs.chat_template_kwargs
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
        ):
            encode_config.update(request.extra_configs.chat_template_kwargs)

        encode_config["reasoning_effort"] = self._normalize_reasoning_effort(
            encode_config.get("reasoning_effort")
        )

        # Filter encode_config to only include parameters accepted by encode_messages()
        # Valid parameters: thinking_mode, context, drop_thinking,
        # add_default_bos_token, reasoning_effort
        valid_params = {
            "thinking_mode",
            "context",
            "drop_thinking",
            "add_default_bos_token",
            "reasoning_effort",
        }
        filtered_config = {k: v for k, v in encode_config.items() if k in valid_params}

        try:
            # Use the encoding module to encode messages
            rendered_prompt = self.encoding_module.encode_messages(
                messages, **filtered_config
            )

            if _dsv4_renderer_debug_enabled():
                logging.info(
                    "[DeepSeekV4RendererDebug] render_prompt "
                    "messages=%d tools=%d thinking_mode=%s filtered_config=%s "
                    "prompt_len=%d prompt_tail=%s",
                    len(messages),
                    len(request.tools or []),
                    thinking_mode,
                    filtered_config,
                    len(rendered_prompt),
                    _preview_text(rendered_prompt[-1024:]),
                )

            logging.debug(
                f"DeepSeek V4.0 rendered prompt (thinking_mode={thinking_mode}): {rendered_prompt[:200]}..."
            )

            return rendered_prompt
        except Exception as e:
            logging.error(f"Failed to render DeepSeek V4.0 prompt: {str(e)}")
            raise ValueError(f"Error rendering DeepSeek V4.0 prompt: {str(e)}")

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """
        Create DSML format detector for tool calls.

        Args:
            request: Chat completion request

        Returns:
            DeepSeekV4Detector if tools are present, None otherwise
        """
        if request.tools:
            # Determine thinking_mode based on whether request is in thinking mode
            thinking_mode = "thinking" if self.in_think_mode(request) else "chat"

            # Pass the encoding module and thinking_mode to detector
            # Detector is created fresh for each request (not singleton)
            return DeepSeekV4Detector(
                encoding_module=self.encoding_module, thinking_mode=thinking_mode
            )
        return None

    def _thinking_mode_for_request(self, request: ChatCompletionRequest) -> str:
        return "thinking" if self.in_think_mode(request) else "chat"

    def _convert_official_tool_calls(
        self,
        detector: Optional[BaseFormatDetector],
        tools: Any,
        parsed_tool_calls: Any,
    ):
        if not detector or not tools or not parsed_tool_calls:
            return None

        sglang_tools = rtp_tools_to_sglang_tools(tools)
        calls = []
        for tool_call in parsed_tool_calls:
            function = tool_call.get("function", {})
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_item = {
                "name": function.get("name"),
                "parameters": arguments,
            }
            calls.extend(
                detector.parse_base_json(
                    tool_item, sglang_tools, start_index=len(calls)
                )
            )

        tool_calls, _ = streaming_parse_result_to_tool_calls(
            StreamingParseResult(calls=calls)
        )
        return tool_calls or None

    async def _parse_full_dsv4_completion(
        self,
        status: ReasoningToolStreamStatus,
        output: GenerateOutput,
    ) -> Optional[OutputDelta]:
        raw_text = status.delta_output_string
        if (
            not raw_text
            or not self.encoding_module
            or not hasattr(self.encoding_module, "parse_message_from_completion_text")
        ):
            return None

        thinking_mode = self._thinking_mode_for_request(status.request)
        try:
            parsed_msg = self.encoding_module.parse_message_from_completion_text(
                raw_text, thinking_mode
            )
        except Exception as e:
            if _dsv4_renderer_debug_enabled():
                logging.warning(
                    "[DeepSeekV4RendererDebug] official_full_parse_failed "
                    "thinking_mode=%s error=%r raw_preview=%s",
                    thinking_mode,
                    e,
                    _preview_text(raw_text),
                )
            return None

        try:
            content = parsed_msg.get("content", "")
            reasoning_content = parsed_msg.get(
                "reasoning_content", parsed_msg.get("reasoning", "")
            )
            tool_calls = self._convert_official_tool_calls(
                status.detector,
                status.request.tools,
                parsed_msg.get("tool_calls", []),
            )
        except Exception as e:
            if _dsv4_renderer_debug_enabled():
                logging.warning(
                    "[DeepSeekV4RendererDebug] official_full_parse_convert_failed "
                    "error=%r parsed_msg=%s",
                    e,
                    _preview_text(str(parsed_msg)),
                )
            return None

        if _dsv4_renderer_debug_enabled():
            logging.info(
                "[DeepSeekV4RendererDebug] official_full_parse_ok "
                "thinking_mode=%s reasoning_len=%d content_len=%d tool_calls=%d",
                thinking_mode,
                len(reasoning_content or ""),
                len(content or ""),
                len(tool_calls or []),
            )

        if tool_calls:
            status.generating_tool_call = True

        status.delta_output_string = ""
        return OutputDelta(
            output_str=DeltaMessage(
                content=content or None,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content or None,
            ),
            logprobs=await self._generate_log_probs(status, output),
            input_length=output.aux_info.input_len,
            output_length=output.aux_info.output_len,
            reuse_length=output.aux_info.reuse_len,
        )

    @override
    async def _process_reasoning_and_tool_calls(
        self,
        status: ReasoningToolStreamStatus,
        output: GenerateOutput,
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        if not is_streaming:
            parsed_delta = await self._parse_full_dsv4_completion(status, output)
            if parsed_delta is not None:
                return parsed_delta

        return await super()._process_reasoning_and_tool_calls(
            status, output, is_streaming
        )

    def _extract_streaming_reasoning_content(
        self,
        reasoning_parser: ReasoningParser,
        text: str,
    ) -> Tuple[str, str]:
        detector = getattr(reasoning_parser, "detector", None)
        if detector is None:
            return reasoning_parser.parse_stream_chunk(text)

        think_start = getattr(detector, "think_start_token", "<think>")
        think_end = getattr(detector, "think_end_token", "</think>")
        dsml_start = "<｜DSML｜tool_calls>"

        detector._buffer += text
        current_text = detector._buffer

        if not getattr(detector, "stripped_think_start", False):
            if think_start in current_text:
                current_text = current_text.replace(think_start, "", 1)
                detector.stripped_think_start = True
                detector._in_reasoning = True

        if not getattr(detector, "_in_reasoning", False):
            if getattr(detector, "_dsv4_after_think", False):
                current_text = current_text.lstrip()
                if not current_text:
                    detector._buffer = ""
                    return "", ""
                detector._buffer = current_text
                detector._dsv4_after_think = False

            hold_len = _longest_suffix_prefix(
                current_text, [think_start, think_end, dsml_start]
            )
            if hold_len:
                detector._buffer = current_text[-hold_len:]
                return "", current_text[:-hold_len]
            detector._buffer = ""
            return "", current_text

        end_idx = current_text.find(think_end)
        dsml_idx = current_text.find(dsml_start)
        if end_idx != -1 and (dsml_idx == -1 or end_idx < dsml_idx):
            reasoning_text = current_text[:end_idx].rstrip()
            normal_text = current_text[end_idx + len(think_end) :].lstrip()
            detector._buffer = ""
            detector._in_reasoning = False
            detector._dsv4_after_think = True
            return reasoning_text, normal_text

        if dsml_idx != -1:
            if _dsv4_renderer_debug_enabled():
                logging.warning(
                    "[DeepSeekV4RendererDebug] implicit_think_end_before_dsml "
                    "text_preview=%s",
                    _preview_text(current_text),
                )
            reasoning_text = (
                current_text[:dsml_idx]
                .replace(think_start, "")
                .replace(think_end, "")
                .strip()
            )
            normal_text = current_text[dsml_idx:]
            detector._buffer = ""
            detector._in_reasoning = False
            return reasoning_text, normal_text

        hold_len = _longest_suffix_prefix(current_text, [think_end, dsml_start])
        if getattr(detector, "stream_reasoning", True):
            if hold_len:
                reasoning_text = current_text[:-hold_len]
                detector._buffer = current_text[-hold_len:]
            else:
                reasoning_text = current_text
                detector._buffer = ""
            return reasoning_text, ""

        return "", ""

    @override
    def _extract_reasoning_content(
        self,
        reasoning_parser: Optional[ReasoningParser],
        text: str,
        is_streaming: bool,
    ) -> Tuple[str, str]:
        if is_streaming and reasoning_parser:
            reasoning_text, remaining_text = self._extract_streaming_reasoning_content(
                reasoning_parser, text
            )
        else:
            reasoning_text, remaining_text = super()._extract_reasoning_content(
                reasoning_parser, text, is_streaming
            )
        if (
            reasoning_parser
            and "<｜DSML｜tool_calls>" in text
            and "<｜DSML｜tool_calls>" not in remaining_text
        ):
            split_result = _split_reasoning_before_dsml(text)
            if split_result is not None:
                reasoning_text, remaining_text = split_result

        if _dsv4_renderer_debug_enabled():
            text_has_dsml = "<｜DSML｜" in text
            remaining_has_dsml = "<｜DSML｜" in remaining_text
            logging.info(
                "[DeepSeekV4RendererDebug] reasoning_extract "
                "streaming=%s parser=%s input_len=%d reasoning_len=%d "
                "remaining_len=%d input_has_dsml=%s remaining_has_dsml=%s "
                "remaining_preview=%s",
                is_streaming,
                type(reasoning_parser).__name__ if reasoning_parser else None,
                len(text),
                len(reasoning_text),
                len(remaining_text),
                text_has_dsml,
                remaining_has_dsml,
                _preview_text(remaining_text),
            )
            if text_has_dsml and not remaining_has_dsml:
                logging.warning(
                    "[DeepSeekV4RendererDebug] reasoning parser consumed DSML "
                    "tool markup; input_preview=%s",
                    _preview_text(text),
                )
        return reasoning_text, remaining_text

    @override
    async def _extract_tool_calls_content(
        self,
        detector: Optional[BaseFormatDetector],
        tools: Any,
        text: str,
        is_streaming: bool,
    ):
        tool_calls, remaining_text = await super()._extract_tool_calls_content(
            detector, tools, text, is_streaming
        )
        if _dsv4_renderer_debug_enabled():
            logging.info(
                "[DeepSeekV4RendererDebug] tool_extract "
                "streaming=%s detector=%s input_len=%d input_has_dsml=%s "
                "tool_calls=%d remaining_len=%d remaining_preview=%s",
                is_streaming,
                type(detector).__name__ if detector else None,
                len(text),
                "<｜DSML｜" in text,
                len(tool_calls or []),
                len(remaining_text),
                _preview_text(remaining_text),
            )
        return tool_calls, remaining_text

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        """
        Create reasoning parser if in thinking mode.

        Args:
            request: Chat completion request

        Returns:
            ReasoningParser if thinking mode is enabled, None otherwise
        """
        if not self.in_think_mode(request):
            return None

        try:
            # Check if the rendered prompt should use thinking mode
            rendered_result = self.render_chat(request)
            if "<think>" in rendered_result.rendered_prompt:
                return ReasoningParser(model_type="deepseek-v3", force_reasoning=True)
        except Exception:
            return None

        return None


register_renderer("deepseek_v4", DeepseekV4Renderer)
