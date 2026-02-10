import importlib.util
import logging
import os
import sys
from typing import Optional

from typing_extensions import override

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs, RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class DeepseekV32Renderer(ReasoningToolBaseRenderer):
    """DeepSeek V3.2 Renderer

    This renderer uses a dedicated Python encoding script instead of Jinja templates.
    The encoding script is loaded from the checkpoint's "encoding" folder.

    Key features:
    1. Loads encoding_dsv32.py from checkpoint/encoding folder
    2. Uses encode_messages function for rendering
    3. Supports thinking mode and tool calls
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
    ):
        # Load the encoding module before calling super().__init__()
        self.encoding_module = self._load_encoding_module(renderer_params.ckpt_path)
        super().__init__(tokenizer, renderer_params)

    def _load_encoding_module(self, ckpt_path: str):
        """
        Load the encoding_dsv32.py module from the checkpoint's encoding folder.

        Args:
            ckpt_path: Path to the checkpoint directory

        Returns:
            The loaded encoding module

        Raises:
            FileNotFoundError: If the encoding script is not found
            ImportError: If the encoding script cannot be loaded
        """
        encoding_folder = os.path.join(ckpt_path, "encoding")
        encoding_script_path = os.path.join(encoding_folder, "encoding_dsv32.py")

        if not os.path.exists(encoding_script_path):
            raise FileNotFoundError(
                f"DeepSeek V3.2 encoding script not found at {encoding_script_path}. "
                f"Please ensure the checkpoint includes the 'encoding' folder with encoding_dsv32.py"
            )

        try:
            spec = importlib.util.spec_from_file_location(
                "encoding_dsv32", encoding_script_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {encoding_script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["encoding_dsv32"] = module
            spec.loader.exec_module(module)

            logging.info(
                f"Successfully loaded DeepSeek V3.2 encoding module from {encoding_script_path}"
            )
            return module
        except Exception as e:
            raise ImportError(
                f"Failed to load DeepSeek V3.2 encoding module from {encoding_script_path}: {str(e)}"
            )

    @override
    def _setup_chat_template(self):
        """
        DeepSeek V3.2 doesn't use Jinja templates.
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

    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        """
        Build prompt string using the DeepSeek V3.2 encoding script.

        Args:
            request: Chat completion request

        Returns:
            str: Rendered prompt string
        """
        # Convert request messages to the format expected by encoding_dsv32
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
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Add reasoning_content if present
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                message_dict["reasoning_content"] = msg.reasoning_content

            messages.append(message_dict)

        # Add tools from request level to the first system message
        # According to encoding_dsv32 format, tools must be attached to a system message
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

        # Filter encode_config to only include parameters accepted by encode_messages()
        # Valid parameters: thinking_mode, context, drop_thinking, add_default_bos_token
        valid_params = {
            "thinking_mode",
            "context",
            "drop_thinking",
            "add_default_bos_token",
        }
        filtered_config = {k: v for k, v in encode_config.items() if k in valid_params}

        try:
            # Use the encoding module to encode messages
            rendered_prompt = self.encoding_module.encode_messages(
                messages, **filtered_config
            )

            logging.debug(
                f"DeepSeek V3.2 rendered prompt (thinking_mode={thinking_mode}): {rendered_prompt[:200]}..."
            )

            return rendered_prompt
        except Exception as e:
            logging.error(f"Failed to render DeepSeek V3.2 prompt: {str(e)}")
            raise ValueError(f"Error rendering DeepSeek V3.2 prompt: {str(e)}")

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        """
        Render chat messages using the DeepSeek V3.2 encoding script.

        Args:
            request: Chat completion request

        Returns:
            RenderedInputs with encoded token IDs and rendered prompt
        """
        prompt = self._build_prompt(request)
        input_ids = self.tokenizer.encode(prompt)
        return RenderedInputs(input_ids=input_ids, rendered_prompt=prompt)

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """
        Create DSML format detector for tool calls.

        Args:
            request: Chat completion request

        Returns:
            DeepSeekV32Detector if tools are present, None otherwise
        """
        if request.tools:
            # Determine thinking_mode based on whether request is in thinking mode
            thinking_mode = "thinking" if self.in_think_mode(request) else "chat"

            # Pass the encoding module and thinking_mode to detector
            # Detector is created fresh for each request (not singleton)
            return DeepSeekV32Detector(
                encoding_module=self.encoding_module, thinking_mode=thinking_mode
            )
        return None

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


# register_renderer("deepseek_v32", DeepseekV32Renderer)
